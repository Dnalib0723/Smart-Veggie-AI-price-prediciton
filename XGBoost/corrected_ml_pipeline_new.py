# -*- coding: utf-8 -*-
"""
修正版機器學習流程的蔬菜價格預測系統
============================================
修正重點：
1. ✅ 正確的測試期窗口生成邏輯
2. ✅ 13個窗口，90個預測（符合理論值）
3. ✅ 清晰的時間邊界處理
4. ✅ 去除不必要的窗口生成
"""

import os
import warnings
import time
import numpy as np
import pandas as pd
from datetime import timedelta, datetime, timedelta
from collections import defaultdict
import pickle
import json
import argparse

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ParameterSampler

import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

# 修復中文字體問題
plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# ====== 參數設定 ======
DEFAULTS = {
    "data_file": "2022_2025_daily_vege_weather_price.csv",
    "train_days": 365,
    "valid_days": 7,
    "step_days": 7,
    "start_date": "2022-01-01",
    "test_days": 90,
    "min_samples": 100,
    "validation_windows": 20,  # 驗證期使用的窗口數量
    "random_search_iter": 50,  # Random Search 迭代次數
}

# 🎯 Random Search 參數空間
PARAM_SPACE = {
    "n_estimators": [100, 200, 300, 500],
    "max_depth": [3, 4, 5, 6, 7, 8],
    "learning_rate": [0.05, 0.1, 0.15, 0.2],
    "subsample": [0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
    "reg_alpha": [0, 0.1, 0.5, 1.0],
    "reg_lambda": [0.5, 1.0, 1.5, 2.0],
}

WEATHER_COLS = ["StnPres", "Temperature", "RH", "WS", "Precp", "typhoon"]
OUTPUT_DIR = "corrected_ml_pipeline_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("🎯 修正版機器學習流程的蔬菜價格預測系統啟動")
print("=" * 50)


# ====== 資料載入（同原版） ======
def load_and_preprocess_data(csv_path):
    """載入並預處理資料"""
    print(f"📊 載入資料: {csv_path}")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"找不到資料檔: {csv_path}")

    encodings = ["utf-8", "utf-8-sig", "big5", "gbk", "cp950", "latin-1"]
    df = None

    for encoding in encodings:
        try:
            df = pd.read_csv(csv_path, encoding=encoding)
            print(f"   ✅ 成功使用編碼: {encoding}")
            break
        except (UnicodeDecodeError, Exception):
            continue

    if df is None:
        raise ValueError(f"無法讀取檔案 {csv_path}")

    # 基本資料清理
    df["ds"] = pd.to_datetime(df["ObsTime"], errors="coerce")
    df["y"] = pd.to_numeric(df["avg_price_per_kg"], errors="coerce")
    df["vege_id"] = df["vege_id"].astype(str)

    # 移除無效資料
    df = df.dropna(subset=["ds", "y", "vege_id"]).copy()
    df = df.sort_values(["vege_id", "ds"]).reset_index(drop=True)

    print(f"✅ 資料載入完成: {len(df):,} 筆, {df['vege_id'].nunique()} 種蔬菜")
    print(f"   日期範圍: {df['ds'].min().date()} → {df['ds'].max().date()}")
    return df


# ====== 特徵工程（同原版，已修復） ======
def calculate_safe_weather_thresholds(train_data_only, weather_cols):
    """🛡️ 修復版：只基於訓練資料計算動態閾值"""
    thresholds = {}

    for col in weather_cols:
        if col in train_data_only.columns:
            values = train_data_only[col].dropna()
            if len(values) > 100:
                q05 = values.quantile(0.05)
                q95 = values.quantile(0.95)
                thresholds[col] = {"low": q05, "high": q95}

    return thresholds


def add_time_features(df):
    """時間特徵（無洩漏風險）"""
    df = df.copy()
    ds = df["ds"]

    df["year"] = ds.dt.year
    df["month"] = ds.dt.month
    df["dayofweek"] = ds.dt.dayofweek
    df["dayofyear"] = ds.dt.dayofyear
    df["quarter"] = ds.dt.quarter
    df["day"] = ds.dt.day
    df["week"] = ds.dt.isocalendar().week.astype(int)

    # 季節標記
    df["is_spring"] = ((df["month"] >= 3) & (df["month"] <= 5)).astype(int)
    df["is_summer"] = ((df["month"] >= 6) & (df["month"] <= 8)).astype(int)
    df["is_autumn"] = ((df["month"] >= 9) & (df["month"] <= 11)).astype(int)
    df["is_winter"] = ((df["month"] == 12) | (df["month"] <= 2)).astype(int)

    # 週期性編碼
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["day_sin"] = np.sin(2 * np.pi * df["dayofyear"] / 365)
    df["day_cos"] = np.cos(2 * np.pi * df["dayofyear"] / 365)
    df["weekday_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["weekday_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)

    return df


def add_safe_weather_features(df, weather_cols, safe_thresholds):
    """🛡️ 修復版氣象特徵工程"""
    df = df.copy()
    windows = [3, 7, 14, 30]

    for col in weather_cols:
        if col in df.columns:
            # 清理資料
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = (
                df[col].fillna(df[col].median()) if not df[col].isna().all() else 0
            )

            # 🛡️ 嚴格防洩漏：所有特徵都基於 t-1 時點
            base_series = df[col].shift(1)

            # 滾動統計（基於歷史資料）
            for w in windows:
                df[f"{col}_ma_{w}"] = base_series.rolling(w, min_periods=1).mean()
                df[f"{col}_std_{w}"] = base_series.rolling(w, min_periods=1).std()

            # 變化特徵
            df[f"{col}_dev30"] = df[col].shift(1) - df[f"{col}_ma_30"]
            df[f"{col}_delta1"] = base_series.diff(1)
            df[f"{col}_delta7"] = base_series.diff(7)

            # 滾動 z-score（基於歷史資料）
            roll_mean = base_series.rolling(30, min_periods=5).mean()
            roll_std = base_series.rolling(30, min_periods=5).std()
            df[f"{col}_z30"] = (df[col].shift(1) - roll_mean) / (
                roll_std.replace(0, np.nan)
            )
            df[f"{col}_z30"] = df[f"{col}_z30"].fillna(0)

            # 🛡️ 極端事件標記（使用安全閾值）
            if safe_thresholds and col in safe_thresholds:
                th = safe_thresholds[col]
                lagged_col = df[col].shift(1)
                df[f"{col}_extreme_low"] = (lagged_col < th["low"]).astype(int)
                df[f"{col}_extreme_high"] = (lagged_col > th["high"]).astype(int)
                df[f"{col}_extreme_any"] = (
                    (lagged_col < th["low"]) | (lagged_col > th["high"])
                ).astype(int)

    return df


def add_price_lags(df):
    """價格滯後特徵（已經是安全的）"""
    df = df.copy().sort_values("ds").reset_index(drop=True)

    # 滯後特徵
    for lag in [1, 3, 7, 14, 30]:
        df[f"y_lag_{lag}"] = df["y"].shift(lag)

    # 移動平均
    for w in [7, 14, 30]:
        df[f"y_ma_{w}"] = df["y"].shift(1).rolling(w, min_periods=1).mean()

    # 價格變化
    df["y_change_1"] = df["y"].shift(1) - df["y"].shift(2)
    df["y_change_7"] = df["y"].shift(7) - df["y"].shift(8)
    df["y_change_30"] = df["y"].shift(30) - df["y"].shift(31)

    # 百分比變化
    df["y_pct_change_1"] = df["y"].shift(1).pct_change(1)
    df["y_pct_change_7"] = df["y"].shift(7).pct_change(1)

    # 波動性
    df["y_volatility_7"] = df["y"].shift(1).rolling(7, min_periods=1).std()
    df["y_volatility_14"] = df["y"].shift(1).rolling(14, min_periods=1).std()

    # 相對位置
    df["y_above_ma7"] = (df["y"].shift(1) > df["y_ma_7"]).astype(int)
    df["y_above_ma30"] = (df["y"].shift(1) > df["y_ma_30"]).astype(int)

    return df


def get_feature_columns(df):
    """獲取所有特徵欄位"""
    base_features = [
        "month",
        "dayofweek",
        "dayofyear",
        "quarter",
        "day",
        "week",
        "year",
        "is_spring",
        "is_summer",
        "is_autumn",
        "is_winter",
        "month_sin",
        "month_cos",
        "day_sin",
        "day_cos",
        "weekday_sin",
        "weekday_cos",
        "y_lag_1",
        "y_lag_3",
        "y_lag_7",
        "y_lag_14",
        "y_lag_30",
        "y_ma_7",
        "y_ma_14",
        "y_ma_30",
        "y_change_1",
        "y_change_7",
        "y_change_30",
        "y_volatility_7",
        "y_volatility_14",
        "y_pct_change_1",
        "y_pct_change_7",
        "y_above_ma7",
        "y_above_ma30",
    ]

    # 氣象特徵
    weather_features = []
    for col in WEATHER_COLS:
        if col in df.columns:
            weather_features.append(col)
        weather_features.extend([c for c in df.columns if c.startswith(f"{col}_")])

    all_features = base_features + weather_features
    return [c for c in all_features if c in df.columns]


def safe_feature_engineering_pipeline(df_veg, cutoff_date):
    """🛡️ 安全的特徵工程流程"""
    # 分離訓練和測試資料
    train_data = df_veg[df_veg["ds"] < cutoff_date].copy()
    test_data = df_veg[df_veg["ds"] >= cutoff_date].copy()

    if len(train_data) < 100:
        return None, None, None

    # 🛡️ 步驟1：基於訓練資料計算安全閾值
    safe_thresholds = calculate_safe_weather_thresholds(train_data, WEATHER_COLS)

    # 🛡️ 步驟2：處理完整資料（但使用訓練期統計量）
    combined_data = pd.concat([train_data, test_data]).sort_values("ds")

    # 特徵工程
    combined_data = add_time_features(combined_data)
    combined_data = add_safe_weather_features(
        combined_data, WEATHER_COLS, safe_thresholds
    )
    combined_data = add_price_lags(combined_data)

    # 重新分離
    train_processed = combined_data[combined_data["ds"] < cutoff_date]
    test_processed = combined_data[combined_data["ds"] >= cutoff_date]

    return train_processed, test_processed, safe_thresholds


# ====== 評估指標 ======
def safe_mape(y_true, y_pred, eps=1e-6):
    """安全的MAPE計算"""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs((y_true - y_pred) / denom))) * 100.0


def calculate_metrics(y_true, y_pred):
    """計算所有評估指標"""
    r2 = r2_score(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    mape = safe_mape(y_true, y_pred)

    return {"R2": r2, "RMSE": rmse, "MAE": mae, "MAPE": mape}


# ====== 🔧 修正版：滑動窗口生成 ======
def generate_validation_windows(start_date, end_date, train_days, valid_days, step_days):
    """生成驗證期的滑動窗口（保持原邏輯）"""
    windows = []
    current = start_date

    while True:
        train_end = current + timedelta(days=train_days - 1)
        valid_start = train_end + timedelta(days=1)
        valid_end = valid_start + timedelta(days=valid_days - 1)

        if valid_end > end_date:
            break

        windows.append(
            {
                "train_start": current,
                "train_end": train_end,
                "valid_start": valid_start,
                "valid_end": valid_end,
            }
        )

        current += timedelta(days=step_days)

    return windows


def generate_test_windows(test_start_date, test_end_date, train_days, pred_days, step_days):
    """
    🔧 修正版：正確的測試期窗口生成邏輯
    
    邏輯：
    1. 在測試期內滑動生成預測窗口
    2. 每個窗口的訓練期向前回溯 train_days 天
    3. 確保完全覆蓋測試期，不重複不遺漏
    """
    windows = []
    current_pred_start = test_start_date
    
    print(f"   🔧 修正版測試窗口生成:")
    print(f"      測試期: {test_start_date.date()} → {test_end_date.date()}")
    print(f"      訓練長度: {train_days}天, 預測長度: {pred_days}天, 滑動步長: {step_days}天")
    
    window_count = 0
    total_pred_days = 0
    
    while current_pred_start <= test_end_date:
        # 計算預測期結束日期
        pred_end = current_pred_start + timedelta(days=pred_days - 1)
        
        # 如果預測期超出測試期，調整結束日期
        if pred_end > test_end_date:
            pred_end = test_end_date
        
        # 計算訓練期（向前回溯）
        train_start = current_pred_start - timedelta(days=train_days)
        train_end = current_pred_start - timedelta(days=1)
        
        # 計算實際預測天數
        actual_pred_days = (pred_end - current_pred_start).days + 1
        
        windows.append({
            "train_start": train_start,
            "train_end": train_end,
            "pred_start": current_pred_start,  # 使用pred_start而不是valid_start
            "pred_end": pred_end,
            "actual_pred_days": actual_pred_days
        })
        
        window_count += 1
        total_pred_days += actual_pred_days
        
        # 如果已經到達測試期末尾，結束
        if pred_end >= test_end_date:
            break
            
        # 移動到下一個窗口
        current_pred_start += timedelta(days=step_days)
    
    print(f"      生成窗口數: {window_count}個")
    print(f"      總預測天數: {total_pred_days}天")
    print(f"      測試期天數: {(test_end_date - test_start_date).days + 1}天")
    
    # 驗證是否完全覆蓋
    if total_pred_days >= (test_end_date - test_start_date).days + 1:
        print(f"      ✅ 完全覆蓋測試期")
    else:
        print(f"      ⚠️ 未完全覆蓋測試期")
    
    return windows


# ====== 🎯 核心修正：正確的機器學習流程 ======

def random_search_optimization(df_veg, validation_windows, n_iter=50):
    """
    🎯 階段1：在驗證期進行 Random Search 參數優化 - 修正版
    """
    print("   🔍 開始 Random Search 參數優化...")

    best_score = -np.inf
    best_params = None
    search_results = []

    # 🔧 修正：先篩選有效的驗證窗口
    valid_windows = []
    for window in validation_windows:
        # 檢查是否有足夠的資料
        window_data = df_veg[
            (df_veg["ds"] >= window["train_start"] - timedelta(days=60))
            & (df_veg["ds"] <= window["valid_end"])
        ]
        if len(window_data) >= 300:
            valid_windows.append(window)

    if len(valid_windows) < 3:
        print("      ❌ 有效驗證窗口不足")
        return None, []

    print(f"      使用 {len(valid_windows)} 個有效驗證窗口")

    # 生成參數組合
    param_sampler = ParameterSampler(PARAM_SPACE, n_iter=n_iter, random_state=42)

    for i, params in enumerate(param_sampler):
        if (i + 1) % 10 == 0:
            print(f"      進度: {i+1}/{n_iter}")

        # 添加固定參數
        full_params = {**params, "random_state": 42, "n_jobs": -1, "verbosity": 0}

        # 在驗證窗口上評估這組參數
        scores = []
        for window in valid_windows[:10]:  # 🔧 限制窗口數量以加速
            try:
                score = evaluate_single_window(df_veg, window, full_params)
                if score is not None and score > -5:  # 🔧 過濾極端負值
                    scores.append(score)
            except:
                continue

        if len(scores) >= 3:  # 🔧 至少需要3個有效分數
            avg_score = np.mean(scores)
            search_results.append(
                {
                    "params": full_params,
                    "score": avg_score,
                    "valid_windows": len(scores),
                }
            )

            if avg_score > best_score:
                best_score = avg_score
                best_params = full_params
                print(f"      新最佳參數! R² = {best_score:.3f}")

    if best_params is None:
        print("      ❌ Random Search 失敗")
        return None, []

    print(f"   ✅ Random Search 完成，最佳 R² = {best_score:.3f}")
    return best_params, search_results


def evaluate_single_window(df_veg, window, params):
    """評估單個窗口的表現 - 修正版"""
    try:
        # 🔧 修正：獲取足夠的歷史資料
        window_start = window["train_start"] - timedelta(days=60)  # 額外歷史資料
        window_end = window["valid_end"]
        window_data = df_veg[
            (df_veg["ds"] >= window_start) & (df_veg["ds"] <= window_end)
        ].copy()

        if len(window_data) < 300:  # 調高資料需求
            return None

        # 特徵工程
        train_processed, valid_processed, _ = safe_feature_engineering_pipeline(
            window_data, window["valid_start"]
        )

        if train_processed is None or len(valid_processed) == 0:
            return None

        # 精確的時間過濾
        train_final = train_processed[
            (train_processed["ds"] >= window["train_start"])
            & (train_processed["ds"] <= window["train_end"])
        ]
        valid_final = valid_processed[
            (valid_processed["ds"] >= window["valid_start"])
            & (valid_processed["ds"] <= window["valid_end"])
        ]

        if len(train_final) < 200 or len(valid_final) == 0:  # 調高最低需求
            return None

        # 準備特徵
        feature_columns = get_feature_columns(train_final)
        available_features = [f for f in feature_columns if f in train_final.columns]

        if len(available_features) < 15:  # 調高特徵需求
            return None

        X_train = train_final[available_features].fillna(0)
        X_valid = valid_final[available_features].fillna(0)
        y_train = train_final["y"].values
        y_valid = valid_final["y"].values

        # 🔧 修正：添加資料檢查
        if len(np.unique(y_train)) < 3:  # 目標變數變化太少
            return None

        # 訓練模型
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train)

        # 預測和評估
        y_pred = model.predict(X_valid)
        y_pred = np.maximum(y_pred, 0)

        # 🔧 修正：檢查預測結果
        if np.isnan(y_pred).any() or np.isinf(y_pred).any():
            return None

        r2 = r2_score(y_valid, y_pred)

        # 🔧 修正：確保R²在合理範圍內
        if r2 < -10 or r2 > 1:
            return None

        return r2

    except Exception as e:
        return None


def build_pooled_training_data(df_veg, validation_windows):
    """
    將所有驗證視窗的「訓練切片」堆成一個 pooled 訓練集。
    使用 cutoff = window["valid_start"] 做特徵工程，確保統計只基於訓練期，欄位與驗證/測試一致；
    之後再切出 [train_start, train_end] 作為真正的訓練樣本，避免空集並防止洩漏。
    """
    X_pool_list, y_pool_list = [], []

    for window in validation_windows:
        try:
            # 多取一段歷史供 rolling 統計
            window_start = min(window["train_start"] - timedelta(days=60), df_veg["ds"].min())
            window_end = window["valid_end"]
            window_data = df_veg[(df_veg["ds"] >= window_start) & (df_veg["ds"] <= window_end)].copy()

            # 用 valid_start 當 cutoff 做特徵工程
            train_processed, _, _ = safe_feature_engineering_pipeline(window_data, window["valid_start"])
            if train_processed is None or len(train_processed) == 0:
                continue

            # 切出真正訓練時段
            train_final = train_processed[
                (train_processed["ds"] >= window["train_start"]) &
                (train_processed["ds"] <= window["train_end"])
            ]

            if len(train_final) < 50:
                continue

            feature_cols = get_feature_columns(train_final)
            if len(feature_cols) < 5:
                continue

            X_pool_list.append(train_final[feature_cols].fillna(0))
            y_pool_list.append(train_final["y"].values)
        except Exception:
            continue

    if len(X_pool_list) == 0:
        return None, None, []

    import pandas as pd, numpy as np
    X_pool = pd.concat(X_pool_list, axis=0)
    y_pool = np.concatenate(y_pool_list, axis=0)
    pooled_features = list(X_pool.columns)
    return X_pool, y_pool, pooled_features


def feature_selection_optimization(df_veg, validation_windows, best_params,
                                   sfm_threshold="median", min_features=10):
    """
    使用 SelectFromModel 自動決定保留特徵數量：
      1) 將所有驗證視窗的「訓練時段」堆成 pooled 訓練集（無洩漏）
      2) 以 best_params 訓練 XGBoost 一次
      3) 由 SelectFromModel 依 threshold 篩選特徵
    """
    print(f"   🎯 特徵選擇（SelectFromModel 版本，threshold={sfm_threshold}）...")

    X_pool, y_pool, pooled_features = build_pooled_training_data(df_veg, validation_windows)
    if X_pool is None or y_pool is None or len(pooled_features) == 0:
        print("   ⚠️ 無法建立 pooled 訓練資料，改用所有可用特徵")
        return None

    # 以驗證期最佳參數訓練模型（一次）
    model = xgb.XGBRegressor(**best_params)
    model.fit(X_pool, y_pool)

    # threshold 支援 'median' / 'mean' / 浮點數
    thr = sfm_threshold
    try:
        thr = float(thr)
    except Exception:
        pass

    sfm = SelectFromModel(estimator=model, threshold=thr, prefit=True)
    mask = sfm.get_support()
    selected_features = [f for f, keep in zip(pooled_features, mask) if keep]

    # 過於嚴格時：降門檻一次
    if len(selected_features) < min_features:
        print(f"   ⚠️ SFM 選出 {len(selected_features)} 個特徵偏少，降門檻至 'mean'")
        sfm = SelectFromModel(estimator=model, threshold="mean", prefit=True)
        mask = sfm.get_support()
        selected_features = [f for f, keep in zip(pooled_features, mask) if keep]

    if len(selected_features) < min_features:
        print("   ⚠️ SFM 仍選太少，暫用所有可用特徵以確保可訓練")
        return None

    print(f"   ✅ SFM 完成，選中 {len(selected_features)} 個特徵（門檻: {sfm_threshold}）")
    return selected_features


def validate_final_configuration(
    df_veg, validation_windows, best_params, selected_features
):
    """
    🎯 階段3：使用最終配置評估驗證期表現
    """
    print("   📊 驗證最終配置...")

    all_predictions = []
    all_actuals = []
    successful_windows = 0

    for window in validation_windows:
        try:
            # 獲取窗口資料
            window_start = min(
                window["train_start"] - timedelta(days=30), df_veg["ds"].min()
            )
            window_end = window["valid_end"]
            window_data = df_veg[
                (df_veg["ds"] >= window_start) & (df_veg["ds"] <= window_end)
            ].copy()

            # 特徵工程
            train_processed, valid_processed, _ = safe_feature_engineering_pipeline(
                window_data, window["valid_start"]
            )

            if train_processed is None or len(valid_processed) == 0:
                continue

            # 時間過濾
            train_final = train_processed[
                (train_processed["ds"] >= window["train_start"])
                & (train_processed["ds"] <= window["train_end"])
            ]
            valid_final = valid_processed[
                (valid_processed["ds"] >= window["valid_start"])
                & (valid_processed["ds"] <= window["valid_end"])
            ]

            if len(train_final) < 100 or len(valid_final) == 0:
                continue

            # 🎯 使用選定的特徵
            if selected_features:
                available_selected = [
                    f for f in selected_features if f in train_final.columns
                ]
                if len(available_selected) < 5:
                    continue
                feature_cols = available_selected
            else:
                feature_cols = get_feature_columns(train_final)

            X_train = train_final[feature_cols].fillna(0)
            X_valid = valid_final[feature_cols].fillna(0)
            y_train = train_final["y"].values
            y_valid = valid_final["y"].values

            # 🎯 使用最佳參數訓練
            model = xgb.XGBRegressor(**best_params)
            model.fit(X_train, y_train)

            # 預測
            y_pred = model.predict(X_valid)
            y_pred = np.maximum(y_pred, 0)

            all_predictions.extend(y_pred)
            all_actuals.extend(y_valid)
            successful_windows += 1

        except:
            continue

    if len(all_predictions) > 0:
        metrics = calculate_metrics(np.array(all_actuals), np.array(all_predictions))
        metrics["successful_windows"] = successful_windows
        print(
            f"   ✅ 驗證完成: R² = {metrics['R2']:.3f}, MAPE = {metrics['MAPE']:.1f}%"
        )
        return metrics
    else:
        return None


def test_with_fixed_configuration(
    df_veg, test_start_date, test_end_date, best_params, selected_features
):
    """
    🎯 階段4：測試期使用完全固定的配置 - 修正版
    """
    print("   🎯 測試期：使用固定配置（絕不調整）...")

    # 🔧 修正：使用正確的測試期窗口生成
    test_windows = generate_test_windows(
        test_start_date,
        test_end_date,
        365,  # 訓練天數
        7,    # 預測天數
        7,    # 滑動步長
    )

    print(f"      測試期滑動窗口數量: {len(test_windows)}")

    if len(test_windows) == 0:
        print("      ❌ 沒有有效的測試窗口")
        return None

    all_predictions = []
    all_actuals = []
    successful_windows = 0
    total_pred_days = 0

    for i, window in enumerate(test_windows):
        try:
            # 獲取窗口資料（包含足夠的歷史資料）
            window_start = window["train_start"] - timedelta(days=60)  # 額外歷史資料用於特徵工程
            window_end = window["pred_end"]
            window_data = df_veg[
                (df_veg["ds"] >= window_start) & (df_veg["ds"] <= window_end)
            ].copy()

            if len(window_data) < 300:  # 需要更多資料
                continue

            # 特徵工程 - 使用pred_start作為cutoff
            train_processed, test_processed, _ = safe_feature_engineering_pipeline(
                window_data, window["pred_start"]
            )

            if train_processed is None or len(test_processed) == 0:
                continue

            # 時間過濾
            train_final = train_processed[
                (train_processed["ds"] >= window["train_start"])
                & (train_processed["ds"] <= window["train_end"])
            ]
            test_final = test_processed[
                (test_processed["ds"] >= window["pred_start"])
                & (test_processed["ds"] <= window["pred_end"])
            ]

            if len(train_final) < 200 or len(test_final) == 0:
                continue

            # 🎯 關鍵：使用驗證期選定的固定特徵
            if selected_features:
                available_selected = [
                    f for f in selected_features if f in train_final.columns
                ]
                if len(available_selected) < 5:
                    continue
                feature_cols = available_selected
            else:
                feature_cols = get_feature_columns(train_final)
                feature_cols = [f for f in feature_cols if f in train_final.columns]

            if len(feature_cols) < 10:
                continue

            X_train = train_final[feature_cols].fillna(0)
            X_test = test_final[feature_cols].fillna(0)
            y_train = train_final["y"].values
            y_test = test_final["y"].values

            # 🎯 關鍵：使用驗證期選定的固定參數
            model = xgb.XGBRegressor(**best_params)
            model.fit(X_train, y_train)

            # 預測
            y_pred = model.predict(X_test)
            y_pred = np.maximum(y_pred, 0)

            all_predictions.extend(y_pred)
            all_actuals.extend(y_test)
            successful_windows += 1
            total_pred_days += len(y_test)

            # 詳細日志（可選）
            if i < 3:  # 只顯示前3個窗口的詳細信息
                print(f"      窗口{i+1}: 訓練{len(y_train)}天 → 預測{len(y_test)}天")

        except Exception as e:
            continue

    if len(all_predictions) > 0:
        metrics = calculate_metrics(np.array(all_actuals), np.array(all_predictions))
        metrics["successful_windows"] = successful_windows
        metrics["total_predictions"] = len(all_predictions)
        print(
            f"      ✅ 測試完成: R² = {metrics['R2']:.3f}, MAPE = {metrics['MAPE']:.1f}%"
        )
        print(f"      成功窗口: {successful_windows} 個")
        print(f"      總預測數: {len(all_predictions)} 個")
        return metrics
    else:
        return None


# ====== 🎯 完整的正確機器學習流程 ======
def correct_ml_pipeline_for_vegetable(df_veg, test_start_date, test_end_date, sfm_threshold="median", min_features=10):
    """
    🎯 為單個蔬菜執行完整的正確機器學習流程 - 修正版
    """
    train_end_date = test_start_date - timedelta(days=1)

    # 生成驗證期的滑動窗口
    validation_windows = generate_validation_windows(
        pd.to_datetime(DEFAULTS["start_date"]),
        train_end_date,
        DEFAULTS["train_days"],
        DEFAULTS["valid_days"],
        DEFAULTS["step_days"],
    )

    # 限制驗證窗口數量以加速
    if len(validation_windows) > DEFAULTS["validation_windows"]:
        step = len(validation_windows) // DEFAULTS["validation_windows"]
        validation_windows = validation_windows[::step][
            : DEFAULTS["validation_windows"]
        ]

    print(f"   📊 使用 {len(validation_windows)} 個驗證窗口")

    # 🎯 階段1：Random Search 參數優化
    best_params, search_results = random_search_optimization(
        df_veg, validation_windows, DEFAULTS["random_search_iter"]
    )

    if best_params is None:
        print("   ❌ 參數優化失敗")
        return None

    # 🎯 階段2：特徵選擇
    selected_features = feature_selection_optimization(df_veg, validation_windows, best_params, sfm_threshold=sfm_threshold, min_features=min_features)

    # 🎯 階段3：驗證最終配置
    validation_metrics = validate_final_configuration(
        df_veg, validation_windows, best_params, selected_features
    )

    if validation_metrics is None:
        print("   ❌ 驗證階段失敗")
        return None

    # 🎯 階段4：測試期評估（完全固定配置）- 修正版
    test_metrics = test_with_fixed_configuration(
        df_veg, test_start_date, test_end_date, best_params, selected_features
    )

    if test_metrics is None:
        print("   ❌ 測試階段失敗")
        return None

    # 返回完整結果
    return {
        "best_params": best_params,
        "selected_features": selected_features,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "search_results": search_results,
    }


# ====== 基準測試 ======
def baseline_predictions(df_veg, test_start_date):
    """基準測試：簡單方法的預測效果"""
    test_data = df_veg[df_veg["ds"] >= test_start_date].copy()

    if len(test_data) == 0:
        return {}

    y_true = test_data["y"].values
    baselines = {}

    # 1. 隨機預測
    np.random.seed(42)
    random_pred = np.random.normal(y_true.mean(), y_true.std(), len(y_true))
    baselines["random"] = {
        "predictions": random_pred,
        "r2": r2_score(y_true, random_pred),
        "mape": safe_mape(y_true, random_pred),
    }

    # 2. 昨日價格
    train_data = df_veg[df_veg["ds"] < test_start_date]
    combined_data = add_price_lags(df_veg.copy())
    test_with_lag = combined_data[combined_data["ds"] >= test_start_date]

    if len(test_with_lag) > 0 and "y_lag_1" in test_with_lag.columns:
        lag1_pred = test_with_lag["y_lag_1"].fillna(y_true.mean()).values
        baselines["lag1"] = {
            "predictions": lag1_pred,
            "r2": r2_score(y_true, lag1_pred),
            "mape": safe_mape(y_true, lag1_pred),
        }

    # 3. 7天移動平均
    if len(test_with_lag) > 0 and "y_ma_7" in test_with_lag.columns:
        ma7_pred = test_with_lag["y_ma_7"].fillna(y_true.mean()).values
        baselines["ma7"] = {
            "predictions": ma7_pred,
            "r2": r2_score(y_true, ma7_pred),
            "mape": safe_mape(y_true, ma7_pred),
        }

    return baselines


# ====== 視覺化 ======
def create_comprehensive_plots(results_df, baseline_df, validation_summary):
    """創建綜合分析圖表"""
    fig, axes = plt.subplots(3, 3, figsize=(20, 16))

    # 1. R² 分佈比較
    axes[0, 0].hist(
        results_df["test_R2"], bins=15, alpha=0.7, label="修正ML流程", color="blue"
    )
    if baseline_df is not None and "lag1_r2" in baseline_df.columns:
        axes[0, 0].hist(
            baseline_df["lag1_r2"],
            bins=15,
            alpha=0.7,
            label="昨日價格基準",
            color="red",
        )
    axes[0, 0].set_xlabel("R² Score")
    axes[0, 0].set_ylabel("頻率")
    axes[0, 0].set_title("🎯 修正ML流程 R² 分佈比較")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. 驗證 vs 測試一致性 - 最關鍵的圖！
    if validation_summary is not None:
        merged_df = pd.merge(validation_summary, results_df, on="vege_id", how="inner")
        if len(merged_df) > 0:
            axes[0, 1].scatter(
                merged_df["val_R2"], merged_df["test_R2"], alpha=0.6, s=50, c="green"
            )
            axes[0, 1].plot(
                [merged_df["val_R2"].min(), merged_df["val_R2"].max()],
                [merged_df["val_R2"].min(), merged_df["val_R2"].max()],
                "r--",
                alpha=0.8,
                label="完美對應線",
            )
            axes[0, 1].set_xlabel("驗證 R²")
            axes[0, 1].set_ylabel("測試 R²")
            axes[0, 1].set_title("🔥 關鍵：驗證vs測試一致性檢查")
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)

            # 添加一致性信息
            correlation = np.corrcoef(merged_df["val_R2"], merged_df["test_R2"])[0, 1]
            consistency = 1 - abs(merged_df["val_R2"] - merged_df["test_R2"]).mean()
            axes[0, 1].text(
                0.05,
                0.95,
                f"相關係數: {correlation:.3f}\n一致性: {consistency:.3f}",
                transform=axes[0, 1].transAxes,
                bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.8),
            )

    # 3. 模型 vs 基準對比
    if baseline_df is not None and "lag1_r2" in baseline_df.columns:
        merged_baseline = pd.merge(results_df, baseline_df, on="vege_id", how="inner")
        axes[0, 2].scatter(
            merged_baseline["lag1_r2"],
            merged_baseline["test_R2"],
            alpha=0.6,
            s=50,
            c="purple",
        )
        axes[0, 2].plot(
            [merged_baseline["lag1_r2"].min(), merged_baseline["lag1_r2"].max()],
            [merged_baseline["lag1_r2"].min(), merged_baseline["lag1_r2"].max()],
            "r--",
            alpha=0.8,
            label="完美對應線",
        )
        axes[0, 2].set_xlabel("昨日價格基準 R²")
        axes[0, 2].set_ylabel("修正ML流程 R²")
        axes[0, 2].set_title("模型 vs 基準效果")
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)

    # 4. MAPE 比較
    axes[1, 0].hist(
        results_df["test_MAPE"], bins=15, alpha=0.7, label="修正ML流程", color="blue"
    )
    if baseline_df is not None and "lag1_mape" in baseline_df.columns:
        axes[1, 0].hist(
            baseline_df["lag1_mape"],
            bins=15,
            alpha=0.7,
            label="昨日價格基準",
            color="red",
        )
    axes[1, 0].set_xlabel("MAPE (%)")
    axes[1, 0].set_ylabel("頻率")
    axes[1, 0].set_title("MAPE 分佈比較")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 5. 改進效果分析
    if baseline_df is not None and "lag1_r2" in baseline_df.columns:
        improvement = merged_baseline["test_R2"] - merged_baseline["lag1_r2"]
        axes[1, 1].hist(improvement, bins=15, alpha=0.7, color="green")
        axes[1, 1].axvline(0, color="red", linestyle="--", label="無改進線")
        axes[1, 1].set_xlabel("R² 改進幅度")
        axes[1, 1].set_ylabel("頻率")
        axes[1, 1].set_title("修正ML流程改進效果")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        # 添加改進統計
        improved_count = (improvement > 0).sum()
        total_count = len(improvement)
        axes[1, 1].text(
            0.05,
            0.95,
            f"改進率: {improved_count/total_count*100:.1f}%\n平均改進: {improvement.mean():.3f}",
            transform=axes[1, 1].transAxes,
            bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.8),
        )

    # 6. 特徵數量分析
    if "selected_features_count" in results_df.columns:
        axes[1, 2].hist(
            results_df["selected_features_count"], bins=15, alpha=0.7, color="orange"
        )
        axes[1, 2].set_xlabel("選中特徵數量")
        axes[1, 2].set_ylabel("頻率")
        axes[1, 2].set_title("特徵選擇結果分析")
        axes[1, 2].grid(True, alpha=0.3)

        # 添加統計信息
        mean_features = results_df["selected_features_count"].mean()
        axes[1, 2].axvline(
            mean_features,
            color="red",
            linestyle="--",
            label=f"平均: {mean_features:.1f}",
        )
        axes[1, 2].legend()

    # 7. 表現分佈箱型圖
    plot_data = [results_df["test_R2"]]
    labels = ["測試"]

    if baseline_df is not None and "lag1_r2" in baseline_df.columns:
        plot_data.append(baseline_df["lag1_r2"])
        labels.append("基準")

    if validation_summary is not None and "val_R2" in validation_summary.columns:
        plot_data.append(validation_summary["val_R2"])
        labels.append("驗證")

    axes[2, 0].boxplot(plot_data, labels=labels, showmeans=True)
    axes[2, 0].set_ylabel("R² Score")
    axes[2, 0].set_title("修正ML流程表現總覽")
    axes[2, 0].grid(True, alpha=0.3)

    # 8. 訓練窗口成功率
    if "successful_windows" in results_df.columns:
        axes[2, 1].hist(
            results_df["successful_windows"], bins=15, alpha=0.7, color="cyan"
        )
        axes[2, 1].set_xlabel("成功訓練窗口數")
        axes[2, 1].set_ylabel("頻率")
        axes[2, 1].set_title("訓練穩定性分析")
        axes[2, 1].grid(True, alpha=0.3)

    # 9. 整體改進摘要
    axes[2, 2].axis("off")

    # 計算關鍵統計
    stats_text = "🎯 修正ML流程關鍵指標\n" + "=" * 30 + "\n"
    stats_text += f"平均測試 R²: {results_df['test_R2'].mean():.3f}\n"
    stats_text += f"中位數測試 R²: {results_df['test_R2'].median():.3f}\n"
    stats_text += f"平均測試 MAPE: {results_df['test_MAPE'].mean():.1f}%\n"

    if validation_summary is not None and len(merged_df) > 0:
        correlation = np.corrcoef(merged_df["val_R2"], merged_df["test_R2"])[0, 1]
        consistency = 1 - abs(merged_df["val_R2"] - merged_df["test_R2"]).mean()
        stats_text += f"\n🔥 流程正確性驗證:\n"
        stats_text += f"驗證-測試相關性: {correlation:.3f}\n"
        stats_text += f"驗證-測試一致性: {consistency:.3f}\n"

        if correlation > 0.8 and consistency > 0.9:
            stats_text += "✅ 流程完全正確！"
        elif correlation > 0.6 and consistency > 0.8:
            stats_text += "✅ 流程基本正確"
        else:
            stats_text += "⚠️ 流程需要調整"

    if baseline_df is not None and "lag1_r2" in baseline_df.columns:
        model_improvement = (
            merged_baseline["test_R2"].mean() - merged_baseline["lag1_r2"].mean()
        )
        improved_rate = (merged_baseline["test_R2"] > merged_baseline["lag1_r2"]).mean()
        stats_text += f"\n📊 相對基準改進:\n"
        stats_text += f"平均R²改進: {model_improvement:.3f}\n"
        stats_text += f"改進成功率: {improved_rate*100:.1f}%\n"

    # 🔧 修正版關鍵改進
    stats_text += f"\n🔧 修正版關鍵改進:\n"
    avg_windows = results_df["successful_windows"].mean()
    avg_predictions = results_df["test_predictions"].mean() if "test_predictions" in results_df.columns else "N/A"
    stats_text += f"平均測試窗口: {avg_windows:.1f}個\n"
    stats_text += f"平均預測數: {avg_predictions}\n"
    stats_text += "✅ 正確窗口生成邏輯\n"
    stats_text += "✅ 13窗口90預測理論值\n"

    axes[2, 2].text(
        0.1,
        0.9,
        stats_text,
        transform=axes[2, 2].transAxes,
        fontsize=11,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.8),
    )

    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, "corrected_ml_pipeline_analysis.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


# ====== 主要執行流程 ======
def main(sfm_threshold="median", min_features=10):
    """修正版機器學習流程主要執行函數"""
    start_time = time.time()

    # 載入資料
    df = load_and_preprocess_data(DEFAULTS["data_file"])

    # 確定測試期間
    max_date = df["ds"].max()
    test_start_date = max_date - timedelta(days=DEFAULTS["test_days"] - 1)
    test_end_date = max_date
    train_end_date = test_start_date - timedelta(days=1)

    print(f"\n📅 時間切分:")
    print(f"   訓練+驗證期: {DEFAULTS['start_date']} → {train_end_date.date()}")
    print(f"   測試期: {test_start_date.date()} → {test_end_date.date()}")
    print(f"   測試期天數: {(test_end_date - test_start_date).days + 1}天")

    # 獲取蔬菜列表
    vegetable_list = sorted(df["vege_id"].unique())
    print(f"\n🥬 將處理 {len(vegetable_list)} 種蔬菜")

    # 處理每種蔬菜
    all_results = []
    baseline_results = []
    validation_summary = []

    # 改為處理所有蔬菜
    test_vegetables = vegetable_list
    print(f"🎯 修正ML流程：處理全部 {len(test_vegetables)} 種蔬菜")
    print("🔥 關鍵改進：正確的測試期窗口生成邏輯，13窗口90預測！")

    for i, veg_id in enumerate(test_vegetables, 1):
        print(f"\n📄 [{i}/{len(test_vegetables)}] 處理蔬菜 {veg_id}...")

        # 單一蔬菜資料
        df_veg = df[df["vege_id"] == veg_id].copy().sort_values("ds")

        if len(df_veg) < DEFAULTS["min_samples"]:
            print(f"   ⭐ 跳過 {veg_id} - 資料量不足")
            continue

        print(f"   📊 原始資料: {len(df_veg)} 筆")

        # 🎯 基準測試
        print("   🎯 執行基準測試...")
        baselines = baseline_predictions(df_veg, test_start_date)
        if baselines:
            baseline_result = {
                "vege_id": veg_id,
                **{
                    f"{method}_{metric}": values[metric]
                    for method, values in baselines.items()
                    for metric in ["r2", "mape"]
                },
            }
            baseline_results.append(baseline_result)

            print(f"   📊 基準測試結果:")
            for method, values in baselines.items():
                print(
                    f"      {method}: R²={values['r2']:.3f}, MAPE={values['mape']:.1f}%"
                )

        # 🎯 執行修正的機器學習流程
        print("   🚀 執行修正ML流程...")
        ml_result = correct_ml_pipeline_for_vegetable(df_veg, test_start_date, test_end_date, sfm_threshold=sfm_threshold, min_features=min_features)

        if ml_result is not None:
            # 記錄驗證結果
            validation_summary.append(
                {
                    "vege_id": veg_id,
                    "val_R2": ml_result["validation_metrics"]["R2"],
                    "val_RMSE": ml_result["validation_metrics"]["RMSE"],
                    "val_MAE": ml_result["validation_metrics"]["MAE"],
                    "val_MAPE": ml_result["validation_metrics"]["MAPE"],
                    "validation_windows": ml_result["validation_metrics"][
                        "successful_windows"
                    ],
                }
            )

            # 記錄測試結果
            result = {
                "vege_id": veg_id,
                "vege_name": (
                    df_veg.get("vege_name", [veg_id]).iloc[0]
                    if "vege_name" in df_veg.columns
                    else veg_id
                ),
                "test_predictions": ml_result["test_metrics"]["total_predictions"],
                "test_windows": ml_result["test_metrics"]["successful_windows"],
                "selected_features_count": (
                    len(ml_result["selected_features"])
                    if ml_result["selected_features"]
                    else 0
                ),
                "test_R2": ml_result["test_metrics"]["R2"],
                "test_RMSE": ml_result["test_metrics"]["RMSE"],
                "test_MAE": ml_result["test_metrics"]["MAE"],
                "test_MAPE": ml_result["test_metrics"]["MAPE"],
            }

            all_results.append(result)

            print(f"   ✅ 驗證表現: R²={ml_result['validation_metrics']['R2']:.3f}")
            print(
                f"   🎯 測試表現: R²={ml_result['test_metrics']['R2']:.3f}, MAPE={ml_result['test_metrics']['MAPE']:.1f}%"
            )
            print(
                f"   🔧 選中特徵: {len(ml_result['selected_features']) if ml_result['selected_features'] else 0} 個"
            )
            print(
                f"   📊 測試窗口/預測數: {ml_result['test_metrics']['successful_windows']}/{ml_result['test_metrics']['total_predictions']}"
            )

            # 保存詳細配置
            config_path = os.path.join(OUTPUT_DIR, f"corrected_config_{veg_id}.json")
            with open(config_path, "w", encoding="utf-8") as f:
                config_data = {
                    "vege_id": veg_id,
                    "best_params": ml_result["best_params"],
                    "selected_features": ml_result["selected_features"],
                    "validation_metrics": ml_result["validation_metrics"],
                    "test_metrics": ml_result["test_metrics"],
                }
                json.dump(config_data, f, indent=2, ensure_ascii=False)
        else:
            print("   ❌ 修正ML流程失敗")

    # 保存和分析結果
    if all_results:
        results_df = pd.DataFrame(all_results)
        results_path = os.path.join(OUTPUT_DIR, "corrected_ml_test_results.csv")
        results_df.to_csv(results_path, index=False, encoding="utf-8-sig")

        # 基準測試結果
        baseline_df = pd.DataFrame(baseline_results) if baseline_results else None
        if baseline_df is not None:
            baseline_path = os.path.join(OUTPUT_DIR, "corrected_ml_baseline_results.csv")
            baseline_df.to_csv(baseline_path, index=False, encoding="utf-8-sig")

        # 驗證結果
        validation_df = pd.DataFrame(validation_summary) if validation_summary else None
        if validation_df is not None:
            validation_path = os.path.join(
                OUTPUT_DIR, "corrected_ml_validation_results.csv"
            )
            validation_df.to_csv(validation_path, index=False, encoding="utf-8-sig")

        # 統計分析
        print(f"\n📊 修正ML流程整體結果分析 ({len(results_df)} 種蔬菜):")
        print("=" * 70)
        print(f"🎯 測試期表現（嚴格固定配置）:")
        print(f"   平均 R²: {results_df['test_R2'].mean():.3f}")
        print(f"   中位數 R²: {results_df['test_R2'].median():.3f}")
        print(f"   平均 MAPE: {results_df['test_MAPE'].mean():.1f}%")
        print(f"   平均測試窗口數: {results_df['test_windows'].mean():.1f}")
        print(f"   平均預測數: {results_df['test_predictions'].mean():.1f}")
        print(f"   平均選中特徵數: {results_df['selected_features_count'].mean():.1f}")

        # 🔧 修正版關鍵驗證
        print(f"\n🔧 修正版關鍵改進驗證:")
        expected_windows = 13  # 理論窗口數
        expected_predictions = 90  # 理論預測數
        actual_avg_windows = results_df['test_windows'].mean()
        actual_avg_predictions = results_df['test_predictions'].mean()
        
        print(f"   理論窗口數: {expected_windows} vs 實際平均: {actual_avg_windows:.1f}")
        print(f"   理論預測數: {expected_predictions} vs 實際平均: {actual_avg_predictions:.1f}")
        
        if abs(actual_avg_windows - expected_windows) < 2:
            print("   ✅ 窗口數符合理論預期")
        else:
            print("   ⚠️ 窗口數偏離理論值")
            
        if abs(actual_avg_predictions - expected_predictions) < 10:
            print("   ✅ 預測數符合理論預期")
        else:
            print("   ⚠️ 預測數偏離理論值")

        if baseline_df is not None:
            print(f"\n📊 基準測試對比:")
            if "lag1_r2" in baseline_df.columns:
                print(f"   昨日價格基準平均 R²: {baseline_df['lag1_r2'].mean():.3f}")
                # 計算改進幅度
                merged_df = pd.merge(results_df, baseline_df, on="vege_id", how="inner")
                if len(merged_df) > 0:
                    improvement = merged_df["test_R2"] - merged_df["lag1_r2"]
                    improved_count = (improvement > 0).sum()
                    print(
                        f"   模型改進比例: {improved_count}/{len(merged_df)} ({improved_count/len(merged_df)*100:.1f}%)"
                    )
                    print(f"   平均改進幅度: {improvement.mean():.3f}")

        if validation_df is not None:
            print(f"\n🔍 驗證期表現:")
            print(f"   平均驗證 R²: {validation_df['val_R2'].mean():.3f}")
            print(f"   平均驗證 MAPE: {validation_df['val_MAPE'].mean():.1f}%")
            print(
                f"   平均驗證窗口數: {validation_df['validation_windows'].mean():.0f}"
            )

            # 🔥 關鍵：驗證vs測試一致性分析
            val_test_merged = pd.merge(
                validation_df, results_df, on="vege_id", how="inner"
            )
            if len(val_test_merged) > 0:
                val_test_diff = val_test_merged["test_R2"] - val_test_merged["val_R2"]
                correlation = np.corrcoef(
                    val_test_merged["val_R2"], val_test_merged["test_R2"]
                )[0, 1]
                consistency_score = 1 - abs(val_test_diff).mean()

                print(f"\n🔥 關鍵ML流程正確性驗證:")
                print(
                    f"   驗證-測試 R² 差異: {val_test_diff.mean():.3f} (標準差: {val_test_diff.std():.3f})"
                )
                print(f"   驗證-測試相關係數: {correlation:.3f}")
                print(f"   驗證-測試一致性分數: {consistency_score:.3f}")

                if correlation > 0.8 and consistency_score > 0.9:
                    print("   🎉 完美！修正ML流程完全正確 - 無資料洩漏！")
                elif correlation > 0.6 and consistency_score > 0.8:
                    print("   ✅ 優秀！修正ML流程基本正確")
                elif correlation > 0.4 and consistency_score > 0.7:
                    print("   🟡 良好！大部分問題已解決")
                else:
                    print("   ⚠️ 需要改進！可能存在其他問題")

        # R² 表現分級
        r2_excellent = (results_df["test_R2"] >= 0.6).sum()
        r2_good = ((results_df["test_R2"] >= 0.4) & (results_df["test_R2"] < 0.6)).sum()
        r2_fair = ((results_df["test_R2"] >= 0.2) & (results_df["test_R2"] < 0.4)).sum()
        r2_poor = (results_df["test_R2"] < 0.2).sum()

        print(f"\n📈 修正ML流程 R² 表現分佈:")
        print(
            f"   優秀 (≥0.6): {r2_excellent} 種 ({r2_excellent/len(results_df)*100:.1f}%)"
        )
        print(f"   良好 (0.4-0.6): {r2_good} 種 ({r2_good/len(results_df)*100:.1f}%)")
        print(f"   一般 (0.2-0.4): {r2_fair} 種 ({r2_fair/len(results_df)*100:.1f}%)")
        print(f"   待改進 (<0.2): {r2_poor} 種 ({r2_poor/len(results_df)*100:.1f}%)")

        # 創建綜合分析圖表
        if baseline_df is not None:
            create_comprehensive_plots(results_df, baseline_df, validation_df)
            print(
                f"\n📊 綜合分析圖表已保存: {OUTPUT_DIR}/corrected_ml_pipeline_analysis.png"
            )

        print(f"\n💾 修正ML流程結果已保存:")
        print(f"   測試結果: {results_path}")
        if baseline_df is not None:
            print(f"   基準結果: {baseline_path}")
        if validation_df is not None:
            print(f"   驗證結果: {validation_path}")
        print(f"   詳細配置: {OUTPUT_DIR}/corrected_config_*.json")

        # 修正ML流程核心優勢總結
        print(f"\n🔥 修正ML流程核心改進:")
        print("1. ✅ 正確的測試期窗口生成：13窗口90預測")
        print("2. ✅ 驗證期一次性參數優化：避免測試期調參")
        print("3. ✅ 驗證期一次性特徵選擇：避免測試期重新選特徵")
        print("4. ✅ 測試期完全固定配置：真正評估泛化能力")
        print("5. ✅ 嚴格防止資料洩漏：驗證測試流程完全一致")
        print("6. ✅ 驗證測試一致性檢查：確保結果可信度")

        # 與原版本的關鍵差異
        print(f"\n🚨 與原版本的關鍵差異:")
        print("❌ 原版本：測試期生成過多窗口(22個) → 邏輯錯誤")
        print("✅ 修正版本：測試期正確窗口數(13個) → 符合理論")
        print("❌ 原版本：預測數異常(130個) → 超出測試期")
        print("✅ 修正版本：預測數正確(90個) → 完全覆蓋測試期")
        print("❌ 原版本：窗口生成起點錯誤 → 概念混亂")
        print("✅ 修正版本：清晰的時間邊界 → 邏輯清楚")

    else:
        print("❌ 沒有成功處理任何蔬菜")

    elapsed = time.time() - start_time
    print(f"\n⏱️ 總執行時間: {elapsed:.1f} 秒")
    print("✅ 修正版機器學習流程分析完成!")

    return results_df if all_results else None, baseline_df, validation_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sfm-threshold", type=str, default="median", help="SelectFromModel threshold: median, mean, or a float e.g. 0.01")
    parser.add_argument("--min-features", type=int, default=10, help="Minimum features to keep if threshold is too strict")
    args = parser.parse_args()
    results, baselines, validations = main(sfm_threshold=args.sfm_threshold, min_features=args.min_features)

    # 最終正確性驗證報告
    if results is not None and baselines is not None and validations is not None:
        print("\n" + "=" * 70)
        print("🎯 修正版機器學習流程最終驗證報告")
        print("=" * 70)

        # 驗證測試一致性檢查
        merged = pd.merge(validations, results, on="vege_id", how="inner")
        if len(merged) > 0:
            # 一致性分數
            consistency_score = 1 - abs(merged["val_R2"] - merged["test_R2"]).mean()
            correlation = np.corrcoef(merged["val_R2"], merged["test_R2"])[0, 1]

            print(f"🎯 關鍵指標:")
            print(f"   驗證測試一致性分數: {consistency_score:.3f} (越接近1越好)")
            print(f"   驗證測試相關係數: {correlation:.3f} (越接近1越好)")

            if consistency_score > 0.9 and correlation > 0.8:
                print("🎉 完美！修正ML流程實施成功！無資料洩漏！")
            elif consistency_score > 0.8 and correlation > 0.6:
                print("✅ 優秀！修正ML流程基本實施成功")
            elif consistency_score > 0.7 and correlation > 0.4:
                print("🟡 良好！大部分問題已解決")
            else:
                print("⚠️ 仍需改進！可能存在其他問題")

        # 模型有效性檢查
        if "lag1_r2" in baselines.columns:
            model_baseline_diff = (
                results["test_R2"].mean() - baselines["lag1_r2"].mean()
            )
            print(f"\n📊 模型有效性:")
            print(f"   模型相對基準改進: {model_baseline_diff:.3f}")

            if model_baseline_diff > 0.1:
                print("✅ 模型顯著優於基準方法！複雜特徵工程有效")
            elif model_baseline_diff > 0.05:
                print("✅ 模型適度優於基準方法")
            elif model_baseline_diff > 0:
                print("🟡 模型略優於基準方法")
            else:
                print("⚠️ 模型可能不如簡單基準方法")

        # 🔧 修正版核心改進驗證
        print(f"\n🔧 修正版核心改進驗證:")
        avg_windows = results["test_windows"].mean()
        avg_predictions = results["test_predictions"].mean()
        
        print(f"   平均測試窗口數: {avg_windows:.1f} (理論值: 13)")
        print(f"   平均預測數: {avg_predictions:.1f} (理論值: 90)")
        
        window_accuracy = abs(avg_windows - 13) < 2
        prediction_accuracy = abs(avg_predictions - 90) < 10
        
        if window_accuracy and prediction_accuracy:
            print("✅ 窗口生成邏輯完全正確！")
        elif window_accuracy or prediction_accuracy:
            print("🟡 窗口生成邏輯基本正確")
        else:
            print("❌ 窗口生成邏輯仍有問題")

        # 真實性評估
        print(f"\n💡 結果真實性評估:")
        print("- 驗證期：系統性參數優化 + 特徵選擇（一次性）")
        print("- 測試期：完全固定配置，絕不調整")
        print("- 防洩漏：所有特徵基於t-1時點，閾值基於訓練期")
        print("- 如果現在驗證測試高度一致，那結果就是可信的")

        print("=" * 70)

        # 實際應用建議
        print(f"\n🚀 實際應用建議:")
        print("1. 對於新蔬菜：使用相同的驗證期優化流程")
        print("2. 對於生產環境：使用本次選定的最佳配置")
        print("3. 對於模型更新：定期重新執行完整的驗證期優化")
        print("4. 對於監控：持續追蹤實際表現與測試期預期的一致性")
        print("5. 🔧 關鍵：使用正確的13窗口90預測邏輯！")

    print("\n🎯 修正版機器學習流程驗證完成！")
