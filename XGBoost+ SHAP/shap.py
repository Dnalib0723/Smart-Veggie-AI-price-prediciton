# -*- coding: utf-8 -*-
"""
enhanced_shap_clean_dynamic_weather_ok_merged_fixed.py
------------------------------------------------------
以 enhanced_shap_clean_dynamic_weather_ok_merged.py 為基礎，修正可能的資料洩漏：
- 將 y_above_ma7 / y_above_ma30 調整為使用 y.shift(1) 與均線比較，避免使用當日目標 y(t)。

其他流程（動態極端氣象特徵、XGBoost 訓練、SHAP 圖輸出與彙總）保持一致。
"""

import os
import warnings
import time
import numpy as np
import pandas as pd
from collections import Counter

# 後端與字型
os.environ["MPLBACKEND"] = "Agg"
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

import xgboost as xgb
import shap

# ====== 使用者檔名設定（與原版一致，可按需更改） ======
# CSV_FILE = "daily_avg_price_vege_low_noise_clean.csv"
CSV_FILE = "daily_avg_price_vege_high_noise_clean.csv"
ID_MAP_FILE = "merged_id_compared.csv"

SCRIPT_NAME = "enhanced_shap_analysis"
OUTPUT_DIR = SCRIPT_NAME
SHAP_DIR = os.path.join(OUTPUT_DIR, "shap_analysis")
INDIVIDUAL_DIR = os.path.join(SHAP_DIR, "individual_vegetables")
DEPENDENCE_DIR = os.path.join(SHAP_DIR, "dependence_plots")
FORCE_DIR = os.path.join(SHAP_DIR, "force_plots")
WATERFALL_DIR = os.path.join(SHAP_DIR, "waterfall_plots")
for d in [
    OUTPUT_DIR,
    SHAP_DIR,
    INDIVIDUAL_DIR,
    DEPENDENCE_DIR,
    FORCE_DIR,
    WATERFALL_DIR,
]:
    os.makedirs(d, exist_ok=True)

# ====== 時間切分（與原版一致） ======
TRAIN_END = pd.to_datetime("2024-04-30")
TEST_START = pd.to_datetime("2024-05-01")
TEST_END = pd.to_datetime("2024-12-31")
SPLIT_METHOD = "time"

# ====== XGBoost 參數（與原版一致） ======
XGB_PARAMS = {
    "n_estimators": 500,
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "n_jobs": -1,
    "verbosity": 0,
}

# ====== SHAP 分析設定（與原版一致） ======
SHAP_SAMPLE_SIZE = 300
BACKGROUND_SIZE = 100
TOP_FEATURES_PER_VEG = 15  # 每個蔬菜抓前 15 名特徵

# ====== 氣象欄位集合（取自 complete_dynamic_weather_ok.py） ======
WEATHER_COLS = ["StnPres", "Temperature", "RH", "WS", "Precp"]

# ====== 中文字型（與原版一致） ======
CJK_CANDIDATES = [
    "Microsoft JhengHei",
    "PingFang TC",
    "Noto Sans CJK TC",
    "Noto Sans TC",
    "SimHei",
    "Arial Unicode MS",
]
available_fonts = {f.name for f in font_manager.fontManager.ttflist}
for font_name in CJK_CANDIDATES:
    if font_name in available_fonts:
        plt.rcParams["font.family"] = font_name
        break
plt.rcParams["axes.unicode_minus"] = False

warnings.filterwarnings("ignore")
plt.ioff()

print("✅ XGBoost imported successfully:", xgb.__version__)
print("✅ SHAP imported successfully:", shap.__version__)


# =====================================================================
# 動態極端氣象特徵（取自 complete_dynamic_weather_ok.py，整合為單檔使用）
# =====================================================================
def calculate_dynamic_weather_thresholds(df, weather_cols):
    """
    基於實際分布計算動態極端天氣閾值，統一使用 5%/95% 分位數法（涵蓋 10% 數據）
    """
    thresholds = {}
    print("🌡️ 計算動態極端天氣閾值...")
    for col in weather_cols:
        if col in df.columns:
            values = df[col].dropna()
            if len(values) > 100:  # 確保有足夠數據
                q05 = values.quantile(0.05)
                q95 = values.quantile(0.95)
                thresholds[col] = {"low": q05, "high": q95, "method": "5%/95%"}
                print(f"✅ {col} 極端閾值: < {q05:.2f} 或 > {q95:.2f} (5%/95%)")
    return thresholds


def add_dynamic_weather_features(df, weather_cols, leak_safe=True):
    """
    增強版氣象特徵工程：rolling、delta、z-score + 動態極端旗標
    leak_safe=True：以 shift(1) 避免未來資訊洩漏
    """
    df = df.copy()
    dynamic_thresholds = calculate_dynamic_weather_thresholds(df, weather_cols)
    windows = [3, 7, 14, 30]
    print("🔧 開始氣象特徵工程...")
    for col in weather_cols:
        if col in df.columns:
            print(f"   處理 {col}...")
            # 清理
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = (
                df[col].fillna(df[col].median()) if not df[col].isna().all() else 0
            )
            base_series = df[col].shift(1) if leak_safe else df[col]
            # 滾動統計
            for w in windows:
                df[f"{col}_ma_{w}"] = base_series.rolling(w, min_periods=1).mean()
                df[f"{col}_std_{w}"] = base_series.rolling(w, min_periods=1).std()
            # 偏差/變化
            df[f"{col}_dev30"] = df[col] - df[f"{col}_ma_30"]
            df[f"{col}_delta1"] = base_series.diff(1)
            df[f"{col}_delta7"] = base_series.diff(7)
            # z-score (30 窗口)
            roll_mean = base_series.rolling(30, min_periods=5).mean()
            roll_std = base_series.rolling(30, min_periods=5).std()
            df[f"{col}_z30"] = (df[col] - roll_mean) / (roll_std.replace(0, np.nan))
            df[f"{col}_z30"] = df[f"{col}_z30"].fillna(0)
            # 極端旗標（使用當日數值與整體分布門檻比較；不屬於洩漏）
            if col in dynamic_thresholds:
                th = dynamic_thresholds[col]
                df[f"{col}_extreme_low"] = (df[col] < th["low"]).astype(int)
                df[f"{col}_extreme_high"] = (df[col] > th["high"]).astype(int)
                df[f"{col}_extreme_any"] = (
                    ((df[col] < th["low"]) | (df[col] > th["high"]))
                ).astype(int)
    print("✅ 氣象特徵工程完成！")
    return df


# =====================================================================
# 保留原版：時間與價格滯後特徵（修正 y_above_ma* 防洩漏）
# =====================================================================
def add_time_features(df):
    df = df.copy()
    ds = df["ds"]
    df["year"] = ds.dt.year
    df["month"] = ds.dt.month
    df["dayofweek"] = ds.dt.dayofweek
    df["dayofyear"] = ds.dt.dayofyear
    df["quarter"] = ds.dt.quarter
    df["day"] = ds.dt.day
    df["week"] = ds.dt.isocalendar().week.astype(int)
    df["is_spring"] = ((df["month"] >= 3) & (df["month"] <= 5)).astype(int)
    df["is_summer"] = ((df["month"] >= 6) & (df["month"] <= 8)).astype(int)
    df["is_autumn"] = ((df["month"] >= 9) & (df["month"] <= 11)).astype(int)
    df["is_winter"] = ((df["month"] == 12) | (df["month"] <= 2)).astype(int)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["day_sin"] = np.sin(2 * np.pi * df["dayofyear"] / 365)
    df["day_cos"] = np.cos(2 * np.pi * df["dayofyear"] / 365)
    df["weekday_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["weekday_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)
    return df


def add_price_lags(df):
    df = df.copy().sort_values("ds").reset_index(drop=True)
    for lag in [1, 3, 7, 14, 30]:
        df[f"y_lag_{lag}"] = df["y"].shift(lag)
    for w in [7, 14, 30]:
        df[f"y_ma_{w}"] = df["y"].shift(1).rolling(w, min_periods=1).mean()
    df["y_change_1"] = df["y"].shift(1) - df["y"].shift(2)
    df["y_change_7"] = df["y"].shift(7) - df["y"].shift(8)
    df["y_change_30"] = df["y"].shift(30) - df["y"].shift(31)
    df["y_pct_change_1"] = df["y"].shift(1).pct_change(1)
    df["y_pct_change_7"] = df["y"].shift(7).pct_change(1)
    df["y_volatility_7"] = df["y"].shift(1).rolling(7, min_periods=1).std()
    df["y_volatility_14"] = df["y"].shift(1).rolling(14, min_periods=1).std()
    # ---- 修正：避免把當日 y(t) 與均線比較造成洩漏，改用 y(t-1) ----
    df["y_above_ma7"] = (df["y"].shift(1) > df["y_ma_7"]).astype(int)
    df["y_above_ma30"] = (df["y"].shift(1) > df["y_ma_30"]).astype(int)
    return df


# =====================================================================
# 新的「特徵清單組合器」：base/price/time + 動態氣象衍生欄位
# =====================================================================
def get_feature_columns_combined(df):
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
    weather_features = []
    for col in WEATHER_COLS:
        if col in df.columns:
            weather_features.append(col)
        weather_features.extend([c for c in df.columns if c.startswith(f"{col}_")])
    all_features = base_features + weather_features
    return [c for c in all_features if c in df.columns]


# =====================================================================
# 輔助：資料載入/分割/清理/評估/訓練
# =====================================================================
def _sanitize_filename(s):
    if s is None:
        return ""
    s = str(s).replace("\u200b", "").replace("\ufeff", "")
    return "".join(c for c in s if c not in r'\/:*?"<>|').strip()


def load_data(csv_path):
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"找不到資料檔：{csv_path}")
    df = pd.read_csv(csv_path)
    print(f"📋 檔案欄位: {list(df.columns)}")
    if "vege_id" in df.columns:
        df["market_vege_id"] = df["vege_id"].astype(str).str.strip()
        print("✅ 使用 vege_id 作為蔬菜識別碼")
    elif "market_vege_id" in df.columns:
        print("✅ 使用 market_vege_id 作為蔬菜識別碼")
    else:
        raise ValueError("CSV 需要 vege_id 或 market_vege_id 欄位")
    df["ds"] = pd.to_datetime(df["ObsTime"], errors="coerce")
    df["y"] = pd.to_numeric(df["avg_price_per_kg"], errors="coerce")
    df = df.dropna(subset=["ds", "y", "market_vege_id"]).copy()
    print(f"✅ 成功載入: {len(df):,} 筆資料")
    print(f"📅 日期範圍: {df['ds'].min().date()} → {df['ds'].max().date()}")
    print(f"🥬 蔬菜種類: {df['market_vege_id'].nunique()} 種")
    return df


def load_id_map(map_path):
    if not os.path.isfile(map_path):
        print(f"⚠️ 找不到 {map_path}")
        return {}
    try:
        m = pd.read_csv(map_path, encoding="utf-8-sig")
        if {"vege_id", "vege_name"}.issubset(m.columns):
            key_col = "vege_id"
        elif {"market_vege_id", "vege_name"}.issubset(m.columns):
            key_col = "market_vege_id"
        else:
            return {}
        m[key_col] = m[key_col].astype(str).str.strip()
        m["vege_name"] = m["vege_name"].astype(str).str.strip()
        m = m.drop_duplicates(subset=[key_col], keep="first")
        return dict(zip(m[key_col], m["vege_name"]))
    except Exception:
        return {}


def split_data(df, method="time"):
    if method == "time":
        train = df[df["ds"] <= TRAIN_END].copy()
        test = df[(df["ds"] >= TEST_START) & (df["ds"] <= TEST_END)].copy()
    else:
        train, test = train_test_split(df, test_size=0.2, random_state=42)
    return train, test


def filter_outliers(train_df, test_df):
    y = train_df["y"].to_numpy()
    if len(y) < 20:
        return train_df, test_df
    q1 = np.nanpercentile(y, 1)
    q99 = np.nanpercentile(y, 99)
    train_filtered = train_df[(train_df["y"] >= q1) & (train_df["y"] <= q99)].copy()
    return train_filtered, test_df


def safe_mape(y_true, y_pred, eps=1e-6):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs((y_true - y_pred) / denom))) * 100.0


def prepare_features(df, feature_cols):
    X = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    median_values = X.median(numeric_only=True)
    X = X.fillna(median_values)
    return X.astype(np.float32)


def train_xgboost_model(X_train, y_train, X_test):
    y_train_log = np.log1p(np.clip(y_train, 0, None))
    model = xgb.XGBRegressor(**XGB_PARAMS)
    model.fit(X_train, y_train_log)
    pred_log = model.predict(X_test)
    pred = np.expm1(pred_log)
    pred = np.maximum(pred, 0.0)
    return model, pred


# =====================================================================
# SHAP 全面分析（保留原版輸出結構）
# =====================================================================
def comprehensive_shap_analysis(
    model, X_train, X_test, feature_names, veg_id, veg_name, save_dirs
):
    try:
        X_test_sample = (
            X_test.sample(n=min(SHAP_SAMPLE_SIZE, len(X_test)), random_state=42)
            if len(X_test) > 0
            else X_test
        )
        X_background = (
            X_train.sample(n=min(BACKGROUND_SIZE, len(X_train)), random_state=42)
            if len(X_train) > 0
            else X_train
        )

        explainer = shap.TreeExplainer(model, X_background)
        shap_values = explainer.shap_values(X_test_sample)

        safe_vid = _sanitize_filename(str(veg_id))
        safe_vname = _sanitize_filename(str(veg_name))
        results = {}

        # summary bar
        try:
            plt.figure(figsize=(12, 8))
            shap.summary_plot(
                shap_values,
                X_test_sample,
                feature_names=feature_names,
                plot_type="bar",
                show=False,
                max_display=20,
            )
            plt.title(
                f"SHAP Feature Importance - {veg_name} ({veg_id})", fontsize=14, pad=20
            )
            plt.tight_layout()
            out = os.path.join(
                save_dirs["individual"], f"summary_bar_{safe_vid}_{safe_vname}.png"
            )
            plt.savefig(out, dpi=150, bbox_inches="tight")
            plt.close()
            results["summary_bar"] = out
        except Exception:
            plt.close("all")

        # summary dot
        try:
            plt.figure(figsize=(12, 8))
            shap.summary_plot(
                shap_values,
                X_test_sample,
                feature_names=feature_names,
                show=False,
                max_display=20,
            )
            plt.title(
                f"SHAP Feature Impact - {veg_name} ({veg_id})", fontsize=14, pad=20
            )
            plt.tight_layout()
            out = os.path.join(
                save_dirs["individual"], f"summary_dot_{safe_vid}_{safe_vname}.png"
            )
            plt.savefig(out, dpi=150, bbox_inches="tight")
            plt.close()
            results["summary_dot"] = out
        except Exception:
            plt.close("all")

        # dependence（選前5特徵）
        try:
            mean_abs = np.mean(np.abs(shap_values), axis=0)
            top_idx = np.argsort(mean_abs)[-5:][::-1]
            deps = []
            for idx in top_idx:
                feat_name = feature_names[idx]
                try:
                    plt.figure(figsize=(10, 6))
                    shap.dependence_plot(
                        idx,
                        shap_values,
                        X_test_sample,
                        feature_names=feature_names,
                        show=False,
                    )
                    plt.title(
                        f"SHAP Dependence - {feat_name}\n{veg_name} ({veg_id})",
                        fontsize=12,
                    )
                    plt.tight_layout()
                    out = os.path.join(
                        save_dirs["dependence"],
                        f"dependence_{safe_vid}_{safe_vname}_{_sanitize_filename(feat_name)}.png",
                    )
                    plt.savefig(out, dpi=150, bbox_inches="tight")
                    plt.close()
                    deps.append(out)
                except Exception:
                    plt.close("all")
            if deps:
                results["dependence"] = deps
        except Exception:
            pass

        # force（最多3筆樣本）
        try:
            forces = []
            n = min(3, len(X_test_sample))
            for i in range(n):
                explanation = shap.Explanation(
                    values=shap_values[i],
                    base_values=explainer.expected_value,
                    data=X_test_sample.iloc[i].values,
                    feature_names=feature_names,
                )
                plt.figure(figsize=(16, 4))
                shap.plots.force(explanation, matplotlib=True, show=False)
                plt.title(
                    f"SHAP Force Plot - Sample {i+1}\n{veg_name} ({veg_id})",
                    fontsize=12,
                )
                plt.tight_layout()
                out = os.path.join(
                    save_dirs["force"],
                    f"force_{safe_vid}_{safe_vname}_sample_{i+1}.png",
                )
                plt.savefig(out, dpi=150, bbox_inches="tight")
                plt.close()
                forces.append(out)
            if forces:
                results["force"] = forces
        except Exception:
            plt.close("all")

        # waterfall（最多3筆樣本）
        try:
            waterfalls = []
            n = min(3, len(X_test_sample))
            for i in range(n):
                explanation = shap.Explanation(
                    values=shap_values[i],
                    base_values=explainer.expected_value,
                    data=X_test_sample.iloc[i].values,
                    feature_names=feature_names,
                )
                plt.figure(figsize=(10, 8))
                shap.plots.waterfall(explanation, show=False)
                plt.title(
                    f"SHAP Waterfall Plot - Sample {i+1}\n{veg_name} ({veg_id})",
                    fontsize=12,
                )
                plt.tight_layout()
                out = os.path.join(
                    save_dirs["waterfall"],
                    f"waterfall_{safe_vid}_{safe_vname}_sample_{i+1}.png",
                )
                plt.savefig(out, dpi=150, bbox_inches="tight")
                plt.close()
                waterfalls.append(out)
            if waterfalls:
                results["waterfall"] = waterfalls
        except Exception:
            plt.close("all")

        # 前 N 重要特徵（以 mean(|SHAP|) 排序）
        try:
            mean_abs = np.mean(np.abs(shap_values), axis=0)
            top_indices = np.argsort(mean_abs)[-TOP_FEATURES_PER_VEG:][::-1]
            top_features = [(feature_names[i], float(mean_abs[i])) for i in top_indices]
            results["top_features"] = top_features
        except Exception:
            results["top_features"] = []

        return explainer, shap_values, X_test_sample, results
    except Exception as e:
        print(f"⚠️ SHAP 失敗: {e}")
        return None, None, None, {}


# =====================================================================
# Top Features 彙總的輔助（改為彙總「所有蔬菜各自前15名」）
# =====================================================================
_WEATHER_KEYS_FOR_TYPE = ["StnPres", "Temperature", "RH", "WS", "Precp"]
_SHORT_TERM_MARKERS = ["_1", "_3", "_5", "_7", "delta1", "delta3"]


def _infer_feature_type(feat: str) -> str:
    if any(w in feat for w in _WEATHER_KEYS_FOR_TYPE):
        return "weather"
    if feat.startswith("y_"):
        return "price"
    return "time"


def _is_short_term(feat: str) -> bool:
    return any(m in feat for m in _SHORT_TERM_MARKERS)


# =====================================================================
# 主流程（保留原版輸出，僅調整 y_above_ma* 定義以及保留其他結構）
# =====================================================================
def main():
    t0 = time.time()
    print("🚀 增強版 XGBoost + SHAP（動態氣象特徵 + 合併單檔 / 修正 anti-leak）")
    print(f"📊 切分方式: {SPLIT_METHOD}")
    print(
        f"🧪 訓練期：到 {TRAIN_END.date()} | 測試期：{TEST_START.date()} → {TEST_END.date()}"
    )

    # 載入資料
    df_all = load_data(CSV_FILE)
    id_map = load_id_map(ID_MAP_FILE)

    vids = sorted(df_all["market_vege_id"].unique())
    all_performance = []
    performance_results = []

    save_dirs = {
        "individual": INDIVIDUAL_DIR,
        "dependence": DEPENDENCE_DIR,
        "force": FORCE_DIR,
        "waterfall": WATERFALL_DIR,
    }

    print(f"\n開始處理 {len(vids)} 種蔬菜...")
    processed_count = 0

    for i, vid in enumerate(vids, 1):
        try:
            sub = df_all[df_all["market_vege_id"] == vid].copy().sort_values("ds")
            if len(sub) < 150:
                print(f"⭐️ [{i}/{len(vids)}] 跳過 {vid} - 資料量不足")
                continue

            train_raw, test_raw = split_data(sub, method=SPLIT_METHOD)
            if len(train_raw) < 100 or len(test_raw) == 0:
                print(f"⭐️ [{i}/{len(vids)}] 跳過 {vid} - 切分後資料不足")
                continue

            _, test_probe = split_data(sub, method="time")
            if SPLIT_METHOD == "time" and len(test_probe) == 0:
                print(f"⭐️ [{i}/{len(vids)}] 跳過 {vid} - 測試集為空")
                continue

            train_raw, test_raw = filter_outliers(train_raw, test_raw)
            veg_name = id_map.get(vid, vid)
            print(
                f"🥬 [{i}/{len(vids)}] {vid} ({veg_name}) - 訓練:{len(train_raw)}, 測試:{len(test_raw)}"
            )

            # 特徵工程：時間 +（動態氣象）+ 價格滯後
            train = add_time_features(train_raw)
            test = add_time_features(test_raw)

            train = add_dynamic_weather_features(train, WEATHER_COLS, leak_safe=True)
            test = add_dynamic_weather_features(test, WEATHER_COLS, leak_safe=True)

            train = add_price_lags(train)
            test = add_price_lags(test)

            feature_cols = get_feature_columns_combined(train)
            print(f"   📊 使用特徵數：{len(feature_cols)}")

            # 建模資料
            X_train = prepare_features(train, feature_cols)
            X_test = prepare_features(test, feature_cols)
            y_train, y_test = train["y"].values, test["y"].values

            # 訓練與預測
            model, pred = train_xgboost_model(X_train, y_train, X_test)
            r2 = r2_score(y_test, pred)
            rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
            mae = float(mean_absolute_error(y_test, pred))
            mape = safe_mape(y_test, pred)
            print(
                f"   📈 測試績效: R²={r2:.3f}, RMSE={rmse:.3f}, MAE={mae:.3f}, MAPE={mape:.1f}%"
            )
            processed_count += 1

            # SHAP 分析
            explainer, shap_values, X_test_sample, shap_results = (
                comprehensive_shap_analysis(
                    model, X_train, X_test, feature_cols, vid, veg_name, save_dirs
                )
            )

            # 總結行
            perf_row = {
                "vege_id": vid,
                "vege_name": veg_name,
                "model": "XGBoost",
                "split_method": SPLIT_METHOD,
                "R2": r2,
                "RMSE": rmse,
                "MAE": mae,
                "MAPE(%)": mape,
                "train_samples": len(X_train),
                "test_samples": len(X_test),
            }
            all_performance.append(perf_row)

            res = {
                "veg_id": str(vid).strip(),
                "veg_name": veg_name,
                "r2": r2,
                "rmse": rmse,
                "mae": mae,
                "mape": mape,
            }
            if shap_values is not None and X_test_sample is not None:
                res.update(
                    {
                        "top_features": shap_results.get("top_features", []),
                        "shap_results": shap_results,
                        "model": model,
                        "feature_names": feature_cols,
                    }
                )
            performance_results.append(res)

        except Exception as e:
            print(f"   ❌ {vid} 失敗: {e}")
            continue

    # ====== 儲存輸出（與原版一致） ======
    print("\n💾 儲存結果...")
    if all_performance:
        perf_df = pd.DataFrame(all_performance).sort_values(["R2"], ascending=False)
        perf_path = os.path.join(OUTPUT_DIR, "performance_summary.csv")
        perf_df.to_csv(perf_path, index=False, encoding="utf-8-sig")
        print(f"📊 {perf_path} 已儲存")

        try:
            print(f"\n📊 測試集統計摘要 (共 {len(perf_df)} 種蔬菜):")
            print(f"   平均 R²: {perf_df['R2'].mean():.3f}")
            print(f"   中位數 R²: {perf_df['R2'].median():.3f}")
            print(f"   平均 RMSE: {perf_df['RMSE'].mean():.3f}")
            print(f"   平均 MAPE: {perf_df['MAPE(%)'].mean():.1f}%")

            r2_high = (perf_df["R2"] >= 0.7).sum()
            r2_mid = ((perf_df["R2"] >= 0.5) & (perf_df["R2"] < 0.7)).sum()
            r2_low = (perf_df["R2"] < 0.5).sum()
            print(f"   R² ≥ 0.7: {r2_high} 種 ({r2_high/len(perf_df)*100:.1f}%)")
            print(f"   R² 0.5-0.7: {r2_mid} 種 ({r2_mid/len(perf_df)*100:.1f}%)")
            print(f"   R² < 0.5: {r2_low} 種 ({r2_low/len(perf_df)*100:.1f}%)")

            # R2 分佈圖
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            ax1.boxplot(
                perf_df["R2"].dropna().values, labels=["XGBoost"], showmeans=True
            )
            ax1.set_ylabel("R²")
            ax1.set_title(f"R² Distribution ({SPLIT_METHOD} split)")
            ax1.grid(True, alpha=0.3)
            ax2.hist(
                perf_df["R2"].dropna().values, bins=20, alpha=0.7, edgecolor="black"
            )
            ax2.set_xlabel("R²")
            ax2.set_ylabel("頻率")
            ax2.set_title("R² Histogram")
            ax2.grid(True, alpha=0.3)
            plt.tight_layout()
            r2_fig = os.path.join(OUTPUT_DIR, f"r2_distribution_{SCRIPT_NAME}.png")
            plt.savefig(r2_fig, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"🖼️ {r2_fig} 已儲存")
        except Exception as e:
            print(f"⚠️ 圖表生成失敗: {e}")

    # ====== Top Features 彙總（改版重點）：彙總「所有蔬菜」的前 TOP_FEATURES_PER_VEG 名 ======
    if performance_results:
        try:
            # 收集每個蔬菜的前 TOP_FEATURES_PER_VEG 名特徵
            per_veg_top_feats = {}
            for r in performance_results:
                vid = str(r.get("veg_id", ""))
                tfs = r.get("top_features", []) or []
                names = [t[0] for t in tfs[:TOP_FEATURES_PER_VEG]]
                if names:
                    per_veg_top_feats[vid] = names

            # 彙總出現次數
            all_feats = []
            for feats in per_veg_top_feats.values():
                all_feats.extend(feats)
            ctr = Counter(all_feats)

            # 產出表格（含分類欄位）
            rows = []
            for feat, cnt in ctr.most_common():
                is_weather = any(w in feat for w in _WEATHER_KEYS_FOR_TYPE)
                rows.append(
                    {
                        "feature_name": feat,
                        "appearance_count": int(cnt),
                        "is_common": cnt >= 2,  # 至少在 2 種蔬菜中出現
                        "is_weather": is_weather,
                        "is_extreme": ("extreme" in feat),
                        "is_short_term": _is_short_term(feat),
                        "feature_type": _infer_feature_type(feat),
                    }
                )
            df_top = pd.DataFrame(rows)
            feature_path = os.path.join(OUTPUT_DIR, "top_features_analysis.csv")
            df_top.to_csv(feature_path, index=False, encoding="utf-8-sig")
            print(
                f"🎯 {feature_path} 已儲存（彙總所有蔬菜各自前 {TOP_FEATURES_PER_VEG} 名特徵）"
            )
        except Exception as e:
            print(f"⚠️ Top Features 彙總失敗: {e}")

        # SHAP 摘要表（保留原版）
        try:
            shap_summary = []
            total_plots = 0
            for r in performance_results:
                sr = r.get("shap_results", {})
                plots_count = (1 if "summary_bar" in sr else 0) + (
                    1 if "summary_dot" in sr else 0
                )
                plots_count += (
                    len(sr.get("dependence", []))
                    + len(sr.get("force", []))
                    + len(sr.get("waterfall", []))
                )
                total_plots += plots_count
                shap_summary.append(
                    {
                        "vege_id": r["veg_id"],
                        "vege_name": r["veg_name"],
                        "r2": r["r2"],
                        "total_plots": plots_count,
                        "summary_plots": (
                            ("summary_bar" in sr) + ("summary_dot" in sr)
                        ),
                        "dependence_plots": len(sr.get("dependence", [])),
                        "force_plots": len(sr.get("force", [])),
                        "waterfall_plots": len(sr.get("waterfall", [])),
                    }
                )
            shap_summary_path = os.path.join(OUTPUT_DIR, "shap_analysis_summary.csv")
            pd.DataFrame(shap_summary).to_csv(
                shap_summary_path, index=False, encoding="utf-8-sig"
            )
            print(f"📋 {shap_summary_path} 已儲存")
            print(f"📈 總共生成了 {total_plots} 個 SHAP 視覺化圖表")
        except Exception as e:
            print(f"⚠️ SHAP 摘要報告生成失敗: {e}")

    print(f"\n📁 所有輸出檔案儲存於：{OUTPUT_DIR}/")
    print(f"   SHAP 個別分析：{INDIVIDUAL_DIR}/")
    print(f"   依賴關係圖：{DEPENDENCE_DIR}/")
    print(f"   力圖：{FORCE_DIR}/")
    print(f"   瀑布圖：{WATERFALL_DIR}/")
    print(f"⏱️ 完成，耗時 {time.time()-t0:.1f} 秒")
    print(f"✅ 成功處理 {processed_count} 種蔬菜")


if __name__ == "__main__":
    main()
