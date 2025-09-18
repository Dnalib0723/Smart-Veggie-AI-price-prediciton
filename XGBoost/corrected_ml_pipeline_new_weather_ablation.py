
# -*- coding: utf-8 -*-
"""
corrected_ml_pipeline_new_weather_ablation.py
（新增）自動生成差異表：比較 original vs no_weather 的 Validation/Test 指標差異
"""
import os
import io
import sys
import json
import time
import math
import argparse
import zipfile
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

try:
    import xgboost as xgb
except Exception as e:
    print("請先安裝 xgboost：pip install xgboost")
    raise

PY7ZR_AVAILABLE = False
try:
    import py7zr  # type: ignore
    PY7ZR_AVAILABLE = True
except Exception:
    pass

WEATHER_PREFIXES = ["StnPres", "Temperature", "RH", "WS", "Precp", "typhoon"]
DEFAULTS = {
    "data_file": "2022_2025_daily_vege_weather_price.csv",
    "train_days": 365,
    "valid_days": 7,
    "step_days": 7,
    "start_date": "2022-01-01",
    "test_days": 90,
    "min_samples": 100,
    "validation_windows": 20,
}
TIME_FEATURES = [
    "year","month","dayofweek","dayofyear","quarter","day","week",
    "is_spring","is_summer","is_autumn","is_winter",
    "month_sin","month_cos","day_sin","day_cos","weekday_sin","weekday_cos",
]
PRICE_FEATURES = [
    "y_lag_1","y_lag_3","y_lag_7","y_lag_14","y_lag_30",
    "y_ma_7","y_ma_14","y_ma_30",
    "y_change_1","y_change_7","y_change_30",
    "y_volatility_7","y_volatility_14",
    "y_pct_change_1","y_pct_change_7",
    "y_above_ma7","y_above_ma30",
]

def _is_weather_feature(name: str) -> bool:
    return any(name == p or name.startswith(p + "_") for p in WEATHER_PREFIXES)

def _read_json_bytes(b: bytes):
    return json.loads(b.decode("utf-8"))

def load_configs_from_dir(dir_path: str) -> Dict[str, dict]:
    configs = {}
    for fn in os.listdir(dir_path):
        if not fn.lower().endswith(".json"):
            continue
        fpath = os.path.join(dir_path, fn)
        try:
            with open(fpath, "rb") as f:
                cfg = _read_json_bytes(f.read())
            vid = str(cfg.get("vege_id", "")).strip()
            if vid:
                configs[vid] = cfg
        except Exception as e:
            print(f"[WARN] 無法解析 {fpath}: {e}")
    return configs

def load_configs_from_zip(zip_path: str) -> Dict[str, dict]:
    configs = {}
    with zipfile.ZipFile(zip_path, "r") as zf:
        for name in zf.namelist():
            if name.lower().endswith(".json"):
                try:
                    with zf.open(name, "r") as f:
                        cfg = _read_json_bytes(f.read())
                    vid = str(cfg.get("vege_id", "")).strip()
                    if vid:
                        configs[vid] = cfg
                except Exception as e:
                    print(f"[WARN] 無法解析 {name} in zip: {e}")
    return configs

def load_configs_from_7z(sevenz_path: str) -> Dict[str, dict]:
    if not PY7ZR_AVAILABLE:
        print("⚠️ 偵測到 .7z，但未安裝 py7zr；請先：pip install py7zr")
        return {}
    tmp_dir = os.path.join(os.path.dirname(sevenz_path), "_tmp_configs_7z")
    try:
        with py7zr.SevenZipFile(sevenz_path, mode="r") as z:
            z.extractall(path=tmp_dir)
        return load_configs_from_dir(tmp_dir)
    except Exception as e:
        print(f"[WARN] 7z 讀取失敗：{e}")
        return {}

def load_all_configs(config_dir: str=None, config_archive: str=None) -> Dict[str, dict]:
    if config_dir and os.path.isdir(config_dir):
        print(f"📥 從資料夾讀取 JSON：{config_dir}")
        return load_configs_from_dir(config_dir)
    if config_archive and os.path.isfile(config_archive):
        lower = config_archive.lower()
        if lower.endswith(".zip"):
            print(f"📥 從 zip 讀取 JSON：{config_archive}")
            return load_configs_from_zip(config_archive)
        if lower.endswith(".7z"):
            print(f"📥 從 7z 讀取 JSON：{config_archive}")
            return load_configs_from_7z(config_archive)
        print(f"[WARN] 不支援的 config 壓縮格式：{config_archive}")
    print("[WARN] 未提供有效的 config 來源")
    return {}

def load_and_preprocess_data(csv_path: str) -> pd.DataFrame:
    encodings = ["utf-8", "utf-8-sig", "big5", "gbk", "cp950", "latin-1"]
    df = None
    for enc in encodings:
        try:
            df = pd.read_csv(csv_path, encoding=enc)
            print(f"   ✅ 成功使用編碼: {enc}")
            break
        except Exception:
            continue
    if df is None:
        raise FileNotFoundError(f"無法讀取資料檔：{csv_path}")
    df["ds"] = pd.to_datetime(df["ObsTime"], errors="coerce")
    df["y"] = pd.to_numeric(df["avg_price_per_kg"], errors="coerce")
    df["vege_id"] = df["vege_id"].astype(str)
    df = df.dropna(subset=["ds", "y", "vege_id"]).sort_values(["vege_id","ds"]).reset_index(drop=True)
    print(f"✅ 資料載入完成: {len(df):,} 筆, {df['vege_id'].nunique()} 種蔬菜")
    print(f"   日期範圍: {df['ds'].min().date()} → {df['ds'].max().date()}")
    return df

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    ds = df["ds"]
    df["year"] = ds.dt.year
    df["month"] = ds.dt.month
    df["dayofweek"] = ds.dt.dayofweek
    df["dayofyear"] = ds.dt.dayofyear
    df["quarter"] = ds.dt.quarter
    df["day"] = ds.dt.day
    df["week"] = ds.dt.isocalendar().week.astype(int)
    df["is_spring"] = ((df["month"]>=3)&(df["month"]<=5)).astype(int)
    df["is_summer"] = ((df["month"]>=6)&(df["month"]<=8)).astype(int)
    df["is_autumn"] = ((df["month"]>=9)&(df["month"]<=11)).astype(int)
    df["is_winter"] = ((df["month"]==12)|(df["month"]<=2)).astype(int)
    df["month_sin"] = np.sin(2*np.pi*df["month"]/12)
    df["month_cos"] = np.cos(2*np.pi*df["month"]/12)
    df["day_sin"]   = np.sin(2*np.pi*df["dayofyear"]/365)
    df["day_cos"]   = np.cos(2*np.pi*df["dayofyear"]/365)
    df["weekday_sin"]=np.sin(2*np.pi*df["dayofweek"]/7)
    df["weekday_cos"]=np.cos(2*np.pi*df["dayofweek"]/7)
    return df

def add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values("ds").reset_index(drop=True)
    for lag in [1,3,7,14,30]:
        df[f"y_lag_{lag}"] = df["y"].shift(lag)
    for w in [7,14,30]:
        df[f"y_ma_{w}"] = df["y"].shift(1).rolling(w, min_periods=1).mean()
    df["y_change_1"] = df["y"].shift(1) - df["y"].shift(2)
    df["y_change_7"] = df["y"].shift(7) - df["y"].shift(8)
    df["y_change_30"] = df["y"].shift(30) - df["y"].shift(31)
    df["y_pct_change_1"] = df["y"].shift(1).pct_change(1)
    df["y_pct_change_7"] = df["y"].shift(7).pct_change(1)
    df["y_volatility_7"]  = df["y"].shift(1).rolling(7, min_periods=1).std()
    df["y_volatility_14"] = df["y"].shift(1).rolling(14, min_periods=1).std()
    df["y_above_ma7"]  = (df["y"].shift(1) > df["y_ma_7"]).astype(int)
    df["y_above_ma30"] = (df["y"].shift(1) > df["y_ma_30"]).astype(int)
    return df

def calc_thresholds(train_df: pd.DataFrame, cols: List[str]) -> dict:
    th = {}
    for c in cols:
        if c in train_df.columns:
            v = pd.to_numeric(train_df[c], errors="coerce").dropna()
            if len(v) >= 50:
                th[c] = {"low": float(v.quantile(0.05)), "high": float(v.quantile(0.95))}
    return th

def add_weather_features(df: pd.DataFrame, cols: List[str], thresholds: dict) -> pd.DataFrame:
    df = df.copy()
    windows = [3,7,14,30]
    for col in cols:
        if col not in df.columns: 
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(df[col].median()) if not df[col].isna().all() else 0
        base = df[col].shift(1)
        for w in windows:
            df[f"{col}_ma_{w}"] = base.rolling(w, min_periods=1).mean()
            df[f"{col}_std_{w}"] = base.rolling(w, min_periods=1).std()
        df[f"{col}_dev30"]   = df[col].shift(1) - df[f"{col}_ma_{30}"]
        df[f"{col}_delta1"]  = base.diff(1)
        df[f"{col}_delta7"]  = base.diff(7)
        roll_mean = base.rolling(30, min_periods=5).mean()
        roll_std  = base.rolling(30, min_periods=5).std()
        df[f"{col}_z30"] = (df[col].shift(1)-roll_mean)/(roll_std.replace(0,np.nan))
        df[f"{col}_z30"] = df[f"{col}_z30"].fillna(0)
        if col in thresholds:
            lagged = df[col].shift(1)
            df[f"{col}_extreme_low"]  = (lagged < thresholds[col]["low"]).astype(int)
            df[f"{col}_extreme_high"] = (lagged > thresholds[col]["high"]).astype(int)
            df[f"{col}_extreme_any"]  = ((lagged < thresholds[col]["low"])|(lagged > thresholds[col]["high"])).astype(int)
    return df

def full_feature_engineering(df_veg: pd.DataFrame, cutoff_date: pd.Timestamp) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_df = df_veg[df_veg["ds"] < cutoff_date].copy()
    test_df  = df_veg[df_veg["ds"] >= cutoff_date].copy()
    if len(train_df) < 50:
        return None, None
    thresholds = calc_thresholds(train_df, WEATHER_PREFIXES)
    combo = pd.concat([train_df, test_df]).sort_values("ds")
    combo = add_time_features(combo)
    combo = add_weather_features(combo, WEATHER_PREFIXES, thresholds)
    combo = add_price_features(combo)
    train_p = combo[combo["ds"] < cutoff_date]
    test_p  = combo[combo["ds"] >= cutoff_date]
    return train_p, test_p

def generate_validation_windows(start_date, end_date, train_days, valid_days, step_days):
    windows = []
    current = start_date
    while True:
        train_end = current + timedelta(days=train_days - 1)
        valid_start = train_end + timedelta(days=1)
        valid_end = valid_start + timedelta(days=valid_days - 1)
        if valid_end > end_date:
            break
        windows.append({"train_start": current, "train_end": train_end, "valid_start": valid_start, "valid_end": valid_end})
        current += timedelta(days=step_days)
    return windows

def generate_test_windows(test_start_date, test_end_date, train_days, pred_days, step_days):
    windows = []
    cur = test_start_date
    while cur <= test_end_date:
        pred_end = cur + timedelta(days=pred_days - 1)
        if pred_end > test_end_date:
            pred_end = test_end_date
        train_start = cur - timedelta(days=train_days)
        train_end   = cur - timedelta(days=1)
        windows.append({"train_start": train_start, "train_end": train_end, "pred_start": cur, "pred_end": pred_end})
        if pred_end >= test_end_date:
            break
        cur += timedelta(days=step_days)
    return windows

def safe_mape(y_true, y_pred, eps=1e-6):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs((y_true - y_pred) / denom))) * 100.0

def calc_metrics(y_true, y_pred) -> dict:
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MAPE": safe_mape(y_true, y_pred),
    }

def validate_fixed_config(df_veg: pd.DataFrame, windows: list, params: dict, used_features: List[str]) -> dict:
    preds, trues, success = [], [], 0
    for w in windows:
        try:
            wstart = min(w["train_start"] - timedelta(days=30), df_veg["ds"].min())
            wdata = df_veg[(df_veg["ds"] >= wstart) & (df_veg["ds"] <= w["valid_end"])].copy()
            train_p, valid_p = full_feature_engineering(wdata, w["valid_start"])
            if train_p is None or len(valid_p) == 0:
                continue
            train_final = train_p[(train_p["ds"]>=w["train_start"]) & (train_p["ds"]<=w["train_end"])]
            valid_final = valid_p[(valid_p["ds"]>=w["valid_start"]) & (valid_p["ds"]<=w["valid_end"])]
            feats = [f for f in used_features if f in train_final.columns]
            if len(train_final)==0 or len(valid_final)==0 or len(feats)<5:
                continue
            Xtr, ytr = train_final[feats].fillna(0), train_final["y"].values
            Xva, yva = valid_final[feats].fillna(0), valid_final["y"].values
            if len(np.unique(ytr)) < 3:
                continue
            model = xgb.XGBRegressor(**params)
            model.fit(Xtr, ytr)
            yhat = np.maximum(model.predict(Xva), 0)
            if np.isnan(yhat).any() or np.isinf(yhat).any():
                continue
            preds.extend(yhat.tolist())
            trues.extend(yva.tolist())
            success += 1
        except Exception:
            continue
    if len(preds)==0:
        return {}
    m = calc_metrics(np.array(trues), np.array(preds))
    m["successful_windows"] = int(success)
    return m

def test_fixed_config(df_veg: pd.DataFrame, test_start: pd.Timestamp, test_end: pd.Timestamp, params: dict, used_features: List[str]) -> dict:
    windows = generate_test_windows(test_start, test_end, 365, 7, 7)
    preds, trues, success = [], [], 0
    for w in windows:
        try:
            wstart = w["train_start"] - timedelta(days=60)
            wdata = df_veg[(df_veg["ds"] >= wstart) & (df_veg["ds"] <= w["pred_end"])].copy()
            train_p, test_p = full_feature_engineering(wdata, w["pred_start"])
            if train_p is None or len(test_p)==0:
                continue
            train_final = train_p[(train_p["ds"]>=w["train_start"]) & (train_p["ds"]<=w["train_end"])]
            test_final  = test_p[(test_p["ds"]>=w["pred_start"]) & (test_p["ds"]<=w["pred_end"])]
            feats = [f for f in used_features if f in train_final.columns]
            if len(train_final)==0 or len(test_final)==0 or len(feats)<5:
                continue
            Xtr, ytr = train_final[feats].fillna(0), train_final["y"].values
            Xte, yte = test_final[feats].fillna(0), test_final["y"].values
            if len(np.unique(ytr)) < 3:
                continue
            model = xgb.XGBRegressor(**params)
            model.fit(Xtr, ytr)
            yhat = np.maximum(model.predict(Xte), 0)
            if np.isnan(yhat).any() or np.isinf(yhat).any():
                continue
            preds.extend(yhat.tolist())
            trues.extend(yte.tolist())
            success += 1
        except Exception:
            continue
    if len(preds)==0:
        return {}
    m = calc_metrics(np.array(trues), np.array(preds))
    m["successful_windows"] = int(success)
    m["total_predictions"] = int(len(preds))
    return m

def baseline_predictions(df_veg: pd.DataFrame, test_start: pd.Timestamp) -> dict:
    test_data = df_veg[df_veg["ds"] >= test_start].copy()
    if len(test_data)==0: 
        return {}
    y_true = test_data["y"].values
    baselines = {}
    np.random.seed(42)
    rand = np.random.normal(y_true.mean(), y_true.std() if y_true.std()>0 else 1.0, len(y_true))
    baselines["random"] = {"r2": float(r2_score(y_true, rand)), "mape": float(safe_mape(y_true, rand))}
    combo = add_price_features(df_veg.copy())
    test_lag = combo[combo["ds"] >= test_start]
    if "y_lag_1" in test_lag.columns:
        lag1 = test_lag["y_lag_1"].fillna(y_true.mean()).values
        baselines["lag1"] = {"r2": float(r2_score(y_true, lag1)), "mape": float(safe_mape(y_true, lag1))}
    if "y_ma_7" in test_lag.columns:
        ma7 = test_lag["y_ma_7"].fillna(y_true.mean()).values
        baselines["ma7"] = {"r2": float(r2_score(y_true, ma7)), "mape": float(safe_mape(y_true, ma7))}
    return baselines

def run_one_setting(df: pd.DataFrame, configs: Dict[str,dict], mode: str, out_dir: str) -> Tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame]:
    os.makedirs(out_dir, exist_ok=True)
    start_dt = pd.to_datetime(DEFAULTS["start_date"])
    max_date = df["ds"].max()
    test_start = max_date - timedelta(days=DEFAULTS["test_days"] - 1)
    test_end   = max_date
    train_end  = test_start - timedelta(days=1)
    val_windows = generate_validation_windows(start_dt, train_end, DEFAULTS["train_days"], DEFAULTS["valid_days"], DEFAULTS["step_days"])
    if len(val_windows) > DEFAULTS["validation_windows"]:
        step = len(val_windows)//DEFAULTS["validation_windows"]
        val_windows = val_windows[::step][:DEFAULTS["validation_windows"]]
    vegetable_list = sorted(df["vege_id"].unique())

    baseline_rows, val_rows, test_rows = [], [], []

    print(f"\n===== 設定：{mode} =====")
    print(f"輸出資料夾：{out_dir}")
    for i, vid in enumerate(vegetable_list, 1):
        if vid not in configs:
            continue
        cfg = configs[vid]
        best_params = cfg.get("best_params", {})
        selected_features = cfg.get("selected_features", []) or []
        if mode == "no_weather":
            used_features = [f for f in selected_features if not _is_weather_feature(f)]
        else:
            used_features = list(selected_features)
        if not used_features:
            used_features = TIME_FEATURES + PRICE_FEATURES

        df_veg = df[df["vege_id"]==vid].copy().sort_values("ds")
        if len(df_veg) < DEFAULTS["min_samples"]:
            continue

        baselines = baseline_predictions(df_veg, test_start)
        if baselines:
            baseline_rows.append({
                "vege_id": vid,
                **{f"{m}_{k}": v for m,vals in baselines.items() for k,v in vals.items()}
            })

        val_metrics = validate_fixed_config(df_veg, val_windows, best_params, used_features)
        test_metrics = test_fixed_config(df_veg, test_start, test_end, best_params, used_features)

        out_json = {
            "vege_id": vid,
            "best_params": best_params,
            "selected_features": selected_features,
            "used_features": used_features,
            "validation_metrics": val_metrics if val_metrics else {},
            "test_metrics": test_metrics if test_metrics else {},
        }
        with open(os.path.join(out_dir, f"corrected_config_{vid}.json"), "w", encoding="utf-8") as f:
            json.dump(out_json, f, ensure_ascii=False, indent=2)

        if val_metrics:
            val_rows.append({"vege_id": vid, **{f"val_{k}": v for k,v in val_metrics.items() if k!='successful_windows'}, "validation_windows": val_metrics.get("successful_windows", 0)})
        if test_metrics:
            test_rows.append({
                "vege_id": vid,
                "test_R2": test_metrics.get("R2", np.nan),
                "test_RMSE": test_metrics.get("RMSE", np.nan),
                "test_MAE": test_metrics.get("MAE", np.nan),
                "test_MAPE": test_metrics.get("MAPE", np.nan),
                "test_windows": test_metrics.get("successful_windows", 0),
                "test_predictions": test_metrics.get("total_predictions", 0),
                "selected_features_count": len(used_features),
            })

    baseline_df = pd.DataFrame(baseline_rows) if baseline_rows else pd.DataFrame()
    validation_df = pd.DataFrame(val_rows) if val_rows else pd.DataFrame()
    test_df = pd.DataFrame(test_rows) if test_rows else pd.DataFrame()

    if not baseline_df.empty:
        baseline_df.to_csv(os.path.join(out_dir, "corrected_ml_baseline_results.csv"), index=False, encoding="utf-8-sig")
    if not validation_df.empty:
        validation_df.to_csv(os.path.join(out_dir, "corrected_ml_validation_results.csv"), index=False, encoding="utf-8-sig")
    if not test_df.empty:
        test_df.to_csv(os.path.join(out_dir, "corrected_ml_test_results.csv"), index=False, encoding="utf-8-sig")

    return baseline_df, validation_df, test_df

def make_diff_tables(output_root: str):
    """
    讀取 original/ 與 no_weather/ 的 CSV，產生差異表：
    - comparison_validation_diff.csv（各 vege_id：no_weather - original 的差值）
    - comparison_test_diff.csv（各 vege_id：no_weather - original 的差值）
    並輸出一份 summary（平均差值）。
    """
    def _load(path):
        return pd.read_csv(path, encoding="utf-8-sig") if os.path.exists(path) else pd.DataFrame()

    ori_val = _load(os.path.join(output_root, "original", "corrected_ml_validation_results.csv"))
    ori_tst = _load(os.path.join(output_root, "original", "corrected_ml_test_results.csv"))
    nw_val  = _load(os.path.join(output_root, "no_weather", "corrected_ml_validation_results.csv"))
    nw_tst  = _load(os.path.join(output_root, "no_weather", "corrected_ml_test_results.csv"))

    os.makedirs(output_root, exist_ok=True)

    # Validation diff
    if not ori_val.empty and not nw_val.empty:
        v = pd.merge(nw_val, ori_val, on="vege_id", suffixes=("_nw", "_ori"))
        # 只對數值欄位做差：nw - ori
        diff_cols = {}
        for col in v.columns:
            if col.endswith("_nw"):
                base = col[:-3]
                ori_col = base + "_ori"
                if ori_col in v.columns:
                    diff_name = f"Δ_{base}"  # Δ_val_R2, Δ_val_RMSE, ...
                    diff_cols[diff_name] = v[col] - v[ori_col]
        v_out = pd.DataFrame({"vege_id": v["vege_id"], **diff_cols})
        v_out.to_csv(os.path.join(output_root, "comparison_validation_diff.csv"), index=False, encoding="utf-8-sig")

        # summary
        summary_v = v_out.drop(columns=["vege_id"]).mean().to_frame("mean_delta").reset_index().rename(columns={"index":"metric"})
        summary_v.to_csv(os.path.join(output_root, "comparison_validation_summary.csv"), index=False, encoding="utf-8-sig")

    # Test diff
    if not ori_tst.empty and not nw_tst.empty:
        t = pd.merge(nw_tst, ori_tst, on="vege_id", suffixes=("_nw", "_ori"))
        diff_cols = {}
        for base in ["test_R2","test_RMSE","test_MAE","test_MAPE","test_windows","test_predictions","selected_features_count"]:
            if base+"_nw" in t.columns and base+"_ori" in t.columns:
                diff_cols[f"Δ_{base}"] = t[base+"_nw"] - t[base+"_ori"]
        t_out = pd.DataFrame({"vege_id": t["vege_id"], **diff_cols})
        t_out.to_csv(os.path.join(output_root, "comparison_test_diff.csv"), index=False, encoding="utf-8-sig")

        summary_t = t_out.drop(columns=["vege_id"]).mean().to_frame("mean_delta").reset_index().rename(columns={"index":"metric"})
        summary_t.to_csv(os.path.join(output_root, "comparison_test_summary.csv"), index=False, encoding="utf-8-sig")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-file", type=str, default=DEFAULTS["data_file"])
    ap.add_argument("--config-dir", type=str, default=None, help="包含 corrected_config_*.json 的資料夾")
    ap.add_argument("--config-archive", type=str, default=None, help="可為 .zip 或 .7z")
    ap.add_argument("--output-root", type=str, default="weather_ablation_results")
    args = ap.parse_args()

    print("🎯 天氣特徵移除對照實驗：啟動")
    os.makedirs(args.output_root, exist_ok=True)

    df = load_and_preprocess_data(args.data_file)

    max_date = df["ds"].max()
    test_start = max_date - timedelta(days=DEFAULTS["test_days"] - 1)
    train_end = test_start - timedelta(days=1)
    print("\n📅 時間切分：")
    print(f"   訓練+驗證期: {DEFAULTS['start_date']} → {train_end.date()}")
    print(f"   測試期      : {test_start.date()} → {max_date.date()}")

    configs = load_all_configs(args.config_dir, args.config_archive)
    if not configs:
        print("❌ 找不到任何 JSON 設定（corrected_config_*.json）")
        return

    run_one_setting(df, configs, "original", os.path.join(args.output_root, "original"))
    run_one_setting(df, configs, "no_weather", os.path.join(args.output_root, "no_weather"))

    # 產生差異表
    make_diff_tables(args.output_root)
    print("\n📊 已輸出差異表：")
    print(f" - {os.path.join(args.output_root, 'comparison_validation_diff.csv')}")
    print(f" - {os.path.join(args.output_root, 'comparison_test_diff.csv')}")
    print(f" - {os.path.join(args.output_root, 'comparison_validation_summary.csv')}")
    print(f" - {os.path.join(args.output_root, 'comparison_test_summary.csv')}")

if __name__ == "__main__":
    main()
