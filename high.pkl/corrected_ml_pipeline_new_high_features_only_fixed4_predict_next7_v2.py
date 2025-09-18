# -*- coding: utf-8 -*-
"""
corrected_ml_pipeline_new_high_features_only_fixed4_predict_next7_v2.py

差異（針對 v1 修正）
------------------
- **predict_date 改為每個 vege_id 自己的最後一天**（而非全體資料的最大日期），
  以避免當某些蔬菜在尾端缺資料時，出現 `ds == predict_date` 找不到列而導致 out-of-bounds。
- `forecast_next_7_days()` 依照傳入的 per-vege `last_date` 建立 carry-forward 天氣與遞迴推進。

其餘功能與參數相同：
- 從 zip 讀每個 vege_id 的 best_params
- 用固定一組特徵訓練 XGBoost
- 遞迴預測 7 天，輸出 output.csv（id, vege_id, predict_date, target_date, predict_price）
- 打包所有模型到單一 PKL
"""

import os
import io
import json
import zipfile
import argparse
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

SELECTED_FEATURES = [
    "y_above_ma30",
    "y_above_ma7",
    "y_lag_1",
    "y_ma_7",
    "y_ma_14",
    "y_ma_30",
    "y_lag_3",
    "y_lag_7",
    "day_sin",
    "y_lag_14",
    "dayofyear",
    "y_lag_30",
    "dayofweek",
    "day_cos",
    "Temperature_ma_30",
    "y_volatility_14",
    "StnPres_std_30",
    "y_volatility_7",
    "y_change_1",
    "Temperature_ma_14",
    "Precp_ma_30",
    "Temperature",
    "StnPres",
    "Temperature_delta1",
]

WEATHER_COLS = ["StnPres", "Temperature", "RH", "WS", "Precp", "typhoon"]

def read_csv_any_encoding(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"找不到資料檔: {csv_path}")
    for enc in ["utf-8", "utf-8-sig", "big5", "gbk", "cp950", "latin-1"]:
        try:
            return pd.read_csv(csv_path, encoding=enc)
        except Exception:
            continue
    return pd.read_csv(csv_path, encoding="latin-1", errors="ignore")

def load_params_from_zip(zip_path: str) -> dict:
    mapping = {}
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"找不到參數壓縮檔: {zip_path}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        for name in zf.namelist():
            if not name.lower().endswith(".json"):
                continue
            with zf.open(name) as f:
                try:
                    data = json.loads(f.read().decode("utf-8"))
                except UnicodeDecodeError:
                    data = json.loads(f.read().decode("utf-8-sig"))
                vid = str(data.get("vege_id", "")).strip()
                best_params = data.get("best_params", None)
                if vid and isinstance(best_params, dict):
                    best_params = {
                        **best_params,
                        "random_state": best_params.get("random_state", 42),
                        "n_jobs": best_params.get("n_jobs", -1),
                        "verbosity": best_params.get("verbosity", 0),
                    }
                    mapping[vid] = best_params
    return mapping

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    ds = df["ds"]
    df["dayofweek"] = ds.dt.dayofweek
    df["dayofyear"] = ds.dt.dayofyear
    df["day_sin"] = np.sin(2 * np.pi * df["dayofyear"] / 365.0)
    df["day_cos"] = np.cos(2 * np.pi * df["dayofyear"] / 365.0)
    return df

def add_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    windows = [3, 7, 14, 30]
    for col in WEATHER_COLS:
        if col not in df.columns:
            continue
        df[col] = pd.to_numeric(df[col], errors="ignore")
        base = df[col].shift(1)
        for w in windows:
            df[f"{col}_ma_{w}"] = base.rolling(w, min_periods=1).mean()
            df[f"{col}_std_{w}"] = base.rolling(w, min_periods=1).std()
        roll_mean = base.rolling(30, min_periods=5).mean()
        roll_std = base.rolling(30, min_periods=5).std()
        df[f"{col}_delta1"] = base.diff(1)
        df[f"{col}_z30"] = (base - roll_mean) / (roll_std.replace(0, np.nan))
        df[f"{col}_z30"] = df[f"{col}_z30"].fillna(0)
    return df

def add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values("ds").reset_index(drop=True)
    for lag in [1, 3, 7, 14, 30]:
        df[f"y_lag_{lag}"] = df["y"].shift(lag)
    for w in [7, 14, 30]:
        df[f"y_ma_{w}"] = df["y"].shift(1).rolling(w, min_periods=1).mean()
    df["y_change_1"] = df["y"].shift(1) - df["y"].shift(2)
    df["y_volatility_7"] = df["y"].shift(1).rolling(7, min_periods=1).std()
    df["y_volatility_14"] = df["y"].shift(1).rolling(14, min_periods=1).std()
    df["y_above_ma7"] = (df["y"] > df["y_ma_7"]).astype(int)
    df["y_above_ma30"] = (df["y"] > df["y_ma_30"]).astype(int)
    return df

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = add_time_features(df)
    if any(col in df.columns for col in WEATHER_COLS):
        df = add_weather_features(df)
    df = add_price_features(df)
    return df

def get_available_features(df: pd.DataFrame) -> list:
    return [c for c in SELECTED_FEATURES if c in df.columns]

def forecast_next_7_days(df_veg: pd.DataFrame, model: XGBRegressor, last_date: pd.Timestamp) -> pd.DataFrame:
    # 若該蔬菜沒有與 last_date 相等的列，改用該蔬菜自己的最後一天
    if (df_veg["ds"] == last_date).sum() == 0:
        last_date = df_veg["ds"].max()
    last_row = df_veg[df_veg["ds"] == last_date].iloc[-1]

    carry_weather = {col: last_row[col] for col in WEATHER_COLS if col in df_veg.columns}
    work_df = df_veg.copy()

    preds = []
    for i in range(1, 8):
        tgt_date = last_date + timedelta(days=i)
        new_row = {c: np.nan for c in work_df.columns}
        new_row["ds"] = tgt_date
        new_row["vege_id"] = work_df["vege_id"].iloc[0]
        for k, v in carry_weather.items():
            new_row[k] = v
        work_df = pd.concat([work_df, pd.DataFrame([new_row])], ignore_index=True)
        work_df = work_df.sort_values("ds").reset_index(drop=True)
        feat_df = build_features(work_df.copy())
        feat_cols = get_available_features(feat_df)
        x_input = feat_df.iloc[[-1]][feat_cols].fillna(0.0)
        y_hat = float(model.predict(x_input)[0])
        if y_hat < 0:
            y_hat = 0.0
        work_df.iloc[-1, work_df.columns.get_loc("y")] = y_hat
        preds.append({"target_date": tgt_date.date(), "predict_price": y_hat})
    return pd.DataFrame(preds), last_date

def main(args):
    df = read_csv_any_encoding(args.data)
    if "ObsTime" in df.columns:
        df["ds"] = pd.to_datetime(df["ObsTime"], errors="coerce")
    elif "ds" in df.columns:
        df["ds"] = pd.to_datetime(df["ds"], errors="coerce")
    else:
        raise ValueError("資料需包含 ObsTime 或 ds 欄位")

    if "avg_price_per_kg" in df.columns:
        df["y"] = pd.to_numeric(df["avg_price_per_kg"], errors="coerce")
    elif "y" not in df.columns:
        raise ValueError("資料需包含 avg_price_per_kg 或 y 欄位")

    if "vege_id" not in df.columns:
        raise ValueError("資料需包含 vege_id 欄位")
    df["vege_id"] = df["vege_id"].astype(str)

    df = df.dropna(subset=["ds", "y", "vege_id"]).copy()
    df = df.sort_values(["vege_id", "ds"]).reset_index(drop=True)

    params_map = load_params_from_zip(args.params_zip)
    if not params_map:
        raise ValueError("無法從 zip 讀到任何 best_params（請確認 zip JSON 結構與 vege_id）")

    out_rows = []
    packed_models = {
        "features": SELECTED_FEATURES,
        "models": {}
    }

    all_veg_ids = sorted(df["vege_id"].unique().tolist())
    running_id = 1
    for vid in all_veg_ids:
        sub = df[df["vege_id"] == vid].copy()
        if len(sub) < 50:
            print(f"[SKIP] vege_id={vid}: 資料量不足 ({len(sub)})")
            continue
        if vid not in params_map:
            print(f"[SKIP] vege_id={vid}: zip 中沒有對應參數 best_params")
            continue

        # 每個蔬菜自己的最後一天為 predict_date
        veg_last_date = sub["ds"].max()

        sub_feat = build_features(sub.copy())
        feat_cols = get_available_features(sub_feat)
        if len(feat_cols) < 8:
            print(f"[SKIP] vege_id={vid}: 可用特徵過少 ({len(feat_cols)})")
            continue

        X = sub_feat[feat_cols].fillna(0.0).values
        y = sub_feat["y"].values
        mask = ~np.isnan(y)
        if mask.sum() < 30:
            print(f"[SKIP] vege_id={vid}: 有效 y 樣本不足")
            continue
        X = X[mask]
        y = y[mask]

        best_params = params_map[vid].copy()
        valid_keys = {
            "n_estimators","max_depth","learning_rate","subsample",
            "colsample_bytree","reg_alpha","reg_lambda",
            "random_state","n_jobs","verbosity",
            "min_child_weight","gamma"
        }
        best_params = {k:v for k,v in best_params.items() if k in valid_keys}

        model = XGBRegressor(**best_params)
        model.fit(X, y)

        base_cols = ["ds","vege_id","y"] + [c for c in sub_feat.columns if c in WEATHER_COLS]
        preds_df, used_last_date = forecast_next_7_days(sub_feat[base_cols].copy(), model, veg_last_date)
        predict_date_str = used_last_date.date().isoformat()

        for _, row in preds_df.iterrows():
            out_rows.append({
                "id": running_id,
                "vege_id": vid,
                "predict_date": predict_date_str,
                "target_date": row["target_date"].isoformat() if hasattr(row["target_date"], "isoformat") else str(row["target_date"]),
                "predict_price": float(row["predict_price"]),
            })
            running_id += 1

        packed_models["models"][vid] = {
            "params": best_params,
            "feature_names": feat_cols,
            "predict_date": predict_date_str,
            "model": model,
        }

    if not out_rows:
        raise RuntimeError("沒有任何可輸出的預測列，請檢查資料與參數對應是否完整。")

    out_df = pd.DataFrame(out_rows, columns=["id","vege_id","predict_date","target_date","predict_price"])
    out_df.to_csv(args.output_csv, index=False, encoding="utf-8-sig")
    print(f"[OK] 已輸出預測結果: {args.output_csv} ({len(out_df)} 列)")

    import pickle
    with open(args.models_pkl, "wb") as f:
        pickle.dump(packed_models, f)
    print(f"[OK] 已打包模型到: {args.models_pkl}（{len(packed_models['models'])} 個 vege 模型）")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="2022_2025_daily_vege_weather_price_HIGH.csv 路徑")
    parser.add_argument("--params_zip", type=str, required=True, help="corrected_ml_pipeline_results_high_0.0.zip 路徑")
    parser.add_argument("--output_csv", type=str, default="output.csv", help="輸出 CSV 路徑（欄位: id, vege_id, predict_date, target_date, predict_price）")
    parser.add_argument("--models_pkl", type=str, default="models_high_next7.pkl", help="輸出 PKL 路徑（打包所有蔬菜模型）")
    args = parser.parse_args()
    main(args)
