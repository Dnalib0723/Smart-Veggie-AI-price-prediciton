# -*- coding: utf-8 -*-
"""
corrected_ml_pipeline_new_fixedonly_skipSFM_autosave_split_predict_next7.py

任務
----
- 從 corrected_ml_pipeline_results_whitelist_RH_0.0.zip 讀取每個 vege_id 的 best_params
- 以固定「白名單」特徵在 2022_2025_daily_vege_weather_price_LOW.csv 針對每個蔬菜訓練 XGBoost
- 以「資料的最後一天」作為 predict_date，往後遞迴預測 7 天（target_date = predict_date+1 ~ +7）
  * 若某個 vege_id 沒有出現這個 predict_date，會自動退回使用該菜自己的最後一天，並在終端顯示提示
- 產出 output.csv 欄位：[id, vege_id, predict_date, target_date, predict_price]
- 將所有蔬菜模型（不同參數、同一組特徵）打包成單一 PKL

使用
----
python corrected_ml_pipeline_new_fixedonly_skipSFM_autosave_split_predict_next7.py \
  --data /path/to/2022_2025_daily_vege_weather_price_LOW.csv \
  --params_zip /path/to/corrected_ml_pipeline_results_whitelist_RH_0.0.zip \
  --output_csv /path/to/output.csv \
  --models_pkl /path/to/models_low_next7.pkl
"""

import os
import io
import json
import zipfile
import argparse
from datetime import timedelta

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

# ===== 固定白名單特徵（與 FIXED_ALLOWED_FEATURES 對齊） =====
SELECTED_FEATURES = [
    # 時間
    "year",
    "month",
    "dayofweek",
    "dayofyear",
    "week",
    "day_sin",
    # 價格
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
    "y_volatility_7",
    "y_volatility_14",
    "y_above_ma7",
    "y_above_ma30",
    # 氣象（t-1 統計）
    "StnPres_std_14",
    "WS_ma_3",
    "WS_ma_7",
    "WS_ma_14",
    "Precp_ma_14",
    "Precp_ma_30",
    "Precp_z30",
]

WEATHER_BASE_COLS = ["StnPres", "WS", "Precp"]


# ---------- 讀檔 ----------
def read_csv_any(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"找不到資料檔: {csv_path}")
    for enc in ["utf-8", "utf-8-sig", "big5", "gbk", "cp950", "latin-1"]:
        try:
            return pd.read_csv(csv_path, encoding=enc)
        except Exception:
            continue
    return pd.read_csv(csv_path, encoding="latin-1", errors="ignore")


def load_params_from_zip(zip_path: str) -> dict:
    """
    回傳 { vege_id(str): best_params(dict) }
    JSON 結構預期包含： "vege_id": "...", "best_params": {...}
    """
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"找不到參數壓縮檔: {zip_path}")
    mapping = {}
    with zipfile.ZipFile(zip_path, "r") as zf:
        for name in zf.namelist():
            if not name.lower().endswith(".json"):
                continue
            with zf.open(name) as f:
                raw = f.read()
                try:
                    data = json.loads(raw.decode("utf-8"))
                except UnicodeDecodeError:
                    data = json.loads(raw.decode("utf-8-sig"))
                vid = str(data.get("vege_id", "")).strip()
                bp = data.get("best_params", None)
                if vid and isinstance(bp, dict):
                    # 增補常用鍵
                    bp = {
                        **bp,
                        "random_state": bp.get("random_state", 42),
                        "n_jobs": bp.get("n_jobs", -1),
                        "verbosity": bp.get("verbosity", 0),
                    }
                    mapping[vid] = bp
    return mapping


# ---------- 特徵工程（與你的固定白名單流程一致的最小版） ----------
def add_time_features_min(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    ds = df["ds"]
    df["year"] = ds.dt.year
    df["month"] = ds.dt.month
    df["dayofweek"] = ds.dt.dayofweek
    df["dayofyear"] = ds.dt.dayofyear
    df["week"] = ds.dt.isocalendar().week.astype(int)
    df["day_sin"] = np.sin(2 * np.pi * df["dayofyear"] / 365.0)
    return df


def add_price_features_min(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values("ds").reset_index(drop=True)
    for lag in [1, 3, 7, 14, 30]:
        df[f"y_lag_{lag}"] = df["y"].shift(lag)
    for w in [7, 14, 30]:
        df[f"y_ma_{w}"] = df["y"].shift(1).rolling(w, min_periods=1).mean()
    df["y_change_1"] = df["y"].shift(1) - df["y"].shift(2)
    df["y_change_7"] = df["y"].shift(7) - df["y"].shift(8)
    df["y_volatility_7"] = df["y"].shift(1).rolling(7, min_periods=1).std()
    df["y_volatility_14"] = df["y"].shift(1).rolling(14, min_periods=1).std()
    df["y_above_ma7"] = (df["y"] > df["y_ma_7"]).astype(int)
    df["y_above_ma30"] = (df["y"] > df["y_ma_30"]).astype(int)
    return df


def add_weather_features_min(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in WEATHER_BASE_COLS:
        if col not in df.columns:
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")
        base = df[col].shift(1)  # t-1 避免洩漏
        if col == "WS":
            for w in [3, 7, 14]:
                df[f"{col}_ma_{w}"] = base.rolling(w, min_periods=1).mean()
        if col == "Precp":
            for w in [14, 30]:
                df[f"{col}_ma_{w}"] = base.rolling(w, min_periods=1).mean()
            roll_mean = base.rolling(30, min_periods=5).mean()
            roll_std = base.rolling(30, min_periods=5).std()
            z = (df[col] - roll_mean) / (roll_std.replace(0, np.nan))
            df["Precp_z30"] = z.fillna(0)
        if col == "StnPres":
            df[f"{col}_std_14"] = base.rolling(14, min_periods=1).std()
    return df


def build_features_min(df: pd.DataFrame) -> pd.DataFrame:
    df = add_time_features_min(df)
    df = add_weather_features_min(df)
    df = add_price_features_min(df)
    return df


def get_feature_list(df: pd.DataFrame) -> list:
    return [c for c in SELECTED_FEATURES if c in df.columns]


# ---------- 遞迴 7 天預測（每個蔬菜） ----------
def forecast_next_7_days(
    df_veg_hist: pd.DataFrame, model: XGBRegressor, global_last_date: pd.Timestamp
):
    """
    df_veg_hist: 單一 vege_id 歷史資料（含 y 與天氣欄位），已排序
    model: 已訓練模型
    global_last_date: 整體資料集最後一天（優先使用）；若該菜當天無資料則退回該菜自己的最後一天
    回傳 (pred_df, used_last_date)
    """
    # 優先用全體最後一天，若該蔬菜沒有那天就改用自身最後一天
    if (df_veg_hist["ds"] == global_last_date).sum() == 0:
        used_last = df_veg_hist["ds"].max()
        print(
            f"   ⚠️ 該菜在全體最後日 {global_last_date.date()} 無資料，改用自身最後日 {used_last.date()} 當 predict_date"
        )
    else:
        used_last = global_last_date

    last_row = df_veg_hist[df_veg_hist["ds"] == used_last].iloc[-1]

    # 天氣欄位 carry-forward
    carry_weather = {
        c: last_row[c] for c in WEATHER_BASE_COLS if c in df_veg_hist.columns
    }

    work_df = df_veg_hist.copy()
    preds = []
    for i in range(1, 8):
        tgt_date = used_last + timedelta(days=i)
        new_row = {c: np.nan for c in work_df.columns}
        new_row["ds"] = tgt_date
        new_row["vege_id"] = work_df["vege_id"].iloc[0]
        for k, v in carry_weather.items():
            new_row[k] = v
        work_df = pd.concat([work_df, pd.DataFrame([new_row])], ignore_index=True)
        work_df = work_df.sort_values("ds").reset_index(drop=True)
        feat_df = build_features_min(work_df.copy())
        feat_cols = get_feature_list(feat_df)
        x_in = feat_df.iloc[[-1]][feat_cols].fillna(0.0)
        y_hat = float(model.predict(x_in)[0])
        if y_hat < 0:
            y_hat = 0.0
        # 寫回 y 供下一日的滯後/移動特徵使用
        work_df.loc[work_df.index[-1], "y"] = y_hat
        preds.append({"target_date": tgt_date.date(), "predict_price": y_hat})
    return pd.DataFrame(preds), used_last


# ---------- 主流程 ----------
def main(args):
    # 讀資料
    df = read_csv_any(args.data)
    # 欄位正規化
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
    df = (
        df.dropna(subset=["ds", "y", "vege_id"])
        .sort_values(["vege_id", "ds"])
        .reset_index(drop=True)
    )

    params_map = load_params_from_zip(args.params_zip)
    if not params_map:
        raise ValueError(
            "無法從 zip 讀到任何 best_params（請檢查 JSON 結構與 vege_id）"
        )

    global_last_date = df["ds"].max()
    print(f"[INFO] 全體資料最後一天 predict_date = {global_last_date.date()}")

    out_rows = []
    packed = {
        "features": SELECTED_FEATURES,
        "predict_date_global": str(global_last_date.date()),
        "models": {},
    }
    running_id = 1

    for vid in sorted(df["vege_id"].unique().tolist()):
        sub = df[df["vege_id"] == vid].copy()
        if len(sub) < 50:
            print(f"[SKIP] vege_id={vid} 樣本不足 ({len(sub)} < 50)")
            continue
        if vid not in params_map:
            print(f"[SKIP] vege_id={vid} 在 zip 無對應 best_params")
            continue

        # 構建特徵
        sub_feat = build_features_min(sub.copy())
        feat_cols = get_feature_list(sub_feat)
        if len(feat_cols) < 8:
            print(f"[SKIP] vege_id={vid} 可用特徵過少 ({len(feat_cols)})")
            continue

        X = sub_feat[feat_cols].fillna(0.0).values
        y = sub_feat["y"].values
        mask = ~np.isnan(y)
        if mask.sum() < 30:
            print(f"[SKIP] vege_id={vid} 有效樣本不足")
            continue
        X = X[mask]
        y = y[mask]

        best_params = {
            k: v
            for k, v in params_map[vid].items()
            if k
            in {
                "n_estimators",
                "max_depth",
                "learning_rate",
                "subsample",
                "colsample_bytree",
                "reg_alpha",
                "reg_lambda",
                "random_state",
                "n_jobs",
                "verbosity",
                "min_child_weight",
                "gamma",
            }
        }
        model = XGBRegressor(**best_params)
        model.fit(X, y)

        base_cols = ["ds", "vege_id", "y"] + [
            c for c in sub_feat.columns if c in WEATHER_BASE_COLS
        ]
        preds_df, used_last = forecast_next_7_days(
            sub_feat[base_cols].copy(), model, global_last_date
        )
        predict_date_str = used_last.date().isoformat()

        for _, r in preds_df.iterrows():
            out_rows.append(
                {
                    "id": running_id,
                    "vege_id": vid,
                    "predict_date": predict_date_str,
                    "target_date": (
                        r["target_date"].isoformat()
                        if hasattr(r["target_date"], "isoformat")
                        else str(r["target_date"])
                    ),
                    "predict_price": float(r["predict_price"]),
                }
            )
            running_id += 1

        packed["models"][vid] = {
            "params": best_params,
            "feature_names": feat_cols,
            "predict_date": predict_date_str,
            "model": model,
        }

    if not out_rows:
        raise RuntimeError("沒有可輸出的預測列，請檢查資料與 zip 參數對應。")

    out_df = pd.DataFrame(
        out_rows,
        columns=["id", "vege_id", "predict_date", "target_date", "predict_price"],
    )
    out_df.to_csv(args.output_csv, index=False, encoding="utf-8-sig")
    print(f"[OK] 已輸出: {args.output_csv}（{len(out_df)} 列）")

    import pickle

    with open(args.models_pkl, "wb") as f:
        pickle.dump(packed, f)
    print(f"[OK] 已打包模型: {args.models_pkl}（{len(packed['models'])} 個 vege 模型）")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data",
        type=str,
        required=True,
        help="2022_2025_daily_vege_weather_price_LOW.csv 路徑",
    )
    ap.add_argument(
        "--params_zip",
        type=str,
        required=True,
        help="corrected_ml_pipeline_results_whitelist_RH_0.0.zip 路徑",
    )
    ap.add_argument(
        "--output_csv", type=str, default="output.csv", help="輸出 CSV 檔路徑"
    )
    ap.add_argument(
        "--models_pkl",
        type=str,
        default="models_low_next7.pkl",
        help="輸出單一 PKL 檔路徑",
    )
    args = ap.parse_args()
    main(args)
