# -*- coding: utf-8 -*-
"""
修復數據洩漏的Prophet回歸變數測試
重點修復特徵創建中的時間洩漏問題
"""

import os
import numpy as np
import pandas as pd
import warnings
from datetime import datetime, timedelta
import json
import time

# 導入必要的庫
try:
    from prophet import Prophet

    PROPHET_AVAILABLE = True
    print("✅ Prophet載入成功")
except ImportError:
    print("❌ Prophet未安裝")
    PROPHET_AVAILABLE = False

try:
    from sklearn.metrics import r2_score, mean_absolute_error

    SKLEARN_AVAILABLE = True
    print("✅ sklearn載入成功")
except ImportError:
    print("❌ sklearn未安裝")
    SKLEARN_AVAILABLE = False

warnings.filterwarnings("ignore")

# ====== 設定 ======
CSV_FILE = "NEW_low_volatility_merged.csv"
OUTPUT_DIR = "fixed_prophet_test_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("🔧 修復數據洩漏的Prophet測試")
print(f"📁 輸出目錄: {OUTPUT_DIR}")

# Prophet配置
PROPHET_PARAMS = {
    "changepoint_prior_scale": 0.02,
    "n_changepoints": 15,
    "seasonality_prior_scale": 2.0,
    "seasonality_mode": "additive",
    "weekly_seasonality": 7,
    "yearly_seasonality": 15,
    "daily_seasonality": False,
    "mcmc_samples": 0,
    "uncertainty_samples": 100,
}


class FixedProphetTester:
    """修復數據洩漏的Prophet測試器"""

    def __init__(self):
        self.prophet_params = PROPHET_PARAMS
        self.test_results = []
        self.failed_vegetables = []
        self.success_vegetables = []

        print("🎯 使用修復後的安全特徵創建")

    def create_safe_features(self, df):
        """
        安全的特徵創建 - 避免數據洩漏

        關鍵原則：
        1. 所有特徵都必須基於歷史數據
        2. 不能使用當期或未來的信息
        3. 移動平均必須正確滯後
        """
        df = df.copy()

        if "y" not in df.columns:
            return df, []

        print("🔧 創建安全特徵（避免數據洩漏）...")
        created_features = []

        try:
            # 確保數據按時間排序
            if "ds" in df.columns:
                df = df.sort_values("ds").reset_index(drop=True)

            # ✅ 特徵1: price_lag_1 - 真正的昨日價格
            df["price_lag_1"] = df["y"].shift(1)
            created_features.append("price_lag_1")
            print("    ✅ price_lag_1: 使用shift(1)確保是昨日價格")

            # ✅ 特徵2: price_change_1 - 昨日相對於前日的變化
            # 注意：這是昨日的變化，不是今日的變化
            df["price_change_1"] = df["y"].shift(1) - df["y"].shift(2)
            # 第一和第二個值會是NaN，這是正確的
            created_features.append("price_change_1")
            print("    ✅ price_change_1: 昨日相對前日的變化")

            # ✅ 特徵3: price_ma_7 - 基於歷史數據的7天移動平均
            # 使用shift(1)確保只用歷史數據
            df["price_ma_7"] = df["y"].shift(1).rolling(window=7, min_periods=1).mean()
            created_features.append("price_ma_7")
            print("    ✅ price_ma_7: 基於歷史數據的移動平均")

            # 🔧 數據清理和驗證
            for feature in created_features[:]:
                if feature in df.columns:
                    try:
                        # 處理無限值
                        df[feature] = df[feature].replace([np.inf, -np.inf], np.nan)

                        # 檢查NaN比例
                        nan_ratio = df[feature].isna().sum() / len(df)
                        print(f"    📊 {feature}: NaN比例 {nan_ratio:.2%}")

                        if nan_ratio > 0.9:  # 超過90%是NaN才移除
                            print(f"    ⚠️ 移除 {feature} (NaN比例過高)")
                            created_features.remove(feature)
                            df.drop(columns=[feature], inplace=True)
                        else:
                            # 填充NaN值 - 注意：只用訓練期間的統計
                            # 這裡暫時用全局中位數，在訓練時會重新計算
                            if df[feature].notna().any():
                                median_val = df[feature].median()
                                if pd.isna(median_val):
                                    median_val = 0
                                df[feature] = df[feature].fillna(median_val)
                            else:
                                df[feature] = 0

                    except Exception as e:
                        print(f"    ❌ 特徵 {feature} 處理失敗: {e}")
                        if feature in created_features:
                            created_features.remove(feature)
                        if feature in df.columns:
                            df.drop(columns=[feature], inplace=True)

            print(f"  ✅ 成功創建 {len(created_features)} 個安全特徵")

            # 🧪 合理性檢查
            self.sanity_check_features(df, created_features)

            return df, created_features

        except Exception as e:
            print(f"  ❌ 特徵創建失敗: {e}")
            return df, []

    def sanity_check_features(self, df, features):
        """合理性檢查 - 確保沒有數據洩漏"""
        print("    🧪 執行合理性檢查...")

        for feature in features:
            if feature in df.columns:
                # 檢查與目標的相關性
                valid_mask = df[feature].notna() & df["y"].notna()
                if valid_mask.sum() > 10:
                    correlation = df.loc[valid_mask, feature].corr(
                        df.loc[valid_mask, "y"]
                    )
                    print(f"      📊 {feature} 與目標相關性: {correlation:.4f}")

                    if correlation > 0.95:
                        print(f"      🚨 警告: {feature} 相關性過高，可能有數據洩漏！")

                # 檢查第一行是否為NaN（滯後特徵應該是）
                if feature in ["price_lag_1", "price_change_1"]:
                    if not pd.isna(df[feature].iloc[0]):
                        print(
                            f"      🚨 警告: {feature} 第一行不是NaN，可能沒有正確滯後！"
                        )
                    else:
                        print(f"      ✅ {feature} 正確滯後（第一行為NaN）")

    def safe_prophet_predict(
        self, train_data, test_data, regressors=None, test_name=""
    ):
        """
        安全的Prophet預測 - 特別注意避免數據洩漏
        """
        try:
            # 準備數據
            train_prophet = train_data[["ds", "y"]].copy()
            test_prophet = test_data[["ds"]].copy()

            # 檢查數據有效性
            if len(train_prophet) < 10 or len(test_prophet) < 3:
                return None, []

            # 建立模型
            model = Prophet(**self.prophet_params)
            added_regressors = []

            # 添加回歸變數
            if regressors:
                for reg in regressors:
                    if reg in train_data.columns:
                        try:
                            train_feature = train_data[reg].copy()
                            test_feature = test_data[reg].copy()

                            # ⚠️ 重要：只使用訓練數據計算統計量
                            train_median = train_feature.median()
                            if pd.isna(train_median):
                                train_median = 0

                            # 填充缺失值
                            train_feature = train_feature.fillna(train_median)
                            test_feature = test_feature.fillna(train_median)

                            # 檢查變異性
                            if train_feature.std() > 1e-8:
                                train_prophet[reg] = train_feature
                                test_prophet[reg] = test_feature

                                model.add_regressor(
                                    reg,
                                    prior_scale=1.0,
                                    standardize=True,
                                    mode="additive",
                                )
                                added_regressors.append(reg)
                                print(f"    ✅ 添加回歸變數: {reg}")

                        except Exception as e:
                            print(f"    ⚠️ 回歸變數 {reg} 添加失敗: {e}")
                            continue

            # 訓練模型
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(train_prophet)

            # 預測
            forecast = model.predict(test_prophet)
            y_pred = forecast["yhat"].values

            # 後處理預測結果
            y_pred = np.where(np.isfinite(y_pred), y_pred, train_data["y"].mean())
            y_pred = np.maximum(y_pred, 0.01)  # 確保價格為正數

            return y_pred, added_regressors

        except Exception as e:
            print(f"    ❌ {test_name} 預測失敗: {e}")
            return None, []

    def evaluate_r2_safe(self, y_true, y_pred):
        """安全的R²計算"""
        try:
            # 移除NaN值
            mask = np.isfinite(y_true) & np.isfinite(y_pred)
            if mask.sum() < 3:
                return -999.0

            y_true_clean = y_true[mask]
            y_pred_clean = y_pred[mask]

            r2 = r2_score(y_true_clean, y_pred_clean)
            return float(r2) if np.isfinite(r2) else -999.0

        except:
            return -999.0

    def test_single_vegetable(self, df_veg, veg_id):
        """測試單一蔬菜"""

        min_samples = 50
        if len(df_veg) < min_samples:
            print(f"   ⚠️ 蔬菜 {veg_id}: 數據量過少 ({len(df_veg)} < {min_samples})")
            return None

        print(f"\n🥬 測試蔬菜ID: {veg_id} (數據量: {len(df_veg)})")

        # 創建安全特徵
        df_veg, created_features = self.create_safe_features(df_veg)

        if len(created_features) < 2:
            print(f"   ❌ 蔬菜 {veg_id}: 有效特徵太少 ({len(created_features)})")
            return None

        # 嚴格的時間序列分割
        if len(df_veg) >= 100:
            split_ratio = 0.8
        elif len(df_veg) >= 60:
            split_ratio = 0.75
        else:
            split_ratio = 0.7

        split_idx = int(len(df_veg) * split_ratio)
        train_data = df_veg.iloc[:split_idx].copy()
        test_data = df_veg.iloc[split_idx:].copy()

        if len(test_data) < 3:
            print(f"   ⚠️ 蔬菜 {veg_id}: 測試數據太少 ({len(test_data)})")
            return None

        print(f"   📊 訓練: {len(train_data)}, 測試: {len(test_data)}")

        # 🔍 額外檢查：確保特徵沒有未來信息
        print("   🔍 檢查特徵安全性...")
        for feature in created_features:
            if feature in train_data.columns:
                # 檢查訓練集最後一天的特徵是否使用了測試集信息
                last_train_feature = train_data[feature].iloc[-1]
                first_test_y = test_data["y"].iloc[0] if len(test_data) > 0 else None

                # 如果特徵值等於測試集第一天的y值，說明有洩漏
                if (
                    first_test_y is not None
                    and abs(last_train_feature - first_test_y) < 1e-6
                ):
                    print(f"      🚨 警告: {feature} 可能使用了測試期間的數據！")

        y_true = test_data["y"].values
        results = {}

        # 🎯 測試配置
        test_configs = [
            {
                "name": "基本Prophet",
                "regressors": None,
                "description": "無回歸變數的基準模型",
            },
            {
                "name": "核心3特徵",
                "regressors": [
                    r
                    for r in ["price_change_1", "price_lag_1", "price_ma_7"]
                    if r in created_features
                ],
                "description": "修復後的3個核心特徵",
            },
            {
                "name": "僅price_lag_1",
                "regressors": (
                    ["price_lag_1"] if "price_lag_1" in created_features else None
                ),
                "description": "僅使用昨日價格",
            },
        ]

        for config in test_configs:
            config_name = config["name"]
            regressors = config["regressors"]

            print(f"   🔄 測試配置: {config_name}")

            try:
                y_pred, used_regressors = self.safe_prophet_predict(
                    train_data, test_data, regressors, config_name
                )

                if y_pred is not None and len(y_pred) == len(y_true):
                    r2 = self.evaluate_r2_safe(y_true, y_pred)

                    if r2 != -999.0:
                        mae = float(mean_absolute_error(y_true, y_pred))

                        results[config_name] = {
                            "r2": r2,
                            "mae": mae,
                            "used_regressors": used_regressors,
                            "description": config["description"],
                        }

                        # 🚨 異常檢測
                        if r2 > 0.95:
                            print(
                                f"   🚨 異常高R²警告: {config_name} R²={r2:.4f} (可能仍有數據洩漏)"
                            )
                        elif r2 > 0.8:
                            print(
                                f"   📈 {config_name}: R²={r2:.4f}, MAE={mae:.4f} ✅ 優秀結果"
                            )
                        elif r2 > 0.5:
                            print(
                                f"   📊 {config_name}: R²={r2:.4f}, MAE={mae:.4f} ✅ 良好結果"
                            )
                        elif r2 > 0.0:
                            print(
                                f"   📉 {config_name}: R²={r2:.4f}, MAE={mae:.4f} ⚠️ 一般結果"
                            )
                        else:
                            print(
                                f"   ❌ {config_name}: R²={r2:.4f}, MAE={mae:.4f} ❌ 差結果"
                            )
                    else:
                        print(f"   ⚠️ {config_name}: R²計算失敗")
                else:
                    print(f"   ❌ {config_name}: 預測失敗")

            except Exception as e:
                print(f"   ❌ {config_name} 測試失敗: {e}")

        # 改善分析
        improvement_analysis = self.analyze_improvement(results)
        if improvement_analysis:
            print(f"   🎯 {improvement_analysis}")

        # 保存結果
        if results:
            result_summary = {
                "veg_id": veg_id,
                "data_samples": len(df_veg),
                "train_samples": len(train_data),
                "test_samples": len(test_data),
                "created_features": created_features,
                "test_results": results,
                "improvement_analysis": improvement_analysis,
            }
            return result_summary
        else:
            print(f"   ❌ 蔬菜 {veg_id}: 所有配置都失敗")
            return None

    def analyze_improvement(self, results):
        """分析改善效果"""
        if "基本Prophet" in results and "核心3特徵" in results:
            basic_r2 = results["基本Prophet"]["r2"]
            enhanced_r2 = results["核心3特徵"]["r2"]
            improvement = enhanced_r2 - basic_r2

            if basic_r2 < 0 and enhanced_r2 >= 0:
                return f"🎉 負轉正: {basic_r2:.4f} → {enhanced_r2:.4f} (+{improvement:.4f})"
            elif improvement > 0.1:
                return f"🚀 顯著改善: {basic_r2:.4f} → {enhanced_r2:.4f} (+{improvement:.4f})"
            elif improvement > 0.01:
                return f"📈 有改善: {basic_r2:.4f} → {enhanced_r2:.4f} (+{improvement:.4f})"
            elif improvement > 0:
                return f"📊 輕微改善: {basic_r2:.4f} → {enhanced_r2:.4f} (+{improvement:.4f})"
            else:
                return (
                    f"➡️ 無改善: {basic_r2:.4f} → {enhanced_r2:.4f} ({improvement:.4f})"
                )

        return None

    def run_fixed_test(self, df_all):
        """運行修復後的測試"""

        # 獲取所有蔬菜ID
        veg_counts = df_all.groupby("vege_id").size().sort_values(ascending=False)
        all_veg_ids = veg_counts.index.tolist()

        print(f"\n🔄 開始修復後的測試 - {len(all_veg_ids)} 種蔬菜")
        print(f"📊 預期結果: R²在0.3-0.8為優秀，0.8+需要檢查是否有洩漏")

        all_results = []
        improvement_stats = {
            "negative_to_positive": 0,
            "significant_improve": 0,
            "moderate_improve": 0,
            "slight_improve": 0,
            "no_improve": 0,
            "failed": 0,
            "suspicious_high_r2": 0,  # 新增：可疑的高R²
        }

        start_time = time.time()

        # 只測試前5種蔬菜，先驗證修復效果
        test_veg_ids = all_veg_ids[:5]
        print(f"🧪 首次驗證：測試前{len(test_veg_ids)}種蔬菜")

        for i, veg_id in enumerate(test_veg_ids, 1):
            print(f"\n[{i}/{len(test_veg_ids)}] " + "=" * 50)

            try:
                veg_data = df_all[df_all["vege_id"] == veg_id].copy()
                veg_data = veg_data.sort_values("ds").reset_index(drop=True)

                result = self.test_single_vegetable(veg_data, veg_id)

                if result:
                    all_results.append(result)
                    self.success_vegetables.append(veg_id)

                    # 統計改善效果
                    test_results = result["test_results"]
                    if "基本Prophet" in test_results and "核心3特徵" in test_results:
                        basic_r2 = test_results["基本Prophet"]["r2"]
                        enhanced_r2 = test_results["核心3特徵"]["r2"]
                        improvement = enhanced_r2 - basic_r2

                        # 檢查是否仍有可疑的高R²
                        if enhanced_r2 > 0.95:
                            improvement_stats["suspicious_high_r2"] += 1
                            print(
                                f"   🚨 蔬菜{veg_id}: R²仍然異常高 ({enhanced_r2:.4f})，需要進一步檢查"
                            )

                        if basic_r2 < 0 and enhanced_r2 >= 0:
                            improvement_stats["negative_to_positive"] += 1
                        elif improvement > 0.1:
                            improvement_stats["significant_improve"] += 1
                        elif improvement > 0.01:
                            improvement_stats["moderate_improve"] += 1
                        elif improvement > 0:
                            improvement_stats["slight_improve"] += 1
                        else:
                            improvement_stats["no_improve"] += 1

                else:
                    self.failed_vegetables.append(veg_id)
                    improvement_stats["failed"] += 1

            except Exception as e:
                print(f"   ❌ 蔬菜 {veg_id} 測試失敗: {e}")
                self.failed_vegetables.append(veg_id)
                improvement_stats["failed"] += 1

        elapsed_time = time.time() - start_time

        print(f"\n⏱️ 測試時間: {elapsed_time:.1f} 秒")
        print(f"✅ 成功測試: {len(all_results)}/{len(test_veg_ids)} 種蔬菜")

        if improvement_stats["suspicious_high_r2"] > 0:
            print(
                f"🚨 仍有 {improvement_stats['suspicious_high_r2']} 種蔬菜R²異常高，需要進一步調查"
            )

        return all_results, improvement_stats

    def print_diagnostic_summary(self, all_results, improvement_stats):
        """打印診斷總結"""
        print("\n" + "=" * 80)
        print("🔧 修復後測試結果診斷")
        print("=" * 80)

        if not all_results:
            print("❌ 無測試結果")
            return

        total_tested = len(all_results)

        print(f"\n📊 測試概況:")
        print(f"  測試蔬菜數: {total_tested}")
        print(f"  可疑高R²: {improvement_stats['suspicious_high_r2']} 種")

        # 分析R²分佈
        r2_values = []
        for result in all_results:
            if "核心3特徵" in result["test_results"]:
                r2 = result["test_results"]["核心3特徵"]["r2"]
                if r2 != -999:
                    r2_values.append(r2)

        if r2_values:
            print(f"\n📊 核心3特徵R²分佈:")
            print(f"  平均R²: {np.mean(r2_values):.4f}")
            print(f"  R²範圍: [{np.min(r2_values):.4f}, {np.max(r2_values):.4f}]")
            print(
                f"  超過0.9的比例: {sum(1 for r2 in r2_values if r2 > 0.9) / len(r2_values) * 100:.1f}%"
            )
            print(
                f"  超過0.95的比例: {sum(1 for r2 in r2_values if r2 > 0.95) / len(r2_values) * 100:.1f}%"
            )

        print(f"\n💡 診斷結論:")
        if improvement_stats["suspicious_high_r2"] == 0:
            print(f"  ✅ 修復成功！沒有異常高的R²值")
        elif improvement_stats["suspicious_high_r2"] < total_tested * 0.3:
            print(f"  ⚠️ 部分修復成功，仍有少數異常值需要調查")
        else:
            print(f"  ❌ 修復不完全，仍有大量異常高R²，需要進一步檢查特徵創建邏輯")

        # 合理性檢查建議
        print(f"\n🎯 下一步建議:")
        if improvement_stats["suspicious_high_r2"] > 0:
            print(f"  1. 手動檢查異常高R²的蔬菜數據")
            print(f"  2. 驗證特徵計算是否真的避免了未來信息")
            print(f"  3. 檢查Prophet模型參數是否合適")
        else:
            print(f"  1. 可以擴展測試到所有29種蔬菜")
            print(f"  2. 結果看起來合理，可以進入生產階段")


def main():
    """主程式"""

    if not PROPHET_AVAILABLE or not SKLEARN_AVAILABLE:
        print("❌ 缺少必要的庫")
        return

    # 載入數據
    print("\n📊 載入數據...")

    if not os.path.exists(CSV_FILE):
        print(f"❌ 找不到數據文件: {CSV_FILE}")
        return

    try:
        df_all = pd.read_csv(CSV_FILE)
        print(f"✅ 成功載入 {len(df_all):,} 筆原始數據")

        # 數據預處理
        df_all["ds"] = pd.to_datetime(df_all["ObsTime"], errors="coerce")
        df_all["y"] = pd.to_numeric(df_all["avg_price_per_kg"], errors="coerce")
        df_all["vege_id"] = df_all["vege_id"].astype(str)

        # 移除無效數據
        initial_count = len(df_all)
        df_all = df_all.dropna(subset=["ds", "y", "vege_id"])
        print(f"✅ 清理後剩餘 {len(df_all):,} 筆有效數據")

        # 蔬菜統計
        veg_counts = df_all.groupby("vege_id").size().sort_values(ascending=False)
        print(f"🥬 總蔬菜種類: {len(veg_counts)}")

    except Exception as e:
        print(f"❌ 數據載入失敗: {e}")
        return

    # 運行修復後的測試
    print(f"\n🚀 開始運行修復後的測試...")
    try:
        tester = FixedProphetTester()
        all_results, improvement_stats = tester.run_fixed_test(df_all)

        # 診斷結果
        tester.print_diagnostic_summary(all_results, improvement_stats)

        print(f"\n📁 如果結果合理，可以擴展到所有29種蔬菜")

    except Exception as e:
        print(f"❌ 測試執行失敗: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    print("=" * 80)
    print("🔧 修復數據洩漏的Prophet測試系統")
    print("=" * 80)
    print("📅 執行時間:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print()

    main()

    print("\n" + "=" * 80)
    print("🏁 診斷測試完成")
    print("=" * 80)
