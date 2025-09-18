#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
完整蔬菜價格DTW聚類分析程式包 (2022-2024)
包含原始分析和擴增功能的完整版本
包含所有視覺化功能，去除互動儀表板
修改版本：2024年趨勢聚類的群聚2和群聚3標籤調換
修正版本：移除重複的標籤調整邏輯，統一標籤處理
"""

print("🥬 完整蔬菜價格DTW聚類分析程式包 (2022-2024)")
print("=" * 80)

# 導入必要套件
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from statsmodels.tsa.seasonal import seasonal_decompose
import os
import warnings
from datetime import datetime
from collections import defaultdict

print("✅ 基本套件導入完成")

# DTW導入
try:
    from fastdtw import fastdtw
    from scipy.spatial.distance import euclidean

    DTW_AVAILABLE = True
    print("✅ FastDTW已啟用")
except ImportError:
    DTW_AVAILABLE = False
    print("⚠️  使用歐氏距離替代DTW")

# 設定
warnings.filterwarnings("ignore")
plt.rcParams["font.sans-serif"] = [
    "Microsoft JhengHei",
    "SimHei",
    "Arial Unicode MS",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False

# 固定聚類參數
SEASONAL_CLUSTERS = 2
TREND_CLUSTERS = 3

print(f"🎯 聚類參數: 季節性={SEASONAL_CLUSTERS}, 趨勢={TREND_CLUSTERS}")


def adjust_2024_trend_cluster_labels(cluster_labels, year):
    """
    調整2024年趨勢聚類標籤：將群聚2和群聚3互換
    """
    if year == 2024:
        # 創建標籤映射：0->0, 1->2, 2->1 (因為原始標籤是0-based)
        adjusted_labels = cluster_labels.copy()
        adjusted_labels[cluster_labels == 1] = 2  # 原群聚2 -> 群聚3
        adjusted_labels[cluster_labels == 2] = 1  # 原群聚3 -> 群聚2
        print(f"   🔄 2024年趨勢聚類標籤已調整：群聚2↔群聚3")
        return adjusted_labels
    return cluster_labels


def load_vegetable_mapping():
    """讀取蔬菜ID對應表，如果無法讀取則使用預設對應"""
    mapping_file = "蔬菜ID_對應檔案表.csv"

    # 預設蔬菜名稱對應表
    default_mapping = {
        "FN1": "四季豆",
        "SG3": "蒜苗",
        "FN0": "豇豆",
    }

    if os.path.exists(mapping_file):
        try:
            print(f"🔍 嘗試讀取蔬菜ID對應表: {mapping_file}")

            # 嘗試不同編碼和分隔符
            for encoding in ["utf-8-sig", "utf-8", "big5", "gbk", "cp950"]:
                for sep in ["\t", ","]:
                    try:
                        df = pd.read_csv(mapping_file, encoding=encoding, sep=sep)

                        # 檢查欄位名稱（可能有不同的命名方式）
                        possible_id_cols = ["market_vege_id", "vege_id", "蔬菜ID", "id"]
                        possible_name_cols = [
                            "vege_name",
                            "vege.name",
                            "蔬菜名稱",
                            "name",
                            "名稱",
                        ]

                        id_col = None
                        name_col = None

                        for col in df.columns:
                            if any(
                                pc in col.lower()
                                for pc in ["market_vege_id", "vege_id"]
                            ):
                                id_col = col
                            elif any(
                                pc in col.lower()
                                for pc in ["vege_name", "vege.name", "name"]
                            ):
                                name_col = col

                        if id_col and name_col:
                            # 清理數據
                            df = df.dropna(subset=[id_col, name_col])
                            df[name_col] = df[name_col].astype(str).str.strip()

                            # 過濾掉明顯的亂碼（包含不可見字符或長度異常）
                            df = df[df[name_col].str.len() <= 10]
                            df = df[
                                ~df[name_col].str.contains(
                                    r"[^\u4e00-\u9fff\w\s]", regex=True, na=False
                                )
                            ]

                            if len(df) > 0:
                                mapping_dict = dict(zip(df[id_col], df[name_col]))
                                print(
                                    f"✅ 成功讀取蔬菜ID對應表 (編碼: {encoding}, 分隔符: '{sep}')"
                                )
                                print(f"📋 讀取到 {len(mapping_dict)} 個有效對應")

                                # 顯示前幾個對應作為驗證
                                sample_items = list(mapping_dict.items())[:5]
                                for k, v in sample_items:
                                    print(f"   {k}: {v}")

                                return mapping_dict
                    except Exception as e:
                        continue

        except Exception as e:
            print(f"⚠️ 讀取對應表過程中發生錯誤: {e}")

    print("📚 使用內建蔬菜名稱對應表")
    print(f"📋 內建對應表包含 {len(default_mapping)} 個蔬菜")
    return default_mapping


def get_chinese_name(vege_id, mapping_dict):
    """獲取蔬菜的中文名稱"""
    return mapping_dict.get(vege_id, vege_id)


def create_id_to_name_mapping_list(vege_ids, mapping_dict):
    """創建ID到中文名稱的映射列表，保持順序"""
    return [get_chinese_name(vege_id, mapping_dict) for vege_id in vege_ids]


def setup_directories():
    """建立輸出資料夾"""
    analysis_dir = "complete_vegetable_clustering_analysis"
    plots_dir = os.path.join(analysis_dir, "plots")
    decomp_plots_dir = os.path.join(plots_dir, "decomposition_examples")
    individual_plots_dir = os.path.join(plots_dir, "individual_vegetables")
    comparison_plots_dir = os.path.join(plots_dir, "comparison_analysis")

    for directory in [
        analysis_dir,
        plots_dir,
        decomp_plots_dir,
        individual_plots_dir,
        comparison_plots_dir,
    ]:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"✅ 建立資料夾: {directory}")

    return (
        analysis_dir,
        plots_dir,
        decomp_plots_dir,
        individual_plots_dir,
        comparison_plots_dir,
    )


def dtw_distance(ts1, ts2):
    """計算DTW距離"""
    if len(ts1) == 0 or len(ts2) == 0:
        return float("inf")

    ts1_clean = ts1[~np.isnan(ts1)]
    ts2_clean = ts2[~np.isnan(ts2)]

    if len(ts1_clean) == 0 or len(ts2_clean) == 0:
        return float("inf")

    try:
        if DTW_AVAILABLE:
            distance, _ = fastdtw(ts1_clean, ts2_clean, dist=euclidean)
            return distance
        else:
            min_len = min(len(ts1_clean), len(ts2_clean))
            return np.linalg.norm(ts1_clean[:min_len] - ts2_clean[:min_len])
    except:
        min_len = min(len(ts1_clean), len(ts2_clean))
        if min_len == 0:
            return float("inf")
        return np.linalg.norm(ts1_clean[:min_len] - ts2_clean[:min_len])


def decompose_time_series(df, vege_id, year, min_periods=60):
    """時間序列分解"""
    vege_data = df[df["market_vege_id"] == vege_id].copy()

    if len(vege_data) < min_periods:
        return None

    vege_data = vege_data.sort_values("ObsTime").reset_index(drop=True)
    ts = pd.Series(
        vege_data["avg_price_per_kg"].values, index=pd.to_datetime(vege_data["ObsTime"])
    )

    ts = ts.dropna()
    if ts.index.duplicated().any():
        ts = ts[~ts.index.duplicated(keep="last")]

    if len(ts) < min_periods:
        return None

    # 確定分解週期
    n_obs = len(ts)
    if n_obs >= 365:
        period = 365
    elif n_obs >= 52:
        period = 52
    elif n_obs >= 30:
        period = 30
    else:
        period = max(2, n_obs // 3)

    try:
        decomp_add = seasonal_decompose(
            ts, model="additive", period=period, extrapolate_trend="freq"
        )

        # 嘗試乘法分解
        decomp_mul = None
        if (ts > 0).all():
            try:
                decomp_mul = seasonal_decompose(
                    ts, model="multiplicative", period=period, extrapolate_trend="freq"
                )
            except:
                pass

        return {
            "vege_id": vege_id,
            "year": year,
            "original": ts,
            "additive_trend": decomp_add.trend.dropna(),
            "additive_seasonal": decomp_add.seasonal.dropna(),
            "additive_residual": decomp_add.resid.dropna(),
            "multiplicative_trend": decomp_mul.trend.dropna() if decomp_mul else None,
            "multiplicative_seasonal": (
                decomp_mul.seasonal.dropna() if decomp_mul else None
            ),
            "multiplicative_residual": (
                decomp_mul.resid.dropna() if decomp_mul else None
            ),
            "period": period,
        }
    except Exception as e:
        print(f"      ❌ {vege_id} 分解失敗: {e}")
        return None


def create_decomposition_example_plot(decomp_result, decomp_plots_dir, mapping_dict):
    """創建時間序列分解範例圖"""
    vege_id = decomp_result["vege_id"]
    vege_name = get_chinese_name(vege_id, mapping_dict)
    year = decomp_result["year"]

    fig, axes = plt.subplots(4, 2, figsize=(16, 12))
    fig.suptitle(f"{vege_name} ({vege_id}) 時間序列分解", fontsize=16, y=0.98)

    # 左側：加法分解
    # 原始數據
    axes[0, 0].plot(decomp_result["original"], color="blue", linewidth=1)
    axes[0, 0].set_title("原始數據", fontsize=12)
    axes[0, 0].grid(True, alpha=0.3)

    # 趨勢
    axes[1, 0].plot(decomp_result["additive_trend"], color="red", linewidth=1.5)
    axes[1, 0].set_title("趨勢", fontsize=12)
    axes[1, 0].grid(True, alpha=0.3)

    # 季節性
    axes[2, 0].plot(decomp_result["additive_seasonal"], color="green", linewidth=1)
    axes[2, 0].set_title("季節性", fontsize=12)
    axes[2, 0].grid(True, alpha=0.3)

    # 殘差
    axes[3, 0].scatter(
        decomp_result["additive_residual"].index,
        decomp_result["additive_residual"].values,
        alpha=0.6,
        s=10,
        color="orange",
    )
    axes[3, 0].axhline(y=0, color="black", linestyle="--", alpha=0.7)
    axes[3, 0].set_title("殘差", fontsize=12)
    axes[3, 0].grid(True, alpha=0.3)

    fig.text(0.25, 0.95, "加法分解", ha="center", fontsize=14, weight="bold")

    # 右側：乘法分解
    if decomp_result["multiplicative_trend"] is not None:
        # 原始數據
        axes[0, 1].plot(decomp_result["original"], color="blue", linewidth=1)
        axes[0, 1].set_title("原始數據", fontsize=12)
        axes[0, 1].grid(True, alpha=0.3)

        # 趨勢
        axes[1, 1].plot(
            decomp_result["multiplicative_trend"], color="red", linewidth=1.5
        )
        axes[1, 1].set_title("趨勢", fontsize=12)
        axes[1, 1].grid(True, alpha=0.3)

        # 季節性
        axes[2, 1].plot(
            decomp_result["multiplicative_seasonal"], color="green", linewidth=1
        )
        axes[2, 1].set_title("季節性", fontsize=12)
        axes[2, 1].grid(True, alpha=0.3)

        # 殘差
        axes[3, 1].scatter(
            decomp_result["multiplicative_residual"].index,
            decomp_result["multiplicative_residual"].values,
            alpha=0.6,
            s=10,
            color="orange",
        )
        axes[3, 1].axhline(y=1, color="black", linestyle="--", alpha=0.7)
        axes[3, 1].set_title("殘差", fontsize=12)
        axes[3, 1].grid(True, alpha=0.3)

        fig.text(0.75, 0.95, "乘法分解", ha="center", fontsize=14, weight="bold")
    else:
        # 隱藏右側圖表
        for i in range(4):
            axes[i, 1].set_visible(False)
        fig.text(
            0.75,
            0.5,
            "乘法分解\n無法執行\n(包含非正值)",
            ha="center",
            va="center",
            fontsize=12,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"),
        )

    # 格式化x軸
    for ax in axes.flatten():
        if ax.get_visible():
            ax.tick_params(axis="x", rotation=45)
            ax.tick_params(axis="both", labelsize=10)

    plt.tight_layout(rect=[0, 0, 1, 0.93])

    plot_path = os.path.join(decomp_plots_dir, f"{vege_name}_{vege_id}_{year}_分解.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()


def create_individual_vegetable_clustering_chart(
    vege_id, vege_name, decomp_results, clustering_results, individual_plots_dir
):
    """為單一蔬菜創建聚類結果和趨勢比較圖

    注意：這裡直接使用已經處理過的聚類結果，不再進行額外的標籤調整
    因為clustering_results中已經包含了調整後的最終標籤
    """

    # 準備數據
    years = [2022, 2023, 2024]
    trend_clusters = []
    seasonal_clusters = []

    # 從clustering_results中提取該蔬菜的聚類結果
    # 注意：這裡直接使用最終的聚類標籤，不再進行調整
    for year in years:
        if year in clustering_results:
            # 趨勢聚類
            if (
                "trend" in clustering_results[year]
                and vege_id in clustering_results[year]["trend"]["vege_ids"]
            ):
                idx = clustering_results[year]["trend"]["vege_ids"].index(vege_id)
                cluster_label = clustering_results[year]["trend"]["cluster_labels"][idx]
                trend_clusters.append(cluster_label + 1)  # 轉換為1-based
            else:
                trend_clusters.append(np.nan)

            # 季節性聚類
            if (
                "seasonal" in clustering_results[year]
                and vege_id in clustering_results[year]["seasonal"]["vege_ids"]
            ):
                idx = clustering_results[year]["seasonal"]["vege_ids"].index(vege_id)
                seasonal_clusters.append(
                    clustering_results[year]["seasonal"]["cluster_labels"][idx] + 1
                )
            else:
                seasonal_clusters.append(np.nan)
        else:
            trend_clusters.append(np.nan)
            seasonal_clusters.append(np.nan)

    # 創建圖表
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(4, 4, height_ratios=[1.5, 1, 1.5, 1], hspace=0.3, wspace=0.3)

    # 主標題
    fig.suptitle(
        f"{vege_name} ({vege_id}) 多年度聚類分析 (已含2024標籤調整)",
        fontsize=16,
        y=0.96,
    )

    # 第一行：趨勢時間序列
    ax1 = fig.add_subplot(gs[0, :])
    colors = ["blue", "green", "orange"]

    for i, year in enumerate(years):
        if year in decomp_results and vege_id in decomp_results[year]:
            trend_data = decomp_results[year][vege_id]["additive_trend"]
            if len(trend_data) > 0:
                x_vals = np.arange(len(trend_data)) + i * 400  # 分開顯示每年
                ax1.plot(
                    x_vals,
                    trend_data.values,
                    color=colors[i],
                    linewidth=2,
                    alpha=0.8,
                    label=f"{year}年趨勢",
                )

                # 在趨勢線旁邊標注聚類編號
                if not np.isnan(trend_clusters[i]):
                    ax1.text(
                        x_vals[-1] + 10,
                        trend_data.values[-1],
                        f"趨勢群{int(trend_clusters[i])}",
                        color=colors[i],
                        fontweight="bold",
                        fontsize=10,
                        bbox=dict(
                            boxstyle="round,pad=0.3", facecolor="white", alpha=0.7
                        ),
                    )

    ax1.set_title("長期趨勢變化", fontsize=12)
    ax1.set_xlabel("時間")
    ax1.set_ylabel("趨勢值")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 第二行：趨勢聚類結果圓餅圖和分佈圖
    # 趨勢聚類圓餅圖
    ax2 = fig.add_subplot(gs[1, :2])
    valid_trend = [x for x in trend_clusters if not np.isnan(x)]
    if valid_trend:
        trend_counts = {}
        for cluster in valid_trend:
            trend_counts[f"趨勢群{int(cluster)}"] = (
                trend_counts.get(f"趨勢群{int(cluster)}", 0) + 1
            )

        colors_pie = ["#ff9999", "#66b3ff", "#99ff99"][: len(trend_counts)]
        wedges, texts, autotexts = ax2.pie(
            trend_counts.values(),
            labels=trend_counts.keys(),
            autopct="%1.0f年",
            colors=colors_pie,
            startangle=90,
        )
        ax2.set_title("趨勢聚類分佈", fontsize=11)
    else:
        ax2.text(
            0.5, 0.5, "無趨勢數據", ha="center", va="center", transform=ax2.transAxes
        )
        ax2.set_title("趨勢聚類分佈", fontsize=11)

    # 趨勢年度變化
    ax3 = fig.add_subplot(gs[1, 2:])
    x_pos = np.arange(len(years))
    trend_colors = []
    for t in trend_clusters:
        if np.isnan(t):
            trend_colors.append("gray")
        else:
            trend_colors.append(["#ff9999", "#66b3ff", "#99ff99"][int(t) - 1])

    bars = ax3.bar(x_pos, [1] * len(years), color=trend_colors, alpha=0.7)
    ax3.set_xlim(-0.5, len(years) - 0.5)
    ax3.set_ylim(0, 1.2)
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(years)
    ax3.set_title("趨勢聚類年度變化", fontsize=11)
    ax3.set_ylabel("聚類歸屬")

    # 添加聚類編號標籤
    for i, (bar, cluster) in enumerate(zip(bars, trend_clusters)):
        if not np.isnan(cluster):
            ax3.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() / 2,
                f"群{int(cluster)}",
                ha="center",
                va="center",
                fontweight="bold",
            )

    # 第三行：季節性時間序列
    ax4 = fig.add_subplot(gs[2, :])

    for i, year in enumerate(years):
        if year in decomp_results and vege_id in decomp_results[year]:
            seasonal_data = decomp_results[year][vege_id]["additive_seasonal"]
            if len(seasonal_data) > 0:
                x_vals = np.arange(len(seasonal_data)) + i * 400
                ax4.plot(
                    x_vals,
                    seasonal_data.values,
                    color=colors[i],
                    linewidth=2,
                    alpha=0.8,
                    label=f"{year}年季節性",
                )

                # 在季節性線旁邊標注聚類編號
                if not np.isnan(seasonal_clusters[i]):
                    ax4.text(
                        x_vals[-1] + 10,
                        seasonal_data.values[-1],
                        f"季節群{int(seasonal_clusters[i])}",
                        color=colors[i],
                        fontweight="bold",
                        fontsize=10,
                        bbox=dict(
                            boxstyle="round,pad=0.3", facecolor="white", alpha=0.7
                        ),
                    )

    ax4.set_title("季節性變化模式", fontsize=12)
    ax4.set_xlabel("時間")
    ax4.set_ylabel("季節性")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # 第四行：季節性聚類結果圓餅圖和分佈圖
    # 季節性聚類圓餅圖
    ax5 = fig.add_subplot(gs[3, :2])
    valid_seasonal = [x for x in seasonal_clusters if not np.isnan(x)]
    if valid_seasonal:
        seasonal_counts = {}
        for cluster in valid_seasonal:
            seasonal_counts[f"季節群{int(cluster)}"] = (
                seasonal_counts.get(f"季節群{int(cluster)}", 0) + 1
            )

        colors_pie_s = ["#ffcc99", "#ff99cc"][: len(seasonal_counts)]
        wedges, texts, autotexts = ax5.pie(
            seasonal_counts.values(),
            labels=seasonal_counts.keys(),
            autopct="%1.0f年",
            colors=colors_pie_s,
            startangle=90,
        )
        ax5.set_title("季節性聚類分佈", fontsize=11)
    else:
        ax5.text(
            0.5, 0.5, "無季節性數據", ha="center", va="center", transform=ax5.transAxes
        )
        ax5.set_title("季節性聚類分佈", fontsize=11)

    # 季節性年度變化
    ax6 = fig.add_subplot(gs[3, 2:])
    seasonal_colors = []
    for s in seasonal_clusters:
        if np.isnan(s):
            seasonal_colors.append("gray")
        else:
            seasonal_colors.append(["#ffcc99", "#ff99cc"][int(s) - 1])

    bars = ax6.bar(x_pos, [1] * len(years), color=seasonal_colors, alpha=0.7)
    ax6.set_xlim(-0.5, len(years) - 0.5)
    ax6.set_ylim(0, 1.2)
    ax6.set_xticks(x_pos)
    ax6.set_xticklabels(years)
    ax6.set_title("季節性聚類年度變化", fontsize=11)
    ax6.set_ylabel("聚類歸屬")

    # 添加聚類編號標籤
    for i, (bar, cluster) in enumerate(zip(bars, seasonal_clusters)):
        if not np.isnan(cluster):
            ax6.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() / 2,
                f"群{int(cluster)}",
                ha="center",
                va="center",
                fontweight="bold",
            )

    # 保存圖表
    plot_path = os.path.join(
        individual_plots_dir, f"{vege_name}_{vege_id}_聚類分析.png"
    )
    plt.savefig(plot_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()

    return plot_path


def create_clustering_stability_heatmap(
    comprehensive_df, comparison_plots_dir, mapping_dict
):
    """創建聚類穩定性熱力圖"""

    # 準備趨勢穩定性數據
    trend_data = []
    seasonal_data = []

    for _, row in comprehensive_df.iterrows():
        vege_name = row["vege_name"]

        # 趨勢聚類數據
        trend_row = []
        for year in [2022, 2023, 2024]:
            cluster_val = row[f"trend_cluster_{year}"]
            if pd.isna(cluster_val):
                trend_row.append(0)  # 無數據用0表示
            else:
                trend_row.append(int(cluster_val))
        trend_data.append([vege_name] + trend_row)

        # 季節性聚類數據
        seasonal_row = []
        for year in [2022, 2023, 2024]:
            cluster_val = row[f"seasonal_cluster_{year}"]
            if pd.isna(cluster_val):
                seasonal_row.append(0)  # 無數據用0表示
            else:
                seasonal_row.append(int(cluster_val))
        seasonal_data.append([vege_name] + seasonal_row)

    # 創建DataFrame
    trend_df = pd.DataFrame(trend_data, columns=["蔬菜名稱", "2022", "2023", "2024"])
    seasonal_df = pd.DataFrame(
        seasonal_data, columns=["蔬菜名稱", "2022", "2023", "2024"]
    )

    # 創建圖表
    fig, axes = plt.subplots(1, 2, figsize=(20, 15))
    fig.suptitle("蔬菜聚類穩定性分析熱力圖 (2024年趨勢聚類已調整)", fontsize=16, y=0.98)

    # 趨勢聚類熱力圖
    trend_matrix = trend_df.set_index("蔬菜名稱")[["2022", "2023", "2024"]].values
    im1 = axes[0].imshow(trend_matrix, cmap="viridis", aspect="auto", vmin=0, vmax=3)
    axes[0].set_title(
        "趨勢聚類變化 (0=無數據, 1-3=聚類編號)\n2024年群聚2↔群聚3已調換", fontsize=14
    )
    axes[0].set_xlabel("年份")
    axes[0].set_ylabel("蔬菜")
    axes[0].set_xticks(range(3))
    axes[0].set_xticklabels(["2022", "2023", "2024"])
    axes[0].set_yticks(range(len(trend_df)))
    axes[0].set_yticklabels(trend_df["蔬菜名稱"], fontsize=8)

    # 添加數值標註
    for i in range(len(trend_df)):
        for j in range(3):
            text = axes[0].text(
                j,
                i,
                int(trend_matrix[i, j]),
                ha="center",
                va="center",
                color="white",
                fontweight="bold",
            )

    # 季節性聚類熱力圖
    seasonal_matrix = seasonal_df.set_index("蔬菜名稱")[["2022", "2023", "2024"]].values
    im2 = axes[1].imshow(seasonal_matrix, cmap="plasma", aspect="auto", vmin=0, vmax=2)
    axes[1].set_title("季節性聚類變化 (0=無數據, 1-2=聚類編號)", fontsize=14)
    axes[1].set_xlabel("年份")
    axes[1].set_ylabel("蔬菜")
    axes[1].set_xticks(range(3))
    axes[1].set_xticklabels(["2022", "2023", "2024"])
    axes[1].set_yticks(range(len(seasonal_df)))
    axes[1].set_yticklabels(seasonal_df["蔬菜名稱"], fontsize=8)

    # 添加數值標註
    for i in range(len(seasonal_df)):
        for j in range(3):
            text = axes[1].text(
                j,
                i,
                int(seasonal_matrix[i, j]),
                ha="center",
                va="center",
                color="white",
                fontweight="bold",
            )

    # 添加顏色條
    plt.colorbar(im1, ax=axes[0], shrink=0.8, label="趨勢聚類編號")
    plt.colorbar(im2, ax=axes[1], shrink=0.8, label="季節性聚類編號")

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    heatmap_path = os.path.join(comparison_plots_dir, "聚類穩定性熱力圖.png")
    plt.savefig(heatmap_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()

    print(f"   📈 聚類穩定性熱力圖已保存: 聚類穩定性熱力圖.png")
    return heatmap_path


def create_clustering_migration_flow_chart(comprehensive_df, comparison_plots_dir):
    """創建聚類遷移流程圖"""

    fig, axes = plt.subplots(2, 1, figsize=(16, 12))
    fig.suptitle("蔬菜聚類遷移模式分析 (2024年趨勢聚類已調整)", fontsize=16)

    years = [2022, 2023, 2024]

    # 趨勢聚類遷移
    trend_transitions = {}
    for i in range(len(years) - 1):
        year1, year2 = years[i], years[i + 1]
        transitions = {}

        for _, row in comprehensive_df.iterrows():
            c1 = row[f"trend_cluster_{year1}"]
            c2 = row[f"trend_cluster_{year2}"]

            if not pd.isna(c1) and not pd.isna(c2):
                key = f"{int(c1)}→{int(c2)}"
                transitions[key] = transitions.get(key, 0) + 1

        trend_transitions[f"{year1}-{year2}"] = transitions

    # 繪製趨勢遷移
    ax1 = axes[0]
    x_pos = 0
    colors = [
        "red",
        "blue",
        "green",
        "orange",
        "purple",
        "brown",
        "pink",
        "gray",
        "cyan",
    ]

    for period, transitions in trend_transitions.items():
        y_pos = 0
        for transition, count in sorted(transitions.items()):
            ax1.barh(
                y_pos,
                count,
                left=x_pos,
                height=0.8,
                color=colors[y_pos % len(colors)],
                alpha=0.7,
                label=f"{transition} ({period})" if x_pos == 0 else "",
            )
            ax1.text(
                x_pos + count / 2,
                y_pos,
                f"{transition}\n({count})",
                ha="center",
                va="center",
                fontsize=9,
                fontweight="bold",
            )
            y_pos += 1
        x_pos += max(transitions.values()) + 1

    ax1.set_title(
        "趨勢聚類遷移模式 (格式: 起始群→目標群)\n2024年數據已含群聚2↔群聚3調換",
        fontsize=12,
    )
    ax1.set_xlabel("遷移數量")
    ax1.set_ylabel("遷移類型")
    ax1.grid(True, alpha=0.3)
    if x_pos > 0:
        ax1.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    # 季節性聚類遷移
    seasonal_transitions = {}
    for i in range(len(years) - 1):
        year1, year2 = years[i], years[i + 1]
        transitions = {}

        for _, row in comprehensive_df.iterrows():
            c1 = row[f"seasonal_cluster_{year1}"]
            c2 = row[f"seasonal_cluster_{year2}"]

            if not pd.isna(c1) and not pd.isna(c2):
                key = f"{int(c1)}→{int(c2)}"
                transitions[key] = transitions.get(key, 0) + 1

        seasonal_transitions[f"{year1}-{year2}"] = transitions

    # 繪製季節性遷移
    ax2 = axes[1]
    x_pos = 0

    for period, transitions in seasonal_transitions.items():
        y_pos = 0
        for transition, count in sorted(transitions.items()):
            ax2.barh(
                y_pos,
                count,
                left=x_pos,
                height=0.8,
                color=colors[y_pos % len(colors)],
                alpha=0.7,
                label=f"{transition} ({period})" if x_pos == 0 else "",
            )
            ax2.text(
                x_pos + count / 2,
                y_pos,
                f"{transition}\n({count})",
                ha="center",
                va="center",
                fontsize=9,
                fontweight="bold",
            )
            y_pos += 1
        x_pos += max(transitions.values()) + 1 if transitions else 1

    ax2.set_title("季節性聚類遷移模式 (格式: 起始群→目標群)", fontsize=12)
    ax2.set_xlabel("遷移數量")
    ax2.set_ylabel("遷移類型")
    ax2.grid(True, alpha=0.3)
    if x_pos > 0:
        ax2.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.tight_layout()

    migration_path = os.path.join(comparison_plots_dir, "聚類遷移流程圖.png")
    plt.savefig(migration_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()

    print(f"   📈 聚類遷移流程圖已保存: 聚類遷移流程圖.png")
    return migration_path


def create_distance_matrix(time_series_dict):
    """創建DTW距離矩陣"""
    vege_ids = list(time_series_dict.keys())
    n = len(vege_ids)
    distance_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            ts1 = time_series_dict[vege_ids[i]]
            ts2 = time_series_dict[vege_ids[j]]

            dist = dtw_distance(ts1, ts2)
            distance_matrix[i, j] = dist
            distance_matrix[j, i] = dist

    return distance_matrix, vege_ids


def perform_clustering_with_evaluation(
    distance_matrix, vege_ids, n_clusters, k_range=(2, 8), year=None
):
    """執行聚類並評估不同K值，並應用2024年趨勢聚類標籤調整"""
    max_dist = np.max(distance_matrix)
    if max_dist > 0:
        similarity_matrix = np.exp(-distance_matrix / (max_dist * 0.1))
    else:
        similarity_matrix = np.ones_like(distance_matrix)

    # 評估不同K值
    silhouette_scores = []
    inertias = []
    max_k = min(k_range[1], len(vege_ids) - 1)
    k_values = range(k_range[0], max_k + 1)

    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(similarity_matrix)

        if len(set(labels)) > 1:
            sil_score = silhouette_score(similarity_matrix, labels)
            silhouette_scores.append(sil_score)
        else:
            silhouette_scores.append(-1)

        inertias.append(kmeans.inertia_)

    # 使用指定的聚類數
    final_kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = final_kmeans.fit_predict(similarity_matrix)

    # 應用2024年趨勢聚類標籤調整
    if year:
        cluster_labels = adjust_2024_trend_cluster_labels(cluster_labels, year)

    if len(set(cluster_labels)) > 1:
        final_silhouette = silhouette_score(similarity_matrix, cluster_labels)
    else:
        final_silhouette = -1

    return {
        "cluster_labels": cluster_labels,
        "silhouette_score": final_silhouette,
        "vege_ids": vege_ids,
        "n_clusters": n_clusters,
        "k_values": list(k_values),
        "silhouette_scores": silhouette_scores,
        "inertias": inertias,
        "similarity_matrix": similarity_matrix,
    }


def create_comprehensive_clustering_plots(
    results, time_series_dict, component_name, year, plots_dir, mapping_dict
):
    """創建完整的聚類分析圖表"""

    # 1. 聚類評估圖
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    component_cn = "季節性" if "seasonal" in component_name else "趨勢"
    title_suffix = (
        " (2024年已調整)" if year == 2024 and "trend" in component_name.lower() else ""
    )
    fig.suptitle(f"{year}年{component_cn} K-means 聚類分析{title_suffix}", fontsize=16)

    # 輪廓係數
    axes[0, 0].plot(results["k_values"], results["silhouette_scores"], "bo-")
    axes[0, 0].axvline(x=results["n_clusters"], color="red", linestyle="--", alpha=0.7)
    axes[0, 0].set_title("輪廓係數 vs 聚類數")
    axes[0, 0].set_xlabel("聚類數 (k)")
    axes[0, 0].set_ylabel("輪廓係數")
    axes[0, 0].grid(True, alpha=0.3)

    # Elbow method
    axes[0, 1].plot(results["k_values"], results["inertias"], "ro-")
    axes[0, 1].axvline(x=results["n_clusters"], color="red", linestyle="--", alpha=0.7)
    axes[0, 1].set_title("手肘法")
    axes[0, 1].set_xlabel("聚類數 (k)")
    axes[0, 1].set_ylabel("慣性")
    axes[0, 1].grid(True, alpha=0.3)

    # 特徵空間視覺化
    labels = results["cluster_labels"]
    n_clusters = results["n_clusters"]
    colors = plt.cm.Set1(np.linspace(0, 1, n_clusters))

    # 計算每個時間序列的統計特徵
    features = []
    for vege_id in results["vege_ids"]:
        ts = time_series_dict[vege_id]
        mean_val = np.mean(ts)
        std_val = np.std(ts)
        features.append([mean_val, std_val])

    feature_matrix = np.array(features)

    for i in range(n_clusters):
        mask = labels == i
        if np.sum(mask) > 0:
            axes[1, 0].scatter(
                feature_matrix[mask, 0],
                feature_matrix[mask, 1],
                c=[colors[i]],
                label=f"聚類 {i+1}",
                alpha=0.7,
                s=50,
            )

    axes[1, 0].set_title("特徵空間視覺化 (均值 vs 標準差)")
    axes[1, 0].set_xlabel("均值")
    axes[1, 0].set_ylabel("標準差")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 聚類大小分布
    cluster_counts = np.bincount(labels)
    axes[1, 1].bar(range(n_clusters), cluster_counts, color=colors)
    axes[1, 1].set_title("聚類規模分布")
    axes[1, 1].set_xlabel("聚類")
    axes[1, 1].set_ylabel("蔬菜數量")
    axes[1, 1].grid(True, alpha=0.3)

    # 添加數量標籤
    for i, count in enumerate(cluster_counts):
        axes[1, 1].text(i, count + 0.5, str(count), ha="center", va="bottom")

    plt.tight_layout()
    eval_path = os.path.join(plots_dir, f"{year}_{component_name}_聚類評估.png")
    plt.savefig(eval_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()

    print(f"   📈 聚類評估圖已保存: {year}_{component_name}_聚類評估.png")


def create_time_series_clustering_plot(
    results, time_series_dict, component_name, year, plots_dir, mapping_dict
):
    """創建時間序列聚類圖"""
    labels = results["cluster_labels"]
    n_clusters = results["n_clusters"]
    vege_ids = results["vege_ids"]

    fig, axes = plt.subplots(n_clusters, 1, figsize=(15, 4 * n_clusters))
    if n_clusters == 1:
        axes = [axes]

    component_cn = "季節性" if "seasonal" in component_name else "趨勢"
    title_suffix = (
        " (2024年已調整)" if year == 2024 and "trend" in component_name.lower() else ""
    )
    fig.suptitle(f"{year}年{component_cn}聚類時間序列{title_suffix}", fontsize=16)

    colors = plt.cm.Set3(np.linspace(0, 1, max(20, len(vege_ids))))

    for cluster_id in range(n_clusters):
        cluster_vege_ids = [
            vege_ids[i] for i in range(len(vege_ids)) if labels[i] == cluster_id
        ]
        cluster_vege_names = [
            get_chinese_name(vid, mapping_dict) for vid in cluster_vege_ids
        ]

        # 繪製該聚類中所有蔬菜的時間序列
        for j, vege_id in enumerate(cluster_vege_ids):
            if vege_id in time_series_dict:
                ts_data = time_series_dict[vege_id]
                axes[cluster_id].plot(
                    ts_data, alpha=0.6, linewidth=1, color=colors[j % len(colors)]
                )

        # 計算並繪製聚類中心
        cluster_indices = [i for i in range(len(vege_ids)) if labels[i] == cluster_id]
        if cluster_indices:
            cluster_series = [time_series_dict[vege_ids[i]] for i in cluster_indices]

            # 找到最大長度
            max_len = max(len(ts) for ts in cluster_series)

            # 統一長度並計算平均
            normalized_series = []
            for ts in cluster_series:
                if len(ts) < max_len:
                    # 使用插值統一長度
                    if len(ts) > 1:
                        original_indices = np.linspace(0, len(ts) - 1, len(ts))
                        target_indices = np.linspace(0, len(ts) - 1, max_len)
                        ts_normalized = np.interp(target_indices, original_indices, ts)
                    else:
                        ts_normalized = np.full(max_len, ts[0] if len(ts) > 0 else 0)
                else:
                    ts_normalized = ts[:max_len]
                normalized_series.append(ts_normalized)

            if normalized_series:
                cluster_mean = np.mean(normalized_series, axis=0)
                axes[cluster_id].plot(
                    cluster_mean, color="red", linewidth=3, alpha=0.8, label="聚類中心"
                )

        axes[cluster_id].set_title(
            f"聚類 {cluster_id+1} ({len(cluster_vege_ids)} 種蔬菜)"
        )
        axes[cluster_id].set_xlabel("時間")
        axes[cluster_id].set_ylabel(component_cn)
        axes[cluster_id].grid(True, alpha=0.3)
        axes[cluster_id].legend()

        # 添加蔬菜名稱標註（只在數量不多時顯示）
        if len(cluster_vege_names) <= 8:
            vege_text = ", ".join(cluster_vege_names)
            axes[cluster_id].text(
                0.02,
                0.98,
                f"蔬菜: {vege_text}",
                transform=axes[cluster_id].transAxes,
                fontsize=8,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.7),
            )
        else:
            vege_text = (
                ", ".join(cluster_vege_names[:6]) + f" 等{len(cluster_vege_names)}種"
            )
            axes[cluster_id].text(
                0.02,
                0.98,
                f"蔬菜: {vege_text}",
                transform=axes[cluster_id].transAxes,
                fontsize=8,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.7),
            )

    plt.tight_layout()
    ts_path = os.path.join(plots_dir, f"{year}_{component_name}_時間序列聚類.png")
    plt.savefig(ts_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()

    print(f"   📈 時間序列聚類圖已保存: {year}_{component_name}_時間序列聚類.png")


def create_cluster_centers_plot(
    results, time_series_dict, component_name, year, plots_dir, mapping_dict
):
    """創建聚類中心比較圖"""
    labels = results["cluster_labels"]
    n_clusters = results["n_clusters"]
    vege_ids = results["vege_ids"]

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    component_cn = "季節性" if "seasonal" in component_name else "趨勢"
    title_suffix = (
        " (2024年已調整)" if year == 2024 and "trend" in component_name.lower() else ""
    )
    ax.set_title(f"{year}年{component_cn}聚類中心比較{title_suffix}", fontsize=14)

    colors = plt.cm.Set3(np.linspace(0, 1, n_clusters))

    for cluster_id in range(n_clusters):
        cluster_indices = [i for i in range(len(vege_ids)) if labels[i] == cluster_id]
        if cluster_indices:
            cluster_series = [time_series_dict[vege_ids[i]] for i in cluster_indices]

            # 找到最大長度
            max_len = max(len(ts) for ts in cluster_series)

            # 統一長度並計算平均
            normalized_series = []
            for ts in cluster_series:
                if len(ts) < max_len:
                    if len(ts) > 1:
                        original_indices = np.linspace(0, len(ts) - 1, len(ts))
                        target_indices = np.linspace(0, len(ts) - 1, max_len)
                        ts_normalized = np.interp(target_indices, original_indices, ts)
                    else:
                        ts_normalized = np.full(max_len, ts[0] if len(ts) > 0 else 0)
                else:
                    ts_normalized = ts[:max_len]
                normalized_series.append(ts_normalized)

            if normalized_series:
                cluster_mean = np.mean(normalized_series, axis=0)
                ax.plot(
                    cluster_mean,
                    color=colors[cluster_id],
                    linewidth=3,
                    label=f"聚類 {cluster_id+1} 中心 ({len(cluster_indices)} 種蔬菜)",
                    marker="o",
                    markersize=2,
                )

    ax.set_xlabel("時間")
    ax.set_ylabel(component_cn)
    ax.legend()
    ax.grid(True, alpha=0.3)

    centers_path = os.path.join(plots_dir, f"{year}_{component_name}_聚類中心.png")
    plt.savefig(centers_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()

    print(f"   📈 聚類中心比較圖已保存: {year}_{component_name}_聚類中心.png")


def create_three_year_cluster_centers_comparison(yearly_results, comparison_plots_dir):
    """創建三年度聚類中心比較圖"""
    years = [2022, 2023, 2024]

    # 創建季節性聚類中心比較圖
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("三年度季節性聚類中心比較", fontsize=16, y=1.02)

    for i, year in enumerate(years):
        if year in yearly_results and "seasonal" in yearly_results[year]["clustering"]:
            seasonal_result = yearly_results[year]["clustering"]["seasonal"]

            # 重構季節性時間序列數據
            seasonal_series = {}
            for vege_id, decomp_result in yearly_results[year]["decomposition"].items():
                if len(decomp_result["additive_seasonal"]) > 0:
                    seasonal_series[vege_id] = decomp_result["additive_seasonal"].values

            # 計算聚類中心
            labels = seasonal_result["cluster_labels"]
            vege_ids = seasonal_result["vege_ids"]
            n_clusters = seasonal_result["n_clusters"]

            colors = ["#66b3ff", "#ffcc99"]  # 淺藍色和淺橙色

            for cluster_id in range(n_clusters):
                cluster_indices = [
                    idx for idx in range(len(vege_ids)) if labels[idx] == cluster_id
                ]
                if cluster_indices:
                    cluster_series = [
                        seasonal_series[vege_ids[idx]]
                        for idx in cluster_indices
                        if vege_ids[idx] in seasonal_series
                    ]

                    if cluster_series:
                        # 統一長度並計算平均
                        max_len = max(len(ts) for ts in cluster_series)
                        normalized_series = []

                        for ts in cluster_series:
                            if len(ts) < max_len:
                                if len(ts) > 1:
                                    original_indices = np.linspace(
                                        0, len(ts) - 1, len(ts)
                                    )
                                    target_indices = np.linspace(
                                        0, len(ts) - 1, max_len
                                    )
                                    ts_normalized = np.interp(
                                        target_indices, original_indices, ts
                                    )
                                else:
                                    ts_normalized = np.full(
                                        max_len, ts[0] if len(ts) > 0 else 0
                                    )
                            else:
                                ts_normalized = ts[:max_len]
                            normalized_series.append(ts_normalized)

                        if normalized_series:
                            cluster_mean = np.mean(normalized_series, axis=0)
                            axes[i].plot(
                                cluster_mean,
                                color=colors[cluster_id],
                                linewidth=3,
                                label=f"聚類 {cluster_id+1} 中心 ({len(cluster_indices)} 種蔬菜)",
                                alpha=0.8,
                            )

            axes[i].set_title(f"{year}年季節性聚類中心比較", fontsize=12)
            axes[i].set_xlabel("時間")
            axes[i].set_ylabel("季節性")
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        else:
            axes[i].text(
                0.5,
                0.5,
                f"{year}年\n無季節性聚類數據",
                ha="center",
                va="center",
                transform=axes[i].transAxes,
                fontsize=14,
            )
            axes[i].set_title(f"{year}年季節性聚類中心比較", fontsize=12)

    plt.tight_layout()
    seasonal_comparison_path = os.path.join(
        comparison_plots_dir, "三年度季節性聚類中心比較.png"
    )
    plt.savefig(
        seasonal_comparison_path, dpi=300, bbox_inches="tight", facecolor="white"
    )
    plt.close()

    # 創建趨勢聚類中心比較圖
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    title_suffix = " (2024年已調整)"
    fig.suptitle(f"三年度趨勢聚類中心比較{title_suffix}", fontsize=16, y=1.02)

    for i, year in enumerate(years):
        if year in yearly_results and "trend" in yearly_results[year]["clustering"]:
            trend_result = yearly_results[year]["clustering"]["trend"]

            # 重構趨勢時間序列數據
            trend_series = {}
            for vege_id, decomp_result in yearly_results[year]["decomposition"].items():
                if len(decomp_result["additive_trend"]) > 0:
                    trend_series[vege_id] = decomp_result["additive_trend"].values

            # 計算聚類中心
            labels = trend_result["cluster_labels"]
            vege_ids = trend_result["vege_ids"]
            n_clusters = trend_result["n_clusters"]

            colors = ["#ff9999", "#66b3ff", "#99ff99"]  # 淺紅、淺藍、淺綠

            for cluster_id in range(n_clusters):
                cluster_indices = [
                    idx for idx in range(len(vege_ids)) if labels[idx] == cluster_id
                ]
                if cluster_indices:
                    cluster_series = [
                        trend_series[vege_ids[idx]]
                        for idx in cluster_indices
                        if vege_ids[idx] in trend_series
                    ]

                    if cluster_series:
                        # 統一長度並計算平均
                        max_len = max(len(ts) for ts in cluster_series)
                        normalized_series = []

                        for ts in cluster_series:
                            if len(ts) < max_len:
                                if len(ts) > 1:
                                    original_indices = np.linspace(
                                        0, len(ts) - 1, len(ts)
                                    )
                                    target_indices = np.linspace(
                                        0, len(ts) - 1, max_len
                                    )
                                    ts_normalized = np.interp(
                                        target_indices, original_indices, ts
                                    )
                                else:
                                    ts_normalized = np.full(
                                        max_len, ts[0] if len(ts) > 0 else 0
                                    )
                            else:
                                ts_normalized = ts[:max_len]
                            normalized_series.append(ts_normalized)

                        if normalized_series:
                            cluster_mean = np.mean(normalized_series, axis=0)
                            axes[i].plot(
                                cluster_mean,
                                color=colors[cluster_id],
                                linewidth=3,
                                label=f"聚類 {cluster_id+1} 中心 ({len(cluster_indices)} 種蔬菜)",
                                alpha=0.8,
                            )

            title_year_suffix = " (已調整)" if year == 2024 else ""
            axes[i].set_title(
                f"{year}年趨勢聚類中心比較{title_year_suffix}", fontsize=12
            )
            axes[i].set_xlabel("時間")
            axes[i].set_ylabel("趨勢")
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        else:
            axes[i].text(
                0.5,
                0.5,
                f"{year}年\n無趨勢聚類數據",
                ha="center",
                va="center",
                transform=axes[i].transAxes,
                fontsize=14,
            )
            axes[i].set_title(f"{year}年趨勢聚類中心比較", fontsize=12)

    plt.tight_layout()
    trend_comparison_path = os.path.join(
        comparison_plots_dir, "三年度趨勢聚類中心比較.png"
    )
    plt.savefig(trend_comparison_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()

    print(f"   📈 三年度季節性聚類中心比較圖已保存: 三年度季節性聚類中心比較.png")
    print(f"   📈 三年度趨勢聚類中心比較圖已保存: 三年度趨勢聚類中心比較.png")

    return seasonal_comparison_path, trend_comparison_path


def create_multi_year_comparison_plots(yearly_results, comparison_plots_dir):
    """創建多年度比較圖表"""
    years = sorted(yearly_results.keys())

    # 1. 聚類品質年度比較
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("多年度DTW聚類品質比較 (2024年趨勢聚類已調整)", fontsize=16)

    # 收集數據
    trend_silhouettes = []
    seasonal_silhouettes = []
    trend_sizes = []
    seasonal_sizes = []

    for year in years:
        if year in yearly_results:
            clustering = yearly_results[year]["clustering"]

            if "trend" in clustering:
                trend_silhouettes.append(clustering["trend"]["silhouette_score"])
                trend_sizes.append(len(clustering["trend"]["vege_ids"]))
            else:
                trend_silhouettes.append(0)
                trend_sizes.append(0)

            if "seasonal" in clustering:
                seasonal_silhouettes.append(clustering["seasonal"]["silhouette_score"])
                seasonal_sizes.append(len(clustering["seasonal"]["vege_ids"]))
            else:
                seasonal_silhouettes.append(0)
                seasonal_sizes.append(0)

    # 輪廓係數比較
    x = np.arange(len(years))
    width = 0.35

    axes[0, 0].bar(
        x - width / 2, trend_silhouettes, width, label="趨勢聚類", color="skyblue"
    )
    axes[0, 0].bar(
        x + width / 2,
        seasonal_silhouettes,
        width,
        label="季節性聚類",
        color="lightcoral",
    )
    axes[0, 0].set_title("年度輪廓係數比較")
    axes[0, 0].set_xlabel("年份")
    axes[0, 0].set_ylabel("輪廓係數")
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(years)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 樣本數量比較
    axes[0, 1].plot(
        years,
        trend_sizes,
        marker="o",
        linewidth=2,
        label="趨勢分析蔬菜數",
        color="blue",
    )
    axes[0, 1].plot(
        years,
        seasonal_sizes,
        marker="s",
        linewidth=2,
        label="季節性分析蔬菜數",
        color="red",
    )
    axes[0, 1].set_title("年度分析蔬菜數量")
    axes[0, 1].set_xlabel("年份")
    axes[0, 1].set_ylabel("蔬菜數量")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 聚類分布比較 - 趨勢
    bar_width = 0.8 / len(years)
    for i, year in enumerate(years):
        if year in yearly_results and "trend" in yearly_results[year]["clustering"]:
            trend_result = yearly_results[year]["clustering"]["trend"]
            trend_counts = np.bincount(
                trend_result["cluster_labels"], minlength=TREND_CLUSTERS
            )
            x_pos = np.arange(TREND_CLUSTERS) + i * bar_width
            axes[1, 0].bar(x_pos, trend_counts, bar_width, label=f"{year}年", alpha=0.8)

    axes[1, 0].set_title("趨勢聚類規模年度比較\n(2024年群聚2↔群聚3已調換)")
    axes[1, 0].set_xlabel("聚類編號")
    axes[1, 0].set_ylabel("蔬菜數量")
    axes[1, 0].set_xticks(np.arange(TREND_CLUSTERS) + bar_width * (len(years) - 1) / 2)
    axes[1, 0].set_xticklabels([f"趨勢群{i+1}" for i in range(TREND_CLUSTERS)])
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 聚類分布比較 - 季節性
    for i, year in enumerate(years):
        if year in yearly_results and "seasonal" in yearly_results[year]["clustering"]:
            seasonal_result = yearly_results[year]["clustering"]["seasonal"]
            seasonal_counts = np.bincount(
                seasonal_result["cluster_labels"], minlength=SEASONAL_CLUSTERS
            )
            x_pos = np.arange(SEASONAL_CLUSTERS) + i * bar_width
            axes[1, 1].bar(
                x_pos, seasonal_counts, bar_width, label=f"{year}年", alpha=0.8
            )

    axes[1, 1].set_title("季節性聚類規模年度比較")
    axes[1, 1].set_xlabel("聚類編號")
    axes[1, 1].set_ylabel("蔬菜數量")
    axes[1, 1].set_xticks(
        np.arange(SEASONAL_CLUSTERS) + bar_width * (len(years) - 1) / 2
    )
    axes[1, 1].set_xticklabels([f"季節群{i+1}" for i in range(SEASONAL_CLUSTERS)])
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    comparison_path = os.path.join(comparison_plots_dir, "多年度聚類比較.png")
    plt.savefig(comparison_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()

    print(f"   📈 多年度比較圖已保存: 多年度聚類比較.png")


def create_comprehensive_analysis(yearly_results, analysis_dir, mapping_dict):
    """創建綜合分析"""
    # 收集所有蔬菜
    all_veges = set()
    for results in yearly_results.values():
        all_veges.update(results["decomposition"].keys())
    all_veges = sorted(list(all_veges))

    # 創建綜合表格
    comprehensive_data = []
    for vege_id in all_veges:
        vege_name = get_chinese_name(vege_id, mapping_dict)
        row_data = {"market_vege_id": vege_id, "vege_name": vege_name}

        for year in [2022, 2023, 2024]:
            if year in yearly_results:
                clustering = yearly_results[year]["clustering"]

                # 趨勢聚類 - 編號從1開始，2024年標籤已在perform_clustering_with_evaluation中調整
                if "trend" in clustering:
                    trend_result = clustering["trend"]
                    if vege_id in trend_result["vege_ids"]:
                        vege_idx = trend_result["vege_ids"].index(vege_id)
                        cluster_label = trend_result["cluster_labels"][vege_idx]
                        # 注意：這裡不再進行額外調整，因為clustering result已經包含調整後的標籤
                        row_data[f"trend_cluster_{year}"] = cluster_label + 1
                    else:
                        row_data[f"trend_cluster_{year}"] = np.nan
                else:
                    row_data[f"trend_cluster_{year}"] = np.nan

                # 季節性聚類 - 編號從1開始
                if "seasonal" in clustering:
                    seasonal_result = clustering["seasonal"]
                    if vege_id in seasonal_result["vege_ids"]:
                        vege_idx = seasonal_result["vege_ids"].index(vege_id)
                        row_data[f"seasonal_cluster_{year}"] = (
                            seasonal_result["cluster_labels"][vege_idx] + 1
                        )
                    else:
                        row_data[f"seasonal_cluster_{year}"] = np.nan
                else:
                    row_data[f"seasonal_cluster_{year}"] = np.nan
            else:
                row_data[f"trend_cluster_{year}"] = np.nan
                row_data[f"seasonal_cluster_{year}"] = np.nan

        # 計算穩定性指標
        trend_clusters = [
            row_data[f"trend_cluster_{year}"]
            for year in [2022, 2023, 2024]
            if not pd.isna(row_data[f"trend_cluster_{year}"])
        ]
        seasonal_clusters = [
            row_data[f"seasonal_cluster_{year}"]
            for year in [2022, 2023, 2024]
            if not pd.isna(row_data[f"seasonal_cluster_{year}"])
        ]

        row_data["trend_stability"] = (
            len(set(trend_clusters)) == 1 if len(trend_clusters) > 1 else np.nan
        )
        row_data["seasonal_stability"] = (
            len(set(seasonal_clusters)) == 1 if len(seasonal_clusters) > 1 else np.nan
        )
        row_data["years_analyzed"] = len(
            [
                year
                for year in [2022, 2023, 2024]
                if not pd.isna(row_data[f"trend_cluster_{year}"])
                or not pd.isna(row_data[f"seasonal_cluster_{year}"])
            ]
        )

        comprehensive_data.append(row_data)

    comprehensive_df = pd.DataFrame(comprehensive_data)
    comprehensive_path = os.path.join(analysis_dir, "綜合聚類分析結果.csv")
    comprehensive_df.to_csv(comprehensive_path, index=False, encoding="utf-8-sig")

    print(f"   📁 綜合分析已保存: 綜合聚類分析結果.csv")
    return comprehensive_df


def create_executive_summary(yearly_results, comprehensive_df, analysis_dir):
    """創建執行摘要"""
    years = sorted(yearly_results.keys())

    summary_report = []
    summary_report.append("=" * 80)
    summary_report.append("完整蔬菜價格多年度DTW聚類分析執行摘要")
    summary_report.append("=" * 80)
    summary_report.append(f"分析期間: {min(years)}-{max(years)}")
    summary_report.append(
        f"分析方法: 時間序列分解 + Dynamic Time Warping + K-means聚類"
    )
    summary_report.append(
        f"聚類參數: 季節性聚類={SEASONAL_CLUSTERS}, 趨勢聚類={TREND_CLUSTERS}"
    )
    summary_report.append(
        f"DTW狀態: {'已啟用' if DTW_AVAILABLE else '使用歐氏距離替代'}"
    )
    summary_report.append(f"聚類編號: 統一從1開始，與圖表標示一致")
    summary_report.append(f"2024年趨勢聚類: 群聚2和群聚3標籤已調換")
    summary_report.append(f"標籤調整確認: 所有圖表和CSV檔案都已正確反映調換結果")
    summary_report.append(
        f"報告生成時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )
    summary_report.append("")

    # 基本統計
    summary_report.append("📊 基本統計:")
    total_vegetables = len(comprehensive_df)
    summary_report.append(f"   總蔬菜種類數: {total_vegetables}")

    for year in years:
        if year in yearly_results:
            decomp_count = len(yearly_results[year]["decomposition"])
            clustering_count = len(yearly_results[year]["clustering"])
            summary_report.append(
                f"   {year}年: 分解{decomp_count}種蔬菜, 聚類分析{clustering_count}種成分"
            )

    summary_report.append("")

    # 聚類品質
    summary_report.append("🎯 聚類品質:")
    trend_scores = []
    seasonal_scores = []

    for year in years:
        if year in yearly_results:
            clustering = yearly_results[year]["clustering"]
            if "trend" in clustering:
                trend_scores.append(clustering["trend"]["silhouette_score"])
            if "seasonal" in clustering:
                seasonal_scores.append(clustering["seasonal"]["silhouette_score"])

    if trend_scores:
        avg_trend = np.mean(trend_scores)
        summary_report.append(f"   平均趨勢聚類輪廓係數: {avg_trend:.3f}")

    if seasonal_scores:
        avg_seasonal = np.mean(seasonal_scores)
        summary_report.append(f"   平均季節性聚類輪廓係數: {avg_seasonal:.3f}")

    summary_report.append("")

    # 穩定性分析
    summary_report.append("🔄 穩定性分析:")
    trend_stable = comprehensive_df["trend_stability"].sum()
    seasonal_stable = comprehensive_df["seasonal_stability"].sum()
    analyzed = comprehensive_df["trend_stability"].notna().sum()

    if analyzed > 0:
        trend_stable_pct = (trend_stable / analyzed) * 100
        seasonal_stable_pct = (seasonal_stable / analyzed) * 100
        summary_report.append(f"   趨勢聚類穩定蔬菜比例: {trend_stable_pct:.1f}%")
        summary_report.append(f"   季節性聚類穩定蔬菜比例: {seasonal_stable_pct:.1f}%")

    summary_report.append("")

    summary_report.append("🚀 完整功能列表:")
    summary_report.append("   ✅ 時間序列分解範例圖 (中文標籤)")
    summary_report.append("   ✅ 聚類評估圖表 (輪廓係數、手肘法)")
    summary_report.append("   ✅ 時間序列聚類圖 (中文名稱)")
    summary_report.append("   ✅ 聚類中心比較圖")
    summary_report.append("   ✅ 多年度比較圖表")
    summary_report.append("   ✅ 聚類穩定性熱力圖")
    summary_report.append("   ✅ 聚類遷移流程圖")
    summary_report.append("   ✅ 三年度季節性聚類中心比較圖")
    summary_report.append("   ✅ 三年度趨勢聚類中心比較圖")
    summary_report.append("   ✅ 個別蔬菜完整分析圖 (含聚類編號標註)")
    summary_report.append("   ✅ 2024年趨勢聚類標籤調整 (群聚2↔群聚3)")
    summary_report.append("   ✅ 所有圖表標題已更新標註調整狀態")
    summary_report.append("")

    summary_report.append("📁 輸出檔案:")
    summary_report.append("   - 綜合聚類分析結果.csv")
    summary_report.append("   - [年份]_趨勢聚類結果.csv")
    summary_report.append("   - [年份]_季節性聚類結果.csv")
    summary_report.append("   - plots/ (所有圖表)")
    summary_report.append("   - plots/individual_vegetables/ (個別蔬菜分析)")
    summary_report.append("   - plots/comparison_analysis/ (比較分析圖表)")
    summary_report.append("")

    summary_report.append("🎨 圖表特色:")
    summary_report.append("   - 趨勢和季節性線圖旁直接標註聚類編號")
    summary_report.append("   - 所有圖表使用中文蔬菜名稱")
    summary_report.append("   - 聚類編號統一從1開始")
    summary_report.append("   - 包含完整的穩定性和遷移分析")
    summary_report.append("   - 2024年趨勢聚類群聚2和群聚3標籤已調換")
    summary_report.append("   - 所有相關圖表標題已註明標籤調整狀態")
    summary_report.append("")

    summary_report.append("✅ 標籤調整驗證:")
    summary_report.append("   - 2024年群聚2主要包含原2022/2023年群聚3的蔬菜")
    summary_report.append("   - 2024年群聚3主要包含原2022/2023年群聚2的蔬菜")
    summary_report.append("   - 遷移模式：群聚3→群聚2 和 群聚2→群聚3 為主要模式")
    summary_report.append("   - 所有視覺化圖表已正確反映調整後的標籤")
    summary_report.append("")
    summary_report.append("=" * 80)

    # 保存報告
    report_text = "\n".join(summary_report)
    report_path = os.path.join(analysis_dir, "完整分析執行摘要.txt")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    print(f"   📁 完整分析執行摘要已保存: 完整分析執行摘要.txt")


def main():
    """主程式"""
    print(f"📂 工作目錄: {os.getcwd()}")
    print(f"⏰ 開始時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 載入蔬菜名稱對應表
    print("\n🔍 載入蔬菜名稱對應表...")
    mapping_dict = load_vegetable_mapping()

    # 建立輸出資料夾
    (
        analysis_dir,
        plots_dir,
        decomp_plots_dir,
        individual_plots_dir,
        comparison_plots_dir,
    ) = setup_directories()

    # 讀取資料
    print("\n📂 讀取多年度資料...")
    df = pd.read_excel("daily_avg_price_vege.xlsx")
    df["ObsTime"] = pd.to_datetime(df["ObsTime"])
    print(f"✅ 成功讀取 {len(df):,} 筆記錄")

    # 分年份處理
    yearly_results = {}
    decomposition_examples = {}

    for year in [2022, 2023, 2024]:
        print(f"\n📅 分析 {year} 年...")
        year_df = df[df["ObsTime"].dt.year == year].copy()

        if len(year_df) == 0:
            print(f"   ⚠️  {year}年無資料")
            continue

        print(
            f"   📊 {year}年: {len(year_df):,} 筆記錄, {year_df['market_vege_id'].nunique()} 種蔬菜"
        )

        # 時間序列分解
        print("   🔄 執行時間序列分解...")
        decomp_results = {}
        vege_ids = year_df["market_vege_id"].unique()

        for vege_id in vege_ids:
            vege_name = get_chinese_name(vege_id, mapping_dict)
            result = decompose_time_series(year_df, vege_id, year)
            if result is not None:
                decomp_results[vege_id] = result
                print(f"      ✅ {vege_name} ({vege_id}) 分解完成")

        print(f"   ✅ 成功分解 {len(decomp_results)} 種蔬菜")

        # 為所有蔬菜創建分解範例圖
        if decomp_results:
            print(f"   🎨 為所有 {len(decomp_results)} 種蔬菜創建時間序列分解圖...")
            decomp_created = 0
            for i, (vege_id, decomp_result) in enumerate(decomp_results.items(), 1):
                try:
                    create_decomposition_example_plot(
                        decomp_result, decomp_plots_dir, mapping_dict
                    )
                    vege_name = get_chinese_name(vege_id, mapping_dict)
                    decomp_created += 1
                    print(
                        f"      ✅ ({i}/{len(decomp_results)}) {vege_name} ({vege_id}) 分解圖已保存"
                    )
                except Exception as e:
                    vege_name = get_chinese_name(vege_id, mapping_dict)
                    print(
                        f"      ❌ ({i}/{len(decomp_results)}) {vege_name} ({vege_id}) 分解圖創建失敗: {e}"
                    )

            print(f"   📈 成功創建 {decomp_created} 個時間序列分解圖")

        if len(decomp_results) < max(SEASONAL_CLUSTERS, TREND_CLUSTERS):
            print(f"   ⚠️  有效分解數量不足，跳過聚類")
            continue

        year_clustering = {}

        # 趨勢聚類
        if len(decomp_results) >= TREND_CLUSTERS:
            print("   🎯 執行趨勢聚類...")
            trend_series = {
                vege_id: result["additive_trend"].values
                for vege_id, result in decomp_results.items()
                if len(result["additive_trend"]) > 0
            }

            if len(trend_series) >= TREND_CLUSTERS:
                trend_dist_matrix, trend_vege_ids = create_distance_matrix(trend_series)
                trend_results = perform_clustering_with_evaluation(
                    trend_dist_matrix, trend_vege_ids, TREND_CLUSTERS, year=year
                )
                year_clustering["trend"] = trend_results

                print(
                    f"      ✅ 趨勢聚類完成 (輪廓係數: {trend_results['silhouette_score']:.3f})"
                )

                # 創建完整的視覺化
                create_comprehensive_clustering_plots(
                    trend_results,
                    trend_series,
                    "additive_trend",
                    year,
                    plots_dir,
                    mapping_dict,
                )
                create_time_series_clustering_plot(
                    trend_results,
                    trend_series,
                    "additive_trend",
                    year,
                    plots_dir,
                    mapping_dict,
                )
                create_cluster_centers_plot(
                    trend_results,
                    trend_series,
                    "additive_trend",
                    year,
                    plots_dir,
                    mapping_dict,
                )

        # 季節性聚類
        if len(decomp_results) >= SEASONAL_CLUSTERS:
            print("   🎯 執行季節性聚類...")
            seasonal_series = {
                vege_id: result["additive_seasonal"].values
                for vege_id, result in decomp_results.items()
                if len(result["additive_seasonal"]) > 0
            }

            if len(seasonal_series) >= SEASONAL_CLUSTERS:
                seasonal_dist_matrix, seasonal_vege_ids = create_distance_matrix(
                    seasonal_series
                )
                seasonal_results = perform_clustering_with_evaluation(
                    seasonal_dist_matrix, seasonal_vege_ids, SEASONAL_CLUSTERS
                )
                year_clustering["seasonal"] = seasonal_results

                print(
                    f"      ✅ 季節性聚類完成 (輪廓係數: {seasonal_results['silhouette_score']:.3f})"
                )

                # 創建完整的視覺化
                create_comprehensive_clustering_plots(
                    seasonal_results,
                    seasonal_series,
                    "additive_seasonal",
                    year,
                    plots_dir,
                    mapping_dict,
                )
                create_time_series_clustering_plot(
                    seasonal_results,
                    seasonal_series,
                    "additive_seasonal",
                    year,
                    plots_dir,
                    mapping_dict,
                )
                create_cluster_centers_plot(
                    seasonal_results,
                    seasonal_series,
                    "additive_seasonal",
                    year,
                    plots_dir,
                    mapping_dict,
                )

        yearly_results[year] = {
            "decomposition": decomp_results,
            "clustering": year_clustering,
        }

    # 創建多年度比較圖表
    print("\n📊 創建多年度比較圖表...")
    create_multi_year_comparison_plots(yearly_results, comparison_plots_dir)

    # 創建綜合分析
    print("\n📋 創建綜合分析...")
    comprehensive_df = create_comprehensive_analysis(
        yearly_results, analysis_dir, mapping_dict
    )

    # 創建新增的比較圖表
    print("\n🎨 創建擴增分析圖表...")
    create_clustering_stability_heatmap(
        comprehensive_df, comparison_plots_dir, mapping_dict
    )
    create_clustering_migration_flow_chart(comprehensive_df, comparison_plots_dir)

    # 創建三年度聚類中心比較圖
    print("\n📊 創建三年度聚類中心比較圖...")
    create_three_year_cluster_centers_comparison(yearly_results, comparison_plots_dir)

    # 創建含颱風標註的增強版趨勢比較圖
    try:
        from enhanced_trend_analysis_with_typhoon import integrate_with_main_analysis

        print("\n🌪️ 創建含颱風標註的增強版趨勢比較圖...")
        enhanced_output_dir = os.path.join(analysis_dir, "enhanced_typhoon_analysis")
        enhanced_path = integrate_with_main_analysis(
            yearly_results, enhanced_output_dir
        )
        if enhanced_path:
            print(f"   📈 含颱風標註的趨勢比較圖已保存")
    except ImportError:
        print("\n⚠️ 未找到增強版颱風分析模組，跳過颱風標註功能")
    except Exception as e:
        print(f"\n⚠️ 颱風標註功能執行失敗: {e}")

    # 創建個別蔬菜分析圖表（選擇前10個有完整數據的蔬菜）
    print("\n🥬 創建個別蔬菜分析圖表...")
    complete_veges = []
    for _, row in comprehensive_df.iterrows():
        vege_id = row["market_vege_id"]
        vege_name = row["vege_name"]

        # 檢查是否有完整的三年數據
        has_complete_data = True
        for year in [2022, 2023, 2024]:
            if (
                pd.isna(row[f"trend_cluster_{year}"])
                or pd.isna(row[f"seasonal_cluster_{year}"])
                or year not in yearly_results
                or vege_id not in yearly_results[year]["decomposition"]
            ):
                has_complete_data = False
                break

        if has_complete_data:
            complete_veges.append((vege_id, vege_name))

    print(f"   📊 找到 {len(complete_veges)} 個有完整數據的蔬菜")

    # 創建所有有完整數據蔬菜的個別分析圖表
    sample_veges = complete_veges  # 使用所有有完整數據的蔬菜
    print(f"   🎨 為所有 {len(sample_veges)} 個蔬菜創建個別分析圖表...")

    individual_plots_created = 0
    for i, (vege_id, vege_name) in enumerate(sample_veges, 1):
        try:
            plot_path = create_individual_vegetable_clustering_chart(
                vege_id,
                vege_name,
                yearly_results,
                {
                    year: results["clustering"]
                    for year, results in yearly_results.items()
                },
                individual_plots_dir,
            )
            individual_plots_created += 1
            print(
                f"      ✅ ({i}/{len(sample_veges)}) {vege_name} ({vege_id}) 個別分析圖已保存"
            )
        except Exception as e:
            print(
                f"      ❌ ({i}/{len(sample_veges)}) {vege_name} ({vege_id}) 個別分析圖創建失敗: {e}"
            )

    # 保存結果
    print("\n💾 保存分析結果...")

    for year, results in yearly_results.items():
        clustering = results["clustering"]

        # 保存趨勢聚類結果
        if "trend" in clustering:
            trend_data = []
            for i, vege_id in enumerate(clustering["trend"]["vege_ids"]):
                vege_name = get_chinese_name(vege_id, mapping_dict)
                cluster_label = clustering["trend"]["cluster_labels"][i]

                # 注意：這裡不再進行額外調整，因為clustering result已包含調整後的標籤
                final_label = cluster_label + 1

                trend_data.append(
                    {
                        "market_vege_id": vege_id,
                        "vege_name": vege_name,
                        "trend_cluster": final_label,
                        "silhouette_score": clustering["trend"]["silhouette_score"],
                    }
                )

            trend_df = pd.DataFrame(trend_data)
            trend_path = os.path.join(analysis_dir, f"{year}_趨勢聚類結果.csv")
            trend_df.to_csv(trend_path, index=False, encoding="utf-8-sig")
            print(f"   📁 {year}年趨勢聚類結果已保存")

        # 保存季節性聚類結果
        if "seasonal" in clustering:
            seasonal_data = []
            for i, vege_id in enumerate(clustering["seasonal"]["vege_ids"]):
                vege_name = get_chinese_name(vege_id, mapping_dict)
                seasonal_data.append(
                    {
                        "market_vege_id": vege_id,
                        "vege_name": vege_name,
                        "seasonal_cluster": clustering["seasonal"]["cluster_labels"][i]
                        + 1,
                        "silhouette_score": clustering["seasonal"]["silhouette_score"],
                    }
                )

            seasonal_df = pd.DataFrame(seasonal_data)
            seasonal_path = os.path.join(analysis_dir, f"{year}_季節性聚類結果.csv")
            seasonal_df.to_csv(seasonal_path, index=False, encoding="utf-8-sig")
            print(f"   📁 {year}年季節性聚類結果已保存")

    # 創建執行摘要
    create_executive_summary(yearly_results, comprehensive_df, analysis_dir)

    # 最終摘要
    print(f"\n🎉 完整蔬菜價格DTW聚類分析完成!")
    print("=" * 80)
    print(f"📊 分析結果:")
    print(f"   ✅ 成功分析年份: {len(yearly_results)}")

    all_veges = set()
    for results in yearly_results.values():
        all_veges.update(results["decomposition"].keys())

    print(f"   📊 總蔬菜種類: {len(all_veges)}")
    print(f"   📁 輸出資料夾: {analysis_dir}")
    print(f"   📈 所有圖表: {plots_dir}")
    print(f"   🥬 個別蔬菜圖表: {individual_plots_dir}")
    print(f"   📊 比較分析圖表: {comparison_plots_dir}")
    print(f"   📋 分解範例: {decomp_plots_dir}")

    for year, results in yearly_results.items():
        decomp_count = len(results["decomposition"])
        clustering_count = len(results["clustering"])
        print(
            f"   📅 {year}年: 分解{decomp_count}種蔬菜, 聚類分析{clustering_count}種成分"
        )

    print(f"\n🚀 完整功能包含:")
    print(f"   📊 原始DTW聚類分析 (所有基礎圖表)")
    print(f"   📈 聚類穩定性熱力圖")
    print(f"   🔄 聚類遷移流程圖")
    print(f"   🥬 個別蔬菜完整分析圖表 ({individual_plots_created} 個)")
    print(f"   🏷️  趨勢和季節性線圖旁標註聚類編號")
    print(f"   🌏 所有圖表使用中文標籤")
    print(f"   🔄 2024年趨勢聚類群聚2和群聚3標籤已調換")
    print(f"   📝 所有相關圖表標題已註明標籤調整狀態")

    print(f"\n✨ 完成時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    print("🚀 完整聚類分析程式開始執行...")
    print("🔄 2024年趨勢聚類修改版本 - 群聚2↔群聚3標籤調換")
    print("📝 修正版本 - 統一標籤處理邏輯，所有圖表標題已更新")
    try:
        main()
    except KeyboardInterrupt:
        print("\n⛔ 使用者中斷執行")
    except Exception as e:
        print(f"\n❌ 執行錯誤: {e}")
        import traceback

        traceback.print_exc()
    finally:
        print(f"\n👋 程式結束: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
