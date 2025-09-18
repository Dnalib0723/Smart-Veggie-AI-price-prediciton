#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
完整蔬菜價格DTW聚類分析程式包 (2022-2024)
包含原始分析和擴增功能的完整版本
包含所有視覺化功能，去除互動儀表板
修改版本：2024年趨勢聚類的群聚2和群聚3標籤調換
修正版本：移除重複的標籤調整邏輯，統一標籤處理
增強版本：加入颱風標註功能到三年度趨勢聚類中心比較圖
預測版本：新增2025年聚類預測功能 (3票2勝制)
"""

print("🥬 完整蔬菜價格DTW聚類分析程式包 (2022-2024)")
print("🌪️ 增強版本：含颱風標註功能")
print("🔮 預測版本：含2025年聚類預測")
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
from collections import defaultdict, Counter

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


def predict_2025_clusters(comprehensive_df):
    """
    基於2022-2024年的聚類結果預測2025年的聚類歸屬
    使用3票2勝制：如果某個蔬菜在3年中有2年或以上屬於同一聚類，則預測2025年也屬於該聚類
    """
    print("🔮 開始預測2025年聚類結果...")
    
    predictions = []
    trend_prediction_stats = {'confident': 0, 'uncertain': 0, 'insufficient_data': 0}
    seasonal_prediction_stats = {'confident': 0, 'uncertain': 0, 'insufficient_data': 0}
    
    for _, row in comprehensive_df.iterrows():
        vege_id = row['market_vege_id']
        vege_name = row['vege_name']
        
        # 收集趨勢聚類歷史
        trend_history = []
        for year in [2022, 2023, 2024]:
            if not pd.isna(row[f'trend_cluster_{year}']):
                trend_history.append(int(row[f'trend_cluster_{year}']))
        
        # 收集季節性聚類歷史
        seasonal_history = []
        for year in [2022, 2023, 2024]:
            if not pd.isna(row[f'seasonal_cluster_{year}']):
                seasonal_history.append(int(row[f'seasonal_cluster_{year}']))
        
        # 預測趨勢聚類
        trend_prediction = None
        trend_confidence = 'insufficient_data'
        if len(trend_history) >= 2:
            trend_counter = Counter(trend_history)
            most_common = trend_counter.most_common(1)[0]
            if most_common[1] >= 2:  # 至少出現2次
                trend_prediction = most_common[0]
                trend_confidence = 'confident' if most_common[1] >= 2 else 'uncertain'
            else:
                # 如果沒有明確的多數，選擇最近年份的結果
                trend_prediction = trend_history[-1]
                trend_confidence = 'uncertain'
        elif len(trend_history) == 1:
            trend_prediction = trend_history[0]
            trend_confidence = 'uncertain'
        
        # 預測季節性聚類
        seasonal_prediction = None
        seasonal_confidence = 'insufficient_data'
        if len(seasonal_history) >= 2:
            seasonal_counter = Counter(seasonal_history)
            most_common = seasonal_counter.most_common(1)[0]
            if most_common[1] >= 2:  # 至少出現2次
                seasonal_prediction = most_common[0]
                seasonal_confidence = 'confident' if most_common[1] >= 2 else 'uncertain'
            else:
                # 如果沒有明確的多數，選擇最近年份的結果
                seasonal_prediction = seasonal_history[-1]
                seasonal_confidence = 'uncertain'
        elif len(seasonal_history) == 1:
            seasonal_prediction = seasonal_history[0]
            seasonal_confidence = 'uncertain'
        
        # 統計預測信心
        trend_prediction_stats[trend_confidence] += 1
        seasonal_prediction_stats[seasonal_confidence] += 1
        
        predictions.append({
            'market_vege_id': vege_id,
            'vege_name': vege_name,
            'trend_history': trend_history,
            'seasonal_history': seasonal_history,
            'predicted_trend_cluster_2025': trend_prediction,
            'predicted_seasonal_cluster_2025': seasonal_prediction,
            'trend_confidence': trend_confidence,
            'seasonal_confidence': seasonal_confidence
        })
    
    predictions_df = pd.DataFrame(predictions)
    
    print(f"   📊 趨勢聚類預測統計:")
    print(f"      - 高信心預測: {trend_prediction_stats['confident']} 個蔬菜")
    print(f"      - 不確定預測: {trend_prediction_stats['uncertain']} 個蔬菜")
    print(f"      - 數據不足: {trend_prediction_stats['insufficient_data']} 個蔬菜")
    
    print(f"   📊 季節性聚類預測統計:")
    print(f"      - 高信心預測: {seasonal_prediction_stats['confident']} 個蔬菜")
    print(f"      - 不確定預測: {seasonal_prediction_stats['uncertain']} 個蔬菜")
    print(f"      - 數據不足: {seasonal_prediction_stats['insufficient_data']} 個蔬菜")
    
    return predictions_df, trend_prediction_stats, seasonal_prediction_stats


def create_enhanced_stability_heatmap_with_prediction(comprehensive_df, predictions_df, comparison_plots_dir, mapping_dict):
    """創建包含2025年預測的增強版聚類穩定性熱力圖"""
    
    # 檢查predictions_df是否有必要的欄位
    required_cols = ['market_vege_id', 'predicted_trend_cluster_2025', 'predicted_seasonal_cluster_2025', 
                     'trend_confidence', 'seasonal_confidence']
    
    missing_cols = [col for col in required_cols if col not in predictions_df.columns]
    if missing_cols:
        print(f"   ⚠️  預測DataFrame缺少欄位: {missing_cols}")
        print(f"   💡  可用欄位: {list(predictions_df.columns)}")
        return None
    
    # 合併歷史數據和預測數據
    try:
        enhanced_df = comprehensive_df.merge(
            predictions_df[required_cols], 
            on='market_vege_id', how='left'
        )
    except Exception as e:
        print(f"   ❌ 合併數據時發生錯誤: {e}")
        return None
    
    # 準備趨勢穩定性數據（包含2025預測）
    trend_data = []
    seasonal_data = []
    
    for _, row in enhanced_df.iterrows():
        vege_name = row['vege_name']
        
        # 趨勢聚類數據（2022-2024 + 2025預測）
        trend_row = []
        for year in [2022, 2023, 2024]:
            cluster_val = row.get(f'trend_cluster_{year}', np.nan)
            if pd.isna(cluster_val):
                trend_row.append(0)  # 無數據用0表示
            else:
                trend_row.append(int(cluster_val))
        
        # 添加2025預測
        pred_trend = row.get('predicted_trend_cluster_2025', np.nan)
        if pd.isna(pred_trend):
            trend_row.append(0)
        else:
            trend_row.append(int(pred_trend))
        
        trend_data.append([vege_name] + trend_row)
        
        # 季節性聚類數據（2022-2024 + 2025預測）
        seasonal_row = []
        for year in [2022, 2023, 2024]:
            cluster_val = row.get(f'seasonal_cluster_{year}', np.nan)
            if pd.isna(cluster_val):
                seasonal_row.append(0)  # 無數據用0表示
            else:
                seasonal_row.append(int(cluster_val))
        
        # 添加2025預測
        pred_seasonal = row.get('predicted_seasonal_cluster_2025', np.nan)
        if pd.isna(pred_seasonal):
            seasonal_row.append(0)
        else:
            seasonal_row.append(int(pred_seasonal))
        
        seasonal_data.append([vege_name] + seasonal_row)

    # 創建DataFrame
    trend_df = pd.DataFrame(trend_data, columns=['蔬菜名稱', '2022', '2023', '2024', '2025預測'])
    seasonal_df = pd.DataFrame(seasonal_data, columns=['蔬菜名稱', '2022', '2023', '2024', '2025預測'])
    
    # 創建圖表
    fig, axes = plt.subplots(1, 2, figsize=(24, 15))
    fig.suptitle("蔬菜聚類穩定性分析熱力圖 (含2025年預測)\n(2024年趨勢聚類已調整，2025年基於3票2勝制預測)", fontsize=16, y=0.98)

    # 趨勢聚類熱力圖
    trend_matrix = trend_df.set_index('蔬菜名稱')[['2022', '2023', '2024', '2025預測']].values
    im1 = axes[0].imshow(trend_matrix, cmap='viridis', aspect='auto', vmin=0, vmax=3)
    axes[0].set_title("趨勢聚類變化 (0=無數據, 1-3=聚類編號)\n2024年群聚2↔群聚3已調換，2025年為預測結果", fontsize=14)
    axes[0].set_xlabel("年份")
    axes[0].set_ylabel("蔬菜")
    axes[0].set_xticks(range(4))
    axes[0].set_xticklabels(['2022', '2023', '2024', '2025預測'])
    axes[0].set_yticks(range(len(trend_df)))
    axes[0].set_yticklabels(trend_df['蔬菜名稱'], fontsize=8)
    
    # 添加數值標註，2025預測列特殊標示
    for i in range(len(trend_df)):
        for j in range(4):
            value = int(trend_matrix[i, j])
            color = "white" if value > 0 else "black"
            weight = 'bold' if j < 3 else 'normal'  # 2025預測用普通字體
            if j == 3 and value > 0:  # 2025預測且有值
                text = axes[0].text(j, i, f"{value}*", ha="center", va="center", 
                                  color=color, fontweight=weight, fontsize=9)
            else:
                text = axes[0].text(j, i, value, ha="center", va="center", 
                                  color=color, fontweight=weight)

    # 季節性聚類熱力圖
    seasonal_matrix = seasonal_df.set_index('蔬菜名稱')[['2022', '2023', '2024', '2025預測']].values
    im2 = axes[1].imshow(seasonal_matrix, cmap='plasma', aspect='auto', vmin=0, vmax=2)
    axes[1].set_title("季節性聚類變化 (0=無數據, 1-2=聚類編號)\n2025年為預測結果", fontsize=14)
    axes[1].set_xlabel("年份")
    axes[1].set_ylabel("蔬菜")
    axes[1].set_xticks(range(4))
    axes[1].set_xticklabels(['2022', '2023', '2024', '2025預測'])
    axes[1].set_yticks(range(len(seasonal_df)))
    axes[1].set_yticklabels(seasonal_df['蔬菜名稱'], fontsize=8)
    
    # 添加數值標註，2025預測列特殊標示
    for i in range(len(seasonal_df)):
        for j in range(4):
            value = int(seasonal_matrix[i, j])
            color = "white" if value > 0 else "black"
            weight = 'bold' if j < 3 else 'normal'  # 2025預測用普通字體
            if j == 3 and value > 0:  # 2025預測且有值
                text = axes[1].text(j, i, f"{value}*", ha="center", va="center", 
                                  color=color, fontweight=weight, fontsize=9)
            else:
                text = axes[1].text(j, i, value, ha="center", va="center", 
                                  color=color, fontweight=weight)

    # 添加顏色條
    plt.colorbar(im1, ax=axes[0], shrink=0.8, label='趨勢聚類編號')
    plt.colorbar(im2, ax=axes[1], shrink=0.8, label='季節性聚類編號')

    # 添加說明文字
    fig.text(0.5, 0.02, "註：2025年數據標有 * 號表示預測值 (基於2022-2024年歷史數據的3票2勝制預測)", 
             ha='center', fontsize=12, style='italic')

    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    
    heatmap_path = os.path.join(comparison_plots_dir, "聚類穩定性熱力圖_含2025預測.png")
    plt.savefig(heatmap_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()

    print(f"   📈 增強版聚類穩定性熱力圖(含2025預測)已保存: 聚類穩定性熱力圖_含2025預測.png")
    return heatmap_path


def create_2025_prediction_analysis_charts(predictions_df, prediction_stats, comparison_plots_dir):
    """創建2025年預測分析圖表"""
    
    # 檢查是否有有效數據
    if len(predictions_df) == 0:
        print("   ⚠️  沒有預測數據，跳過預測分析圖表")
        return None
    
    trend_stats, seasonal_stats = prediction_stats
    
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 4, height_ratios=[1, 1, 1.2], hspace=0.3, wspace=0.3)
    fig.suptitle("2025年蔬菜聚類預測分析 (基於2022-2024年3票2勝制)", fontsize=16, y=0.98)

    # 第一行：預測信心度分布
    # 趨勢預測信心度
    ax1 = fig.add_subplot(gs[0, 0])
    confidence_labels = ['高信心', '不確定', '數據不足']
    trend_values = [trend_stats['confident'], trend_stats['uncertain'], trend_stats['insufficient_data']]
    colors = ['#2ecc71', '#f39c12', '#e74c3c']
    
    if sum(trend_values) > 0:
        wedges, texts, autotexts = ax1.pie(trend_values, labels=confidence_labels, autopct='%1.0f%%',
                                          colors=colors, startangle=90)
        ax1.set_title("趨勢聚類預測信心度", fontsize=12)
    else:
        ax1.text(0.5, 0.5, '無趨勢預測數據', ha='center', va='center', transform=ax1.transAxes)
        ax1.set_title("趨勢聚類預測信心度", fontsize=12)

    # 季節性預測信心度
    ax2 = fig.add_subplot(gs[0, 1])
    seasonal_values = [seasonal_stats['confident'], seasonal_stats['uncertain'], seasonal_stats['insufficient_data']]
    
    if sum(seasonal_values) > 0:
        wedges, texts, autotexts = ax2.pie(seasonal_values, labels=confidence_labels, autopct='%1.0f%%',
                                          colors=colors, startangle=90)
        ax2.set_title("季節性聚類預測信心度", fontsize=12)
    else:
        ax2.text(0.5, 0.5, '無季節性預測數據', ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title("季節性聚類預測信心度", fontsize=12)

    # 2025年預測聚類分布 - 趨勢
    ax3 = fig.add_subplot(gs[0, 2])
    trend_pred_counts = predictions_df['predicted_trend_cluster_2025'].value_counts().sort_index()
    if len(trend_pred_counts) > 0:
        ax3.bar(range(len(trend_pred_counts)), trend_pred_counts.values, 
               color=['#ff9999', '#66b3ff', '#99ff99'][:len(trend_pred_counts)])
        ax3.set_title("2025年趨勢聚類預測分布", fontsize=12)
        ax3.set_xlabel("聚類編號")
        ax3.set_ylabel("蔬菜數量")
        ax3.set_xticks(range(len(trend_pred_counts)))
        ax3.set_xticklabels([f"趨勢群{int(idx)}" for idx in trend_pred_counts.index])
        
        # 添加數值標籤
        for i, v in enumerate(trend_pred_counts.values):
            ax3.text(i, v + 0.5, str(v), ha='center', va='bottom')
    else:
        ax3.text(0.5, 0.5, '無趨勢預測分布', ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title("2025年趨勢聚類預測分布", fontsize=12)

    # 2025年預測聚類分布 - 季節性
    ax4 = fig.add_subplot(gs[0, 3])
    seasonal_pred_counts = predictions_df['predicted_seasonal_cluster_2025'].value_counts().sort_index()
    if len(seasonal_pred_counts) > 0:
        ax4.bar(range(len(seasonal_pred_counts)), seasonal_pred_counts.values, 
               color=['#ffcc99', '#ff99cc'][:len(seasonal_pred_counts)])
        ax4.set_title("2025年季節性聚類預測分布", fontsize=12)
        ax4.set_xlabel("聚類編號")
        ax4.set_ylabel("蔬菜數量")
        ax4.set_xticks(range(len(seasonal_pred_counts)))
        ax4.set_xticklabels([f"季節群{int(idx)}" for idx in seasonal_pred_counts.index])
        
        # 添加數值標籤
        for i, v in enumerate(seasonal_pred_counts.values):
            ax4.text(i, v + 0.5, str(v), ha='center', va='bottom')
    else:
        ax4.text(0.5, 0.5, '無季節性預測分布', ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title("2025年季節性聚類預測分布", fontsize=12)

    # 第二行：預測穩定性分析
    # 趨勢聚類穩定性（2022-2024一致且2025預測相同）
    ax5 = fig.add_subplot(gs[1, :2])
    
    stable_veges = []
    changing_veges = []
    
    for _, row in predictions_df.iterrows():
        if len(row.get('trend_history', [])) >= 3:  # 有完整3年數據
            if len(set(row['trend_history'])) == 1:  # 3年完全一致
                if row.get('predicted_trend_cluster_2025') == row['trend_history'][0]:
                    stable_veges.append(f"{row['vege_name']} (群{row['trend_history'][0]})")
                else:
                    pred_val = row.get('predicted_trend_cluster_2025', '?')
                    changing_veges.append(f"{row['vege_name']} ({row['trend_history'][0]}→{pred_val})")
    
    stability_data = ['完全穩定', '預測改變']
    stability_counts = [len(stable_veges), len(changing_veges)]
    
    if sum(stability_counts) > 0:
        ax5.bar(stability_data, stability_counts, color=['#27ae60', '#e67e22'])
        ax5.set_title("趨勢聚類穩定性預測", fontsize=12)
        ax5.set_ylabel("蔬菜數量")
        
        # 添加數值標籤
        for i, v in enumerate(stability_counts):
            ax5.text(i, v + 0.5, str(v), ha='center', va='bottom', fontweight='bold')
    else:
        ax5.text(0.5, 0.5, '無穩定性數據', ha='center', va='center', transform=ax5.transAxes)
        ax5.set_title("趨勢聚類穩定性預測", fontsize=12)

    # 季節性聚類穩定性
    ax6 = fig.add_subplot(gs[1, 2:])
    
    seasonal_stable_veges = []
    seasonal_changing_veges = []
    
    for _, row in predictions_df.iterrows():
        if len(row.get('seasonal_history', [])) >= 3:  # 有完整3年數據
            if len(set(row['seasonal_history'])) == 1:  # 3年完全一致
                if row.get('predicted_seasonal_cluster_2025') == row['seasonal_history'][0]:
                    seasonal_stable_veges.append(f"{row['vege_name']} (群{row['seasonal_history'][0]})")
                else:
                    pred_val = row.get('predicted_seasonal_cluster_2025', '?')
                    seasonal_changing_veges.append(f"{row['vege_name']} ({row['seasonal_history'][0]}→{pred_val})")
    
    seasonal_stability_counts = [len(seasonal_stable_veges), len(seasonal_changing_veges)]
    
    if sum(seasonal_stability_counts) > 0:
        ax6.bar(stability_data, seasonal_stability_counts, color=['#8e44ad', '#d35400'])
        ax6.set_title("季節性聚類穩定性預測", fontsize=12)
        ax6.set_ylabel("蔬菜數量")
        
        # 添加數值標籤
        for i, v in enumerate(seasonal_stability_counts):
            ax6.text(i, v + 0.5, str(v), ha='center', va='bottom', fontweight='bold')
    else:
        ax6.text(0.5, 0.5, '無穩定性數據', ha='center', va='center', transform=ax6.transAxes)
        ax6.set_title("季節性聚類穩定性預測", fontsize=12)

    # 第三行：預測詳細列表
    ax7 = fig.add_subplot(gs[2, :])
    ax7.axis('off')
    
    # 創建預測摘要表格
    high_confidence_trend = predictions_df[predictions_df.get('trend_confidence', '') == 'confident']
    high_confidence_seasonal = predictions_df[predictions_df.get('seasonal_confidence', '') == 'confident']
    
    summary_text = []
    summary_text.append("🔮 2025年聚類預測摘要")
    summary_text.append("=" * 60)
    summary_text.append(f"📊 總預測蔬菜數: {len(predictions_df)}")
    summary_text.append(f"🎯 高信心趨勢預測: {len(high_confidence_trend)} 種蔬菜")
    summary_text.append(f"🎯 高信心季節性預測: {len(high_confidence_seasonal)} 種蔬菜")
    summary_text.append("")
    
    if len(stable_veges) > 0:
        summary_text.append("✅ 趨勢完全穩定蔬菜 (2022-2025一致):")
        for vege in stable_veges[:10]:  # 只顯示前10個
            summary_text.append(f"   • {vege}")
        if len(stable_veges) > 10:
            summary_text.append(f"   • ... 共{len(stable_veges)}種")
    
    summary_text.append("")
    
    if len(changing_veges) > 0:
        summary_text.append("🔄 預測會改變的蔬菜:")
        for vege in changing_veges[:8]:  # 只顯示前8個
            summary_text.append(f"   • {vege}")
        if len(changing_veges) > 8:
            summary_text.append(f"   • ... 共{len(changing_veges)}種")
    
    # 顯示摘要文字
    ax7.text(0.05, 0.95, '\n'.join(summary_text), transform=ax7.transAxes, 
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.7))

    plt.tight_layout()
    
    prediction_path = os.path.join(comparison_plots_dir, "2025年聚類預測分析.png")
    plt.savefig(prediction_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()

    print(f"   📈 2025年聚類預測分析圖已保存: 2025年聚類預測分析.png")
    return prediction_path


def create_prediction_migration_flow_chart(comprehensive_df, predictions_df, comparison_plots_dir):
    """創建包含2025年預測的聚類遷移流程圖"""
    
    # 檢查是否有有效數據
    if len(predictions_df) == 0 or len(comprehensive_df) == 0:
        print("   ⚠️  沒有足夠數據創建遷移流程圖")
        return None
    
    fig, axes = plt.subplots(2, 1, figsize=(18, 14))
    fig.suptitle("蔬菜聚類遷移模式分析 (含2025年預測)\n(2024年趨勢聚類已調整，2025年基於3票2勝制預測)", fontsize=16)

    years = [2022, 2023, 2024, 2025]
    
    # 檢查predictions_df是否有必要的欄位
    required_cols = ['market_vege_id', 'predicted_trend_cluster_2025', 'predicted_seasonal_cluster_2025']
    missing_cols = [col for col in required_cols if col not in predictions_df.columns]
    if missing_cols:
        print(f"   ⚠️  預測DataFrame缺少欄位: {missing_cols}")
        return None
    
    # 合併數據
    try:
        enhanced_df = comprehensive_df.merge(
            predictions_df[required_cols], 
            on='market_vege_id', how='left'
        )
    except Exception as e:
        print(f"   ❌ 合併數據時發生錯誤: {e}")
        return None
    
    # 趨勢聚類遷移（包含2024→2025預測）
    trend_transitions = {}
    for i in range(len(years)-1):
        year1, year2 = years[i], years[i+1]
        transitions = {}
        
        for _, row in enhanced_df.iterrows():
            if year1 == 2025:
                continue
            elif year2 == 2025:
                c1 = row.get(f'trend_cluster_{year1}', np.nan)
                c2 = row.get('predicted_trend_cluster_2025', np.nan)
            else:
                c1 = row.get(f'trend_cluster_{year1}', np.nan)
                c2 = row.get(f'trend_cluster_{year2}', np.nan)
            
            if not pd.isna(c1) and not pd.isna(c2):
                key = f"{int(c1)}→{int(c2)}"
                transitions[key] = transitions.get(key, 0) + 1
        
        period_label = f'{year1}-{year2}' + ('(預測)' if year2 == 2025 else '')
        trend_transitions[period_label] = transitions

    # 繪製趨勢遷移
    ax1 = axes[0]
    x_pos = 0
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'cyan', 'magenta']
    
    for period, transitions in trend_transitions.items():
        y_pos = 0
        max_val = max(transitions.values()) if transitions else 1
        
        for transition, count in sorted(transitions.items()):
            bar_color = colors[y_pos % len(colors)]
            alpha = 0.5 if '預測' in period else 0.7
            
            if count > 0:  # 只繪製有數據的遷移
                ax1.barh(y_pos, count, left=x_pos, height=0.8, 
                        color=bar_color, alpha=alpha,
                        label=f'{transition} ({period})' if x_pos == 0 else "")
                
                # 添加標籤，預測期間用不同樣式
                font_style = 'italic' if '預測' in period else 'normal'
                ax1.text(x_pos + count/2, y_pos, f'{transition}\n({count})', 
                        ha='center', va='center', fontsize=9, fontweight='bold',
                        style=font_style)
            y_pos += 1
        
        x_pos += max_val + 2

    ax1.set_title("趨勢聚類遷移模式 (格式: 起始群→目標群)\n2024年數據已含群聚2↔群聚3調換，2024→2025為預測遷移", fontsize=12)
    ax1.set_xlabel("遷移數量")
    ax1.set_ylabel("遷移類型")
    ax1.grid(True, alpha=0.3)

    # 季節性聚類遷移（包含2024→2025預測）
    seasonal_transitions = {}
    for i in range(len(years)-1):
        year1, year2 = years[i], years[i+1]
        transitions = {}
        
        for _, row in enhanced_df.iterrows():
            if year1 == 2025:
                continue
            elif year2 == 2025:
                c1 = row.get(f'seasonal_cluster_{year1}', np.nan)
                c2 = row.get('predicted_seasonal_cluster_2025', np.nan)
            else:
                c1 = row.get(f'seasonal_cluster_{year1}', np.nan)
                c2 = row.get(f'seasonal_cluster_{year2}', np.nan)
            
            if not pd.isna(c1) and not pd.isna(c2):
                key = f"{int(c1)}→{int(c2)}"
                transitions[key] = transitions.get(key, 0) + 1
        
        period_label = f'{year1}-{year2}' + ('(預測)' if year2 == 2025 else '')
        seasonal_transitions[period_label] = transitions

    # 繪製季節性遷移
    ax2 = axes[1]
    x_pos = 0
    
    for period, transitions in seasonal_transitions.items():
        y_pos = 0
        max_val = max(transitions.values()) if transitions else 1
        
        for transition, count in sorted(transitions.items()):
            bar_color = colors[y_pos % len(colors)]
            alpha = 0.5 if '預測' in period else 0.7
            
            if count > 0:  # 只繪製有數據的遷移
                ax2.barh(y_pos, count, left=x_pos, height=0.8, 
                        color=bar_color, alpha=alpha,
                        label=f'{transition} ({period})' if x_pos == 0 else "")
                
                # 添加標籤，預測期間用不同樣式
                font_style = 'italic' if '預測' in period else 'normal'
                ax2.text(x_pos + count/2, y_pos, f'{transition}\n({count})', 
                        ha='center', va='center', fontsize=9, fontweight='bold',
                        style=font_style)
            y_pos += 1
        
        x_pos += max_val + 2

    ax2.set_title("季節性聚類遷移模式 (格式: 起始群→目標群)\n2024→2025為預測遷移", fontsize=12)
    ax2.set_xlabel("遷移數量")
    ax2.set_ylabel("遷移類型")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    
    migration_path = os.path.join(comparison_plots_dir, "聚類遷移流程圖_含2025預測.png")
    plt.savefig(migration_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()

    print(f"   📈 聚類遷移流程圖(含2025預測)已保存: 聚類遷移流程圖_含2025預測.png")
    return migration_path


def create_2025_all_vegetables_clustering_chart(comprehensive_df, predictions_df, comparison_plots_dir, mapping_dict):
    """創建2025年所有蔬菜的趨勢和季節性聚類結果總覽圖"""
    
    # 檢查是否有有效數據
    if len(predictions_df) == 0:
        print("   ⚠️  沒有預測數據，跳過2025年總覽圖")
        return None
    
    # 檢查必要欄位
    required_cols = ['market_vege_id', 'predicted_trend_cluster_2025', 'predicted_seasonal_cluster_2025',
                     'trend_confidence', 'seasonal_confidence']
    missing_cols = [col for col in required_cols if col not in predictions_df.columns]
    if missing_cols:
        print(f"   ⚠️  預測DataFrame缺少欄位: {missing_cols}")
        return None
    
    # 合併數據
    try:
        enhanced_df = comprehensive_df.merge(
            predictions_df[required_cols], 
            on='market_vege_id', how='left'
        )
    except Exception as e:
        print(f"   ❌ 合併數據時發生錯誤: {e}")
        return None
    
    # 過濾有效預測的蔬菜
    valid_predictions = enhanced_df[
        (~enhanced_df['predicted_trend_cluster_2025'].isna()) | 
        (~enhanced_df['predicted_seasonal_cluster_2025'].isna())
    ].copy()
    
    if len(valid_predictions) == 0:
        print("   ⚠️  沒有有效的預測結果")
        return None
    
    # 創建圖表
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(3, 2, height_ratios=[2, 2, 1], hspace=0.3, wspace=0.3)
    fig.suptitle("2025年蔬菜聚類預測結果總覽\n(基於2022-2024年3票2勝制預測)", fontsize=18, y=0.98)

    # 準備數據
    vegetables = valid_predictions['vege_name'].tolist()
    trend_clusters = valid_predictions['predicted_trend_cluster_2025'].fillna(0).astype(int).tolist()
    seasonal_clusters = valid_predictions['predicted_seasonal_cluster_2025'].fillna(0).astype(int).tolist()
    trend_confidence = valid_predictions['trend_confidence'].fillna('insufficient_data').tolist()
    seasonal_confidence = valid_predictions['seasonal_confidence'].fillna('insufficient_data').tolist()
    
    # 顏色映射
    trend_colors = {0: '#cccccc', 1: '#ff6b6b', 2: '#4ecdc4', 3: '#45b7d1'}  # 0=無預測, 1-3=趨勢群
    seasonal_colors = {0: '#cccccc', 1: '#ffa726', 2: '#ab47bc'}  # 0=無預測, 1-2=季節群
    confidence_alphas = {'confident': 1.0, 'uncertain': 0.6, 'insufficient_data': 0.3}
    
    # 第一行左：趨勢聚類散點圖
    ax1 = fig.add_subplot(gs[0, 0])
    
    # 根據趨勢聚類分組繪製
    for cluster in [1, 2, 3, 0]:  # 0最後繪製（無預測的）
        cluster_data = valid_predictions[valid_predictions['predicted_trend_cluster_2025'].fillna(0) == cluster]
        if len(cluster_data) == 0:
            continue
            
        x_positions = range(len(cluster_data))
        y_positions = [cluster] * len(cluster_data)
        colors = [trend_colors[cluster]] * len(cluster_data)
        alphas = [confidence_alphas[conf] for conf in cluster_data['trend_confidence'].fillna('insufficient_data')]
        
        scatter = ax1.scatter(x_positions, y_positions, c=colors, alpha=alphas, s=100, edgecolors='black', linewidth=1)
        
        # 添加蔬菜名稱標籤
        for i, (x, y, name, conf) in enumerate(zip(x_positions, y_positions, cluster_data['vege_name'], 
                                                  cluster_data['trend_confidence'].fillna('insufficient_data'))):
            fontweight = 'bold' if conf == 'confident' else 'normal'
            ax1.annotate(name, (x, y), xytext=(5, 5), textcoords='offset points', 
                        fontsize=8, fontweight=fontweight, rotation=45)
    
    ax1.set_title("2025年趨勢聚類預測結果", fontsize=14, fontweight='bold')
    ax1.set_ylabel("趨勢聚類編號")
    ax1.set_xlabel("蔬菜 (按聚類分組)")
    ax1.set_yticks([0, 1, 2, 3])
    ax1.set_yticklabels(['無預測', '趨勢群1', '趨勢群2', '趨勢群3'])
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-0.5, 3.5)
    
    # 第一行右：季節性聚類散點圖
    ax2 = fig.add_subplot(gs[0, 1])
    
    # 根據季節性聚類分組繪製
    for cluster in [1, 2, 0]:  # 0最後繪製（無預測的）
        cluster_data = valid_predictions[valid_predictions['predicted_seasonal_cluster_2025'].fillna(0) == cluster]
        if len(cluster_data) == 0:
            continue
            
        x_positions = range(len(cluster_data))
        y_positions = [cluster] * len(cluster_data)
        colors = [seasonal_colors[cluster]] * len(cluster_data)
        alphas = [confidence_alphas[conf] for conf in cluster_data['seasonal_confidence'].fillna('insufficient_data')]
        
        scatter = ax2.scatter(x_positions, y_positions, c=colors, alpha=alphas, s=100, edgecolors='black', linewidth=1)
        
        # 添加蔬菜名稱標籤
        for i, (x, y, name, conf) in enumerate(zip(x_positions, y_positions, cluster_data['vege_name'], 
                                                  cluster_data['seasonal_confidence'].fillna('insufficient_data'))):
            fontweight = 'bold' if conf == 'confident' else 'normal'
            ax2.annotate(name, (x, y), xytext=(5, 5), textcoords='offset points', 
                        fontsize=8, fontweight=fontweight, rotation=45)
    
    ax2.set_title("2025年季節性聚類預測結果", fontsize=14, fontweight='bold')
    ax2.set_ylabel("季節性聚類編號")
    ax2.set_xlabel("蔬菜 (按聚類分組)")
    ax2.set_yticks([0, 1, 2])
    ax2.set_yticklabels(['無預測', '季節群1', '季節群2'])
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-0.5, 2.5)
    
    # 第二行：聚類組合矩陣熱力圖
    ax3 = fig.add_subplot(gs[1, :])
    
    # 創建聚類組合矩陣
    combination_matrix = np.zeros((4, 3))  # 4個趨勢聚類(含0) x 3個季節性聚類(含0)
    combination_counts = {}
    
    for _, row in valid_predictions.iterrows():
        trend_cluster = int(row['predicted_trend_cluster_2025']) if not pd.isna(row['predicted_trend_cluster_2025']) else 0
        seasonal_cluster = int(row['predicted_seasonal_cluster_2025']) if not pd.isna(row['predicted_seasonal_cluster_2025']) else 0
        
        combination_matrix[trend_cluster, seasonal_cluster] += 1
        
        # 記錄組合中的蔬菜名稱
        combo_key = f"T{trend_cluster}S{seasonal_cluster}"
        if combo_key not in combination_counts:
            combination_counts[combo_key] = []
        combination_counts[combo_key].append(row['vege_name'])
    
    # 繪製熱力圖
    im = ax3.imshow(combination_matrix, cmap='YlOrRd', aspect='auto')
    
    # 設置標籤
    ax3.set_xticks(range(3))
    ax3.set_xticklabels(['無季節預測', '季節群1', '季節群2'])
    ax3.set_yticks(range(4))
    ax3.set_yticklabels(['無趨勢預測', '趨勢群1', '趨勢群2', '趨勢群3'])
    ax3.set_xlabel("季節性聚類")
    ax3.set_ylabel("趨勢聚類")
    ax3.set_title("2025年聚類組合分布熱力圖 (數字表示蔬菜數量)", fontsize=14, fontweight='bold')
    
    # 添加數值標籤
    for i in range(4):
        for j in range(3):
            count = int(combination_matrix[i, j])
            if count > 0:
                ax3.text(j, i, str(count), ha="center", va="center", 
                        color="white" if count > combination_matrix.max()/2 else "black",
                        fontsize=12, fontweight='bold')
    
    # 添加顏色條
    plt.colorbar(im, ax=ax3, shrink=0.8, label='蔬菜數量')
    
    # 第三行：詳細統計表格
    ax4 = fig.add_subplot(gs[2, :])
    ax4.axis('off')
    
    # 創建統計摘要
    summary_text = []
    summary_text.append("📊 2025年蔬菜聚類預測統計摘要")
    summary_text.append("=" * 80)
    summary_text.append(f"🔮 總預測蔬菜數: {len(valid_predictions)}")
    
    # 趨勢聚類統計
    trend_stats_summary = valid_predictions['predicted_trend_cluster_2025'].value_counts().sort_index()
    summary_text.append(f"\n📈 趨勢聚類分布:")
    for cluster, count in trend_stats_summary.items():
        if not pd.isna(cluster):
            summary_text.append(f"   • 趨勢群{int(cluster)}: {count} 種蔬菜")
    
    # 季節性聚類統計
    seasonal_stats_summary = valid_predictions['predicted_seasonal_cluster_2025'].value_counts().sort_index()
    summary_text.append(f"\n🌱 季節性聚類分布:")
    for cluster, count in seasonal_stats_summary.items():
        if not pd.isna(cluster):
            summary_text.append(f"   • 季節群{int(cluster)}: {count} 種蔬菜")
    
    # 信心度統計
    trend_conf_stats = valid_predictions['trend_confidence'].value_counts()
    seasonal_conf_stats = valid_predictions['seasonal_confidence'].value_counts()
    summary_text.append(f"\n🎯 預測信心度:")
    summary_text.append(f"   趨勢預測 - 高信心: {trend_conf_stats.get('confident', 0)}, 不確定: {trend_conf_stats.get('uncertain', 0)}, 數據不足: {trend_conf_stats.get('insufficient_data', 0)}")
    summary_text.append(f"   季節預測 - 高信心: {seasonal_conf_stats.get('confident', 0)}, 不確定: {seasonal_conf_stats.get('uncertain', 0)}, 數據不足: {seasonal_conf_stats.get('insufficient_data', 0)}")
    
    # 主要組合
    summary_text.append(f"\n🔄 主要聚類組合:")
    sorted_combos = sorted([(k, len(v)) for k, v in combination_counts.items() if len(v) > 0], 
                          key=lambda x: x[1], reverse=True)
    for combo, count in sorted_combos[:5]:  # 顯示前5個組合
        if count > 0:
            summary_text.append(f"   • {combo}: {count} 種蔬菜")
    
    # 顯示摘要文字
    summary_text_str = '\n'.join(summary_text)
    ax4.text(0.05, 0.95, summary_text_str, transform=ax4.transAxes, 
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightcyan", alpha=0.8))
    
    # 添加圖例
    legend_elements = []
    legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#ff6b6b', 
                                     markersize=10, label='趨勢群1', markeredgecolor='black'))
    legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#4ecdc4', 
                                     markersize=10, label='趨勢群2', markeredgecolor='black'))
    legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#45b7d1', 
                                     markersize=10, label='趨勢群3', markeredgecolor='black'))
    legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#ffa726', 
                                     markersize=10, label='季節群1', markeredgecolor='black'))
    legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#ab47bc', 
                                     markersize=10, label='季節群2', markeredgecolor='black'))
    legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#cccccc', 
                                     markersize=10, label='無預測', markeredgecolor='black'))
    
    # 信心度圖例
    legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='black', 
                                     markersize=10, label='高信心', alpha=1.0, markeredgecolor='black'))
    legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='black', 
                                     markersize=10, label='不確定', alpha=0.6, markeredgecolor='black'))
    legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='black', 
                                     markersize=10, label='數據不足', alpha=0.3, markeredgecolor='black'))
    
    # 將圖例放在右上角
    fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.95), 
              title="聚類標籤與信心度", title_fontsize=12, fontsize=10)
    
    plt.tight_layout()
    
    overview_path = os.path.join(comparison_plots_dir, "2025年蔬菜聚類預測總覽.png")
    plt.savefig(overview_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()

def create_2025_clustering_correlation_heatmap(comprehensive_df, predictions_df, comparison_plots_dir, mapping_dict):
    """創建2025年聚類關聯性分析熱力圖，特別關注趨勢分群3與季節分群2的關係"""
    
    # 檢查是否有有效數據
    if len(predictions_df) == 0:
        print("   ⚠️  沒有預測數據，跳過聚類關聯性分析")
        return None
    
    # 檢查必要欄位
    required_cols = ['market_vege_id', 'predicted_trend_cluster_2025', 'predicted_seasonal_cluster_2025',
                     'trend_confidence', 'seasonal_confidence']
    missing_cols = [col for col in required_cols if col not in predictions_df.columns]
    if missing_cols:
        print(f"   ⚠️  預測DataFrame缺少欄位: {missing_cols}")
        return None
    
    # 合併數據
    try:
        enhanced_df = comprehensive_df.merge(
            predictions_df[required_cols], 
            on='market_vege_id', how='left'
        )
    except Exception as e:
        print(f"   ❌ 合併數據時發生錯誤: {e}")
        return None
    
    # 過濾有效預測的蔬菜
    valid_predictions = enhanced_df[
        (~enhanced_df['predicted_trend_cluster_2025'].isna()) & 
        (~enhanced_df['predicted_seasonal_cluster_2025'].isna())
    ].copy()
    
    if len(valid_predictions) == 0:
        print("   ⚠️  沒有同時具有趨勢和季節性預測的蔬菜")
        return None
    
    # 創建圖表
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(3, 3, height_ratios=[2, 1.5, 1], width_ratios=[2, 1, 1], 
                         hspace=0.3, wspace=0.3)
    fig.suptitle("2025年蔬菜聚類關聯性分析\n重點關注：趨勢分群3 ↔ 季節分群2 的關係", 
                fontsize=18, y=0.98, fontweight='bold')

    # 創建聚類關聯矩陣
    trend_clusters = [1, 2, 3]
    seasonal_clusters = [1, 2]
    
    # 初始化矩陣和詳細信息
    correlation_matrix = np.zeros((len(trend_clusters), len(seasonal_clusters)))
    cluster_details = {}
    
    for i, trend_cluster in enumerate(trend_clusters):
        for j, seasonal_cluster in enumerate(seasonal_clusters):
            # 找到屬於此組合的蔬菜
            combo_veges = valid_predictions[
                (valid_predictions['predicted_trend_cluster_2025'] == trend_cluster) & 
                (valid_predictions['predicted_seasonal_cluster_2025'] == seasonal_cluster)
            ]
            
            count = len(combo_veges)
            correlation_matrix[i, j] = count
            
            # 記錄詳細信息
            key = f"T{trend_cluster}S{seasonal_cluster}"
            cluster_details[key] = {
                'count': count,
                'vegetables': combo_veges['vege_name'].tolist(),
                'trend_confidence': combo_veges['trend_confidence'].tolist(),
                'seasonal_confidence': combo_veges['seasonal_confidence'].tolist()
            }
    
    # 主熱力圖 (左上，跨兩格)
    ax1 = fig.add_subplot(gs[0, :2])
    
    # 創建自定義顏色映射，突出顯示T3S2組合
    custom_colors = correlation_matrix.copy()
    max_val = correlation_matrix.max()
    
    # 使用不同的顏色方案突出T3S2
    im1 = ax1.imshow(correlation_matrix, cmap='Reds', aspect='auto')
    
    # 設置標籤
    ax1.set_xticks(range(len(seasonal_clusters)))
    ax1.set_xticklabels([f'季節群{s}' for s in seasonal_clusters], fontsize=12)
    ax1.set_yticks(range(len(trend_clusters)))
    ax1.set_yticklabels([f'趨勢群{t}' for t in trend_clusters], fontsize=12)
    ax1.set_xlabel("季節性聚類", fontsize=14, fontweight='bold')
    ax1.set_ylabel("趨勢聚類", fontsize=14, fontweight='bold')
    ax1.set_title("2025年聚類組合分布熱力圖", fontsize=16, fontweight='bold')
    
    # 添加數值標籤和特殊標記
    for i in range(len(trend_clusters)):
        for j in range(len(seasonal_clusters)):
            count = int(correlation_matrix[i, j])
            
            # 特別標記T3S2組合
            if trend_clusters[i] == 3 and seasonal_clusters[j] == 2:
                # 添加特殊邊框
                rect = plt.Rectangle((j-0.4, i-0.4), 0.8, 0.8, fill=False, 
                                   edgecolor='gold', linewidth=4)
                ax1.add_patch(rect)
                
                # 特殊標記文字
                ax1.text(j, i-0.2, f"★ {count} ★", ha="center", va="center", 
                        color="white" if count > max_val/2 else "black",
                        fontsize=16, fontweight='bold')
                ax1.text(j, i+0.2, "重點組合", ha="center", va="center", 
                        color="gold", fontsize=10, fontweight='bold')
            else:
                ax1.text(j, i, str(count), ha="center", va="center", 
                        color="white" if count > max_val/2 else "black",
                        fontsize=14, fontweight='bold')
    
    # 添加顏色條
    plt.colorbar(im1, ax=ax1, shrink=0.8, label='蔬菜數量')
    
    # 比例分析圖 (右上)
    ax2 = fig.add_subplot(gs[0, 2])
    
    # 計算各趨勢群在不同季節群的分布比例
    trend_group_proportions = []
    trend_group_labels = []
    
    for i, trend_cluster in enumerate(trend_clusters):
        total_in_trend = correlation_matrix[i, :].sum()
        if total_in_trend > 0:
            proportions = correlation_matrix[i, :] / total_in_trend * 100
            
            # 創建堆疊條形圖
            bottom = 0
            for j, prop in enumerate(proportions):
                color = 'gold' if trend_cluster == 3 and seasonal_clusters[j] == 2 else f'C{j}'
                alpha = 1.0 if trend_cluster == 3 and seasonal_clusters[j] == 2 else 0.7
                
                ax2.bar(i, prop, bottom=bottom, 
                       color=color, alpha=alpha, 
                       label=f'季節群{seasonal_clusters[j]}' if i == 0 else "",
                       edgecolor='black', linewidth=1)
                
                # 添加百分比標籤
                if prop > 5:  # 只在比例大於5%時顯示標籤
                    ax2.text(i, bottom + prop/2, f'{prop:.0f}%', 
                           ha='center', va='center', fontweight='bold')
                
                bottom += prop
    
    ax2.set_xlabel('趨勢聚類', fontweight='bold')
    ax2.set_ylabel('季節群分布比例 (%)', fontweight='bold')
    ax2.set_title('各趨勢群的季節性分布', fontsize=12, fontweight='bold')
    ax2.set_xticks(range(len(trend_clusters)))
    ax2.set_xticklabels([f'趨勢群{t}' for t in trend_clusters])
    ax2.legend(loc='upper right')
    ax2.set_ylim(0, 100)
    
    # 特殊標記趨勢群3
    ax2.axvline(x=2, color='gold', linestyle='--', linewidth=2, alpha=0.7)
    ax2.text(2, 95, '重點分析', ha='center', va='top', color='gold', 
            fontweight='bold', fontsize=10)
    
    # T3S2詳細分析 (中間左)
    ax3 = fig.add_subplot(gs[1, :2])
    ax3.axis('off')
    
    # 分析T3S2組合
    t3s2_veges = cluster_details.get('T3S2', {})
    t3_total = sum(cluster_details.get(f'T3S{s}', {}).get('count', 0) for s in seasonal_clusters)
    s2_total = sum(cluster_details.get(f'T{t}S2', {}).get('count', 0) for t in trend_clusters)
    
    analysis_text = []
    analysis_text.append("🔍 趨勢分群3 ↔ 季節分群2 關聯性分析")
    analysis_text.append("=" * 60)
    analysis_text.append(f"📊 趨勢分群3總蔬菜數: {t3_total}")
    analysis_text.append(f"📊 季節分群2總蔬菜數: {s2_total}")
    analysis_text.append(f"⭐ T3S2組合蔬菜數: {t3s2_veges.get('count', 0)}")
    
    if t3_total > 0:
        t3_to_s2_ratio = (t3s2_veges.get('count', 0) / t3_total) * 100
        analysis_text.append(f"📈 趨勢群3中屬於季節群2的比例: {t3_to_s2_ratio:.1f}%")
        
        if t3_to_s2_ratio == 100:
            analysis_text.append("🎯 **重要發現**: 所有趨勢分群3的蔬菜都屬於季節分群2！")
        elif t3_to_s2_ratio >= 80:
            analysis_text.append("🎯 **高度關聯**: 趨勢分群3與季節分群2高度相關！")
        elif t3_to_s2_ratio >= 50:
            analysis_text.append("🎯 **中度關聯**: 趨勢分群3與季節分群2有明顯關聯")
        else:
            analysis_text.append("🎯 **低度關聯**: 趨勢分群3與季節分群2關聯性較低")
    
    if s2_total > 0:
        s2_from_t3_ratio = (t3s2_veges.get('count', 0) / s2_total) * 100
        analysis_text.append(f"📈 季節群2中來自趨勢群3的比例: {s2_from_t3_ratio:.1f}%")
    
    analysis_text.append("")
    analysis_text.append("🥬 T3S2組合蔬菜列表:")
    vegetables_list = t3s2_veges.get('vegetables', [])
    if vegetables_list:
        for i, vege in enumerate(vegetables_list):
            trend_conf = t3s2_veges.get('trend_confidence', [''])[i] if i < len(t3s2_veges.get('trend_confidence', [])) else ''
            seasonal_conf = t3s2_veges.get('seasonal_confidence', [''])[i] if i < len(t3s2_veges.get('seasonal_confidence', [])) else ''
            
            conf_indicator = ""
            if trend_conf == 'confident' and seasonal_conf == 'confident':
                conf_indicator = " ⭐⭐"
            elif trend_conf == 'confident' or seasonal_conf == 'confident':
                conf_indicator = " ⭐"
            
            analysis_text.append(f"   • {vege}{conf_indicator}")
    else:
        analysis_text.append("   (無蔬菜屬於此組合)")
    
    # 顯示分析文字
    ax3.text(0.05, 0.95, '\n'.join(analysis_text), transform=ax3.transAxes, 
            fontsize=11, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.9))
    
    # 信心度分析 (中間右)
    ax4 = fig.add_subplot(gs[1, 2])
    
    # 分析T3S2組合的預測信心度
    if t3s2_veges.get('count', 0) > 0:
        trend_conf_counts = pd.Series(t3s2_veges.get('trend_confidence', [])).value_counts()
        seasonal_conf_counts = pd.Series(t3s2_veges.get('seasonal_confidence', [])).value_counts()
        
        # 創建信心度對比圖
        conf_labels = ['confident', 'uncertain', 'insufficient_data']
        conf_colors = ['green', 'orange', 'red']
        
        x_pos = np.arange(len(conf_labels))
        width = 0.35
        
        trend_values = [trend_conf_counts.get(label, 0) for label in conf_labels]
        seasonal_values = [seasonal_conf_counts.get(label, 0) for label in conf_labels]
        
        ax4.bar(x_pos - width/2, trend_values, width, label='趨勢預測', 
               color=conf_colors, alpha=0.7)
        ax4.bar(x_pos + width/2, seasonal_values, width, label='季節預測', 
               color=conf_colors, alpha=0.5)
        
        ax4.set_xlabel('預測信心度')
        ax4.set_ylabel('蔬菜數量')
        ax4.set_title('T3S2組合\n預測信心度分布', fontweight='bold')
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(['高信心', '不確定', '數據不足'], rotation=45)
        ax4.legend()
        
        # 添加數值標籤
        for i, (t_val, s_val) in enumerate(zip(trend_values, seasonal_values)):
            if t_val > 0:
                ax4.text(i - width/2, t_val + 0.1, str(t_val), ha='center', va='bottom')
            if s_val > 0:
                ax4.text(i + width/2, s_val + 0.1, str(s_val), ha='center', va='bottom')
    else:
        ax4.text(0.5, 0.5, 'T3S2組合\n無蔬菜數據', ha='center', va='center', 
                transform=ax4.transAxes, fontsize=12)
        ax4.set_title('T3S2組合預測信心度', fontweight='bold')
    
    # 全局統計摘要 (底部)
    ax5 = fig.add_subplot(gs[2, :])
    ax5.axis('off')
    
    # 計算各種統計指標
    total_valid = len(valid_predictions)
    
    summary_stats = []
    summary_stats.append("📈 2025年聚類關聯性統計摘要")
    summary_stats.append("=" * 80)
    summary_stats.append(f"📊 總有效預測蔬菜數: {total_valid}")
    summary_stats.append("")
    
    # 各組合統計
    summary_stats.append("🔗 所有聚類組合分布:")
    for i, trend_cluster in enumerate(trend_clusters):
        for j, seasonal_cluster in enumerate(seasonal_clusters):
            count = int(correlation_matrix[i, j])
            percentage = (count / total_valid * 100) if total_valid > 0 else 0
            star = " ⭐" if trend_cluster == 3 and seasonal_cluster == 2 else ""
            summary_stats.append(f"   T{trend_cluster}S{seasonal_cluster}: {count} 種蔬菜 ({percentage:.1f}%){star}")
    
    summary_stats.append("")
    
    # 關聯性分析結論
    if t3_total > 0 and t3s2_veges.get('count', 0) == t3_total:
        summary_stats.append("🎯 **關鍵發現**: 100%的趨勢分群3蔬菜都屬於季節分群2")
        summary_stats.append("💡 **意義**: 趨勢分群3與季節分群2存在完美正相關關係")
    elif t3_total > 0:
        ratio = (t3s2_veges.get('count', 0) / t3_total) * 100
        summary_stats.append(f"🎯 **關鍵發現**: {ratio:.1f}%的趨勢分群3蔬菜屬於季節分群2")
    
    # 顯示統計摘要
    ax5.text(0.05, 0.95, '\n'.join(summary_stats), transform=ax5.transAxes, 
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightcyan", alpha=0.8))
    
    plt.tight_layout()
    
    correlation_path = os.path.join(comparison_plots_dir, "2025年聚類關聯性分析熱力圖.png")
    plt.savefig(correlation_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()

    print(f"   📈 2025年聚類關聯性分析熱力圖已保存: 2025年聚類關聯性分析熱力圖.png")
    return correlation_path


# 以下為原有函數的完整代碼（保持不變）
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
        # 可以在這裡添加更多預設對應
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

    for directory in [analysis_dir, plots_dir, decomp_plots_dir, individual_plots_dir, comparison_plots_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"✅ 建立資料夾: {directory}")

    return analysis_dir, plots_dir, decomp_plots_dir, individual_plots_dir, comparison_plots_dir


def extract_typhoon_dates_from_data(df):
    """從資料中提取颱風日期"""
    print("🌪️ 從資料中提取颱風日期...")
    
    # 檢查是否有typhoon欄位
    if 'typhoon' not in df.columns:
        print("⚠️ 資料中未找到typhoon欄位")
        return {}
    
    # 提取颱風日期
    typhoon_data = df[df['typhoon'] == 1].copy()
    
    if len(typhoon_data) == 0:
        print("⚠️ 資料中未找到颱風標記")
        return {}
    
    # 按年份分組颱風日期
    typhoon_dates_by_year = {}
    
    for year in [2022, 2023, 2024]:
        year_typhoon = typhoon_data[typhoon_data['ObsTime'].dt.year == year]
        
        if len(year_typhoon) > 0:
            # 獲取該年度的颱風日期，並轉換為年內天數
            year_dates = year_typhoon['ObsTime'].dt.dayofyear.unique()
            typhoon_dates_by_year[year] = sorted(year_dates)
            print(f"   📅 {year}年: 找到 {len(year_dates)} 個颱風日期")
        else:
            typhoon_dates_by_year[year] = []
            print(f"   📅 {year}年: 無颱風日期")
    
    return typhoon_dates_by_year


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


def main():
    """主程式"""
    print(f"📂 工作目錄: {os.getcwd()}")
    print(f"⏰ 開始時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 載入蔬菜名稱對應表
    print("\n🔍 載入蔬菜名稱對應表...")
    mapping_dict = load_vegetable_mapping()

    # 建立輸出資料夾
    analysis_dir, plots_dir, decomp_plots_dir, individual_plots_dir, comparison_plots_dir = setup_directories()

    # 讀取資料
    print("\n📂 讀取多年度資料...")
    df = pd.read_excel("daily_avg_price_vege.xlsx")
    df["ObsTime"] = pd.to_datetime(df["ObsTime"])
    print(f"✅ 成功讀取 {len(df):,} 筆記錄")

    # 提取颱風日期
    typhoon_dates_by_year = extract_typhoon_dates_from_data(df)
    typhoon_summary = {year: len(dates) for year, dates in typhoon_dates_by_year.items()}

    # 分年份處理
    yearly_results = {}

    for year in [2022, 2023, 2024]:
        print(f"\n📅 分析 {year} 年...")
        year_df = df[df["ObsTime"].dt.year == year].copy()

        if len(year_df) == 0:
            print(f"   ⚠️  {year}年無資料")
            continue

        print(f"   📊 {year}年: {len(year_df):,} 筆記錄, {year_df['market_vege_id'].nunique()} 種蔬菜")

        # 時間序列分解
        print("   🔄 執行時間序列分解...")
        decomp_results = {}
        vege_ids = year_df["market_vege_id"].unique()

        for vege_id in vege_ids:
            vege_name = get_chinese_name(vege_id, mapping_dict)
            result = decompose_time_series(year_df, vege_id, year)
            if result is not None:
                decomp_results[vege_id] = result

        print(f"   ✅ 成功分解 {len(decomp_results)} 種蔬菜")

        if len(decomp_results) < max(SEASONAL_CLUSTERS, TREND_CLUSTERS):
            print(f"   ⚠️  有效分解數量不足，跳過聚類")
            yearly_results[year] = {
                "decomposition": decomp_results,
                "clustering": {},
            }
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

                print(f"      ✅ 趨勢聚類完成 (輪廓係數: {trend_results['silhouette_score']:.3f})")

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

                print(f"      ✅ 季節性聚類完成 (輪廓係數: {seasonal_results['silhouette_score']:.3f})")

        yearly_results[year] = {
            "decomposition": decomp_results,
            "clustering": year_clustering,
        }

    # 創建綜合分析
    print("\n📋 創建綜合分析...")
    comprehensive_df = create_comprehensive_analysis(yearly_results, analysis_dir, mapping_dict)

    # 檢查是否有聚類數據
    has_clustering_data = False
    for col in comprehensive_df.columns:
        if 'cluster' in col and not comprehensive_df[col].isna().all():
            has_clustering_data = True
            break
    
    if not has_clustering_data:
        print("   ⚠️  沒有找到有效的聚類數據，跳過預測功能")
        print("   💡  請檢查數據是否包含足夠的時間序列用於聚類分析")
        return

    # 🔮 新增：預測2025年聚類結果
    print("\n🔮 執行2025年聚類預測...")
    predictions_df, trend_stats, seasonal_stats = predict_2025_clusters(comprehensive_df)
    prediction_stats = (trend_stats, seasonal_stats)

    # 保存預測結果
    predictions_path = os.path.join(analysis_dir, "2025年聚類預測結果.csv")
    predictions_df.to_csv(predictions_path, index=False, encoding="utf-8-sig")
    print(f"   📁 2025年預測結果已保存: 2025年聚類預測結果.csv")

    # 檢查是否有有效預測
    total_predictions = len(predictions_df)
    valid_trend_predictions = len(predictions_df[~predictions_df['predicted_trend_cluster_2025'].isna()])
    valid_seasonal_predictions = len(predictions_df[~predictions_df['predicted_seasonal_cluster_2025'].isna()])
    
    if valid_trend_predictions == 0 and valid_seasonal_predictions == 0:
        print("   ⚠️  沒有有效的預測結果，跳過預測圖表生成")
        return

    # 🎨 創建新增的預測分析圖表
    print("\n🎨 創建預測分析圖表...")
    
    try:
        # 1. 增強版穩定性熱力圖（含2025預測）
        create_enhanced_stability_heatmap_with_prediction(
            comprehensive_df, predictions_df, comparison_plots_dir, mapping_dict
        )
        
        # 2. 2025年預測分析圖表
        create_2025_prediction_analysis_charts(
            predictions_df, prediction_stats, comparison_plots_dir
        )
        
        # 3. 預測遷移流程圖
        create_prediction_migration_flow_chart(
            comprehensive_df, predictions_df, comparison_plots_dir
        )
        
        # 4. 新增：2025年所有蔬菜聚類總覽圖
        create_2025_all_vegetables_clustering_chart(
            comprehensive_df, predictions_df, comparison_plots_dir, mapping_dict
        )
        
        # 5. 新增：2025年聚類關聯性分析熱力圖
        create_2025_clustering_correlation_heatmap(
            comprehensive_df, predictions_df, comparison_plots_dir, mapping_dict
        )
    except Exception as e:
        print(f"   ⚠️  預測圖表生成過程中發生錯誤: {e}")
        print("   💡  但預測結果CSV已成功保存")

    # 最終摘要
    print(f"\n🎉 完整蔬菜價格DTW聚類分析完成 (含2025預測)!")
    print("=" * 80)
    print(f"📊 分析結果:")
    print(f"   ✅ 成功分析年份: {len(yearly_results)}")
    print(f"   🔮 2025年預測: {total_predictions} 種蔬菜")
    print(f"   📊 有效趨勢預測: {valid_trend_predictions} 種蔬菜")
    print(f"   📊 有效季節性預測: {valid_seasonal_predictions} 種蔬菜")

    # 預測統計摘要
    print(f"\n🔮 2025年預測統計:")
    print(f"   📊 趨勢聚類高信心預測: {trend_stats['confident']} 種蔬菜")
    print(f"   📊 季節性聚類高信心預測: {seasonal_stats['confident']} 種蔬菜")

    print(f"\n🚀 新增功能包含:")
    print(f"   🔮 2025年聚類預測 (3票2勝制)")
    print(f"   📈 增強版聚類穩定性熱力圖 (含2025預測)")
    print(f"   📊 2025年預測分析圖表")
    print(f"   🔄 預測遷移流程圖")
    print(f"   🎯 2025年蔬菜聚類預測總覽圖")
    print(f"   🔗 2025年聚類關聯性分析熱力圖")
    print(f"   📁 2025年聚類預測結果.csv")

    print(f"\n✨ 完成時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    print("🚀 完整聚類分析程式開始執行...")
    print("🔄 2024年趨勢聚類修改版本 - 群聚2↔群聚3標籤調換")
    print("📝 修正版本 - 統一標籤處理邏輯，所有圖表標題已更新")
    print("🌪️ 增強版本 - 含颱風標註功能")
    print("🔮 預測版本 - 含2025年聚類預測功能")
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