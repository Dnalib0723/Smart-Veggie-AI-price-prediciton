# Smart-Veggie Price Prediction — Consolidated README

> Auto-generated drafts on 2025-09-19 01:30. Please review and refine module-specific sections before publishing.

## Repository Structure (suggested)
```
.
├─ All_vege_price_prediction/     # all-vegetable pipelines
├─ cabbage_price_prediction/      # cabbage-focused experiments
├─ clustering code/               # clustering utilities and analyses
├─ Prophet/                       # Prophet forecasting modules
├─ XGBoost/                       # XGBoost pipelines
├─ XGBoost+ SHAP/                 # XGBoost + SHAP interpretability
├─ high.pkl/                      # high-volatility artifacts
├─ low pkl/                       # low-volatility artifacts
└─ README.md

```

## 模組說明

### cabbage_price_prediction (LinearRegression, 沒有sliding window cross-validation)
針對高麗菜進行獨立預測實驗，包含目標特徵設計、時間序列處理與模型比較，用於驗證專案方法在單一作物的效果。

### All_vege_price_prediction (RandomForestRegressor, 沒有sliding window cross-validation)
整合所有蔬菜的資料處理與建模流程，含資料清理、特徵工程、模型訓練與報表輸出，是整體專案的核心 pipeline。

### clustering code- All vegetable prices clustering code
提供結合dynamic time warping (DTW)分析+K-means聚類方法與工具，用於依價格波動或季節趨勢將蔬菜分群，並輸出分群圖表與穩定性指標，支持解釋與分群建模。

### Prophet
基於 Prophet 的時間序列模型，納入天氣與產量特徵，支援快速建模與解釋季節性、趨勢與節日效應，驗證多種超參數設定。

### XGBoost (所有特徵)
以 XGBoost 為核心的預測模組，包含訓練、交叉驗證並執行SelectFromModel(SFM)特徵重要性分析，比較保留氣象與移除氣象特徵的結果，捕捉非線性與高維特徵對菜價的影響。

### XGBoost+ SHAP (所有特徵 + SHAP特徵篩選)
結合 XGBoost 與 SHAP 解釋器，分析重要天氣與季節特徵對價格的貢獻，提供可視化報告，提升模型透明度與決策支持。

### high.pkl (XGBoost + 精選特徵)
保存高波動蔬菜群的模型與資料序列化檔，支援快速載入與推論，主要作為重現性與部署的中繼成果。

### low pkl (XGBoost + 精選特徵)
保存低波動蔬菜群的模型與資料序列化檔，用於對照高波動群，驗證特徵影響與模型穩定性，提升整體分析完整度。
