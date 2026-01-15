# LIBE-based Material Screening Demo
## A Model–Policy–Agent Ready PoC for Battery Electrolyte R&D

---

# Quick Setup (無 Python 環境可忽略)

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate.bat

python -m pip install -U pip
pip install -r requirements.txt
```

請先移除 `data\outputs`、`data\processed` 中的所有檔案，並擇一執行：

```powershell
# 需要有本地 Python 環境
.\setup.ps1
.\run_all.ps1

## Notes
- Run the PowerShell scripts from the project root (this folder). You do not need to move the `.ps1` files.
- If PowerShell blocks script execution, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in the same window, then re-run the scripts.
- Outputs are written under `data/processed/` and `data/outputs/`.
```

或

```bash
# 依序執行:
python src\00_inspect_libe.py
python src\00b_inspect_nested.py
python src\01_build_table.py
python src\02_train_rank.py
python src\03_explain_and_export.py
python src\04_family_ranking.py
python src\05_rule_feedback_retrain.py
```

---

## 1. 專案目的（Why this demo exists）
本專案示範一個以電池材料研發（R&D）為核心的 AI 決策支援系統原型，整體設計對齊材料科學與 AI 系統工程，目標並非直接發現新材料，而是建立一套可擴展、可解釋、可接軌 Agentic AI 的材料篩選與決策流程。

核心理念包含：
- 材料科學知識 + 大規模資料結構化
- 候選材料空間的矩陣化表示（Material Space）
- AI 高通量篩選（High-throughput screening）
- 以研發決策支援為核心，而非製造端控制
- 可自然延伸為 Agentic AI（自動觸發 DFT / 實驗 / 回饋）

---

## 2. 系統整體設計理念（High-level Design）

### 為何不採用端到端生成模型
在材料研發中，真正困難的不是生成分子，而是：
- 如何在大量候選中進行有效比較
- 如何同時考量物理量與研發策略
- 如何在不破壞物理一致性的前提下導入決策偏好

因此本專案採用分層式設計：
資料層 → 表徵層 → 模型層 → 決策層（Rule / Policy） → Agentic 擴展

---

## 3. 資料來源與前處理（Data Layer）

### 3.1 資料來源：LIBE Dataset
LIBE（Lithium-Ion Battery Electrolyte）資料集包含約 17,000 筆分子與反應中間體，提供：
- 原子組成（elements）
- 鍵結關係（bonds）
- 三維座標（xyz）
- 多種熱力學性質（thermo）

### 3.2 為何不能直接使用
LIBE 並非穩定電解質清單，而是反應空間資料，包含：
- H₂、LiH 等碎片
- Radical 與中間體
- 不適合作為直接研發候選

因此需先進行結構重建與語義篩選。

---

## 4. 分子結構重建與標籤設計

### 4.1 結構重建
- 使用 elements + bonds 重建 RDKit Mol
- 轉為 canonical SMILES
- 僅保留 charge = 0 分子

### 4.2 標籤設計：per-atom free energy
直接使用總自由能會導致尺寸效應主導排序，因此採用：
free_energy_per_atom = free_energy / number_atoms

此設計可讓模型學習化學結構差異，而非分子大小。

---

## 5. 特徵工程（Feature Engineering）

採用可解釋且成熟的 RDKit descriptors：
- MolWt
- TPSA
- MolLogP
- NumHDonors / NumHAcceptors
- HeavyAtomCount
- RingCount
- NumRotatableBonds

刻意避免使用黑盒 GNN，以維持可解釋性與決策透明度。

---

## 6. 模型層（Model Layer）

- 模型：XGBoost Regressor
- 任務：預測 per-atom free energy
- 評估：MAE + SHAP

修正後模型顯示：
- MolWt 不再壓倒性主導
- TPSA、MolLogP、Donor 等化學語義浮現

---

## 7. 高通量排序與 Family-level 聚合

### 7.1 Family 定義
依元素組合（如 C-F-N-S、C-N-O-S）進行分群

### 7.2 設計目的
- 避免同族構型重複佔榜
- 將決策焦點提升至材料族群層級

---

## 8. Rule / Policy 決策層（Demo 核心）

### 8.1 為何需要 Rule
模型僅反映物理可行性，實際研發仍需考量：
- 溶解性
- 安全性
- 專案策略

這些以 Rule / Policy 層獨立表達。

### 8.2 Rule-based Re-ranking
最終決策分數定義為：
final_score = y_pred + policy_bonus

Rule 僅影響決策邊界候選，不推翻物理模型。

---

## 9. Demo 成果解讀

- 大多數 family 排名穩定
- 僅在邊界區域出現 1-in / 1-out 調整
- 顯示決策系統穩定且可信

---

## 10. Agentic AI 延伸說明

未來 Agent 可：
1. 讀取模型排序
2. 依 Policy 選擇候選
3. 觸發 DFT / 實驗
4. 回寫結果
5. 週期性再訓練或 rerank

本 Demo 的 Rule 即為 Agent 的 policy prototype。

---

## 11. 專案定位總結

本專案不是：
- 商用系統
- 製造控制系統
- 全自動材料發現引擎

而是：
材料 AI 決策支援 + Agentic AI 擴展就緒的 PoC。
