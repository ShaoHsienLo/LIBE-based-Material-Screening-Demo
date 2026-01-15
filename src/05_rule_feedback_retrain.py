from pathlib import Path
import pandas as pd
import numpy as np
import joblib

from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

DATA = Path("data/processed/libe_like.csv")
OUT = Path("data/outputs")
OUT.mkdir(parents=True, exist_ok=True)

DESCRIPTOR_FNS = {
    "MolWt": Descriptors.MolWt,
    "TPSA": Descriptors.TPSA,
    "NumHDonors": Descriptors.NumHDonors,
    "NumHAcceptors": Descriptors.NumHAcceptors,
    "MolLogP": Descriptors.MolLogP,
    "NumRotatableBonds": Descriptors.NumRotatableBonds,
    "RingCount": Descriptors.RingCount,
    "HeavyAtomCount": Descriptors.HeavyAtomCount,
}

# ===== 自定義 rule（你可以改這裡）=====
RULE = {
    # 例：電解質/添加劑常見的「可溶/可極性」窗口（示意，非真實規格）
    "tpsa_min": 10.0,
    "tpsa_max": 80.0,
    "logp_max": 2.0,
    "donor_max": 1,

    # 觸發 DFT/實驗的數量（從目前預測 topN 中挑 K 個）
    "screen_topN": 200,
    "select_K": 20,
}

# 回饋如何作用到再訓練：sample weight
WEIGHT_POS = 3.0   # 被 rule 選中（建議送 DFT/實驗）→ 加權
WEIGHT_NEG = 1.0   # 未選中 → 普通權重


def featurize_smiles(smiles: str) -> dict | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    feats = {}
    for k, fn in DESCRIPTOR_FNS.items():
        try:
            feats[k] = float(fn(mol))
        except Exception:
            feats[k] = np.nan
    return feats


def build_features(df: pd.DataFrame):
    feat_rows, ok_idx = [], []
    for i, s in enumerate(df["smiles"].astype(str).tolist()):
        f = featurize_smiles(s)
        if f is None:
            continue
        feat_rows.append(f)
        ok_idx.append(i)
    dfx = pd.DataFrame(feat_rows).reset_index(drop=True)
    df_ok = df.iloc[ok_idx].reset_index(drop=True)
    return df_ok, dfx


def train_model(X, y, sample_weight=None, seed=42):
    Xtr, Xva, ytr, yva, wtr, wva = train_test_split(
        X, y, sample_weight if sample_weight is not None else np.ones_like(y),
        test_size=0.2, random_state=seed
    )
    model = XGBRegressor(
        n_estimators=800,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(Xtr, ytr, sample_weight=wtr)
    pred = model.predict(Xva)
    mae = mean_absolute_error(yva, pred)
    return model, mae


def apply_rule(df_scored: pd.DataFrame) -> pd.DataFrame:
    """
    df_scored: 必須含 y_pred + descriptors 欄位（MolWt/TPSA/...）
    回傳 df_scored 增加 feedback_label 與 sample_weight
    """
    # 從預測最好的 topN 中選 K 個符合 rule 的候選作為「送 DFT/實驗」
    topN = df_scored.head(int(RULE["screen_topN"])).copy()

    cond = (
        (topN["TPSA"] >= RULE["tpsa_min"]) &
        (topN["TPSA"] <= RULE["tpsa_max"]) &
        (topN["MolLogP"] <= RULE["logp_max"]) &
        (topN["NumHDonors"] <= RULE["donor_max"])
    )
    chosen = topN[cond].head(int(RULE["select_K"]))["id"].tolist()
    chosen = set(chosen)

    df_scored["feedback_label"] = df_scored["id"].apply(lambda x: 1 if x in chosen else 0)
    df_scored["sample_weight"] = df_scored["feedback_label"].apply(lambda v: WEIGHT_POS if v == 1 else WEIGHT_NEG)

    return df_scored


def family_ranking(df_scored: pd.DataFrame) -> pd.DataFrame:
    # family key：用 elements signature（此處用 smiles 的字母序簡化版；如需更準可改）
    df_scored = df_scored.copy()
    df_scored["family"] = df_scored["smiles"].astype(str).str.replace(r"[^A-Za-z]", "", regex=True)

    fam = (
        df_scored.groupby("family", as_index=False)
        .agg(
            best_y_pred=("y_pred", "min"),
            best_id=("id", "first"),
            best_smiles=("smiles", "first"),
            count=("id", "count"),
            feedback_hits=("feedback_label", "sum"),
        )
        .sort_values("best_y_pred", ascending=True)
        .reset_index(drop=True)
    )
    return fam


def main():
    if not DATA.exists():
        raise SystemExit(f"Missing {DATA.resolve()}")

    df = pd.read_csv(DATA)
    df_ok, dfx = build_features(df)

    # ===== baseline 訓練（沒有 feedback）=====
    y = df_ok["label_y"].astype(float).values
    X = dfx.values
    base_model, base_mae = train_model(X, y, sample_weight=None)
    print(f"[BASE] valid MAE = {base_mae:.6f}")

    # baseline 預測與排序（越低越好）
    base_pred = base_model.predict(X)
    base_scored = df_ok.copy()
    base_scored["y_pred"] = base_pred
    for c in dfx.columns:
        base_scored[c] = dfx[c].values
    base_scored = base_scored.sort_values("y_pred", ascending=True).reset_index(drop=True)

    base_top20 = base_scored.head(20)[["id","smiles","label_y","y_pred"]].copy()
    base_top20.to_csv(OUT / "top20_before_feedback.csv", index=False)

    # ===== 套用 rule，產生 feedback_label 與 sample_weight =====
    scored_w_fb = apply_rule(base_scored)
    scored_w_fb.to_csv(OUT / "scored_with_feedback.csv", index=False)
    print("[RULE] feedback positives:", int(scored_w_fb["feedback_label"].sum()))

    # ===== retrain：用 sample_weight 把 rule 選中的候選拉近模型偏好 =====
    w = scored_w_fb["sample_weight"].astype(float).values
    rt_model, rt_mae = train_model(X, y, sample_weight=w)
    print(f"[RETRAIN] valid MAE = {rt_mae:.6f}")

    rt_pred = rt_model.predict(X)
    rt_scored = df_ok.copy()
    rt_scored["y_pred"] = rt_pred
    for c in dfx.columns:
        rt_scored[c] = dfx[c].values
    rt_scored = rt_scored.sort_values("y_pred", ascending=True).reset_index(drop=True)

    rt_top20 = rt_scored.head(20)[["id","smiles","label_y","y_pred"]].copy()
    rt_top20.to_csv(OUT / "top20_after_feedback.csv", index=False)
    
    # ===== Rule-based Re-rank (Policy Layer) =====
    # 目標：不改變模型，只用 rule/policy 對候選做決策偏好調整（符合 MU/Agentic 的分層設計）
    # 說明：y_pred 越低越好；若 feedback_label==1（符合 rule、建議送 DFT/實驗），給一個負的 bonus 讓它往前排
    POLICY_ALPHA = 1000.0  # policy 權重（越大，rule 影響越強；建議先用 200~800 試）

    # 注意：rt_scored 目前沒有 feedback_label，需要沿用 rule 選中的 id 清單
    chosen_ids = set(scored_w_fb[scored_w_fb["feedback_label"] == 1]["id"].tolist())
    rt_scored["feedback_label"] = rt_scored["id"].apply(lambda x: 1 if x in chosen_ids else 0)

    # policy bonus：符合 rule 的候選往前（更小更好）
    rt_scored["policy_bonus"] = rt_scored["feedback_label"].apply(
        lambda v: -POLICY_ALPHA if v == 1 else 0.0
    )

    # 最終決策分數：模型預測 + policy bonus
    rt_scored["final_score"] = rt_scored["y_pred"] + rt_scored["policy_bonus"]

    # 依 final_score 排名（越低越好）
    reranked = rt_scored.sort_values("final_score", ascending=True).reset_index(drop=True)

    # 輸出：可直接拿去做簡報對照
    reranked.head(20)[
        ["id", "smiles", "label_y", "y_pred", "feedback_label", "policy_bonus", "final_score"]
    ].to_csv(OUT / "top20_rule_reranked.csv", index=False)

    # 額外輸出：比較 before/after 的差異（哪些被 policy 拉進 top20）
    before_ids = set(rt_top20["id"].tolist())
    after_ids = set(reranked.head(20)["id"].tolist())
    added_by_policy = sorted(list(after_ids - before_ids))
    removed_by_policy = sorted(list(before_ids - after_ids))

    pd.DataFrame([{
        "policy_alpha": POLICY_ALPHA,
        "top20_added_by_policy": ";".join(added_by_policy[:100]),
        "top20_removed_by_policy": ";".join(removed_by_policy[:100]),
        "added_count": len(added_by_policy),
        "removed_count": len(removed_by_policy),
    }]).to_csv(OUT / "policy_rerank_summary.csv", index=False)


    # ===== family ranking before/after =====
    # before：把 feedback_label 也加上，方便看 family 中被選中的數量
    base_scored_fb = scored_w_fb.copy()
    fam_before = family_ranking(base_scored_fb)
    fam_before.to_csv(OUT / "family_before_feedback.csv", index=False)

    # after：用 retrain 後的排序，但 feedback 標記沿用 rule 選中的 id（示範 workflow）
    rt_scored["feedback_label"] = rt_scored["id"].apply(lambda x: 1 if x in set(scored_w_fb[scored_w_fb["feedback_label"]==1]["id"]) else 0)
    fam_after = family_ranking(rt_scored)
    fam_after.to_csv(OUT / "family_after_feedback.csv", index=False)

    # ===== 差異摘要 =====
    # 看 top20 變化：新增/移除項目
    before_ids = set(base_top20["id"])
    after_ids = set(rt_top20["id"])
    added = sorted(list(after_ids - before_ids))
    removed = sorted(list(before_ids - after_ids))

    summary = {
        "base_valid_mae": base_mae,
        "retrain_valid_mae": rt_mae,
        "feedback_positive_count": int(scored_w_fb["feedback_label"].sum()),
        "top20_added_ids": ";".join(added[:50]),
        "top20_removed_ids": ";".join(removed[:50]),
    }
    pd.DataFrame([summary]).to_csv(OUT / "feedback_retrain_summary.csv", index=False)

    # 存模型
    joblib.dump(rt_model, OUT / "model_retrained.joblib")
    print("Saved outputs to:", OUT.resolve())
    print(" - top20_before_feedback.csv / top20_after_feedback.csv")
    print(" - scored_with_feedback.csv")
    print(" - family_before_feedback.csv / family_after_feedback.csv")
    print(" - feedback_retrain_summary.csv")
    print(" - model_retrained.joblib")

if __name__ == "__main__":
    main()
