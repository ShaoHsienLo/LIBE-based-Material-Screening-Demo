from pathlib import Path
import pandas as pd
import numpy as np
import shap
import joblib
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Descriptors

OUT = Path("data/outputs")
DATA = Path("data/processed/libe_like.csv")

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

def featurize(smiles: str) -> list[float] | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    vals = []
    for k in DESCRIPTOR_FNS.keys():
        vals.append(float(DESCRIPTOR_FNS[k](mol)))
    return vals

def main():
    bundle = joblib.load(OUT / "model.joblib")
    model = bundle["model"]
    feature_names = bundle["feature_names"]

    df = pd.read_csv(DATA)
    rows = []
    keep = []
    for i, s in enumerate(df["smiles"].astype(str).tolist()):
        v = featurize(s)
        if v is None:
            continue
        rows.append(v)
        keep.append(i)
    X = np.array(rows)
    df_ok = df.iloc[keep].reset_index(drop=True)

    # 建 explainer（TreeExplainer 對 XGBoost 很合適）
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # Global summary plot
    plt.figure()
    shap.summary_plot(shap_values, features=X, feature_names=feature_names, show=False)
    gpath = OUT / "shap_summary.png"
    plt.tight_layout()
    plt.savefig(gpath, dpi=200)
    plt.close()
    print("Saved:", gpath.resolve())

    # Local explanation for Top 5 predicted
    y_pred = model.predict(X)
    df_ok["y_pred"] = y_pred
    df_ok = df_ok.sort_values("y_pred", ascending=False).reset_index(drop=True)
    top5 = df_ok.head(5).copy()

    # For each top item, export top contributing features
    explain_rows = []
    for rank, row in top5.iterrows():
        idx = rank  # because top5 from sorted df_ok; but shap_values aligned with df_ok rows before sorting
        # We need original index mapping:
        # find matching by id+smiles in df_ok_unsorted:
        # easiest: recompute mapping from sorted indices
    # Build mapping properly
    order = np.argsort(-y_pred)  # indices in df_ok (unsorted) ordered by desc
    for k in range(5):
        orig_i = int(order[k])
        sv = shap_values[orig_i]
        contrib = sorted(zip(feature_names, sv), key=lambda x: abs(x[1]), reverse=True)[:5]
        explain_rows.append({
            "rank": k+1,
            "id": df_ok.loc[k, "id"],
            "smiles": df_ok.loc[k, "smiles"],
            "y_pred": float(df_ok.loc[k, "y_pred"]),
            "top_features": "; ".join([f"{n}:{v:+.4f}" for n, v in contrib]),
        })

    exp_df = pd.DataFrame(explain_rows)
    exp_path = OUT / "top5_explanations.csv"
    exp_df.to_csv(exp_path, index=False)
    print("Saved:", exp_path.resolve())

if __name__ == "__main__":
    main()
