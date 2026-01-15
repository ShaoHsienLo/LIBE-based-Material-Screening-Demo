from pathlib import Path
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
import joblib

DATA = Path("data/processed/libe_like.csv")
OUT = Path("data/outputs")
OUT.mkdir(parents=True, exist_ok=True)

# 一組簡單、可解釋的 descriptors（你可自行增減）
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

def main():
    if not DATA.exists():
        raise SystemExit(f"Missing {DATA.resolve()}")

    df = pd.read_csv(DATA)
    print("Loaded:", df.shape)

    feat_rows = []
    ok_idx = []
    for i, s in enumerate(df["smiles"].astype(str).tolist()):
        f = featurize_smiles(s)
        if f is None:
            continue
        feat_rows.append(f)
        ok_idx.append(i)

    dfx = pd.DataFrame(feat_rows)
    df_ok = df.iloc[ok_idx].reset_index(drop=True)
    dfx = dfx.reset_index(drop=True)

    y = df_ok["label_y"].astype(float).values
    X = dfx.values
    feature_names = list(dfx.columns)

    # Train/valid split
    Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.2, random_state=42)

    model = XGBRegressor(
        n_estimators=600,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(Xtr, ytr)

    pred = model.predict(Xva)
    mae = mean_absolute_error(yva, pred)
    print(f"Valid MAE: {mae:.6f}")

    # Score all candidates
    score = model.predict(X)
    df_scored = df_ok.copy()
    df_scored["y_pred"] = score

    # 排序方向：假設越大越好；若你的 label_y 是「越小越好」記得改成 ascending=True
    # df_scored = df_scored.sort_values("y_pred", ascending=False).reset_index(drop=True)
    df_scored = df_scored.sort_values("y_pred", ascending=True).reset_index(drop=True)

    topk = df_scored.head(20).copy()
    top_path = OUT / "top20.csv"
    topk.to_csv(top_path, index=False)
    print("Saved:", top_path.resolve())

    # Save model + feature names
    joblib.dump({"model": model, "feature_names": feature_names}, OUT / "model.joblib")
    print("Saved:", (OUT / "model.joblib").resolve())

if __name__ == "__main__":
    main()
