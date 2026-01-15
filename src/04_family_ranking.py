from pathlib import Path
import pandas as pd

IN_TOP = Path("data/outputs/top20.csv")
OUT_DIR = Path("data/outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# family 定義方式（擇一）
FAMILY_MODE = "formula"   # "formula" or "elements_set" or "canonical_smiles"

def elements_set_from_smiles(smiles: str) -> str:
    # 這裡用簡單方式：取字母序元素 token（不做 RDKit 解析，避免額外成本）
    # 若你要更準確，可改用 RDKit 解析原子元素
    import re
    tokens = re.findall(r"\[.*?\]|Br|Cl|Si|Na|Li|Mg|Al|Ca|Fe|Zn|Cu|I|F|P|S|O|N|C|H", smiles)
    # 移除括號形式內容（簡化）
    elems = []
    for t in tokens:
        if t.startswith("["):
            # [LiH] 這種先粗略當 Li/H
            inner = re.sub(r"[\[\]0-9+\-@Hh]", "", t)
            if inner:
                elems.append(inner)
        else:
            elems.append(t)
    elems = sorted(set(elems))
    return "-".join(elems)

def main():
    if not IN_TOP.exists():
        raise SystemExit(f"Missing {IN_TOP.resolve()}")

    df = pd.read_csv(IN_TOP)

    # 產生 family key
    if FAMILY_MODE == "formula":
        # 你現在 top20.csv 沒有 formula 欄位，用「smiles 略簡」作為 pseudo-formula family
        # 實務上推薦：在 01_build_table.py 中把 elements / composition 輸出成欄位，這裡會更準。
        df["family"] = df["smiles"].astype(str).str.replace(r"[^A-Za-z]", "", regex=True)
    elif FAMILY_MODE == "elements_set":
        df["family"] = df["smiles"].astype(str).apply(elements_set_from_smiles)
    elif FAMILY_MODE == "canonical_smiles":
        df["family"] = df["smiles"].astype(str)
    else:
        raise ValueError("Unknown FAMILY_MODE")

    # family-level 彙總：以最好的 y_pred 作為該 family 的代表（你的排序越低越好）
    fam = (
        df.groupby("family", as_index=False)
          .agg(
              best_y_pred=("y_pred", "min"),
              best_id=("id", "first"),
              best_smiles=("smiles", "first"),
              count=("id", "count")
          )
          .sort_values("best_y_pred", ascending=True)
          .reset_index(drop=True)
    )

    out_path = OUT_DIR / "family_ranking.csv"
    fam.to_csv(out_path, index=False)
    print("Saved:", out_path.resolve())
    print(fam.head(15))

if __name__ == "__main__":
    main()
