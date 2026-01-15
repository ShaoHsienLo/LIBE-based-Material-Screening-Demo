import json
from pathlib import Path
import pandas as pd
import numpy as np

from rdkit import Chem

RAW_JSON = Path("data/raw/libe.json")
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 你目前用的 target（自由能，eV）
TARGET_PATH = "thermo.shifted_rrho_eV.free_energy"

# ---- 修正開關 ----
FILTER_CHARGE0 = True            # 建議保留：避免 charge 分配造成 SMILES 不穩
NORMALIZE_MODE = "per_atom"      # "per_atom" 或 "per_heavy_atom" 或 "none"

# ---- 建議的資料篩選（B 部分也在這裡，可先不動）----
MIN_ATOMS = 5                    # 避免 H2/LiH 這種極小碎片主導
ALLOWED_ELEMENTS = {"C", "H", "O", "F", "N", "S", "P", "Cl", "Br", "I", "Li"}  # 可依你用途調整
REQUIRE_CARBON = True            # 電解質/有機物 screening 通常要有 C（若你要無機/鹽類可關掉）


def get_by_path(d: dict, path: str):
    cur = d
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def build_smiles_from_graph(elements, bonds):
    """
    elements: list[str] e.g. ["C","O","H"...]
    bonds: list like [[i, j, order], ...]
    """
    if not elements or not bonds:
        return None

    rw = Chem.RWMol()

    for el in elements:
        rw.AddAtom(Chem.Atom(str(el)))

    for b in bonds:
        if not isinstance(b, (list, tuple)) or len(b) < 3:
            continue
        i, j, order = int(b[0]), int(b[1]), b[2]
        if i < 0 or j < 0 or i >= len(elements) or j >= len(elements):
            continue

        bt = Chem.BondType.SINGLE
        try:
            o = float(order)
        except Exception:
            o = 1.0

        if o == 2.0:
            bt = Chem.BondType.DOUBLE
        elif o == 3.0:
            bt = Chem.BondType.TRIPLE
        elif o == 1.5:
            bt = Chem.BondType.AROMATIC

        try:
            rw.AddBond(i, j, bt)
        except Exception:
            continue

    mol = rw.GetMol()
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return None

    try:
        return Chem.MolToSmiles(mol, isomericSmiles=False)
    except Exception:
        return None


def normalize_label(y_free_eV: float, n_atoms: int, n_heavy: int) -> float:
    if NORMALIZE_MODE == "none":
        return float(y_free_eV)
    if NORMALIZE_MODE == "per_atom":
        return float(y_free_eV) / max(int(n_atoms), 1)
    if NORMALIZE_MODE == "per_heavy_atom":
        return float(y_free_eV) / max(int(n_heavy), 1)
    raise ValueError(f"Unknown NORMALIZE_MODE: {NORMALIZE_MODE}")


def main():
    if not RAW_JSON.exists():
        raise SystemExit(f"Missing {RAW_JSON.resolve()}")

    with RAW_JSON.open("r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []
    skipped_smiles = 0
    skipped_filter = 0

    for rec in data:
        mid = rec.get("molecule_id")
        charge = rec.get("charge", None)
        if FILTER_CHARGE0 and (charge is not None) and (int(charge) != 0):
            continue

        elements = rec.get("elements", None)
        bonds = rec.get("bonds", None)
        n_atoms = rec.get("number_atoms", None)

        if elements is None or bonds is None or n_atoms is None:
            skipped_filter += 1
            continue

        # ---- B：篩選（可先留著，MIN_ATOMS=5 已能有效去掉 H2/LiH 類）----
        if int(n_atoms) < int(MIN_ATOMS):
            skipped_filter += 1
            continue

        # 元素白名單
        if any((e not in ALLOWED_ELEMENTS) for e in elements):
            skipped_filter += 1
            continue

        if REQUIRE_CARBON and ("C" not in set(elements)):
            skipped_filter += 1
            continue

        smiles = build_smiles_from_graph(elements, bonds)
        if not smiles:
            skipped_smiles += 1
            continue

        y = get_by_path(rec, TARGET_PATH)
        if y is None or isinstance(y, (dict, list)):
            continue

        try:
            y = float(y)
        except Exception:
            continue

        # heavy atoms：非 H 的原子數
        n_heavy = sum(1 for e in elements if e != "H")

        y_norm = normalize_label(y, int(n_atoms), int(n_heavy))

        rows.append({
            "id": mid,
            "smiles": smiles,
            "label_y": y_norm,      # <-- 關鍵：已正規化
            "charge": int(charge) if charge is not None else np.nan,
            "number_atoms": int(n_atoms),
            "heavy_atoms": int(n_heavy),
            "free_energy_eV": y,    # 保留原值方便你日後對照
        })

    out = pd.DataFrame(rows)
    print("Built rows:", out.shape)
    print("skipped_smiles:", skipped_smiles)
    print("skipped_filter:", skipped_filter)

    out_path = OUT_DIR / "libe_like.csv"
    out.to_csv(out_path, index=False)
    print("Saved:", out_path.resolve())
    print("\nHead:")
    print(out.head(5))


if __name__ == "__main__":
    main()
