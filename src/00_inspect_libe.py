from pathlib import Path
import pandas as pd

RAW_DIR = Path("data/raw")

def try_read_file(p: Path, nrows=5):
    suf = p.suffix.lower()
    try:
        if suf in [".csv"]:
            df = pd.read_csv(p, nrows=nrows)
            return df
        if suf in [".json"]:
            df = pd.read_json(p, lines=False)
            return df.head(nrows)
        if suf in [".jsonl"]:
            df = pd.read_json(p, lines=True)
            return df.head(nrows)
        if suf in [".parquet"]:
            df = pd.read_parquet(p)
            return df.head(nrows)
    except Exception as e:
        return f"[READ_FAIL] {p.name}: {e}"
    return f"[SKIP] {p.name} (unsupported ext {suf})"

def main():
    if not RAW_DIR.exists():
        raise SystemExit(f"RAW_DIR not found: {RAW_DIR.resolve()}")

    files = sorted([p for p in RAW_DIR.rglob("*") if p.is_file()])
    print(f"Found {len(files)} files under {RAW_DIR.resolve()}\n")

    # Print top candidates by extension
    exts = {}
    for p in files:
        exts[p.suffix.lower()] = exts.get(p.suffix.lower(), 0) + 1
    print("Extensions summary:")
    for k, v in sorted(exts.items(), key=lambda x: (-x[1], x[0])):
        print(f"  {k or '[noext]'}: {v}")
    print()

    # Try to read a few tabular-ish files
    candidates = [p for p in files if p.suffix.lower() in [".csv", ".json", ".jsonl", ".parquet"]]
    print(f"Tabular candidates: {len(candidates)}\n")

    for p in candidates[:10]:
        print("="*80)
        print(f"FILE: {p.relative_to(RAW_DIR)}")
        out = try_read_file(p)
        if isinstance(out, pd.DataFrame):
            print("COLUMNS:", list(out.columns)[:50])
            print(out.head(3))
        else:
            print(out)

if __name__ == "__main__":
    main()
