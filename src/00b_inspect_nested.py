import json
from pathlib import Path
from collections import Counter
import itertools

P = Path("data/raw/libe.json")

def walk_keys(obj, prefix=""):
    """yield key paths like thermo.gibbs_free_energy"""
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{prefix}.{k}" if prefix else k
            yield p
            yield from walk_keys(v, p)
    elif isinstance(obj, list):
        # list 不展開太深，避免爆量；只看前幾個元素的結構
        for i, v in enumerate(obj[:3]):
            yield from walk_keys(v, f"{prefix}[]")

def get_by_path(d, path):
    cur = d
    for part in path.split("."):
        if part.endswith("[]"):
            # 跳過 list 標記
            part = part[:-2]
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur

def main():
    if not P.exists():
        raise SystemExit(f"Missing {P.resolve()}")

    with P.open("r", encoding="utf-8") as f:
        data = json.load(f)

    print("Total records:", len(data))

    # 掃 key path 出現次數（聚焦 thermo）
    cnt = Counter()
    for rec in data[:2000]:  # 先掃前2000筆足夠看輪廓
        thermo = rec.get("thermo", {})
        for kp in walk_keys(thermo, "thermo"):
            cnt[kp] += 1

    print("\nTop thermo key paths (by frequency):")
    for kp, n in cnt.most_common(50):
        print(f"{n:5d}  {kp}")

    # 額外：列出第一筆 thermo 的結構（方便你直觀看）
    print("\nSample record thermo (keys only):")
    sample = data[0].get("thermo", {})
    # 只印第一層 keys
    if isinstance(sample, dict):
        for k in sample.keys():
            print(" -", k)
    else:
        print(type(sample))

if __name__ == "__main__":
    main()
