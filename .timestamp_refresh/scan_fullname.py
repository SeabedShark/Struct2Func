# scan_fullname.py
import pandas as pd, math, json

CSV = "./data/test.csv"   # <- 改成你的
ID_COL = "accession"      # <- 如果没有就用索引

df = pd.read_csv(CSV)
if ID_COL not in df.columns:
    df[ID_COL] = range(len(df))

# 兼容多种 Full Name 列名写法
FULLNAME_CANDIDATES = [
    "Full Name", "Full_Name", "full_name", "fullname", "FullName"
]
full_name_col = None
for c in FULLNAME_CANDIDATES:
    if c in df.columns:
        full_name_col = c
        break
if full_name_col is None:
    # 若不存在相关列，创建一个空列以便统一处理
    full_name_col = "Full Name"
    df[full_name_col] = ""

def is_bad(x):
    if x is None: return True
    # pandas NA/NaN
    try:
        if pd.isna(x): return True
    except Exception:
        pass
    # 列表/字典/bytes 等非常见类型
    if isinstance(x, (list, tuple, dict, set, bytes, bytearray)): return True
    # 剩下不是 str 的一律标记（数字等需要先转成 str）
    return not isinstance(x, str)

bad_mask = df[full_name_col].apply(is_bad)
bad = df.loc[bad_mask, [ID_COL, full_name_col]]
print(f"Total rows: {len(df)}, bad full_name: {len(bad)}")
print(bad.head(50).to_string(index=False))
