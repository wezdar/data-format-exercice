from __future__ import annotations

import csv
import json
import random
import time
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import List, TypedDict

# ---------- Paths ----------
BASE = Path(__file__).parent.resolve()
CSV_PATH = BASE / "users.csv"
JSON_PATH = BASE / "users.json"
YAML_PATH = BASE / "users.yaml"
XML_PATH = BASE / "users.xml"

# ---------- Pretty-print helpers ----------
def print_title(title: str):
    bar = "=" * len(title)
    print(f"\n{title}\n{bar}")

def print_rows_as_table(rows: list[dict]):
    """Print a list[dict] as a fixed-width table in console (no pandas required)."""
    if not rows:
        print("(no data)")
        return
    # Build column order from union of keys
    cols = list({k for r in rows for k in r.keys()})
    # Column widths
    widths = {c: max(len(c), max(len(str(r.get(c, ""))) for r in rows)) for c in cols}
    # Header
    header = "  ".join(f"{c:{widths[c]}}" for c in cols)
    print(header)
    print("-" * len(header))
    # Rows
    for r in rows:
        print("  ".join(f"{str(r.get(c, '')):{widths[c]}}" for c in cols))

# ---------- Optional libs ----------
try:
    import numpy as np
    NUMPY_OK = True
except Exception:
    NUMPY_OK = False

try:
    import pandas as pd
    PANDAS_OK = True
except Exception:
    PANDAS_OK = False

# Try PyYAML, else fallback to a tiny parser (works for this file structure)
try:
    import yaml  # type: ignore
    YAML_OK = True
except Exception:
    YAML_OK = False

# ---------- Structures (TypedDict, namedtuple, dataclass, Pydantic if available) ----------
from collections import namedtuple

class UserTD(TypedDict):
    id: int
    name: str
    email: str
    signup_date: str
    is_active: bool
    roles: List[str]

UserNT = namedtuple("UserNT", "id name email signup_date is_active roles")

@dataclass
class UserDC:
    id: int
    name: str
    email: str
    signup_date: str
    is_active: bool
    roles: List[str]

try:
    from pydantic import BaseModel, EmailStr, Field
    class UserModel(BaseModel):
        id: int
        name: str
        email: EmailStr
        signup_date: str
        is_active: bool = Field(default=True)
        roles: List[str] = Field(default_factory=list)
    PYDANTIC_OK = True
except Exception:
    class UserModel:  # minimal shim
        def __init__(self, **data): self.__dict__.update(data)
    PYDANTIC_OK = False

# ---------- Decorator ----------
def timeit(fn):
    @wraps(fn)
    def w(*a, **k):
        t0 = time.perf_counter()
        r = fn(*a, **k)
        t1 = time.perf_counter()
        print(f"{fn.__name__} took {(t1 - t0) * 1000:.2f} ms")
        return r
    return w

# ---------- Bench list vs NumPy ----------
@timeit
def scalar_vec_list(s: float, data: list[float]) -> list[float]:
    return [s * x for x in data]

if NUMPY_OK:
    @timeit
    def scalar_vec_numpy(s: float, arr):
        return s * arr

def run_benchmark():
    N = 100_000
    scalar = 3.14159
    py_list = [random.random() for _ in range(N)]
    scalar_vec_list(scalar, py_list)
    if NUMPY_OK:
        np_arr = np.array(py_list, dtype=float)
        scalar_vec_numpy(scalar, np_arr)
    else:
        print("NumPy not available; skipped NumPy benchmark.")

# ---------- Loaders for each format ----------
def load_csv_rows() -> list[dict]:
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def load_json_rows() -> list[dict]:
    with open(JSON_PATH, encoding="utf-8") as f:
        data = json.load(f)
        # Expecting a list[dict]
        return list(data)

def load_yaml_rows() -> list[dict]:
    if YAML_OK:
        with open(YAML_PATH, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return list(data.get("users", []))
    # Fallback: parse our simple YAML structure
    rows: list[dict] = []
    cur: dict | None = None
    with open(YAML_PATH, encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if line.strip() == "users:":
                continue
            if line.strip().startswith("- id:"):
                if cur: rows.append(cur)
                cur = {"roles": []}
                cur["id"] = int(line.split(":")[1].strip())
            elif line.strip().startswith("name:") and cur is not None:
                cur["name"] = line.split(":", 1)[1].strip()
            elif line.strip().startswith("email:") and cur is not None:
                cur["email"] = line.split(":", 1)[1].strip()
            elif line.strip().startswith("signup_date:") and cur is not None:
                cur["signup_date"] = line.split(":", 1)[1].strip()
            elif line.strip().startswith("is_active:") and cur is not None:
                cur["is_active"] = line.split(":", 1)[1].strip().lower() == "true"
            elif line.strip() == "roles:" and cur is not None:
                # next lines are role items
                pass
            elif line.strip().startswith("- ") and line.startswith("      ") and cur is not None:
                # YAML role line like "      - admin"
                cur["roles"].append(line.strip()[2:])
    if cur: rows.append(cur)
    return rows

def load_xml_rows() -> list[dict]:
    from xml.etree import ElementTree as ET
    tree = ET.parse(XML_PATH)
    root = tree.getroot()
    rows: list[dict] = []
    for user_el in root.findall("user"):
        row = {"id": int(user_el.get("id", "0")), "roles": []}
        for tag in ["name", "email", "signup_date", "is_active"]:
            el = user_el.find(tag)
            if el is not None:
                if tag == "is_active":
                    row[tag] = (el.text or "").strip().lower() == "true"
                else:
                    row[tag] = (el.text or "").strip()
        roles_el = user_el.find("roles")
        if roles_el is not None:
            for r in roles_el.findall("role"):
                if r.text:
                    row["roles"].append(r.text.strip())
        rows.append(row)
    return rows

# ---------- Display helpers using pandas if available ----------
def print_with_optional_pandas(rows: list[dict], title: str):
    print_title(title)
    if PANDAS_OK:
        df = pd.DataFrame(rows)
        # Normalize roles (list -> ; joined) for nice printing
        if "roles" in df.columns:
            df["roles"] = df["roles"].apply(lambda x: ";".join(x) if isinstance(x, list) else x)
        print(df.to_string(index=False))
    else:
        print_rows_as_table(rows)

# ---------- Main ----------
def main():
    # 1) Show JSON
    json_rows = load_json_rows()
    print_with_optional_pandas(json_rows, "JSON → rows")

    # 2) Show CSV
    csv_rows = load_csv_rows()
    # Convert roles "a;b;c" -> list[str] for uniformity
    for r in csv_rows:
        r["id"] = int(r["id"])
        r["is_active"] = str(r["is_active"]).strip().lower() in {"true", "1", "yes"}
        r["roles"] = str(r["roles"]).split(";") if r.get("roles") else []
    print_with_optional_pandas(csv_rows, "CSV → rows")

    # 3) Show YAML
    yaml_rows = load_yaml_rows()
    print_with_optional_pandas(yaml_rows, f"YAML → rows (parser={'PyYAML' if YAML_OK else 'fallback'})")

    # 4) Show XML
    xml_rows = load_xml_rows()
    print_with_optional_pandas(xml_rows, "XML → rows")

    # 5) Benchmark
    print_title("Benchmarks")
    run_benchmark()

if __name__ == "__main__":
    main()
