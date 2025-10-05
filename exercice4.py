
import pandas as pd
import numpy as np
from functools import partial


data = {
    "user_id": [10, 11, 12, 13, 14, 14],
    "name": ["lina", "yassine", "noor", "fatma", "omar", "zoe"],
    "age": [np.nan, 27, 31, "twenty", 45, 28],
    "join_date": [
        "15/07/2021",
        "2020-02-30",
        "2024.12.31",
        "2019-03-01",
        "March 5, 2023",
        "2023-06-10"
    ],
    "score": ["100", " 88 ", "eighty", "95.5", "", "105"]
}

df_raw = pd.DataFrame(data)
df_raw.to_csv("users.csv", index=False)
print(" Example CSV file 'users.csv' (v2) created successfully!\n")


series_example = pd.Series([420, 360, 510], index=["X", "Y", "Z"], name="Custom Series V2")
print(" Pandas Series with custom index:")
print(series_example, "\n")


columns = ["user_id", "name", "age", "join_date", "score"]
df = pd.read_csv("users.csv", usecols=columns)
print(" DataFrame loaded from CSV:")
print(df, "\n")


print(" Data Types (dtypes):")
print(df.dtypes, "\n")

print(" Head (first 3 rows):")
print(df.head(3), "\n")

print(" Tail (last 2 rows):")
print(df.tail(2), "\n")

print("Summary statistics (describe):")
print(df.describe(include='all'), "\n")


print(" Rows 1 to 3 (iloc[1:4]):")
print(df.iloc[1:4], "\n")

print("Selected columns ['name', 'age']:")
print(df[["name", "age"]], "\n")


print(" Rows where age > 25:")
print(df[pd.to_numeric(df["age"], errors="coerce") > 25], "\n")

print("Rows where 20 <= age <= 30:")
age_num = pd.to_numeric(df["age"], errors="coerce")
print(df[age_num.between(20, 30)], "\n")


print(" Fully duplicated rows:")
print(df[df.duplicated(keep=False)], "\n")

print(" Number of unique user IDs:", df["user_id"].nunique(), "\n")

print("Drop duplicates based on 'user_id' (keep first):")
df_no_dupes = df.drop_duplicates(subset=["user_id"], keep="first")
print(df_no_dupes, "\n")


df["score"] = pd.to_numeric(df["score"], errors="coerce")
df["join_date"] = pd.to_datetime(df["join_date"], errors="coerce", dayfirst=True)


print("🔧 Data types after safe conversion:")
print(df.dtypes, "\n")


def fill_missing_age(age):
    """Replace missing or invalid ages with a default value of 26."""
    val = pd.to_numeric(age, errors="coerce")
    return 26 if pd.isna(val) else val

df["age"] = df["age"].apply(fill_missing_age)
print(" After filling/normalizing age values (.apply):")
print(df, "\n")


def clean_types(df_in: pd.DataFrame) -> pd.DataFrame:

    df_out = df_in.copy()
    df_out["score"] = pd.to_numeric(df_out["score"], errors="coerce")
    df_out["join_date"] = pd.to_datetime(df_out["join_date"], errors="coerce")
    df_out["age"] = pd.to_numeric(df_out["age"], errors="coerce")
    # Report:
    print(" .pipe(clean_types) — data types:")
    print(df_out.dtypes)
    print("\n .pipe(clean_types) — null counts:")
    print(df_out.isna().sum(), "\n")
    return df_out

df_cleaned = df.pipe(clean_types)
print(" After cleaning pipeline (.pipe): data types")
print(df_cleaned.dtypes, "\n")
print(" Null counts after cleaning:")
print(df_cleaned.isnull().sum(), "\n")


def drop_low_scores(df_in: pd.DataFrame, threshold: float) -> pd.DataFrame:

    out = df_in[df_in["score"] >= threshold].copy()
    print(f" .pipe(drop_low_scores) — threshold = {threshold}, rows kept = {len(out)}")
    return out


df_final = df_cleaned.pipe(partial(drop_low_scores, threshold=90.0))

print("\n Final DataFrame (score >= 90):")
print(df_final, "\n")

print(" Data processing pipeline (v2) completed successfully!")
