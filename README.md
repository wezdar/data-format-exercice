# Data Formats & Python Structures Demo

This folder contains example files and a Python script that:
- Writes **JSON**, **CSV**, **YAML**, **XML** user data (see `users.*`).
- Defines the same `User` via **TypedDict**, **namedtuple**, **dataclass**, and **Pydantic** (if available).
- Implements a `@timeit` decorator for timing.
- Compares scalar–vector multiplication on a Python list vs NumPy array.
- Loads the CSV into a Pandas DataFrame.

## Files
- `users.json`, `users.csv`, `users.yaml`, `users.xml`
- `main.py` (all code in one place)
- `README.md` (this file)

## Run locally
```bash
python3 main.py
```

## Commit & push (run in your local repo)
```bash
git add users.json users.csv users.yaml users.xml main.py README.md
git commit -m "Add data formats demo, timing decorator, NumPy vs list benchmark, and CSV loading"
git branch -M main  # optional if you want 'main'
git remote add origin YOUR_REPO_URL  # skip if already added
git push -u origin main
```
