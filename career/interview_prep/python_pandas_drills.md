# Python / Pandas Drills: 30 Patterns for DS Interviews

Pandas is the second most common technical screen after SQL. Interviewers test whether you can manipulate data quickly and readably. Performance matters less in junior interviews; correctness and idiomatic code matter more.

Each drill has a scenario, the expected idiomatic solution, and a gotcha note.

## How to use this file

Open a Jupyter notebook or Python REPL. For each drill, write the solution in under 5 minutes without peeking. Then compare. If you struggled, rewrite it twice.

## Dataset assumption

Unless noted, use this small DataFrame for warm-up drills:

```python
import pandas as pd
import numpy as np
df = pd.DataFrame({
    "user_id": [1, 1, 2, 2, 3, 3, 3],
    "date": pd.to_datetime(["2026-01-01","2026-01-05","2026-01-01","2026-02-10","2026-01-02","2026-01-15","2026-02-20"]),
    "amount": [10, 20, 15, 50, 5, 8, 40],
    "channel": ["web","web","mobile","mobile","web","mobile","mobile"],
})
```

---

## Basics (1 to 6)

### 1. Filter, select, sort
Get rows where `amount > 10`, select `user_id` and `amount`, sort descending by amount.

```python
df.loc[df.amount > 10, ["user_id", "amount"]].sort_values("amount", ascending=False)
```

Gotcha: `df[df.amount > 10]` works but `.loc` is clearer when you also select columns.

### 2. Count unique per column
Count distinct users and distinct channels.

```python
df.nunique()
# or
df["user_id"].nunique(), df["channel"].nunique()
```

### 3. Group-by aggregate with named columns
Total amount and number of orders per user.

```python
df.groupby("user_id").agg(total=("amount","sum"), n=("amount","size")).reset_index()
```

Gotcha: `size` counts rows (including NaN); `count` skips NaN. For orders, `size` is usually right.

### 4. Group-by multiple columns
Revenue by user and channel.

```python
df.groupby(["user_id","channel"])["amount"].sum().reset_index()
```

### 5. Top N per group
Top 2 highest amounts per user.

```python
df.sort_values("amount", ascending=False).groupby("user_id").head(2)
```

### 6. Pivot
Pivot so rows are users, columns are channels, values are total amount.

```python
df.pivot_table(index="user_id", columns="channel", values="amount", aggfunc="sum", fill_value=0)
```

Gotcha: `pivot` (without `_table`) fails on duplicate (index, column) pairs. Use `pivot_table` defensively.

## Merging and joining (7 to 11)

### 7. Left join
Given `users(user_id, name)`, attach user names to `df`.

```python
users = pd.DataFrame({"user_id":[1,2,3,4], "name":["Ana","Luis","Maria","Jose"]})
df.merge(users, on="user_id", how="left")
```

### 8. Anti-join (users who never bought)
Users in `users` not in `df`.

```python
users[~users.user_id.isin(df.user_id)]
```

Or the merge-indicator pattern:

```python
users.merge(df, on="user_id", how="left", indicator=True).query("_merge=='left_only'")
```

### 9. Cross join
All combinations of users and months.

```python
months = pd.DataFrame({"month": pd.date_range("2026-01", "2026-03", freq="MS")})
users.merge(months, how="cross")
```

Gotcha: Before pandas 1.2, `how="cross"` did not exist. Use `users.assign(k=1).merge(months.assign(k=1), on="k").drop(columns="k")`.

### 10. Merge on different keys
Left key `customer_id`, right key `user_id`.

```python
left.merge(right, left_on="customer_id", right_on="user_id", how="left")
```

### 11. Validate merge
Ensure the join is one-to-one.

```python
left.merge(right, on="user_id", how="left", validate="one_to_one")
```

`validate` raises if the assumption is wrong. Use in production pipelines.

## Reshape and clean (12 to 17)

### 12. Wide to long
A DataFrame `wide` with columns `user_id, jan, feb, mar`. Convert to long with columns `user_id, month, value`.

```python
wide.melt(id_vars="user_id", var_name="month", value_name="value")
```

### 13. Long to wide
Inverse with `pivot_table` (see drill 6).

### 14. Handle missing values
Fill missing `amount` with group mean by user.

```python
df["amount"] = df["amount"].fillna(df.groupby("user_id")["amount"].transform("mean"))
```

`transform` returns a series aligned to the original index. Key pattern.

### 15. Drop duplicates, keep latest
Keep the most recent row per user.

```python
df.sort_values("date").drop_duplicates("user_id", keep="last")
```

### 16. String manipulation
Split `email` into `local` and `domain` parts.

```python
df[["local","domain"]] = df["email"].str.split("@", expand=True)
```

### 17. Type casting safely
Convert `amount` from string (with `$` and commas) to float.

```python
df["amount"] = (df["amount"].astype(str)
                            .str.replace(r"[\$,]","", regex=True)
                            .astype(float))
```

## Time series (18 to 22)

### 18. Resample to daily
Daily revenue across all users.

```python
df.set_index("date").resample("D")["amount"].sum().fillna(0)
```

### 19. Rolling 7-day sum
7-day trailing revenue.

```python
df.set_index("date")["amount"].resample("D").sum().rolling(7, min_periods=1).sum()
```

Gotcha: `rolling(7)` returns NaN for the first 6 rows unless `min_periods=1`. Confirm with interviewer which behavior is expected.

### 20. Shift and lag
Compare today's amount to yesterday's per user.

```python
df = df.sort_values(["user_id","date"])
df["prev_amount"] = df.groupby("user_id")["amount"].shift(1)
df["diff"] = df["amount"] - df["prev_amount"]
```

### 21. Expanding window
Cumulative amount per user.

```python
df["cumulative"] = df.sort_values("date").groupby("user_id")["amount"].cumsum()
```

### 22. Business days vs calendar days
Days between two dates excluding weekends.

```python
np.busday_count(df["start"].values.astype("datetime64[D]"),
                df["end"].values.astype("datetime64[D]"))
```

## Apply, map, vectorization (23 to 26)

### 23. Replace `apply` with `np.where`
Create a flag column where `amount > 20`.

```python
# slow
df["big"] = df["amount"].apply(lambda x: 1 if x > 20 else 0)
# fast
df["big"] = np.where(df["amount"] > 20, 1, 0)
```

Interviewers listen for "I'd avoid apply when I can use a vectorized operation."

### 24. Conditional with `np.select`
Tier users by amount: small (<10), medium (10 to 30), large (>30).

```python
conds = [df.amount < 10, df.amount.between(10,30), df.amount > 30]
choices = ["small","medium","large"]
df["tier"] = np.select(conds, choices, default="unknown")
```

### 25. Apply a function row-wise (when truly needed)
Compute a non-vectorizable custom score.

```python
def score(row):
    return row["amount"] * (1.1 if row["channel"]=="mobile" else 1.0)
df["score"] = df.apply(score, axis=1)
```

Last resort. Prefer vectorized arithmetic with `np.where`.

### 26. Map categorical
Replace channel codes with human-readable names.

```python
mapping = {"web":"Web","mobile":"App"}
df["channel_display"] = df["channel"].map(mapping).fillna("Unknown")
```

## Performance and memory (27 to 30)

### 27. Dtype audit
Reduce memory footprint.

```python
df.info(memory_usage="deep")
df["channel"] = df["channel"].astype("category")
df["amount"] = pd.to_numeric(df["amount"], downcast="float")
```

Category dtype saves memory for low-cardinality strings. `downcast` moves int64 to int32 where possible.

### 28. Read large CSV in chunks

```python
chunks = []
for chunk in pd.read_csv("big.csv", chunksize=500_000):
    chunk["amount"] = chunk["amount"].astype("float32")
    chunks.append(chunk[chunk.amount > 0])
df = pd.concat(chunks, ignore_index=True)
```

### 29. Parquet vs CSV
Why is `pd.read_parquet` faster than `pd.read_csv`?

Answer: Parquet is columnar, compressed, and types are embedded. CSV is row-wise, uncompressed by default, and types must be inferred. For 1GB+ files, parquet is usually 5 to 10x faster and 3 to 5x smaller.

### 30. Groupby with `observed=True`
When grouping on a categorical column, `observed=False` (default) includes all categories, even empty ones, which can explode memory.

```python
df.groupby("channel", observed=True)["amount"].sum()
```

---

## Common interview anti-patterns

1. Using `for` loops to iterate over DataFrames. `iterrows` is slow and non-idiomatic.
2. Chained indexing `df["a"]["b"] = x`. Triggers SettingWithCopyWarning and does not always update the DataFrame. Use `df.loc[..., "b"] = x`.
3. Forgetting that `groupby` returns a GroupBy object; you must call an aggregation to get a DataFrame.
4. Using `inplace=True` everywhere. Modern pandas guidance favors explicit reassignment.
5. Reaching for `apply` before checking if `np.where`, `np.select`, or vectorized arithmetic would work.
6. Ignoring the default join type. `merge` defaults to inner; if you wanted left, you lose rows silently.

## Speed tips that pay off in interviews

- `df.query("amount > 10")` reads faster than `df[df.amount > 10]` for complex filters.
- Pre-sort before `groupby(...).head()` to avoid re-sorting per group.
- `pd.Categorical` for low-cardinality strings in the index.
- Use `assign` for readable chained transformations instead of many intermediate variables.
- Benchmark with `%timeit` in a notebook; interviewers like when you can quantify a claim.

## Typical interview question format

"Here is a DataFrame of orders. Find the user who had the highest total spend in March 2026, and show their top 3 products by revenue, with the product name."

Solve top-down:

```python
march = df[df.date.between("2026-03-01", "2026-03-31")]
top_user = (march.groupby("user_id")["amount"].sum()
                 .sort_values(ascending=False).head(1).index[0])
top_products = (march[march.user_id == top_user]
                    .groupby("product_id")["amount"].sum()
                    .sort_values(ascending=False).head(3)
                    .reset_index()
                    .merge(products, on="product_id"))
```

Narrate every step. Interviewers hire for communication as much as correctness.
