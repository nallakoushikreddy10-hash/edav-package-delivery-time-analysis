import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="whitegrid")
os.makedirs("charts", exist_ok=True)

# 1. Load dataset
CSV_FILE = "Package_Delivery.csv"

if os.path.exists(CSV_FILE):
    df = pd.read_csv(CSV_FILE)
else:
    print(f"'{CSV_FILE}' not found - generating SAMPLE data for demonstration.")
    rng = np.random.default_rng(42)
    n = 3000
    carriers = rng.choice(["FastShip", "QuickPost", "EcoMove", "SkyCargo"], n)
    methods = rng.choice(["Standard", "Express", "Economy"], n, p=[0.5, 0.2, 0.3])
    regions = rng.choice(["North", "South", "East", "West", "Central"], n)
    distance = rng.uniform(20, 2500, n)
    weight = rng.gamma(2, 2, n)
    order_date = pd.to_datetime("2025-01-01") + pd.to_timedelta(rng.integers(0, 365, n), unit="D")
    ship_date = order_date + pd.to_timedelta(rng.integers(0, 4, n), unit="D")
    base = {"Express": 1.5, "Standard": 4, "Economy": 7}
    days = (np.array([base[m] for m in methods]) + distance / 700
            + rng.normal(0, 1.2, n) + (order_date.month.isin([11, 12]) * 1.5))
    days = np.clip(np.round(days), 1, None)
    est_days = np.array([base[m] for m in methods]) + 2
    df = pd.DataFrame({
        "Order_Date": order_date,
        "Ship_Date": ship_date,
        "Delivery_Date": ship_date + pd.to_timedelta(days, unit="D"),
        "Estimated_Delivery_Date": ship_date + pd.to_timedelta(est_days, unit="D"),
        "Region": regions,
        "Carrier": carriers,
        "Shipping_Method": methods,
        "Weight": weight.round(2),
        "Distance_km": distance.round(1),
    })
    # introduce some missing values
    df.loc[rng.choice(n, 90, replace=False), "Delivery_Date"] = pd.NaT
    df.loc[rng.choice(n, 60, replace=False), "Weight"] = np.nan

print("Shape:", df.shape)
print(df.head())

# 2. Normalize column names to match usage below
rename_map = {
    "Order_Date": "order_date",
    "Ship_Date": "ship_date",
    "Delivery_Date": "delivery_date",
    "Estimated_Delivery_Date": "estimated_date",
    "Region": "region",
    "Carrier": "carrier",
    "Shipping_Method": "method",
    "Weight": "weight",
    "Distance_km": "distance",
}
df = df.rename(columns=rename_map)

# 3. Clean data types
for col in ["order_date", "ship_date", "delivery_date", "estimated_date"]:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")
for col in ["weight", "distance"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# 4. Missing data check
missing = df.isna().sum().sort_values(ascending=False)
print("\nMissing values per column:\n", missing)

plt.figure(figsize=(8, 4))
sns.barplot(x=missing.values, y=missing.index, color="steelblue")
plt.title("Missing Values per Column")
plt.xlabel("Count of missing values")
plt.tight_layout()
plt.savefig("charts/01_missing_values.png", dpi=150)
plt.show()

# 5. Feature engineering
df["delivery_days"] = (df["delivery_date"] - df["ship_date"]).dt.days
df["delay_days"] = (df["delivery_date"] - df["estimated_date"]).dt.days
df["is_late"] = df["delay_days"] > 0
df["day_of_week"] = df["order_date"].dt.day_name()
df["month"] = df["order_date"].dt.month
df["is_weekend"] = df["order_date"].dt.dayofweek >= 5

# remove impossible durations and rows with no delivery date
before = len(df)
df = df.dropna(subset=["delivery_days"])
df = df[df["delivery_days"] >= 0]
print(f"\nRemoved {before - len(df)} rows with missing/negative delivery time.")

print("\nDelivery days summary:\n", df["delivery_days"].describe())
print(f"\nLate delivery rate: {df['is_late'].mean():.1%}")

# 6. Distribution of delivery time
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
sns.histplot(df["delivery_days"], bins=20, kde=True, ax=axes[0], color="teal")
axes[0].set_title("Distribution of Delivery Time")
axes[0].set_xlabel("Delivery days")
sns.ecdfplot(df["delivery_days"], ax=axes[1], color="darkorange")
axes[1].set_title("ECDF of Delivery Time")
axes[1].set_xlabel("Delivery days")
plt.tight_layout()
plt.savefig("charts/02_delivery_distribution.png", dpi=150)
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.boxplot(data=df, x="carrier", y="delivery_days", ax=axes[0])
axes[0].set_title("Delivery Time by Carrier")
sns.violinplot(data=df, x="method", y="delivery_days", ax=axes[1])
axes[1].set_title("Delivery Time by Shipping Method")
plt.tight_layout()
plt.savefig("charts/03_carrier_method_comparison.png", dpi=150)
plt.show()

monthly = df.groupby("month")["delivery_days"].mean().reset_index()
plt.figure(figsize=(8, 4))
sns.lineplot(data=monthly, x="month", y="delivery_days", marker="o")
plt.title("Average Delivery Time by Month")
plt.xticks(range(1, 13))
plt.ylabel("Average delivery days")
plt.tight_layout()
plt.savefig("charts/04_monthly_trend.png", dpi=150)
plt.show()

order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
pivot = df.pivot_table(index="day_of_week", columns="month",
                       values="delivery_days", aggfunc="mean").reindex(order)
plt.figure(figsize=(10, 4))
sns.heatmap(pivot, cmap="YlOrRd", annot=True, fmt=".1f")
plt.title("Average Delivery Days: Day of Week vs Month")
plt.tight_layout()
plt.savefig("charts/05_heatmap_day_month.png", dpi=150)
plt.show()

region_avg = df.groupby("region")["delivery_days"].mean().sort_values()
plt.figure(figsize=(7, 4))
sns.barplot(x=region_avg.values, y=region_avg.index, color="seagreen")
plt.title("Average Delivery Time by Region")
plt.xlabel("Average delivery days")
plt.tight_layout()
plt.savefig("charts/06_region_average.png", dpi=150)
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
sns.scatterplot(data=df, x="distance", y="delivery_days", alpha=0.3, ax=axes[0])
axes[0].set_title("Distance vs Delivery Time")
sns.scatterplot(data=df, x="weight", y="delivery_days", alpha=0.3, ax=axes[1], color="purple")
axes[1].set_title("Weight vs Delivery Time")
plt.tight_layout()
plt.savefig("charts/07_scatter_relationships.png", dpi=150)
plt.show()

num_cols = ["delivery_days", "delay_days", "distance", "weight"]
plt.figure(figsize=(6, 5))
sns.heatmap(df[num_cols].corr(), annot=True, cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("charts/08_correlation_heatmap.png", dpi=150)
plt.show()

late_pct = (pd.crosstab(df["carrier"], df["is_late"], normalize="index") * 100)
late_pct.plot(kind="bar", stacked=True, figsize=(8, 4), color=["#4caf50", "#e53935"])
plt.title("Late vs On-time Deliveries by Carrier (%)")
plt.ylabel("Percentage")
plt.legend(["On time", "Late"])
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("charts/09_late_vs_ontime.png", dpi=150)
plt.show()

print("\n===== KEY FINDINGS =====")
print("Average delivery time:", round(df["delivery_days"].mean(), 2), "days")
print("Fastest carrier (avg):", df.groupby("carrier")["delivery_days"].mean().idxmin())
print("Slowest region (avg):", df.groupby("region")["delivery_days"].mean().idxmax())
print("Late delivery rate:", f"{df['is_late'].mean():.1%}")
print("Correlation (distance vs delivery days):",
      round(df["distance"].corr(df["delivery_days"]), 2))
