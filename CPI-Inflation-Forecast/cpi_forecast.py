# Can a simple model forecast South African headline inflation better than
# "it stays where it is"? Rolling-origin backtest on Stats SA's official rates.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HORIZONS = [1, 3, 6, 12]
TRAIN_FROM = "2000-01"   # inflation targeting was adopted in February 2000
TEST_FROM = "2010-01"    # first forecast origin; every origin trains only on data before it

cpi = pd.read_csv("cpi_headline.csv", index_col="month", parse_dates=["month"])
y = cpi["yoy_rate_pct"]
index = cpi["index_dec2024_100"]

# What is known at the end of month t
features = pd.DataFrame({
    "rate": y,
    "change_1m": y - y.shift(1),
    "change_3m": y - y.shift(3),
    "change_12m": y - y.shift(12),
    # Price growth over the last 3 months, annualised: it moves before the yearly rate does
    "momentum_3m": ((index / index.shift(3)) ** 4 - 1) * 100,
})


def diebold_mariano(e_model, e_base, h):
    """Two-sided test that two forecasts have the same mean absolute error.
    Errors h months apart overlap, so the variance uses h - 1 autocovariances."""
    d = np.abs(e_model) - np.abs(e_base)
    n = len(d)
    d_c = d - d.mean()
    var = d_c @ d_c / n
    for k in range(1, h):
        var += 2 * (d_c[k:] @ d_c[:-k]) / n
    t = d.mean() / np.sqrt(var / n)
    return t, 2 * stats.t.sf(abs(t), df=n - 1)


results, kept = [], {}
for h in HORIZONS:
    target = y.shift(-h) - y   # the model predicts the change over the next h months
    data = features.assign(target=target).loc[TRAIN_FROM:].dropna(subset=features.columns)
    origins = data.loc[TEST_FROM:].dropna(subset=["target"]).index
    rows = []
    for t in origins:
        # Only pairs whose outcome was already known at month t
        train = data.loc[: t - pd.DateOffset(months=h)].dropna(subset=["target"])
        model = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
        model.fit(train[features.columns], train["target"])
        x = data.loc[[t], features.columns]
        rows.append({
            "origin": t,
            "actual": y[t + pd.DateOffset(months=h)],
            "Ridge on recent changes": y[t] + model.predict(x)[0],
            "No change (last rate)": y[t],
            "Average of last 12 months": y.loc[t - pd.DateOffset(months=11): t].mean(),
        })
    frame = pd.DataFrame(rows).set_index("origin")
    kept[h] = frame
    errors = {name: frame[name] - frame["actual"] for name in frame.columns if name != "actual"}
    t_stat, p = diebold_mariano(errors["Ridge on recent changes"].to_numpy(),
                                errors["No change (last rate)"].to_numpy(), h)
    results.append({
        "horizon": h,
        "forecasts": len(frame),
        **{name: e.abs().mean() for name, e in errors.items()},
        "p (Ridge vs no change)": p,
    })

table = pd.DataFrame(results).set_index("horizon")
print(f"Official rates {y.index[0]:%b %Y} to {y.index[-1]:%b %Y}; training from {TRAIN_FROM}, "
      f"forecasts made every month from {TEST_FROM}")
print("\nMean absolute error in percentage points of inflation, by months ahead")
print(table.round(3).to_string())

# The last 12 months, where inflation turned up
recent = kept[3].loc["2025-06":]
print("\nThree months ahead, latest forecasts against what happened")
print(recent.round(2).to_string())

# Chart: 3-month-ahead forecasts against the outcome, and error by horizon
h = 3
frame = kept[h]
target_month = frame.index + pd.DateOffset(months=h)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.8), gridspec_kw={"width_ratios": [2, 1]})
ax1.plot(target_month, frame["actual"], color="#222222", linewidth=1.6, label="Official rate")
ax1.plot(target_month, frame["Ridge on recent changes"], color="#2c6e9b", linewidth=1,
         label="Ridge, made 3 months earlier")
ax1.plot(target_month, frame["No change (last rate)"], color="#e67e22", linewidth=1, linestyle="--",
         label="No change, made 3 months earlier")
ax1.set_ylabel("Headline CPI, % year on year")
ax1.set_title("Forecasts made 3 months ahead, 2010 to 2026")
ax1.legend(fontsize=8)

names = ["Ridge on recent changes", "No change (last rate)", "Average of last 12 months"]
colours = ["#2c6e9b", "#e67e22", "#999999"]
width = 0.26
for i, (name, colour) in enumerate(zip(names, colours)):
    ax2.bar(np.arange(len(HORIZONS)) + (i - 1) * width, table[name], width, color=colour, label=name)
ax2.set_xticks(range(len(HORIZONS)))
ax2.set_xticklabels([f"{h} m" for h in HORIZONS])
ax2.set_xlabel("Months ahead")
ax2.set_ylabel("Mean absolute error (points)")
ax2.set_title("Error by horizon")
ax2.legend(fontsize=8)
fig.tight_layout()
fig.savefig("cpi_forecast.png", dpi=120)
print("\nChart saved to cpi_forecast.png")
