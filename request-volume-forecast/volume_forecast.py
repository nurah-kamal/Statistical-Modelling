# Forecast daily request volume from day of week, month and SA public holidays,
# and check it against naive baselines on a time-based split.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from sa_calendar import is_holiday

TEST_FROM = pd.Timestamp("2025-01-01")

# One row per day, including days with no requests
requests = pd.read_csv("requests.csv", parse_dates=["opened_on"])
days = pd.date_range("2023-01-01", "2025-12-31", freq="D")
daily = requests.groupby("opened_on").size().reindex(days, fill_value=0).rename("requests").to_frame()
daily["day_of_week"] = daily.index.day_of_week
daily["month"] = daily.index.month
daily["holiday"] = [int(is_holiday(d.date())) for d in daily.index]

features = ["day_of_week", "month", "holiday"]
train = daily[daily.index < TEST_FROM]
test = daily[daily.index >= TEST_FROM]
print(f"Training on {len(train)} days ({train.index[0].date()} to {train.index[-1].date()}), "
      f"testing on {len(test)} days ({test.index[0].date()} to {test.index[-1].date()})")

# Models
encoder = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), features)])
models = {
    "Poisson regression": Pipeline([("encode", encoder),
                                    ("model", PoissonRegressor(alpha=1e-4, max_iter=1000))]),
    "Gradient boosting": HistGradientBoostingRegressor(loss="poisson", categorical_features=[0, 1, 2],
                                                      max_iter=200, learning_rate=0.05, random_state=42),
}

# Rolling time-series cross-validation inside the training years
print("\nRolling cross-validation on 2023-2024 (5 folds, each test fold after its training data)")
for name, model in models.items():
    scores = []
    for fit_idx, val_idx in TimeSeriesSplit(n_splits=5).split(train):
        model.fit(train.iloc[fit_idx][features], train.iloc[fit_idx]["requests"])
        scores.append(mean_absolute_error(train.iloc[val_idx]["requests"],
                                          model.predict(train.iloc[val_idx][features])))
    print(f"  {name}: MAE {np.mean(scores):.2f} (folds {', '.join(f'{s:.2f}' for s in scores)})")

forecasts = {}
for name, model in models.items():
    model.fit(train[features], train["requests"])
    forecasts[name] = model.predict(test[features])

# Baselines
forecasts["Same day last week"] = daily["requests"].shift(7).loc[test.index].to_numpy()
seasonal = train.groupby(["day_of_week", "month"])["requests"].mean()
forecasts["Seasonal average (weekday x month)"] = seasonal.loc[
    list(zip(test["day_of_week"], test["month"]))].to_numpy()
forecasts["Training mean"] = np.full(len(test), train["requests"].mean())

# The rate the synthetic data was drawn from (scripts/make-sample.js in sa-turnaround).
# No model can beat this on average; it shows how much of the error is pure chance.
MONTH_FACTOR = [1.35, 1.15, 1.05, 0.95, 1, 1, 0.95, 1, 1, 1, 0.95, 0.7]
working = (test["day_of_week"] < 5) & (test["holiday"] == 0)
closed_factor = np.where(test["day_of_week"] == 6, 0.12, 0.2)
true_rate = 7 * np.array(MONTH_FACTOR)[test["month"] - 1] * np.where(working, 1, closed_factor)
forecasts["True generating rate (not available in practice)"] = true_rate

print("\nMean absolute error on 2025, in requests per day")
results = pd.Series({name: mean_absolute_error(test["requests"], f) for name, f in forecasts.items()})
for name, mae in results.items():
    print(f"  {name}: {mae:.2f}")

holidays_2025 = test["holiday"] == 1
weekday_holidays = holidays_2025 & (test["day_of_week"] < 5)
print(f"\nOn the {weekday_holidays.sum()} weekday public holidays in 2025 (actual mean "
      f"{test.loc[weekday_holidays, 'requests'].mean():.1f} requests)")
for name in ["Poisson regression", "Same day last week", "Seasonal average (weekday x month)"]:
    print(f"  {name}: MAE {mean_absolute_error(test.loc[weekday_holidays, 'requests'], forecasts[name][weekday_holidays]):.2f}")

# Chart: March to May 2025 (Easter, Freedom Day, Workers' Day), and MAE by method
window = (test.index >= "2025-03-01") & (test.index <= "2025-05-31")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.8), gridspec_kw={"width_ratios": [2, 1]})
ax1.bar(test.index[window], test["requests"][window], color="#cccccc", label="Actual")
ax1.plot(test.index[window], forecasts["Poisson regression"][window], color="#2c6e9b", label="Poisson regression")
ax1.plot(test.index[window], forecasts["Same day last week"][window], color="#e67e22", linestyle="--",
         linewidth=1, label="Same day last week")
for d in test.index[window & holidays_2025.to_numpy()]:
    ax1.axvline(d, color="#c0392b", linewidth=0.8, alpha=0.6)
ax1.set_title("Daily requests, March to May 2025 (red lines are public holidays)")
ax1.set_ylabel("Requests opened")
ax1.legend(fontsize=8)

order = results.drop("True generating rate (not available in practice)").sort_values()
ax2.barh(order.index, order.values, color=["#2c6e9b" if "regression" in n or "boosting" in n else "#999999"
                                         for n in order.index])
ax2.axvline(results["True generating rate (not available in practice)"], color="#c0392b", linestyle="--",
            label="Floor: the true rate")
ax2.invert_yaxis()
ax2.set_xlabel("MAE on 2025 (requests per day)")
ax2.set_title("Error by method")
ax2.legend(fontsize=8)
fig.tight_layout()
fig.savefig("volume_forecast.png", dpi=120)
print("\nChart saved to volume_forecast.png")
