# Did the second sign-off slow refunds?
# Permutation tests and bootstrap intervals on synthetic service-desk data.
from datetime import date

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sa_calendar import working_days_between

CHANGE = pd.Timestamp("2025-07-01")
N_RESAMPLES = 10_000
rng = np.random.default_rng(42)

# Load the requests and measure each closed one in working days
data = pd.read_csv("requests.csv", parse_dates=["opened_on", "closed_on"])
data = data.dropna(subset=["closed_on"]).copy()
data["working_days"] = [
    working_days_between(o.date(), c.date()) for o, c in zip(data["opened_on"], data["closed_on"])
]
data["late"] = data["working_days"] > data["target_working_days"]
data["quarter"] = data["opened_on"].dt.to_period("Q").astype(str)


def permutation_test(a, b, n=N_RESAMPLES):
    """Two-sided p-value for the difference in means between a and b."""
    observed = b.mean() - a.mean()
    pooled = np.concatenate([a, b])
    diffs = np.empty(n)
    for i in range(n):
        rng.shuffle(pooled)
        diffs[i] = pooled[len(a):].mean() - pooled[:len(a)].mean()
    p = (np.sum(np.abs(diffs) >= abs(observed)) + 1) / (n + 1)
    return observed, p, diffs


def bootstrap_ci(values, n=N_RESAMPLES):
    means = rng.choice(values, size=(n, len(values)), replace=True).mean(axis=1)
    return np.percentile(means, [2.5, 97.5])


def bootstrap_diff_ci(a, b, n=N_RESAMPLES):
    da = rng.choice(a, size=(n, len(a)), replace=True).mean(axis=1)
    db = rng.choice(b, size=(n, len(b)), replace=True).mean(axis=1)
    return np.percentile(db - da, [2.5, 97.5])


def compare(name, before, after, labels=("before", "after")):
    print(f"\n{name}")
    results = {}
    for measure, scale, unit in [("late", 100, "% late"), ("working_days", 1, " working days")]:
        a = before[measure].to_numpy(dtype=float)
        b = after[measure].to_numpy(dtype=float)
        diff, p, diffs = permutation_test(a, b)
        lo_a, hi_a = bootstrap_ci(a) * scale
        lo_b, hi_b = bootstrap_ci(b) * scale
        lo_d, hi_d = bootstrap_diff_ci(a, b) * scale
        print(f"  {measure}: {labels[0]} {a.mean() * scale:.1f}{unit} (95% CI {lo_a:.1f} to {hi_a:.1f}, n={len(a)}), "
              f"{labels[1]} {b.mean() * scale:.1f}{unit} (95% CI {lo_b:.1f} to {hi_b:.1f}, n={len(b)})")
        print(f"    difference {diff * scale:+.1f} (95% CI {lo_d:+.1f} to {hi_d:+.1f}), permutation p = {p:.4f}")
        results[measure] = (diff, p, diffs)
    return results


# 1. Refunds before and after the second sign-off
refunds = data[data["category"] == "Refund"]
refund_results = compare("Refunds, opened before vs from 1 July 2025",
                         refunds[refunds["opened_on"] < CHANGE], refunds[refunds["opened_on"] >= CHANGE])

# 2. Placebo: the other categories across the same date
print("\nOther categories across 1 July 2025 (late rate only)")
for category in ["Account change", "Billing query", "Document request", "Complaint"]:
    rows = data[data["category"] == category]
    a = rows.loc[rows["opened_on"] < CHANGE, "late"].to_numpy(dtype=float)
    b = rows.loc[rows["opened_on"] >= CHANGE, "late"].to_numpy(dtype=float)
    diff, p, _ = permutation_test(a, b)
    print(f"  {category}: {a.mean() * 100:.1f}% -> {b.mean() * 100:.1f}% ({diff * 100:+.1f} points), p = {p:.4f}")

# 3. The complaints bump: Q2-Q3 2025 against every other quarter
complaints = data[data["category"] == "Complaint"]
bump = complaints["quarter"].isin(["2025Q2", "2025Q3"])
complaint_results = compare("Complaints, other quarters vs 2025 Q2-Q3", complaints[~bump], complaints[bump],
                             labels=("other quarters", "2025 Q2-Q3"))

# The Q2-Q3 2025 window was picked after looking at the quarterly table, so the test
# above is too generous. A fairer question: if late requests were spread at random over
# time, how often would the biggest bump in any two neighbouring quarters, in any of the
# five categories, be at least this big? Shuffle the late flags within each category,
# find the biggest bump each time, and compare.
def biggest_bump(late, quarter_codes, n_quarters):
    biggest = -np.inf
    for start in range(n_quarters - 1):
        inside = (quarter_codes == start) | (quarter_codes == start + 1)
        biggest = max(biggest, late[inside].mean() - late[~inside].mean())
    return biggest


quarters = sorted(data["quarter"].unique())
code = {q: i for i, q in enumerate(quarters)}
groups = {c: (g["late"].to_numpy(dtype=float), g["quarter"].map(code).to_numpy())
          for c, g in data[data["category"] != "Refund"].groupby("category")}
observed_bump = complaint_results["late"][0]
largest = sorted(((biggest_bump(*g, len(quarters)), c) for c, g in groups.items()), reverse=True)
print("\nBiggest two-quarter bump in each category except refunds: "
      + ", ".join(f"{c} {b * 100:+.1f}" for b, c in largest))

N_SCAN = 2000
max_complaints, max_any = np.empty(N_SCAN), np.empty(N_SCAN)
for i in range(N_SCAN):
    bumps = {}
    for c, (late, codes) in groups.items():
        bumps[c] = biggest_bump(rng.permutation(late), codes, len(quarters))
    max_complaints[i] = bumps["Complaint"]
    max_any[i] = max(bumps.values())
p_complaints = (np.sum(max_complaints >= observed_bump) + 1) / (N_SCAN + 1)
p_any = (np.sum(max_any >= observed_bump) + 1) / (N_SCAN + 1)
print(f"Scan test, complaints only (any two neighbouring quarters): p = {p_complaints:.3f}")
print(f"Scan test, any of the four non-refund categories: p = {p_any:.3f}")

# A real change should still be there in the next quarter, which played no part in
# spotting it. Compare 2025 Q4 alone with everything before 2025 Q2.
print("\nDoes it last? 2025 Q4 against all quarters before 2025 Q2")
for category in ["Refund", "Complaint"]:
    rows = data[data["category"] == category]
    a = rows.loc[rows["quarter"] < "2025Q2", "late"].to_numpy(dtype=float)
    b = rows.loc[rows["quarter"] == "2025Q4", "late"].to_numpy(dtype=float)
    diff, p, _ = permutation_test(a, b)
    print(f"  {category}: {a.mean() * 100:.1f}% -> {b.mean() * 100:.1f}% ({diff * 100:+.1f} points, n={len(b)}), p = {p:.4f}")

# Because the data is synthetic, the true complaint late rate is known. In
# scripts/make-sample.js a complaint takes 1, 2, 3, 3, 4, 5 or 7 working days, and 4%
# wait another 3 to 8 days, with nothing that changes over time.
effort = np.array([1, 2, 3, 3, 4, 5, 7])
p_late_plain = np.mean(effort > 5)
p_late_waiting = np.mean([(e + d) > 5 for e in effort for d in range(3, 9)])
true_rate = 0.96 * p_late_plain + 0.04 * p_late_waiting
print(f"\nTrue complaint late rate in the generator: {true_rate * 100:.1f}% in every quarter")

# Chart: quarterly refund late rate with bootstrap intervals, and the permutation null
quarterly = refunds.groupby("quarter")["late"]
labels = list(quarterly.groups.keys())
rates = quarterly.mean().to_numpy() * 100
cis = np.array([bootstrap_ci(g.to_numpy(dtype=float), n=2000) * 100 for _, g in quarterly])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))
colours = ["#c0392b" if q >= "2025Q3" else "#2c6e9b" for q in labels]
ax1.errorbar(range(len(labels)), rates, yerr=[rates - cis[:, 0], cis[:, 1] - rates], fmt="none",
             ecolor="#888888", capsize=3)
ax1.scatter(range(len(labels)), rates, c=colours, zorder=3)
ax1.set_xticks(range(len(labels)))
ax1.set_xticklabels(labels, rotation=45, ha="right")
ax1.set_ylabel("Refunds late (%)")
ax1.set_title("Refund late rate by quarter opened, with 95% bootstrap intervals")
ax1.axvline(labels.index("2025Q3") - 0.5, color="#c0392b", linestyle="--", linewidth=1)

diff, p, diffs = refund_results["late"]
cdiff, cp, cdiffs = complaint_results["late"]
ax2.hist(diffs * 100, bins=50, color="#2c6e9b", alpha=0.6, label="Refunds: shuffled labels")
ax2.hist(cdiffs * 100, bins=50, color="#999999", alpha=0.6, label="Complaints: shuffled labels")
ax2.axvline(diff * 100, color="#c0392b", linewidth=2, label=f"Refunds observed ({diff * 100:+.1f})")
ax2.axvline(cdiff * 100, color="#333333", linewidth=2, linestyle="--",
            label=f"Complaints Q2-Q3 2025 observed ({cdiff * 100:+.1f})")
ax2.set_xlabel("Difference in late rate (percentage points)")
ax2.set_ylabel("Shuffles")
ax2.set_title("Permutation test: observed difference against chance")
ax2.legend(fontsize=8)
fig.tight_layout()
fig.savefig("sign_off_test.png", dpi=120)
print("\nChart saved to sign_off_test.png")
