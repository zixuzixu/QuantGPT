"""市场下跌期间: 夜间策略减亏还是扩大亏损?

对 7 次真实下跌事件逐段计算三条腿的累计收益 (MU 与 SPY),
并汇总熊市状态 (SPY<MA200) 下的几何年化收益对比。

用法: .venv/bin/python research/overnight/crisis_analysis.py
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from backtest_overnight import ANN, decompose, fetch

OUT = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(OUT, "fig")

plt.rcParams.update({
    "figure.dpi": 150, "axes.grid": True, "grid.alpha": 0.3,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10, "legend.frameon": False,
})

EPISODES = {
    "Dotcom\n00-02": ("2000-03-10", "2002-10-09"),
    "GFC\n07-09": ("2007-10-09", "2009-03-09"),
    "2011\ncorrection": ("2011-05-02", "2011-10-03"),
    "2015-16\nselloff": ("2015-07-20", "2016-02-11"),
    "2018Q4": ("2018-10-01", "2018-12-24"),
    "COVID\n2020": ("2020-02-19", "2020-03-23"),
    "2022\nbear": ("2022-01-03", "2022-10-12"),
}


def cum(r: pd.Series) -> float:
    return float(np.expm1(np.log1p(r).sum()) * 100)


results = {}
for tkr in ["MU", "SPY"]:
    rets = decompose(fetch(tkr))
    rows = []
    for name, (a, b) in EPISODES.items():
        w = rets.loc[a:b]
        rows.append({
            "episode": name.replace("\n", " "),
            "days": len(w),
            "overnight_pct": round(cum(w["overnight"]), 1),
            "intraday_pct": round(cum(w["intraday"]), 1),
            "buyhold_pct": round(cum(w["cc"]), 1),
            "worst_single_overnight_pct": round(float(w["overnight"].min()) * 100, 1),
        })
    results[tkr] = pd.DataFrame(rows)
    print(f"\n=== {tkr} 下跌事件分解 ===")
    print(results[tkr].to_string(index=False))

# 熊市状态汇总 (几何年化): 夜间 vs 买入持有
spy = fetch("SPY")
bull = (spy["Close"] > spy["Close"].rolling(200).mean()).shift(1)
mu_rets = decompose(fetch("MU")).join(bull.rename("bull")).dropna()
mu_rets = mu_rets[mu_rets.index >= "2000-01-01"]
bear = ~mu_rets["bull"].astype(bool)
print("\n=== MU 熊市状态汇总 (SPY<MA200, since 2000) ===")
for leg in ["overnight", "intraday", "cc"]:
    r = mu_rets.loc[bear, leg]
    geo_ann = (np.expm1(np.log1p(r).mean() * ANN)) * 100
    print(f"  {leg:10s} 算术日均 {r.mean()*1e4:6.2f} bps | 几何年化 {geo_ann:7.1f}% | 日波动 {r.std()*1e4:.0f} bps")

# 图15: 事件柱状图
fig, axes = plt.subplots(2, 1, figsize=(10, 8))
for ax, tkr in zip(axes, ["MU", "SPY"]):
    df = results[tkr]
    x = np.arange(len(df))
    ax.bar(x - 0.27, df["overnight_pct"], 0.27, label="Overnight leg", color="#1a7f37")
    ax.bar(x, df["intraday_pct"], 0.27, label="Intraday leg", color="#cf222e")
    ax.bar(x + 0.27, df["buyhold_pct"], 0.27, label="Buy & Hold", color="#57606a")
    ax.set_xticks(x); ax.set_xticklabels(list(EPISODES.keys()), fontsize=8)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("Cumulative return %")
    ax.set_title(f"{tkr}: Return Decomposition in 7 Market Declines")
    ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(f"{FIG}/fig15_crisis.png"); plt.close(fig)

results["MU"].to_csv(f"{OUT}/crisis_mu.csv", index=False)
results["SPY"].to_csv(f"{OUT}/crisis_spy.csv", index=False)
print("\nfig15_crisis.png written")
