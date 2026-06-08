"""牛熊/涨跌状态下的夜间策略表现 + 全横截面深度回测.

状态定义 (全部用 t-1 收盘可得信息, 无前视):
- bull_spy : SPY Close(t-1) > SPY MA200(t-1)        市场牛熊
- up_own   : Close(t-1) > 自身 MA200(t-1)            个股趋势
- prev_up  : 前一日总收益 cc(t-1) > 0                短期涨跌

输出: fig/fig11~fig14.png + regime_universe.csv + regime_data.json
用法: .venv/bin/python research/overnight/regime_analysis.py
"""

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from backtest_overnight import ANN, UNIVERSE, decompose, fetch

OUT = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(OUT, "fig")

plt.rcParams.update({
    "figure.dpi": 150, "axes.grid": True, "grid.alpha": 0.3,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10, "legend.frameon": False,
})


def bucket_stats(r: pd.Series) -> dict:
    n = len(r)
    if n < 50:
        return {"n": n, "mean_bps": np.nan, "t": np.nan, "sharpe": np.nan}
    mean, std = r.mean(), r.std()
    return {
        "n": int(n),
        "mean_bps": round(mean * 1e4, 2),
        "t": round(mean / std * np.sqrt(n), 2),
        "sharpe": round(mean / std * np.sqrt(ANN), 2),
    }


# ---------- SPY 牛熊状态 ----------
spy = fetch("SPY")
spy_ma = spy["Close"].rolling(200).mean()
bull_spy = (spy["Close"] > spy_ma).shift(1).rename("bull_spy")  # t-1 收盘判定

# ---------- 全横截面: 各状态下的夜间收益 ----------
CONDS = ["bull", "bear", "own_up", "own_down", "prev_up", "prev_down"]
rows, mu_pack = [], None
for t in UNIVERSE:
    try:
        df = fetch(t)
    except Exception as e:
        print(f"  {t}: failed {e}", file=sys.stderr)
        continue
    rets = decompose(df)
    ma = df["Close"].rolling(200).mean()
    flags = pd.DataFrame({
        "bull_spy": bull_spy,
        "up_own": (df["Close"] > ma).shift(1),
        "prev_up": (rets["cc"] > 0).shift(1),
    })
    data = rets.join(flags).dropna()
    data = data[data.index >= "2000-01-01"]
    if t == "MU":
        mu_pack = data  # 留给深度图
    row = {"ticker": t, "n": len(data)}
    masks = {
        "bull": data["bull_spy"].astype(bool),
        "bear": ~data["bull_spy"].astype(bool),
        "own_up": data["up_own"].astype(bool),
        "own_down": ~data["up_own"].astype(bool),
        "prev_up": data["prev_up"].astype(bool),
        "prev_down": ~data["prev_up"].astype(bool),
    }
    for name, m in masks.items():
        s = bucket_stats(data.loc[m, "overnight"])
        row[f"on_{name}_bps"] = s["mean_bps"]
        row[f"on_{name}_t"] = s["t"]
        row[f"on_{name}_n"] = s["n"]
    # 日内腿牛熊对照
    for name in ["bull", "bear"]:
        s = bucket_stats(data.loc[masks[name], "intraday"])
        row[f"in_{name}_bps"] = s["mean_bps"]
    rows.append(row)
    print(f"  {t} done")

uni = pd.DataFrame(rows).set_index("ticker")
uni.to_csv(f"{OUT}/regime_universe.csv")

# ---------- 图11: MU 夜间策略 牛市only vs 熊市only ----------
mu = mu_pack
fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
ax = axes[0]
eq_all = (1 + mu["overnight"]).cumprod()
eq_bull = (1 + mu["overnight"].where(mu["bull_spy"].astype(bool), 0)).cumprod()
eq_bear = (1 + mu["overnight"].where(~mu["bull_spy"].astype(bool), 0)).cumprod()
eq_all.plot(ax=ax, label="Overnight: all days", color="#1a7f37", lw=1.3, logy=True)
eq_bull.plot(ax=ax, label="Overnight: bull only (SPY>MA200)", color="#0969da", lw=1.1, logy=True)
eq_bear.plot(ax=ax, label="Overnight: bear only (SPY<MA200)", color="#cf222e", lw=1.1, logy=True)
ax.set_title("MU Overnight Strategy by Market Regime (since 2000, no costs, log scale)")
ax.set_ylabel("Growth of $1"); ax.legend()

ax = axes[1]
eq_up = (1 + mu["overnight"].where(mu["up_own"].astype(bool), 0)).cumprod()
eq_dn = (1 + mu["overnight"].where(~mu["up_own"].astype(bool), 0)).cumprod()
eq_up.plot(ax=ax, label="Overnight: MU above own MA200", color="#0969da", lw=1.1, logy=True)
eq_dn.plot(ax=ax, label="Overnight: MU below own MA200", color="#cf222e", lw=1.1, logy=True)
ax.set_title("MU Overnight Strategy by Own Trend")
ax.set_ylabel("Growth of $1"); ax.set_xlabel(""); ax.legend()
fig.tight_layout(); fig.savefig(f"{FIG}/fig11_regime_equity.png"); plt.close(fig)

mu_summary = {}
for name, m in {
    "bull": mu["bull_spy"].astype(bool), "bear": ~mu["bull_spy"].astype(bool),
    "own_up": mu["up_own"].astype(bool), "own_down": ~mu["up_own"].astype(bool),
    "prev_up": mu["prev_up"].astype(bool), "prev_down": ~mu["prev_up"].astype(bool),
}.items():
    mu_summary[name] = {
        "overnight": bucket_stats(mu.loc[m, "overnight"]),
        "intraday": bucket_stats(mu.loc[m, "intraday"]),
    }

# ---------- 图12: 横截面 牛熊夜间收益对比 ----------
u = uni.sort_values("on_bull_bps", ascending=False)
fig, ax = plt.subplots(figsize=(10, 4.5))
x = np.arange(len(u))
ax.bar(x - 0.2, u["on_bull_bps"], 0.4, label="Overnight mean, bull (SPY>MA200)", color="#0969da")
ax.bar(x + 0.2, u["on_bear_bps"], 0.4, label="Overnight mean, bear (SPY<MA200)", color="#cf222e")
ax.set_xticks(x); ax.set_xticklabels(u.index, rotation=60)
ax.axhline(0, color="k", lw=0.8)
ax.set_ylabel("Mean daily overnight return (bps)")
ax.set_title("Overnight Leg: Bull vs Bear Market (since 2000, no costs)")
ax.legend()
fig.tight_layout(); fig.savefig(f"{FIG}/fig12_bull_bear.png"); plt.close(fig)

# ---------- 图13: 状态×标的 热力图 (夜间日均 bps) ----------
hm = uni[[f"on_{c}_bps" for c in CONDS]].copy()
hm.columns = ["Bull", "Bear", "OwnUp", "OwnDown", "PrevUp", "PrevDown"]
hm = hm.loc[uni["on_bull_t"].sort_values(ascending=False).index]
fig, ax = plt.subplots(figsize=(7.5, 7))
vmax = np.nanmax(np.abs(hm.values))
im = ax.imshow(hm.values, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")
ax.set_xticks(range(len(hm.columns))); ax.set_xticklabels(hm.columns, rotation=30)
ax.set_yticks(range(len(hm))); ax.set_yticklabels(hm.index, fontsize=8)
for i in range(hm.shape[0]):
    for j in range(hm.shape[1]):
        v = hm.values[i, j]
        if not np.isnan(v):
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=7)
ax.set_title("Overnight Mean Return (bps/day) by Regime")
ax.grid(False)
fig.tight_layout(); fig.savefig(f"{FIG}/fig13_regime_heatmap.png"); plt.close(fig)

# ---------- 图14: MU 条件收益柱状图 (夜间 vs 日内, 六状态) ----------
fig, ax = plt.subplots(figsize=(9, 4))
conds = ["bull", "bear", "own_up", "own_down", "prev_up", "prev_down"]
labels = ["Bull\n(SPY>MA200)", "Bear\n(SPY<MA200)", "Own up\n(>MA200)",
          "Own down\n(<MA200)", "Prev day\nup", "Prev day\ndown"]
on_v = [mu_summary[c]["overnight"]["mean_bps"] for c in conds]
in_v = [mu_summary[c]["intraday"]["mean_bps"] for c in conds]
x = np.arange(len(conds))
ax.bar(x - 0.2, on_v, 0.4, label="Overnight", color="#1a7f37")
ax.bar(x + 0.2, in_v, 0.4, label="Intraday", color="#cf222e")
for i, v in enumerate(on_v):
    ax.text(i - 0.2, v + (1 if v >= 0 else -3), f"{v:.0f}", ha="center", fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
ax.axhline(0, color="k", lw=0.8)
ax.set_ylabel("Mean daily return (bps)")
ax.set_title("MU: Conditional Mean Returns by Regime (since 2000, no costs)")
ax.legend()
fig.tight_layout(); fig.savefig(f"{FIG}/fig14_mu_conditional.png"); plt.close(fig)

# ---------- 汇总 ----------
agg = {c: {
    "mean_on_bps": round(float(uni[f"on_{c}_bps"].mean()), 2),
    "n_positive": int((uni[f"on_{c}_bps"] > 0).sum()),
    "n_sig": int((uni[f"on_{c}_t"] > 2).sum()),
} for c in CONDS}

out = {"mu": mu_summary, "universe_agg": agg,
       "bear_days_pct": round(float((~bull_spy.dropna().astype(bool)).mean() * 100), 1)}
with open(f"{OUT}/regime_data.json", "w") as f:
    json.dump(out, f, indent=1, ensure_ascii=False)

print("\n=== MU 条件统计 (夜间腿) ===")
for c in conds:
    s = mu_summary[c]["overnight"]
    print(f"  {c:10s} n={s['n']:5d} mean={s['mean_bps']:7.2f} bps  t={s['t']:5.2f}  sharpe={s['sharpe']:5.2f}")
print("\n=== 横截面汇总 (22票, 夜间腿日均bps) ===")
print(json.dumps(agg, indent=1))
print("\nSPY 熊市天数占比:", out["bear_days_pct"], "%")
