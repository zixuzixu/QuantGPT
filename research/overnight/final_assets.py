"""最终评估报告的补充图表: 趋势过滤策略 / 多夜空日价差 / 全策略对比.

补齐之前讨论但未绘制的内容, 并用精确的策略级 (而非条件) 统计.
用法: .venv/bin/python research/overnight/final_assets.py
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
from backtest_overnight import ANN, decompose, fetch

OUT = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(OUT, "fig")

plt.rcParams.update({
    "figure.dpi": 150, "axes.grid": True, "grid.alpha": 0.3,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10, "legend.frameon": False,
})


def stats(r: pd.Series) -> dict:
    r = r.dropna()
    n, mean, std = len(r), r.mean(), r.std()
    eq = (1 + r).cumprod()
    years = n / ANN
    cagr = eq.iloc[-1] ** (1 / years) - 1 if eq.iloc[-1] > 0 else -1.0
    mdd = (eq / eq.cummax() - 1).min()
    dn = r[r < 0].std()
    return {
        "mean_bps": round(mean * 1e4, 2),
        "t": round(mean / std * np.sqrt(n), 2) if std > 0 else np.nan,
        "sharpe": round(mean / std * np.sqrt(ANN), 2) if std > 0 else np.nan,
        "sortino": round(mean / dn * np.sqrt(ANN), 2) if dn > 0 else np.nan,
        "cagr_pct": round(cagr * 100, 1),
        "total_x": round(float(eq.iloc[-1]), 2),
        "mdd_pct": round(mdd * 100, 1),
        "vol_pct": round(std * np.sqrt(ANN) * 100, 1),
        "win_pct": round((r > 0).mean() * 100, 1),
    }


# ---------- 数据 ----------
df = fetch("MU")
rets = decompose(df)
ma = df["Close"].rolling(200).mean()
above = (df["Close"] > ma).shift(1).reindex(rets.index)  # t-1 收盘判定, 无前视
rets = rets.assign(above=above)
r2000 = rets[rets.index >= "2000-01-01"].copy()


def net_leg(r, c):
    return (1 + r) * (1 - c / 1e4) ** 2 - 1


def spread_net(on, inn, c):
    # 多夜 + 空日: 每天 4 笔成交, 成本约 4c (近似)
    return on - inn - 4 * c / 1e4


# ---------- 全样本 (1984~) 策略集 ----------
variants = {}
for c in [0, 2, 5]:
    variants[f"夜间 {c}bps"] = net_leg(rets["overnight"], c)
    variants[f"价差(多夜空日) {c}bps"] = spread_net(rets["overnight"], rets["intraday"], c)
# 趋势过滤夜间 (跌破自身MA200则空仓, 收益0)
for c in [0, 2, 5]:
    tf = net_leg(rets["overnight"], c).where(rets["above"].fillna(False), 0.0)
    variants[f"趋势过滤夜间 {c}bps"] = tf
variants["买入持有"] = rets["cc"]
variants["日内 0bps"] = rets["intraday"]

tbl = {k: stats(v) for k, v in variants.items()}
with open(f"{OUT}/final_stats.json", "w") as f:
    json.dump(tbl, f, indent=1, ensure_ascii=False)

print("=== 全策略统计 (MU, 全样本) ===")
hdr = f"{'策略':22s}{'日均bps':>8s}{'t':>6s}{'Sharpe':>8s}{'CAGR%':>8s}{'累计x':>14s}{'MaxDD%':>8s}{'波动%':>7s}"
print(hdr)
for k, s in tbl.items():
    print(f"{k:22s}{s['mean_bps']:8.1f}{s['t']:6.1f}{s['sharpe']:8.2f}{s['cagr_pct']:8.1f}{s['total_x']:14.1f}{s['mdd_pct']:8.0f}{s['vol_pct']:7.0f}")

# ---------- figA: 全策略累计净值 (2bps, log) ----------
fig, ax = plt.subplots(figsize=(10, 5))
plot_set = [("夜间 2bps", "Overnight 2bps", "#1a7f37", "-"),
            ("趋势过滤夜间 2bps", "Trend-filtered overnight 2bps", "#0969da", "-"),
            ("价差(多夜空日) 2bps", "Long-ON/Short-IN spread 2bps", "#8250df", "-"),
            ("买入持有", "Buy & Hold", "#57606a", "--"),
            ("日内 0bps", "Intraday 0bps", "#cf222e", ":")]
for key, lbl, col, ls in plot_set:
    (1 + variants[key]).cumprod().plot(ax=ax, label=lbl, color=col, ls=ls, lw=1.3, logy=True)
ax.set_title("MU: All Strategy Variants — Growth of $1 (log scale)")
ax.set_ylabel("Growth of $1"); ax.set_xlabel(""); ax.legend()
fig.tight_layout(); fig.savefig(f"{FIG}/figA_all_strategies.png"); plt.close(fig)

# ---------- figB: 趋势过滤 vs 纯夜间 (净值+回撤) ----------
fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
for key, lbl, col in [("夜间 2bps", "Overnight 2bps", "#1a7f37"),
                      ("趋势过滤夜间 2bps", "Trend-filtered overnight 2bps", "#0969da")]:
    eq = (1 + variants[key]).cumprod()
    eq.plot(ax=axes[0], label=lbl, color=col, lw=1.2, logy=True)
    (eq / eq.cummax() - 1).plot(ax=axes[1], label=lbl, color=col, lw=1)
axes[0].set_title("Trend Filter Effect: Equity (log)"); axes[0].set_ylabel("Growth of $1"); axes[0].legend()
axes[1].set_title("Drawdown"); axes[1].set_ylabel("Drawdown"); axes[1].set_xlabel(""); axes[1].legend(loc="lower left")
fig.tight_layout(); fig.savefig(f"{FIG}/figB_trend_filter.png"); plt.close(fig)

# ---------- figC: 价差策略 成本敏感性 + 牛熊拆分 ----------
spy = fetch("SPY")
bull = (spy["Close"] > spy["Close"].rolling(200).mean()).shift(1).reindex(r2000.index).fillna(False)
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
ax = axes[0]
for c in [0, 2, 5]:
    (1 + spread_net(rets["overnight"], rets["intraday"], c)).cumprod().plot(
        ax=ax, label=f"{c} bps/side (×4/day)", lw=1.2, logy=True)
(1 + rets["cc"]).cumprod().plot(ax=ax, label="Buy & Hold", color="k", ls="--", lw=1)
ax.set_title("Long-ON/Short-IN Spread: Cost Sensitivity"); ax.set_ylabel("Growth of $1"); ax.set_xlabel(""); ax.legend()
ax = axes[1]
sp = spread_net(r2000["overnight"], r2000["intraday"], 2)
eq_bull = (1 + sp.where(bull.astype(bool), 0)).cumprod()
eq_bear = (1 + sp.where(~bull.astype(bool), 0)).cumprod()
eq_bull.plot(ax=ax, label="Spread in bull only", color="#0969da", lw=1.2, logy=True)
eq_bear.plot(ax=ax, label="Spread in bear only", color="#cf222e", lw=1.2, logy=True)
ax.set_title("Spread Strategy by Regime (2 bps, since 2000)"); ax.set_ylabel("Growth of $1"); ax.set_xlabel(""); ax.legend()
fig.tight_layout(); fig.savefig(f"{FIG}/figC_spread.png"); plt.close(fig)

# 价差牛熊统计
sp_bull, sp_bear = sp[bull.astype(bool)], sp[~bull.astype(bool)]
spread_regime = {"bull": stats(sp_bull), "bear": stats(sp_bear)}
print("\n=== 价差策略 牛熊 (2bps) ===")
print("bull:", spread_regime["bull"]["mean_bps"], "bps  t", spread_regime["bull"]["t"])
print("bear:", spread_regime["bear"]["mean_bps"], "bps  t", spread_regime["bear"]["t"])

# ---------- figD: Sharpe 对比柱状 (0/2/5 bps) ----------
fig, ax = plt.subplots(figsize=(10, 4.5))
groups = ["夜间", "趋势过滤夜间", "价差(多夜空日)"]
glabels = ["Overnight", "Trend-filtered\novernight", "Long-ON/\nShort-IN spread"]
x = np.arange(len(groups))
for i, c in enumerate([0, 2, 5]):
    vals = [tbl[f"{g} {c}bps"]["sharpe"] for g in groups]
    ax.bar(x + (i - 1) * 0.25, vals, 0.25, label=f"{c} bps/side")
ax.axhline(tbl["买入持有"]["sharpe"], color="k", ls="--", lw=1, label=f"Buy&Hold ({tbl['买入持有']['sharpe']})")
ax.set_xticks(x); ax.set_xticklabels(glabels)
ax.set_ylabel("Sharpe ratio"); ax.set_title("MU: Sharpe by Strategy and Cost (full sample)")
ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(f"{FIG}/figD_sharpe.png"); plt.close(fig)

# ---------- figE: 风险-收益散点 ----------
fig, ax = plt.subplots(figsize=(8, 5.5))
pts = {"夜间 0bps": "Overnight 0", "夜间 2bps": "Overnight 2", "夜间 5bps": "Overnight 5",
       "趋势过滤夜间 2bps": "Trend-filt 2", "价差(多夜空日) 0bps": "Spread 0",
       "价差(多夜空日) 2bps": "Spread 2", "价差(多夜空日) 5bps": "Spread 5",
       "买入持有": "Buy&Hold", "日内 0bps": "Intraday 0"}
for key, lbl in pts.items():
    s = tbl[key]
    ax.scatter(s["vol_pct"], s["cagr_pct"], s=60)
    ax.annotate(lbl, (s["vol_pct"], s["cagr_pct"]), fontsize=8, xytext=(5, 3), textcoords="offset points")
ax.axhline(0, color="k", lw=0.8)
ax.set_xlabel("Annualized volatility %"); ax.set_ylabel("CAGR %")
ax.set_title("Risk vs Return Across Strategy Variants (MU, full sample)")
fig.tight_layout(); fig.savefig(f"{FIG}/figE_risk_return.png"); plt.close(fig)

out = {"variants": tbl, "spread_regime": spread_regime}
with open(f"{OUT}/final_stats.json", "w") as f:
    json.dump(out, f, indent=1, ensure_ascii=False)
print("\nfigA-E + final_stats.json written")
