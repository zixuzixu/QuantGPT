"""为 LaTeX 报告生成全套图表与统计数据.

输出: research/overnight/fig/*.png + report_data.json
用法: .venv/bin/python research/overnight/make_report_assets.py
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
from backtest_overnight import ANN, COST_BPS, UNIVERSE, decompose, fetch, net_returns

OUT = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(OUT, "fig")
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 150, "axes.grid": True, "grid.alpha": 0.3,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10, "legend.frameon": False,
})
C = {"overnight": "#1a7f37", "intraday": "#cf222e", "cc": "#57606a"}
LBL = {"overnight": "Overnight (close→next open)",
       "intraday": "Intraday (open→close)",
       "cc": "Buy & Hold"}


def ext_stats(r: pd.Series) -> dict:
    """扩展绩效指标: 含 Sortino / Calmar."""
    n, mean, std = len(r), r.mean(), r.std()
    downside = r[r < 0].std()
    equity = (1 + r).cumprod()
    years = n / ANN
    cagr = equity.iloc[-1] ** (1 / years) - 1
    mdd = (equity / equity.cummax() - 1).min()
    return {
        "n_days": int(n),
        "mean_bps": round(mean * 1e4, 2),
        "t_stat": round(mean / std * np.sqrt(n), 2),
        "sharpe": round(mean / std * np.sqrt(ANN), 2),
        "sortino": round(mean / downside * np.sqrt(ANN), 2) if downside > 0 else None,
        "cagr_pct": round(cagr * 100, 2),
        "calmar": round(cagr / abs(mdd), 2) if mdd < 0 else None,
        "total_x": round(float(equity.iloc[-1]), 2),
        "max_dd_pct": round(mdd * 100, 1),
        "win_rate_pct": round((r > 0).mean() * 100, 1),
        "skew": round(float(r.skew()), 2),
        "kurt": round(float(r.kurt()), 1),
    }


# ---------- MU 数据 ----------
df = fetch("MU")
rets = decompose(df)
data = {"start": str(rets.index[0].date()), "end": str(rets.index[-1].date()),
        "n_days": len(rets)}

# 表1: 三条腿 + 成本档
rows = {}
for leg in ["overnight", "intraday", "cc"]:
    for c in (COST_BPS if leg != "cc" else [0]):
        r = net_returns(rets[leg], c) if leg != "cc" else rets[leg]
        rows[f"{leg}_{c}"] = ext_stats(r)
data["strategies"] = rows

# 图1: 累计净值 (log)
fig, ax = plt.subplots(figsize=(9, 4.5))
for leg in ["overnight", "intraday", "cc"]:
    (1 + rets[leg]).cumprod().plot(ax=ax, label=LBL[leg], color=C[leg], lw=1.2, logy=True)
ax.set_title("MU: Cumulative Growth of $1 (no costs, log scale)")
ax.set_ylabel("Growth of $1"); ax.set_xlabel(""); ax.legend()
fig.tight_layout(); fig.savefig(f"{FIG}/fig1_equity.png"); plt.close(fig)

# 图2: 回撤曲线
fig, ax = plt.subplots(figsize=(9, 3.5))
for leg, c, label in [("overnight", 0, "Overnight 0 bps"),
                      ("overnight", 2, "Overnight 2 bps/side"),
                      ("cc", 0, "Buy & Hold")]:
    r = net_returns(rets[leg], c) if leg != "cc" else rets[leg]
    eq = (1 + r).cumprod()
    dd = eq / eq.cummax() - 1
    dd.plot(ax=ax, label=label, lw=1)
ax.set_title("MU: Drawdown"); ax.set_ylabel("Drawdown"); ax.set_xlabel(""); ax.legend(loc="lower left")
fig.tight_layout(); fig.savefig(f"{FIG}/fig2_drawdown.png"); plt.close(fig)

# 图3: 分年度柱状图 (since 2000)
yearly = rets.groupby(rets.index.year).apply(
    lambda g: pd.Series({k: np.expm1(np.log1p(g[k]).sum()) * 100
                         for k in ["overnight", "intraday", "cc"]}))
y = yearly[yearly.index >= 2000]
fig, ax = plt.subplots(figsize=(10, 4))
x = np.arange(len(y))
for i, leg in enumerate(["overnight", "intraday", "cc"]):
    ax.bar(x + (i - 1) * 0.28, y[leg], 0.28, label=LBL[leg], color=C[leg])
ax.set_xticks(x); ax.set_xticklabels(y.index, rotation=60)
ax.axhline(0, color="k", lw=0.8)
ax.set_title("MU: Annual Returns by Leg (%)"); ax.set_ylabel("%"); ax.legend()
fig.tight_layout(); fig.savefig(f"{FIG}/fig3_yearly.png"); plt.close(fig)
data["yearly"] = {int(k): {c: round(v, 1) for c, v in row.items()} for k, row in y.iterrows()}

# 图4: 滚动 252 日 Sharpe + 夜间-日内收益差
fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
roll_sharpe = (rets["overnight"].rolling(ANN).mean() /
               rets["overnight"].rolling(ANN).std() * np.sqrt(ANN))
roll_sharpe.plot(ax=axes[0], color=C["overnight"], lw=1)
axes[0].axhline(0, color="k", lw=0.8)
axes[0].set_title("MU: Rolling 1y Sharpe of Overnight Leg")
spread = (np.log1p(rets["overnight"]).rolling(ANN).sum() -
          np.log1p(rets["intraday"]).rolling(ANN).sum()) * 100
spread.plot(ax=axes[1], color="#0969da", lw=1)
axes[1].axhline(0, color="k", lw=0.8)
axes[1].set_title("MU: Rolling 1y Overnight − Intraday Spread (log %, cumulative)")
fig.tight_layout(); fig.savefig(f"{FIG}/fig4_rolling.png"); plt.close(fig)
data["rolling_sharpe_neg_pct"] = round(float((roll_sharpe.dropna() < 0).mean() * 100), 1)

# 图5: 月度收益热力图 (夜间策略, 2bps, since 2008)
r2 = net_returns(rets["overnight"], 2)
r2 = r2[r2.index >= "2008-01-01"]
monthly = np.expm1(np.log1p(r2).groupby([r2.index.year, r2.index.month]).sum()) * 100
mtab = monthly.unstack()
fig, ax = plt.subplots(figsize=(9, 5.5))
vmax = np.nanmax(np.abs(mtab.values))
im = ax.imshow(mtab.values, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")
ax.set_xticks(range(12)); ax.set_xticklabels(
    ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
ax.set_yticks(range(len(mtab))); ax.set_yticklabels(mtab.index)
for i in range(mtab.shape[0]):
    for j in range(mtab.shape[1]):
        v = mtab.values[i, j]
        if not np.isnan(v):
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=7)
ax.set_title("MU Overnight Strategy: Monthly Returns % (2 bps/side, since 2008)")
ax.grid(False)
fig.tight_layout(); fig.savefig(f"{FIG}/fig5_heatmap.png"); plt.close(fig)

# 图6: 日收益分布
fig, ax = plt.subplots(figsize=(9, 3.8))
bins = np.linspace(-0.08, 0.08, 121)
for leg in ["overnight", "intraday"]:
    ax.hist(rets[leg].clip(-0.08, 0.08), bins=bins, alpha=0.55,
            label=f"{LBL[leg]} (mean {rets[leg].mean()*1e4:.1f} bps)", color=C[leg])
ax.set_title("MU: Daily Return Distribution (clipped at ±8%)")
ax.set_xlabel("Daily return"); ax.legend()
fig.tight_layout(); fig.savefig(f"{FIG}/fig6_dist.png"); plt.close(fig)

# 图7: 成本敏感性
fig, ax = plt.subplots(figsize=(9, 4.2))
for c in COST_BPS:
    (1 + net_returns(rets["overnight"], c)).cumprod().plot(
        ax=ax, label=f"{c} bps/side", lw=1.2, logy=True)
(1 + rets["cc"]).cumprod().plot(ax=ax, label="Buy & Hold", color="k", ls="--", lw=1)
ax.set_title("MU Overnight Strategy: Cost Sensitivity (log scale)")
ax.set_ylabel("Growth of $1"); ax.set_xlabel(""); ax.legend()
fig.tight_layout(); fig.savefig(f"{FIG}/fig7_cost.png"); plt.close(fig)

# 图8: 盈利天数曲线
windows = [5, 10, 21, 42, 63, 126, 252, 504, 756]
fig, ax = plt.subplots(figsize=(9, 4))
dtp_data = {}
for c in COST_BPS:
    logr = np.log1p(net_returns(rets["overnight"], c))
    pct = [float((logr.rolling(n).sum().dropna() > 0).mean() * 100) for n in windows]
    dtp_data[c] = dict(zip([str(w) for w in windows], [round(p, 1) for p in pct]))
    ax.plot(windows, pct, marker="o", label=f"{c} bps/side")
ax.axhline(50, color="k", lw=0.8, ls=":")
ax.axhline(80, color="k", lw=0.8, ls="--")
ax.set_xscale("log"); ax.set_xticks(windows); ax.set_xticklabels(windows)
ax.set_xlabel("Holding window (trading days)"); ax.set_ylabel("% of windows positive")
ax.set_title("MU Overnight Strategy: Probability of Profit vs Holding Period")
ax.legend()
fig.tight_layout(); fig.savefig(f"{FIG}/fig8_days_to_profit.png"); plt.close(fig)
data["days_to_profit"] = dtp_data

# ---------- 横截面 ----------
uni_rows = []
for t in UNIVERSE:
    try:
        rr = decompose(fetch(t))
    except Exception as e:
        print(f"  {t}: failed {e}", file=sys.stderr)
        continue
    rr = rr[rr.index >= "2000-01-01"]
    vol = rr["cc"].std() * np.sqrt(ANN) * 100
    so, si = ext_stats(rr["overnight"]), ext_stats(rr["intraday"])
    uni_rows.append({
        "ticker": t, "since": str(rr.index[0].date()), "ann_vol_pct": round(vol, 1),
        "on_mean_bps": so["mean_bps"], "on_t": so["t_stat"], "on_cagr": so["cagr_pct"],
        "in_mean_bps": si["mean_bps"], "in_t": si["t_stat"], "in_cagr": si["cagr_pct"],
    })
    print(f"  {t} done")
uni = pd.DataFrame(uni_rows).sort_values("on_t", ascending=False)
uni.to_csv(f"{OUT}/universe_report.csv", index=False)
data["universe"] = uni.to_dict("records")

# 图9: 横截面柱状图
fig, ax = plt.subplots(figsize=(10, 4.5))
x = np.arange(len(uni))
ax.bar(x - 0.2, uni["on_cagr"], 0.4, label="Overnight CAGR %", color=C["overnight"])
ax.bar(x + 0.2, uni["in_cagr"], 0.4, label="Intraday CAGR %", color=C["intraday"])
ax.set_xticks(x); ax.set_xticklabels(uni["ticker"], rotation=60)
ax.axhline(0, color="k", lw=0.8)
ax.set_title("Overnight vs Intraday CAGR by Ticker (since 2000, no costs)")
ax.legend()
fig.tight_layout(); fig.savefig(f"{FIG}/fig9_universe.png"); plt.close(fig)

# 图10: 波动率 vs 夜间 t-stat 散点
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(uni["ann_vol_pct"], uni["on_t"], color="#0969da")
for _, row in uni.iterrows():
    ax.annotate(row["ticker"], (row["ann_vol_pct"], row["on_t"]),
                fontsize=8, xytext=(4, 3), textcoords="offset points")
ax.axhline(2, color="k", ls="--", lw=0.8)
ax.text(uni["ann_vol_pct"].max(), 2.1, "t = 2", ha="right", fontsize=8)
ax.set_xlabel("Annualized volatility % (close-to-close, since 2000)")
ax.set_ylabel("Overnight-leg t-statistic")
ax.set_title("Overnight Effect Strength vs Volatility")
fig.tight_layout(); fig.savefig(f"{FIG}/fig10_scatter.png"); plt.close(fig)
corr = float(np.corrcoef(uni["ann_vol_pct"], uni["on_t"])[0, 1])
data["vol_t_corr"] = round(corr, 2)

with open(f"{OUT}/report_data.json", "w") as f:
    json.dump(data, f, indent=1, ensure_ascii=False)
print("\nAll assets written to", FIG)
print(json.dumps({k: v for k, v in data.items() if k != "universe"}, indent=1)[:2000])
