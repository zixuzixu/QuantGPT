"""最近一个月 MU: 夜间策略 vs 买入持有逐日对比.

回答: 只持夜盘会错过上涨吗? 下跌时亏更多吗?
用法: .venv/bin/python research/overnight/recent_month.py
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf

OUT = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(OUT, "fig")

PERIOD = os.environ.get("PERIOD", "2mo")
TAIL = int(os.environ.get("TAIL", "23"))
df = yf.download("MU", period=PERIOD, auto_adjust=True, progress=False)
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)
df = df[["Open", "Close"]].dropna().tail(TAIL)

d = pd.DataFrame(index=df.index)
d["overnight"] = df["Open"] / df["Close"].shift(1) - 1   # 昨收->今开
d["intraday"] = df["Close"] / df["Open"] - 1             # 今开->今收
d["cc"] = df["Close"] / df["Close"].shift(1) - 1         # 买入持有日收益
d = d.dropna()

print(f"MU: {d.index[0].date()} ~ {d.index[-1].date()}, {len(d)} 个交易日\n")
if len(d) <= 25:
    view = d.copy()
    for c in ["overnight", "intraday", "cc"]:
        view[c] = (view[c] * 100).round(2)
    print(view.to_string())

# 累计
on_cum = np.expm1(np.log1p(d["overnight"]).sum()) * 100
in_cum = np.expm1(np.log1p(d["intraday"]).sum()) * 100
bh_cum = np.expm1(np.log1p(d["cc"]).sum()) * 100
print(f"\n累计: 夜间 {on_cum:+.1f}% | 日内 {in_cum:+.1f}% | 买入持有 {bh_cum:+.1f}%")

# 上涨日 vs 下跌日拆解 (按买入持有当日方向)
up = d[d["cc"] > 0]
dn = d[d["cc"] < 0]
print(f"\n上涨日 ({len(up)}天): 夜间累计 {np.expm1(np.log1p(up['overnight']).sum())*100:+.1f}% | "
      f"买入持有 {np.expm1(np.log1p(up['cc']).sum())*100:+.1f}%  "
      f"-> 夜间捕获了上涨日 {up['overnight'].mean()/up['cc'].mean()*100:.0f}% 的日均涨幅")
print(f"下跌日 ({len(dn)}天): 夜间累计 {np.expm1(np.log1p(dn['overnight']).sum())*100:+.1f}% | "
      f"买入持有 {np.expm1(np.log1p(dn['cc']).sum())*100:+.1f}%  "
      f"-> 夜间在下跌日{'减亏' if dn['overnight'].mean()>dn['cc'].mean() else '扩亏'}")

# 错过上涨 / 下跌少亏 统计
big_up = d[d["cc"] > 0.03]
big_dn = d[d["cc"] < -0.03]
print(f"\n大涨日 (买入持有>+3%, {len(big_up)}天): 当日 夜间 vs 买入持有")
for idx, row in big_up.iterrows():
    print(f"  {idx.date()}: 夜间 {row['overnight']*100:+.1f}% vs 买入持有 {row['cc']*100:+.1f}% "
          f"({'错过' if row['overnight']<row['cc'] else '捕获'} {abs(row['overnight']-row['cc'])*100:.1f}pp)")
print(f"大跌日 (买入持有<-3%, {len(big_dn)}天):")
for idx, row in big_dn.iterrows():
    print(f"  {idx.date()}: 夜间 {row['overnight']*100:+.1f}% vs 买入持有 {row['cc']*100:+.1f}% "
          f"({'少亏' if row['overnight']>row['cc'] else '多亏'} {abs(row['overnight']-row['cc'])*100:.1f}pp)")

# 图
fig, axes = plt.subplots(2, 1, figsize=(11, 8))
ax = axes[0]
(1 + d["overnight"]).cumprod().sub(1).mul(100).plot(ax=ax, marker="o", label="Overnight only", color="#1a7f37")
(1 + d["intraday"]).cumprod().sub(1).mul(100).plot(ax=ax, marker="s", label="Intraday only", color="#cf222e")
(1 + d["cc"]).cumprod().sub(1).mul(100).plot(ax=ax, marker="^", label="Buy & Hold", color="#57606a")
ax.axhline(0, color="k", lw=0.8)
ax.set_title(f"MU: cumulative return % ({d.index[0].date()} ~ {d.index[-1].date()}, {len(d)} days)")
ax.set_ylabel("Cumulative %"); ax.legend()

ax = axes[1]
x = np.arange(len(d))
ax.bar(x - 0.27, d["overnight"]*100, 0.27, label="Overnight", color="#1a7f37")
ax.bar(x, d["intraday"]*100, 0.27, label="Intraday", color="#cf222e")
ax.bar(x + 0.27, d["cc"]*100, 0.27, label="Buy&Hold", color="#57606a")
step = max(1, len(d) // 22)
ax.set_xticks(x[::step]); ax.set_xticklabels([i.strftime("%m-%d") for i in d.index[::step]], rotation=60, fontsize=7)
ax.axhline(0, color="k", lw=0.8)
ax.set_title("Daily return by leg"); ax.set_ylabel("%"); ax.legend(fontsize=8)
OUTNAME = os.environ.get("OUTNAME", "fig16_recent_month")
fig.tight_layout(); fig.savefig(f"{FIG}/{OUTNAME}.png"); plt.close(fig)
print(f"\n{OUTNAME}.png written")
