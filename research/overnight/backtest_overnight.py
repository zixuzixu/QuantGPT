"""MU 夜间(收盘买→次日开盘卖) vs 日内(开盘买→收盘卖) 策略回测.

复现 "A Tug of War: Overnight versus Intraday Expected Returns" 现象:
1. MU 单票深度回测 (含成本敏感性 + 盈利天数分析)
2. 多股票横截面扫描

用法: .venv/bin/python research/overnight/backtest_overnight.py
输出: research/overnight/ 下的 CSV / PNG / summary.txt
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

UNIVERSE = [
    # 半导体
    "MU", "NVDA", "AMD", "INTC", "TSM", "AVGO", "QCOM", "TXN",
    # 大盘科技
    "AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "NFLX",
    # 非科技
    "JPM", "XOM", "KO", "WMT", "BA",
    # 指数 ETF
    "SPY", "QQQ",
]

COST_BPS = [0, 2, 5, 10]  # 单边成本 (佣金+滑点), 每天 2 笔成交
ANN = 252


def fetch(ticker: str) -> pd.DataFrame:
    """下载日线 OHLC, auto_adjust 处理分红拆股."""
    df = yf.download(ticker, period="max", auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "Close"]].dropna()
    df = df[(df["Open"] > 0) & (df["Close"] > 0)]
    return df


def decompose(df: pd.DataFrame) -> pd.DataFrame:
    """夜间收益 = Open_t/Close_{t-1}-1; 日内收益 = Close_t/Open_t-1."""
    out = pd.DataFrame(index=df.index)
    out["overnight"] = df["Open"] / df["Close"].shift(1) - 1
    out["intraday"] = df["Close"] / df["Open"] - 1
    out["cc"] = df["Close"] / df["Close"].shift(1) - 1  # 买入持有日收益
    return out.dropna()


def net_returns(r: pd.Series, cost_bps: float) -> pd.Series:
    """每天一买一卖, 两笔单边成本."""
    c = cost_bps / 1e4
    return (1 + r) * (1 - c) ** 2 - 1


def stats(r: pd.Series) -> dict:
    n = len(r)
    mean, std = r.mean(), r.std()
    tstat = mean / std * np.sqrt(n) if std > 0 else np.nan
    equity = (1 + r).cumprod()
    years = n / ANN
    cagr = equity.iloc[-1] ** (1 / years) - 1 if years > 0 else np.nan
    mdd = (equity / equity.cummax() - 1).min()
    return {
        "n_days": n,
        "mean_bps": mean * 1e4,
        "t_stat": tstat,
        "sharpe": mean / std * np.sqrt(ANN) if std > 0 else np.nan,
        "cagr_pct": cagr * 100,
        "total_x": equity.iloc[-1],
        "max_dd_pct": mdd * 100,
        "win_rate_pct": (r > 0).mean() * 100,
    }


def days_to_profit(r: pd.Series) -> pd.DataFrame:
    """滚动 N 天累计收益为正的窗口占比."""
    logr = np.log1p(r)
    rows = []
    for n in [5, 10, 21, 42, 63, 126, 252, 504, 756]:
        if n >= len(r):
            break
        roll = logr.rolling(n).sum().dropna()
        rows.append({
            "window_days": n,
            "pct_positive": (roll > 0).mean() * 100,
            "median_ret_pct": (np.expm1(roll.median())) * 100,
            "worst_ret_pct": (np.expm1(roll.min())) * 100,
        })
    return pd.DataFrame(rows)


def run_mu():
    df = fetch("MU")
    rets = decompose(df)
    print(f"MU 数据: {rets.index[0].date()} ~ {rets.index[-1].date()}, {len(rets)} 个交易日")

    # 1) 三条腿 + 成本敏感性
    rows = []
    for leg in ["overnight", "intraday", "cc"]:
        costs = COST_BPS if leg != "cc" else [0]  # 买入持有不计每日成本
        for c in costs:
            r = net_returns(rets[leg], c) if leg != "cc" else rets[leg]
            s = stats(r)
            s["strategy"], s["cost_bps"] = leg, c
            rows.append(s)
    res = pd.DataFrame(rows).set_index(["strategy", "cost_bps"])
    res.to_csv(f"{OUT}/mu_strategies.csv")

    # 2) 分年度: 夜间 vs 日内 年化收益
    yearly = rets.groupby(rets.index.year).apply(
        lambda g: pd.Series({
            "overnight_pct": (np.expm1(np.log1p(g["overnight"]).sum())) * 100,
            "intraday_pct": (np.expm1(np.log1p(g["intraday"]).sum())) * 100,
            "buyhold_pct": (np.expm1(np.log1p(g["cc"]).sum())) * 100,
        })
    )
    yearly.to_csv(f"{OUT}/mu_yearly.csv")

    # 3) 盈利天数分析 (夜间策略, 各成本档)
    dtp = {}
    for c in COST_BPS:
        dtp[c] = days_to_profit(net_returns(rets["overnight"], c))
        dtp[c]["cost_bps"] = c
    dtp_all = pd.concat(dtp.values(), ignore_index=True)
    dtp_all.to_csv(f"{OUT}/mu_days_to_profit.csv", index=False)

    # 4) 图: 累计净值 (log)
    fig, axes = plt.subplots(2, 1, figsize=(11, 10))
    ax = axes[0]
    for leg, label in [("overnight", "Overnight (buy close, sell next open)"),
                       ("intraday", "Intraday (buy open, sell close)"),
                       ("cc", "Buy & Hold")]:
        (1 + rets[leg]).cumprod().plot(ax=ax, label=label, logy=True)
    ax.set_title("MU: Overnight vs Intraday vs Buy&Hold (no costs, log scale)")
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1]
    for c in COST_BPS:
        (1 + net_returns(rets["overnight"], c)).cumprod().plot(
            ax=ax, label=f"Overnight, {c} bps/side", logy=True)
    ax.set_title("MU Overnight strategy: cost sensitivity")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{OUT}/mu_curves.png", dpi=120)
    plt.close(fig)

    return res, yearly, dtp_all


def run_universe():
    rows = []
    for t in UNIVERSE:
        try:
            rets = decompose(fetch(t))
        except Exception as e:  # 单票失败不中断扫描
            print(f"  {t}: 下载失败 {e}", file=sys.stderr)
            continue
        rets = rets[rets.index >= "2000-01-01"]  # 统一起点, 避免超长历史主导
        for leg in ["overnight", "intraday"]:
            s = stats(rets[leg])
            s["ticker"], s["leg"] = t, leg
            s["since"] = str(rets.index[0].date())
            rows.append(s)
        print(f"  {t}: done ({len(rets)} days)")
    df = pd.DataFrame(rows)
    pivot = df.pivot(index="ticker", columns="leg",
                     values=["mean_bps", "t_stat", "sharpe", "cagr_pct"])
    pivot.columns = [f"{a}_{b}" for a, b in pivot.columns]
    pivot["overnight_effect"] = (pivot["t_stat_overnight"] > 2) & (pivot["mean_bps_overnight"] > pivot["mean_bps_intraday"])
    pivot = pivot.sort_values("t_stat_overnight", ascending=False)
    pivot.to_csv(f"{OUT}/universe_scan.csv")

    # 图: 各票夜间 vs 日内 年化收益
    fig, ax = plt.subplots(figsize=(12, 7))
    x = np.arange(len(pivot))
    ax.bar(x - 0.2, pivot["cagr_pct_overnight"], 0.4, label="Overnight CAGR %")
    ax.bar(x + 0.2, pivot["cagr_pct_intraday"], 0.4, label="Intraday CAGR %")
    ax.set_xticks(x); ax.set_xticklabels(pivot.index, rotation=60)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_title("Overnight vs Intraday CAGR by ticker (since 2000, no costs)")
    ax.legend(); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(f"{OUT}/universe_scan.png", dpi=120)
    plt.close(fig)
    return pivot


if __name__ == "__main__":
    print("=== Phase A: MU 单票回测 ===")
    res, yearly, dtp = run_mu()
    print("\n--- MU 策略统计 ---")
    print(res.round(2).to_string())
    print("\n--- MU 盈利天数分析 (夜间策略) ---")
    print(dtp.round(1).to_string(index=False))
    print("\n--- MU 最近 12 年分年度 (%) ---")
    print(yearly.tail(12).round(1).to_string())

    print("\n=== Phase B: 多股票扫描 (since 2000) ===")
    pivot = run_universe()
    print("\n--- 横截面结果 ---")
    print(pivot.round(2).to_string())
