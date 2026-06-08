"""Triple-Gate Overnight Drift (TGOD) — 可执行量化策略.

策略规则全部来自 research/overnight/ 的回测发现:
- backtest_overnight.py:  现象在 MU/AMD/NVDA/... 上 t > 5
- regime_analysis.py:     熊市效应崩塌, 上升趋势 Sharpe 2.14
- crisis_analysis.py:     跳空型熊市无法减亏, 必须空仓

设计规格见同目录 STRATEGY.md.
用法:
    .venv/bin/python research/overnight/strategy.py              # 回测
    .venv/bin/python research/overnight/strategy.py --live       # 出今日信号
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

HERE = os.path.dirname(os.path.abspath(__file__))


# ---------- 策略参数 ----------
@dataclass
class TGODConfig:
    universe: tuple[str, ...] = (
        "MU", "AMD", "AVGO", "GOOGL", "NVDA",
        "TSLA", "TSM", "AAPL", "META", "MSFT", "QQQ",
    )
    benchmark: str = "SPY"
    start: str = "2010-01-01"
    end: Optional[str] = None  # None = 今天

    # Gates
    ma_window: int = 200
    vol_window: int = 20
    vol_cap_pct: float = 0.80         # 跳过 vol > 80 分位的夜
    earnings_blackout_days: int = 1   # 财报前后 N 天不持仓

    # 仓位
    max_per_name: float = 0.10        # 单票 10% NAV
    gross_cap: float = 1.00           # 总仓位 100%

    # 成本
    commission_bps: float = 0.5       # 单边竞价单 ~0.5 bps (IBKR 大票)
    slippage_bps: float = 1.5         # 单边竞价滑点
    # 总成本 = (commission + slippage) × 2 = 4 bps/round-trip

    # 风控
    gap_kill_pct: float = -0.08       # 单夜 -8% 触发
    gap_kill_days: int = 3            # 暂停 3 天
    losing_streak: int = 5            # 5 连败暂停
    streak_pause_days: int = 3


# ---------- 数据 ----------
def fetch_panel(tickers: list[str], start: str, end: Optional[str]) -> dict[str, pd.DataFrame]:
    raw = yf.download(
        tickers, start=start, end=end,
        auto_adjust=True, progress=False, group_by="ticker", threads=True,
    )
    panel = {}
    for t in tickers:
        try:
            df = raw[t].dropna().copy() if len(tickers) > 1 else raw.dropna().copy()
        except KeyError:
            continue
        if df.empty:
            continue
        df["overnight"] = df["Open"] / df["Close"].shift(1) - 1.0
        df["intraday"] = df["Close"] / df["Open"] - 1.0
        df["total"] = df["Close"] / df["Close"].shift(1) - 1.0
        df["ma"] = df["Close"].rolling(200).mean()
        df["vol20"] = df["total"].rolling(20).std()
        df["vol20_pct"] = df["vol20"].rolling(252, min_periods=60).rank(pct=True)
        panel[t] = df
    return panel


def fetch_earnings_dates(tickers: list[str]) -> dict[str, set[pd.Timestamp]]:
    """yfinance 提供有限的历史财报日, 用作粗略 blackout."""
    out: dict[str, set[pd.Timestamp]] = {}
    for t in tickers:
        try:
            cal = yf.Ticker(t).get_earnings_dates(limit=40)
            if cal is None or cal.empty:
                out[t] = set()
                continue
            dates = pd.to_datetime(cal.index.date)
            out[t] = set(dates)
        except Exception:
            out[t] = set()
    return out


# ---------- 信号 ----------
def compute_signals(panel: dict[str, pd.DataFrame],
                    spy: pd.DataFrame,
                    earnings: dict[str, set[pd.Timestamp]],
                    cfg: TGODConfig) -> pd.DataFrame:
    """每日 EOD 输出每只票的 take_position (0/1) — 表示当晚是否持有."""
    spy_bull = (spy["Close"] > spy["ma"]).rename("spy_bull")
    rows = []
    for t, df in panel.items():
        sig = pd.DataFrame(index=df.index)
        sig["ticker"] = t
        sig["overnight"] = df["overnight"]                              # 次日开盘实现的收益
        sig["g1_universe"] = True                                       # 已经在 universe 里
        sig["g2_bull"] = spy_bull.reindex(df.index).fillna(False)
        sig["g3_trend"] = (df["Close"] > df["ma"])
        # G4: 财报 blackout — 财报当日 ± N 天不持仓
        bl_days = cfg.earnings_blackout_days
        e_dates = earnings.get(t, set())
        if e_dates:
            mask = pd.Series(False, index=df.index)
            for d in e_dates:
                d = pd.Timestamp(d).normalize()
                lo = d - pd.Timedelta(days=bl_days)
                hi = d + pd.Timedelta(days=bl_days)
                mask |= (df.index >= lo) & (df.index <= hi)
            sig["g4_no_earnings"] = ~mask
        else:
            sig["g4_no_earnings"] = True
        # G5: vol cap
        sig["g5_vol_ok"] = (df["vol20_pct"] < cfg.vol_cap_pct) | df["vol20_pct"].isna()
        sig["take_position"] = (sig[["g1_universe", "g2_bull", "g3_trend",
                                     "g4_no_earnings", "g5_vol_ok"]].all(axis=1)).astype(int)
        rows.append(sig)
    return pd.concat(rows).reset_index().rename(columns={"index": "date", "Date": "date"})


# ---------- 组合回测 ----------
@dataclass
class BacktestResult:
    equity: pd.Series
    daily_pnl: pd.Series
    positions: pd.DataFrame
    metrics: dict
    trades_per_day: pd.Series


def backtest(signals: pd.DataFrame, cfg: TGODConfig) -> BacktestResult:
    """组合层回测: 每日选取 take_position=1 的票, 等权分配, 受 max_per_name 与 gross_cap 约束."""
    cost_rt = (cfg.commission_bps + cfg.slippage_bps) * 2 / 1e4  # round-trip 总成本 (decimal)
    sig = signals.copy()
    sig["date"] = pd.to_datetime(sig["date"]).dt.normalize()

    # 把 (date, ticker) 的 take_position 信号在 *下一夜* 兑现
    # signals 行索引日期 d 的含义: "d 收盘 EOD 判定, d→d+1 开盘的夜间收益已在 overnight 列里"
    # 所以决策日 d 的 take_position 直接乘以 overnight (overnight 已对齐为 d 开盘相对 d-1 收盘)
    # 注意: overnight 列对应"昨晚的夜间收益", 即 d 行的 overnight = Open[d]/Close[d-1]-1
    # 我们要预测"d→d+1 的夜间收益", 即用 d 行信号 × d+1 行 overnight
    sig = sig.sort_values(["ticker", "date"])
    sig["overnight_next"] = sig.groupby("ticker")["overnight"].shift(-1)
    sig = sig.dropna(subset=["overnight_next"])

    # 每天活跃票数 & 权重
    daily = sig.pivot_table(index="date", columns="ticker",
                            values="take_position", aggfunc="last").fillna(0)
    pnl_mat = sig.pivot_table(index="date", columns="ticker",
                              values="overnight_next", aggfunc="last").fillna(0)

    n_active = daily.sum(axis=1).replace(0, np.nan)
    weights = daily.div(n_active, axis=0).fillna(0)               # 等权
    weights = weights.clip(upper=cfg.max_per_name)
    gross = weights.sum(axis=1).clip(upper=cfg.gross_cap)
    # 若 raw_gross > cap, 按比例缩
    scale = (cfg.gross_cap / weights.sum(axis=1)).where(weights.sum(axis=1) > cfg.gross_cap, 1.0)
    weights = weights.mul(scale, axis=0)

    # 跳空 kill switch
    raw_pnl = (weights * pnl_mat).sum(axis=1)
    # 单夜最差仓位收益 (检测是否触发 gap_kill)
    worst_today = (pnl_mat * (weights > 0)).replace(0, np.nan).min(axis=1).fillna(0)

    kill_until = pd.Timestamp.min
    streak_until = pd.Timestamp.min
    loss_streak = 0
    pnl_records = []
    weight_records = []
    trade_records = []

    for d in raw_pnl.index:
        if d <= max(kill_until, streak_until):
            # 暂停
            pnl_records.append(0.0)
            weight_records.append(weights.loc[d] * 0.0)
            trade_records.append(0)
            continue
        w = weights.loc[d]
        active = (w > 0).sum()
        # 假设每天全部平仓重开: 单日交易额 = 2 × gross (买入+卖出)
        cost = w.sum() * cost_rt
        net = raw_pnl.loc[d] - cost
        pnl_records.append(net)
        weight_records.append(w)
        trade_records.append(int(active))

        # 触发 kill switch?
        if worst_today.loc[d] <= cfg.gap_kill_pct:
            kill_until = d + pd.Timedelta(days=cfg.gap_kill_days)
        # 连败?
        if net < 0:
            loss_streak += 1
        else:
            loss_streak = 0
        if loss_streak >= cfg.losing_streak:
            streak_until = d + pd.Timedelta(days=cfg.streak_pause_days)
            loss_streak = 0

    daily_pnl = pd.Series(pnl_records, index=raw_pnl.index, name="net_pnl")
    equity = (1 + daily_pnl).cumprod()
    positions = pd.DataFrame(weight_records, index=raw_pnl.index)
    trades = pd.Series(trade_records, index=raw_pnl.index, name="active_names")

    ann = 252
    metrics = dict(
        n_days=int(len(daily_pnl)),
        active_days=int((trades > 0).sum()),
        active_pct=float((trades > 0).mean() * 100),
        avg_names_when_active=float(trades.replace(0, np.nan).mean()),
        mean_bps=float(daily_pnl.mean() * 1e4),
        std_bps=float(daily_pnl.std() * 1e4),
        cagr_pct=float(equity.iloc[-1] ** (ann / len(daily_pnl)) - 1) * 100,
        sharpe=float(daily_pnl.mean() / daily_pnl.std() * np.sqrt(ann)),
        sortino=float(daily_pnl.mean() / daily_pnl[daily_pnl < 0].std() * np.sqrt(ann)),
        max_dd_pct=float(((equity / equity.cummax()) - 1).min() * 100),
        total_x=float(equity.iloc[-1]),
        win_rate_pct=float((daily_pnl[daily_pnl != 0] > 0).mean() * 100),
        t_stat=float(daily_pnl.mean() / (daily_pnl.std() / np.sqrt(len(daily_pnl)))),
    )

    return BacktestResult(equity=equity, daily_pnl=daily_pnl,
                          positions=positions, metrics=metrics,
                          trades_per_day=trades)


# ---------- 基准 ----------
def benchmark_buyhold_equal(panel: dict[str, pd.DataFrame],
                            tickers: tuple[str, ...]) -> pd.Series:
    """等权 buy-and-hold 基准."""
    rets = pd.DataFrame({t: panel[t]["total"] for t in tickers if t in panel}).dropna()
    eq = (1 + rets.mean(axis=1)).cumprod()
    return eq


# ---------- 实时信号 ----------
def live_signal(cfg: TGODConfig) -> pd.DataFrame:
    """打印今天 EOD 应该持仓的票."""
    end = pd.Timestamp.today().strftime("%Y-%m-%d")
    start = (pd.Timestamp.today() - pd.Timedelta(days=600)).strftime("%Y-%m-%d")
    panel = fetch_panel(list(cfg.universe) + [cfg.benchmark], start, end)
    spy = panel.pop(cfg.benchmark)
    earnings = fetch_earnings_dates(list(cfg.universe))
    sig = compute_signals(panel, spy, earnings, cfg)
    today = sig["date"].max()
    rows = sig[(sig["date"] == today)].copy()
    return rows[["ticker", "g2_bull", "g3_trend", "g4_no_earnings",
                 "g5_vol_ok", "take_position"]]


# ---------- 主函数 ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true", help="打印今日 EOD 信号")
    ap.add_argument("--start", default="2010-01-01")
    ap.add_argument("--end", default=None)
    args = ap.parse_args()

    cfg = TGODConfig(start=args.start, end=args.end)

    if args.live:
        df = live_signal(cfg)
        print("\n=== TGOD 今日 EOD 信号 ===")
        print(df.to_string(index=False))
        held = df[df["take_position"] == 1]["ticker"].tolist()
        print(f"\n应持仓: {held} (等权)")
        return

    print("拉取数据...")
    panel = fetch_panel(list(cfg.universe) + [cfg.benchmark], cfg.start, cfg.end)
    spy = panel.pop(cfg.benchmark)

    print("拉取财报日...")
    earnings = fetch_earnings_dates(list(cfg.universe))

    print("计算信号...")
    sig = compute_signals(panel, spy, earnings, cfg)

    print("回测策略...")
    res = backtest(sig, cfg)

    bh = benchmark_buyhold_equal(panel, cfg.universe).reindex(res.equity.index).ffill()
    bh_cagr = (bh.iloc[-1] ** (252 / len(bh)) - 1) * 100
    bh_dd = ((bh / bh.cummax()) - 1).min() * 100

    print("\n=== TGOD 策略表现 ===")
    print(f"区间: {res.equity.index[0].date()} → {res.equity.index[-1].date()}")
    print(f"总交易日: {res.metrics['n_days']}")
    print(f"活跃天数: {res.metrics['active_days']} ({res.metrics['active_pct']:.1f}%)")
    print(f"活跃时平均持仓票数: {res.metrics['avg_names_when_active']:.1f}")
    print(f"日均收益: {res.metrics['mean_bps']:.2f} bps  (t = {res.metrics['t_stat']:.2f})")
    print(f"CAGR (含成本): {res.metrics['cagr_pct']:.2f}%")
    print(f"Sharpe: {res.metrics['sharpe']:.2f}  Sortino: {res.metrics['sortino']:.2f}")
    print(f"MaxDD: {res.metrics['max_dd_pct']:.2f}%  胜率: {res.metrics['win_rate_pct']:.1f}%")
    print(f"累计倍数: {res.metrics['total_x']:.2f}×")
    print()
    print(f"基准: 11票等权 buy-and-hold")
    print(f"  CAGR {bh_cagr:.2f}%  MaxDD {bh_dd:.2f}%  累计 {bh.iloc[-1]:.2f}×")

    # 保存结果
    out_dir = HERE
    res.equity.to_csv(os.path.join(out_dir, "tgod_equity.csv"))
    res.daily_pnl.to_csv(os.path.join(out_dir, "tgod_daily_pnl.csv"))
    pd.DataFrame([res.metrics]).to_csv(os.path.join(out_dir, "tgod_metrics.csv"), index=False)

    # 画图
    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 1, figsize=(11, 8), gridspec_kw={"height_ratios": [2, 1]})
        ax1 = axes[0]
        ax1.semilogy(res.equity.index, res.equity, label="TGOD (含成本)", lw=2, color="#1f77b4")
        ax1.semilogy(bh.index, bh, label="11票等权 buy-and-hold", lw=1.5, color="#888", linestyle="--")
        ax1.set_title("TGOD vs Buy-and-Hold (累计净值, 对数轴)")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2 = axes[1]
        dd = (res.equity / res.equity.cummax()) - 1
        bh_dd_series = (bh / bh.cummax()) - 1
        ax2.fill_between(dd.index, dd * 100, 0, color="#d62728", alpha=0.5, label="TGOD")
        ax2.plot(bh_dd_series.index, bh_dd_series * 100, color="#888", lw=1, label="Buy-Hold")
        ax2.set_title("回撤 (%)")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "fig", "fig16_tgod.png"), dpi=130, bbox_inches="tight")
        print(f"\n图已保存: fig/fig16_tgod.png")
    except Exception as e:
        print(f"画图失败 (非关键): {e}")


if __name__ == "__main__":
    main()
