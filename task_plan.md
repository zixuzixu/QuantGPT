# Task Plan: MU 夜间 vs 日内策略回测

## Goal
验证 overnight-intraday return gap 论文现象：
1. MU 上是否生效（夜间策略 vs 日内策略 vs 买入持有）
2. 持有多少天能大概率盈利（含交易成本）
3. 在哪些股票上生效（横截面扫描 ~20 只）
4. 输出"真正回测需要做什么"清单

## Success Criteria
- [ ] MU 夜间/日内收益拆分：累计净值、Sharpe、t-stat、分年度表现
- [ ] 成本敏感性：0 / 2 / 5 / 10 bps 单边成本下策略是否存活
- [ ] 盈利天数：滚动 N 天窗口正收益概率，找出 ≥80% 胜率的最小 N
- [ ] 多股票扫描：哪些票夜间效应显著（t>2）
- [ ] 中文报告 + 图表

## Phases
| # | Phase | Status |
|---|-------|--------|
| 1 | 环境检查 + 规划文件 | complete |
| 2 | 写回测脚本 research/overnight/backtest_overnight.py | complete |
| 3 | MU 单票回测 + 成本敏感性 + 盈利天数分析 | complete |
| 4 | 多股票横截面扫描 (22 票) | complete |
| 5 | 图表 + 中文报告 | complete |
| 6 | 牛熊状态分析 (regime_analysis.py) | complete |
| 7 | 下跌期间减亏/扩亏 (crisis_analysis.py) | complete |
| 8 | 近端样本外 1月/6月 (recent_month.py) | complete |
| 9 | 全策略评估+对冲构想 (final_assets.py) + 总报告 final_report.pdf 19页 | complete |

## Decisions
- 数据源：yfinance（已装 1.3.0），auto_adjust=True 处理分红拆股
- 夜间收益 = Open_t/Close_{t-1} - 1；日内收益 = Close_t/Open_t - 1
- 成本模型：每天 2 笔成交，单边成本 = 佣金+滑点，扫 0/2/5/10 bps
- 输出目录：research/overnight/

## Errors Encountered
| Error | Attempt | Resolution |
|-------|---------|------------|
