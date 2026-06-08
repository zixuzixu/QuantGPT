# Progress Log

## 2026-06-07: MU 夜间/日内策略回测
- 写了 research/overnight/backtest_overnight.py (yfinance 日线, 夜间/日内拆分, 成本敏感性, 盈利天数, 22票扫描)
- 一次运行成功, 输出: mu_strategies.csv / mu_yearly.csv / mu_days_to_profit.csv / universe_scan.csv / 2张PNG
- 结论见 findings.md: MU 夜间效应成立 (t=9.15), 盈亏平衡成本 ~7.8bps/单边, 高波动成长股普遍生效, 价值股反转

## 2026-06-07 (续): LaTeX 图文报告
- make_report_assets.py 生成 10 张图 (fig/) + report_data.json + universe_report.csv
- report.tex 用 ctexart + fontset=none + Noto CJK SC, latexmk -xelatex 编译, 10页 A4
- 新增统计: Sortino/Calmar/偏度/峰度, 滚动1年Sharpe负值占比 18.6%, 波动率-夜间t值相关 0.51

## Lessons Learned
- SendUserFile 需要绝对路径或确认 cwd — Bash 的 cd 改变了工作目录, 相对路径解析失败一次
- 论文式"天文数字收益"复现时务必同时报告成本敏感性, 0→10bps 单边成本足以把 280万× 变成归零
- LaTeX: \neg 是内置数学符号不能 \newcommand, 改名 \negv; ctexart 在本机用 fontset=none + 手动 \setCJKmainfont{Noto Serif CJK SC} 最稳
