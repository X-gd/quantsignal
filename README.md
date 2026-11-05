# QuantSignal · AI量化买卖点提醒 Agent

免费、可自主优化迭代的金融量化 Web 应用：对 **A股 / 宽基指数 / ETF / 场外基金** 生成可解释的
**买点 / 卖点提醒**，并通过 walk-forward 滚动寻优每天自动进化参数。

> ⚠️ 免责声明：所有输出为量化模型参考信号，基于历史数据计算，不构成投资建议。市场有风险，投资需谨慎。

## 核心设计

| 模块 | 方案 |
|------|------|
| 信号核心 | 五维 10 因子加权投票（趋势×2、动量×2、情绪×2、量价×2、估值×1、ATR止损×1），每个信号可单独解释 |
| ML 增强 | 梯度提升择时模型输出 0-100 分，微调投票阈值（sklearn 缺失时自动降级） |
| 自主优化 | walk-forward 滚动回测 + 网格寻优（3 种权重风格 × 3 组阈值），只有样本外更优的参数才晋级，参数次日生效 |
| 回测口径 | 年化收益 / 最大回撤 / 夏普 / 胜率 / 盈亏比 / 信号次数 / 平均持有天数 / 分年度收益；基准=买入持有 |
| 提醒规则 | 信号状态翻转去重才触发单条提醒 + 每日汇总；卖点优先展示 |
| 数据源 | 东方财富公开接口（无需密钥），akshare 可选后备 |
| 用户体系 | 免登录，自选存 localStorage，同步码跨设备迁移 |

## 架构与免费部署

```
浏览器 SPA ──REST──> FastAPI(Render免费层) ──读──> SQLite
     ↑                                        ↑写
     └──── 静态导出 export.json <── GitHub Actions 每日 17:30（免费算力）
                                        │
                                  东财免费数据源
```

### 方案 A（推荐，全免费）：GitHub Pages + Actions

1. 把本项目推到 GitHub 仓库
2. 仓库 Settings → Pages → 选择分支部署 `frontend/`（或在 Actions 里加一步上传）
3. Actions 会自动每天 17:30（北京时间）跑流水线，把更新后的
   `data/quant.db` 与 `data/export.json` 提交回仓库
4. 前端自动按 `/api/export` → `data/export.json` → 内置演示数据 三级降级加载数据

### 方案 B：Render 后端（可选）

- 新建 Web Service → 启动命令 `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`
- 环境变量 `QUANT_DB=/data/quant.db`（挂载免费磁盘保数据持久）
- 前端与后端同源部署时走 `/api/export` 实时接口

## 本地运行

```bash
pip install -r requirements.txt

# 1) 跑一次真实流水线（需要能访问东方财富）
python -m quant.run_daily --db data/quant.db --export data/export.json --optimize

# 2) 启动后端
uvicorn backend.main:app --reload

# 3) 前端（任选其一）
cd frontend && python -m http.server 8080     # 直接打开 http://localhost:8080
```

无网络环境可直接体验离线演示模式：前端在拿不到 API / export.json 时自动使用
`frontend/demo-data.js`（模拟 3 年行情，已跑通因子→投票→walk-forward 全链路）。

## 目录结构

```
quant/            核心包
  data_fetch.py   东财免费数据抓取（A股/指数/ETF/基金净值 + 搜索）
  factors.py      10 因子计算（每个信号可解释）
  signals.py      加权投票 + ML 择时增强
  backtest.py     事件回测 + walk-forward 滚动寻优
  db.py           SQLite 存储
  export.py       前端数据导出
  run_daily.py    每日流水线入口
backend/main.py   FastAPI 服务
frontend/         单页应用（三区：自选 / K线信号 / 通知中心）
demo/make_demo.py 演示数据生成 + 流水线自测
.github/workflows/daily.yml  每日定时任务（UTC 09:30 = 北京 17:30）
```

## 二次开发建议

- 加因子：在 `factors.py` 增加，并在 `signals.py::DEFAULT_WEIGHTS` 与三种风格中登记权重
- 加市场（港股/美股）：在 `data_fetch.py` 增加 secid 映射即可
- 调优化强度：`backtest.py::THRESHOLDS` 扩网格，或把网格搜索换成贝叶斯优化
