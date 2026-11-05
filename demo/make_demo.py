"""生成演示数据：模拟 3 年 A股/指数/ETF/场外基金行情 → 跑通
因子 → 投票 → walk-forward 寻优 → 通知 全链路 → 输出 frontend/demo-data.js。

    python -m demo.make_demo

用途：无网络/无部署时前端也能完整演示；同时作为流水线的端到端自测。
"""
from __future__ import annotations

import datetime
import json
from pathlib import Path

import numpy as np
import pandas as pd

from quant import db
from quant import factors as F
from quant import signals as S
from quant.backtest import walk_forward
from quant.export import build_export

UNIVERSE = [
    ("000300", "沪深300", "index", 3900, 14.5, 101),
    ("399006", "创业板指", "index", 2100, 32.0, 102),
    ("510300", "沪深300ETF", "etf", 3.9, 13.0, 103),
    ("600519", "贵州茅台", "stock", 1650, 26.0, 104),
    ("000001", "平安银行", "stock", 11.5, 5.5, 105),
    ("110022", "易方达消费行业", "fund", 4.6, None, 106),
]


def _trading_dates(n: int) -> list[str]:
    days = pd.bdate_range(end=datetime.date.today(), periods=n)
    return [d.strftime("%Y-%m-%d") for d in days]


def simulate(code: str, name: str, itype: str, base: float, pe0: float | None,
             seed: int, n: int = 760):
    rng = np.random.default_rng(seed)
    dates = _trading_dates(n)
    closes, pes = [], []
    price = base
    pe = pe0
    i = 0
    while i < n:
        regime_len = int(rng.integers(35, 90))
        bull = rng.random() > 0.45
        mu = 0.0011 if bull else -0.0009
        sigma = rng.uniform(0.011, 0.02)
        if itype == "fund":
            sigma *= 0.8
        for _ in range(regime_len):
            if i >= n:
                break
            ret = mu + sigma * rng.standard_normal()
            price = max(price * (1 + ret), base * 0.35)
            closes.append(price)
            if pe0 is not None:
                pe = pe + 0.9 * (pe0 * (0.8 + 0.5 * (price / base) ** 2) - pe) + pe0 * 0.01 * rng.standard_normal()
                pes.append(max(pe, 3.0))
            i += 1
    closes = np.array(closes[:n])
    opens = np.empty(n)
    opens[0] = closes[0]
    opens[1:] = closes[:-1] * (1 + rng.normal(0, 0.003, n - 1))
    spread = np.abs(rng.normal(0.008, 0.004, n))
    highs = np.maximum(opens, closes) * (1 + spread)
    lows = np.minimum(opens, closes) * (1 - spread)
    vol_base = {"index": 2.0e8, "etf": 8.0e6, "stock": 3.0e7, "fund": 0}.get(itype, 1e7)
    prev_ret = np.zeros(n)
    prev_ret[1:] = np.abs(closes[1:] / closes[:-1] - 1)
    vol = vol_base * np.exp(rng.normal(0, 0.35, n)) * (1 + prev_ret * 30)
    if itype == "fund":
        vol = np.zeros(n)
    df = pd.DataFrame(dict(date=dates, open=opens, high=highs, low=lows,
                           close=closes, volume=vol))
    pe_series = pd.Series(pes[:n], index=dates) if pe0 is not None else None
    return name, df, pe_series


def main(out_path: str = "frontend/demo-data.js"):
    conn = db.connect("data/demo.db")
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    signals_all = []
    notifications = []
    optimizations = []
    for code, name, itype, base, pe0, seed in UNIVERSE:
        conn.execute("INSERT OR REPLACE INTO instruments(code,name,type) VALUES(?,?,?)",
                     (code, name, itype))
        name2, df, pe = simulate(code, name, itype, base, pe0, seed)
        conn.executemany(
            "INSERT OR REPLACE INTO daily_bars(code,date,open,high,low,close,volume,pe) "
            "VALUES(?,?,?,?,?,?,?,?)",
            [(code, str(r["date"]), r["open"], r["high"], r["low"], r["close"], r["volume"],
              None if pe is None else float(pe.loc[str(r["date"])])
              if str(r["date"]) in (pe.index if pe is not None else []) else None)
             for _, r in df.iterrows()])
        hist = S.factor_score_history(df, pe)
        res = walk_forward(df, hist)
        params = res["params"]
        oos = res["oos_metrics"]
        conn.execute(
            "INSERT INTO param_versions(created,code,params,oos_sharpe,oos_maxdd,"
            "oos_metrics,fold_reports) VALUES(?,?,?,?,?,?,?)",
            (now, code, json.dumps(params, ensure_ascii=False),
             oos.get("sharpe"), oos.get("max_drawdown"),
             json.dumps(oos, ensure_ascii=False),
             json.dumps(res["fold_reports"], ensure_ascii=False)))
        totals = []
        for dt, row in hist.iterrows():
            t = sum(params["weights"].get(str(int(c)), 1.0) * row[str(int(c))]
                    for c in params["weights"])
            totals.append((dt, round(float(t), 2)))
        prev = "HOLD"
        cutoff = totals[-90][0] if len(totals) > 90 else totals[0][0]
        for dt, t in totals:
            act = S.vote(t, params)
            if act != prev:
                fl = F.compute_factors(df[df["date"] <= dt])
                if act in ("BUY", "SELL") and dt >= cutoff:
                    pri = 1 if act == "SELL" else 2
                    top = sorted(fl, key=lambda f: abs(f["score"]) *
                                 params["weights"].get(str(f["id"]), 1.0),
                                 reverse=True)[:3]
                    reason = "；".join(f"{x['name']}：{x['explain']}" for x in top)
                    emoji = "🟢 买点" if act == "BUY" else "🔴 卖点"
                    notifications.append(dict(
                        ts=dt, kind="signal", code=code, name=name, action=act,
                        message=f"{emoji}【{name} {code}】综合分 {t:+.1f} 触发"
                                f"{'买入' if act == 'BUY' else '卖出/减仓'}信号。主要依据：{reason}",
                        priority=pri))
                signals_all.append((code, dt, act, t, 1,
                                    json.dumps(fl, ensure_ascii=False)))
            prev = act
        optimizations.append(
            f"🔧 walk-forward 寻优【{name}({code})】→ {params['profile']} 风格，"
            f"买/卖阈值 {params['buy_threshold']}/{params['sell_threshold']}，"
            f"样本外夏普 {oos.get('sharpe', '—')}，最大回撤 {oos.get('max_drawdown', '—')}，"
            f"{res['n_folds']} 折验证")
    conn.executemany(
        "INSERT OR REPLACE INTO signals(code,date,action,score,params_version,factors) "
        "VALUES(?,?,?,?,?,?)", signals_all)
    notifications.sort(key=lambda x: x["ts"], reverse=True)
    for nt in notifications:
        conn.execute(
            "INSERT INTO notifications(ts,kind,code,name,action,message,priority) "
            "VALUES(?,?,?,?,?,?,?)",
            (nt["ts"], nt["kind"], nt["code"], nt["name"], nt["action"],
             nt["message"], nt["priority"]))
    digest = f"📋 每日汇总：演示数据已生成，含 {len(UNIVERSE)} 只标的的信号与回测。本页为离线演示模式，部署后由每日流水线实时更新。"
    conn.execute("INSERT INTO notifications(ts,kind,code,name,action,message,priority) "
                 "VALUES(?,?,?,?,?,?,?)", (now, "digest", None, None, None, digest, 3))
    for m in optimizations:
        conn.execute("INSERT INTO optimize_logs(ts,message) VALUES(?,?)", (now, m))
    conn.commit()

    data = build_export(conn)
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("window.DEMO_DATA = " + json.dumps(data, ensure_ascii=False) + ";",
                 encoding="utf-8")
    n_sig = sum(1 for c in data["details"].values()
                for s in c["signals"] if s["action"] in ("BUY", "SELL"))
    print(f"OK 演示数据已写入 {p} | 标的 {len(data['details'])} | BUY/SELL 信号 {n_sig} 条 | "
          f"通知 {len(notifications)+1} 条 | 优化日志 {len(optimizations)} 条")
    print("样本外回测摘要：")
    for code, d in data["details"].items():
        bt = d.get("backtest") or {}
        print(f"  {d['name']}: 夏普={bt.get('sharpe')} 回撤={bt.get('max_drawdown')} "
              f"胜率={bt.get('win_rate')} 交易数={bt.get('n_trades')} "
              f"参数v{d.get('param_version')} 今日={d['action_today']}({d['score_today']:+.1f})")


if __name__ == "__main__":
    main()
