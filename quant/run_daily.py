"""每日流水线入口（GitHub Actions / 本地均可运行）：

    python -m quant.run_daily --db data/quant.db --export data/export.json [--optimize]

流程：抓取行情 → 10因子打分 → 加权投票 → 信号状态翻转去重 → 写通知中心
     →（可选/--optimize 或周五）walk-forward 寻优生成参数新版本 → 优化日志
"""
from __future__ import annotations

import argparse
import datetime
import json

from . import data_fetch, db
from . import factors as F
from . import signals as S
from .backtest import walk_forward
from .export import build_export

DEFAULT_WATCHLIST = [
    ("000300", "沪深300", "index"),
    ("399006", "创业板指", "index"),
    ("510300", "沪深300ETF", "etf"),
    ("600519", "贵州茅台", "stock"),
    ("000001", "平安银行", "stock"),
    ("110022", "易方达消费行业", "fund"),
]


def _today() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d")


def _now() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M")


def run(db_path: str = "data/quant.db", export_path: str | None = None,
        optimize: bool = False, codes: list[str] | None = None,
        bar_size: int = 800) -> dict:
    conn = db.connect(db_path)
    watch = [(r["code"], r["name"], r["type"]) for r in db.get_watchlist(conn)]
    if not watch:
        for code, name, itype in DEFAULT_WATCHLIST:
            db.add_watch(conn, code, name, itype, _today())
        watch = [(c, n, t) for c, n, t in DEFAULT_WATCHLIST]
    if codes:
        watch = [w for w in watch if w[0] in set(codes)] or watch

    glob = db.latest_params(conn)
    params = (glob or {}).get("params") or S.DEFAULT_PARAMS
    version = (glob or {}).get("version", 0)

    digest = {"BUY": [], "SELL": [], "HOLD": [], "fail": []}
    for code, name, itype in watch:
        fetched_name, df = data_fetch.fetch_daily(code, itype, bar_size)
        if df is None or len(df) < 70:
            digest["fail"].append(f"{name}({code}) 数据获取失败或不足")
            continue
        if fetched_name:
            name = fetched_name
            conn.execute("UPDATE instruments SET name=? WHERE code=?", (name, code))
            conn.execute("UPDATE watchlist SET name=? WHERE code=?", (name, code))
        db.upsert_bars(conn, code, df)

        fl = F.compute_factors(df)
        if not fl:
            continue
        total = S.total_score(fl, params)
        action = S.vote(total, params)
        top = sorted(fl, key=lambda f: abs(f["score"]) * params.get("weights", {}).get(
            str(f["id"]), 1.0), reverse=True)[:3]
        reason = "；".join(f"{t['name']}：{t['explain']}" for t in top)
        db.save_signal(conn, code, _today(), action, total, version, fl)

        prev = None
        row = conn.execute(
            "SELECT action FROM signals WHERE code=? AND date<? ORDER BY date DESC LIMIT 1",
            (code, _today())).fetchone()
        prev = row["action"] if row else None
        if action in ("BUY", "SELL"):
            pri = 1 if action == "SELL" else 2
            emoji = "🟢 买点" if action == "BUY" else "🔴 卖点"
            db.add_notification(conn, _now(), "signal", code, name, action,
                                f"{emoji}【{name} {code}】综合分 {total:+.1f} 触发{ '买入' if action=='BUY' else '卖出/减仓' }信号。主要依据：{reason}",
                                pri)
        digest[action].append(f"{name}({code}) {total:+.1f}")
        if action in ("BUY", "SELL"):
            digest[action][-1] += f" → {'买入' if action=='BUY' else '卖出'}信号"

    summary = "；".join(filter(None, [
        f"买点 {len(digest['BUY'])} 个" if digest["BUY"] else "",
        f"卖点 {len(digest['SELL'])} 个" if digest["SELL"] else "",
        f"持币观望 {len(digest['HOLD'])} 只" if digest["HOLD"] else "",
    ])) or "今日无信号"
    if digest["fail"]:
        summary += f"；{len(digest['fail'])} 只数据异常"
    db.add_notification(conn, _now(), "digest", None, None, None,
                        f"📋 每日汇总（{_today()}）：{summary}。参数版本 v{version}。", 3)

    opt_msg = None
    if optimize or datetime.datetime.now().weekday() == 4:
        opt_msg = run_optimize(conn, db_path, bar_size)

    result = dict(date=_today(), digest=digest, summary=summary, optimize=opt_msg)
    if export_path:
        data = build_export(conn)
        import pathlib
        p = pathlib.Path(export_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        # 同步一份到 frontend/data/，供 GitHub Pages 静态站直接读取
        fp = pathlib.Path("frontend") / "data" / "export.json"
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    conn.close()
    return result


def run_optimize(conn, db_path: str, bar_size: int = 800) -> str:
    """对自选池逐只跑 walk-forward，落库参数版本与优化日志。"""
    watch = [(r["code"], r["name"], r["type"]) for r in db.get_watchlist(conn)]
    msgs = []
    for code, name, itype in watch[:8]:
        df = db.load_bars(conn, code, bar_size)
        if df is None or len(df) < 400:
            continue
        hist = S.factor_score_history(df)
        if hist.empty:
            continue
        res = walk_forward(df, hist)
        oos = res["oos_metrics"]
        db.save_param_version(conn, _now(), code, res["params"], oos, res["fold_reports"])
        m = oos or {}
        msgs.append(
            f"{name}({code}) → {res['params']['profile']} 风格，买/卖阈值 "
            f"{res['params']['buy_threshold']}/{res['params']['sell_threshold']}，"
            f"样本外夏普 {m.get('sharpe', '—')}，最大回撤 {m.get('max_drawdown', '—')}，"
            f"{res['n_folds']} 折验证")
    msg = f"🔧 walk-forward 寻优完成（{_now()}）：" + "；".join(msgs) if msgs else \
        f"🔧 walk-forward 寻优：数据不足，本轮跳过"
    db.add_optimize_log(conn, _now(), msg)
    return msg


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/quant.db")
    ap.add_argument("--export", default=None)
    ap.add_argument("--optimize", action="store_true")
    ap.add_argument("--codes", nargs="*", default=None)
    args = ap.parse_args()
    out = run(args.db, args.export, args.optimize, args.codes)
    print(json.dumps(out, ensure_ascii=False, indent=2, default=str))
