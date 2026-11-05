"""把数据库内容打包成前端可直接消费的 JSON（API /api/export 与静态导出共用）。"""
from __future__ import annotations

import datetime
import json

from . import factors as F
from . import signals as S
from .backtest import buy_hold_metrics
from .db import latest_params, load_bars


def build_export(conn, bar_limit: int = 400) -> dict:
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    codes = [dict(r) for r in conn.execute(
        "SELECT code,name,type FROM instruments ORDER BY code").fetchall()]
    glob = latest_params(conn)  # 全局参数版本（code IS NULL）
    details = {}
    for inst in codes:
        code = inst["code"]
        df = load_bars(conn, code, bar_limit)
        if df is None:
            continue
        pv = latest_params(conn, code) or glob
        params = (pv or {}).get("params") or S.DEFAULT_PARAMS
        fl = F.compute_factors(df)
        timing = None
        total = S.total_score(fl, params, timing) if fl else 0.0
        action = S.vote(total, params) if fl else "HOLD"
        sigs = []
        for r in conn.execute(
                "SELECT date,action,score,factors FROM signals WHERE code=? "
                "ORDER BY date DESC LIMIT 80", (code,)).fetchall():
            sigs.append(dict(date=r["date"], action=r["action"], score=r["score"],
                             factors=json.loads(r["factors"] or "[]")))
        bt = pv.get("oos_metrics") if pv else None
        details[code] = dict(
            name=inst["name"], type=inst["type"],
            bars=[[r["date"], r["open"], r["high"], r["low"], r["close"], r["volume"]]
                  for _, r in df.iterrows()],
            factors_today=fl, score_today=total, action_today=action,
            signals=sigs, param_version=(pv or {}).get("version"),
            params=params, backtest=bt,
            benchmark=buy_hold_metrics(df),
        )
    notes = [dict(r) for r in conn.execute(
        "SELECT * FROM notifications ORDER BY id DESC LIMIT 120").fetchall()]
    opts = [dict(r) for r in conn.execute(
        "SELECT * FROM optimize_logs ORDER BY id DESC LIMIT 60").fetchall()]
    return dict(generated=now, instruments=codes, details=details,
                notifications=notes, optimizations=opts)
