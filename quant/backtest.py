"""事件驱动回测 + walk-forward 滚动寻优（样本外约束防过拟合）。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .signals import DEFAULT_PARAMS, DEFAULT_WEIGHTS

PROFILES: dict[str, dict] = {
    "balanced": DEFAULT_WEIGHTS,
    "trend": {1: 1.6, 2: 1.2, 3: 1.4, 4: 1.0, 5: 0.6, 6: 0.6, 7: 1.0, 8: 0.8, 9: 0.6, 10: 2.0},
    "reversion": {1: 0.8, 2: 0.6, 3: 0.8, 4: 0.6, 5: 1.4, 6: 1.2, 7: 1.0, 8: 1.4, 9: 1.2, 10: 2.0},
}
THRESHOLDS = [(1.5, -1.5), (2.0, -2.0), (2.5, -2.5)]
TRADING_DAYS = 244


def profile_totals(score_hist: pd.DataFrame) -> dict[str, pd.Series]:
    """每种权重风格一条综合分序列。"""
    out = {}
    if score_hist.empty:
        return out
    for name, w in PROFILES.items():
        keys = {str(k): v for k, v in w.items()}
        cols = [c for c in score_hist.columns if str(c).isdigit()]
        tot = sum(score_hist[c] * keys.get(str(c), 1.0) for c in cols)
        out[name] = tot
    return out


def simulate(df: pd.DataFrame, totals: pd.Series, buy_thr: float, sell_thr: float,
             cost: float = 0.001) -> dict:
    """次日开盘买入/卖出的事件回测；返回权益曲线与交易明细。"""
    df = df.sort_values("date").reset_index(drop=True)
    tot = totals.reindex(df["date"].values)
    tot = pd.Series(tot).fillna(0.0).values if tot is not None else np.zeros(len(df))
    opens = df["open"].astype(float).values
    closes = df["close"].astype(float).values
    dates = df["date"].values
    n = len(df)
    equity = np.ones(n)
    pos = 0
    entry_px = 0.0
    entry_i = 0
    trades: list[dict] = []
    pending = None
    for i in range(n):
        if i > 0:
            equity[i] = equity[i - 1] * (closes[i] / closes[i - 1]) if pos == 1 else equity[i - 1]
        act, pending = pending, None
        if act == "BUY" and pos == 0:
            pos, entry_px, entry_i = 1, opens[i] * (1 + cost), i
        elif act == "SELL" and pos == 1:
            px = opens[i] * (1 - cost)
            trades.append(dict(entry_date=str(dates[entry_i]), exit_date=str(dates[i]),
                               ret=float(px / entry_px - 1), days=int(i - entry_i)))
            pos = 0
        if i + 1 < n and pos == 0 and tot[i] >= buy_thr:
            pending = "BUY"
        elif i + 1 < n and pos == 1 and tot[i] <= sell_thr:
            pending = "SELL"
    if pos == 1:
        trades.append(dict(entry_date=str(dates[entry_i]), exit_date=str(dates[-1]),
                           ret=float(closes[-1] * (1 - cost) / entry_px - 1),
                           days=int(n - 1 - entry_i)))
    return {"equity": equity, "dates": dates, "trades": trades}


def metrics(equity, dates, trades: list[dict]) -> dict:
    eq = np.asarray(equity, dtype=float)
    total_ret = float(eq[-1] / eq[0] - 1) if eq[0] > 0 else 0.0
    years = len(eq) / TRADING_DAYS
    annual = float((eq[-1] / eq[0]) ** (1 / years) - 1) if years > 0 and eq[0] > 0 else 0.0
    peak = np.maximum.accumulate(eq)
    max_dd = float((eq / peak - 1).min())
    r = pd.Series(eq).pct_change().dropna()
    sharpe = float(r.mean() / r.std() * np.sqrt(TRADING_DAYS)) if len(r) > 1 and r.std() > 0 else 0.0
    rets = [t["ret"] for t in trades]
    wins = [x for x in rets if x > 0]
    losses = [x for x in rets if x <= 0]
    gross_win, gross_loss = sum(wins), abs(sum(losses))
    yearly = {}
    if len(eq) > 1:
        s = pd.Series(eq, index=pd.to_datetime(pd.Index([str(d)[:10] for d in dates])))
        ye = s.groupby(s.index.year).last()
        y0 = s.groupby(s.index.year).first()
        yearly = {str(y): round(float(ye[y] / y0[y] - 1), 4) for y in ye.index}
    return {
        "total_return": round(total_ret, 4),
        "annual_return": round(annual, 4),
        "max_drawdown": round(max_dd, 4),
        "sharpe": round(sharpe, 2),
        "win_rate": round(len(wins) / len(rets), 3) if rets else None,
        "profit_factor": round(gross_win / gross_loss, 2) if gross_loss > 0 else None,
        "n_trades": len(rets),
        "avg_hold_days": round(float(np.mean([t["days"] for t in trades])), 1) if trades else None,
        "yearly": yearly,
    }


def fitness(m: dict) -> float:
    """适应度 = 样本外夏普 − 回撤惩罚（交易过少降权）。"""
    base = m["sharpe"] - abs(m["max_drawdown"]) * 2.0
    if m["n_trades"] < 3:
        base -= 1.0
    wr = m["win_rate"] if m["win_rate"] is not None else 0.5
    return round(base + 0.2 * (wr - 0.5), 3)


def buy_hold_metrics(df: pd.DataFrame) -> dict:
    eq = (df["close"].astype(float) / float(df["close"].iloc[0])).values
    return metrics(eq, df["date"].values, [])


def _date_at(df: pd.DataFrame, i: int):
    return df["date"].iloc[i]


def walk_forward(df: pd.DataFrame, score_hist: pd.DataFrame, folds: int = 3,
                 train: int = 250, test: int = 125) -> dict:
    """滚动窗口：训练段网格寻优 → 紧邻测试段样本外评估；返回最终参数与样本外表现。"""
    df = df.sort_values("date").reset_index(drop=True)
    if score_hist.empty or len(df) < train + test:
        w = PROFILES["balanced"]
        return dict(params=dict(DEFAULT_PARAMS, weights=w), oos_metrics={},
                    fold_reports=[], n_folds=0)
    totals_by_profile = profile_totals(score_hist)
    combos = [(p, bt, st) for p in PROFILES for (bt, st) in THRESHOLDS]
    fold_reports = []
    oos_folds = []
    best_fitness, best_combo = None, None
    n = len(df)
    for k in range(folds):
        tr_end = n - (folds - k) * test
        tr_start = max(0, tr_end - train)
        te_start, te_end = tr_end, min(n, tr_end + test)
        if te_end - te_start < 30 or tr_end - tr_start < 120:
            continue
        scores = []
        for (p, bt, st) in combos:
            tot = totals_by_profile[p]
            t_tr = tot[(tot.index >= _date_at(df, tr_start)) & (tot.index < _date_at(df, tr_end))]
            if len(t_tr) < 50:
                continue
            r = simulate(df.iloc[tr_start:tr_end], tot, bt, st)
            m = metrics(r["equity"], r["dates"], r["trades"])
            scores.append((fitness(m), p, bt, st))
        if not scores:
            continue
        scores.sort(reverse=True)
        f0, p0, bt0, st0 = scores[0]
        tot = totals_by_profile[p0]
        r = simulate(df.iloc[te_start:te_end], tot, bt0, st0)
        m = metrics(r["equity"], r["dates"], r["trades"])
        fold_reports.append(dict(
            fold=k + 1,
            chosen=dict(profile=p0, buy_threshold=bt0, sell_threshold=st0),
            train_fitness=f0, oos_metrics=m))
        oos_folds.append(m)
        if best_fitness is None or f0 > best_fitness:
            best_fitness, best_combo = f0, dict(profile=p0, buy_threshold=bt0, sell_threshold=st0)
    if not best_combo:
        best_combo = dict(profile="balanced", buy_threshold=2.0, sell_threshold=-2.0)
    w = PROFILES[best_combo["profile"]]
    final_params = dict(DEFAULT_PARAMS, profile=best_combo["profile"], weights=w,
                        buy_threshold=best_combo["buy_threshold"],
                        sell_threshold=best_combo["sell_threshold"])
    return dict(params=final_params, oos_metrics=aggregate_oos(oos_folds),
                fold_reports=fold_reports, n_folds=len(fold_reports))


def aggregate_oos(fold_metrics: list[dict]) -> dict:
    if not fold_metrics:
        return {}
    keys = ["total_return", "annual_return", "max_drawdown", "sharpe", "win_rate",
            "profit_factor", "n_trades", "avg_hold_days"]
    out = {}
    for k in keys:
        vals = [m[k] for m in fold_metrics if m.get(k) is not None]
        if vals:
            out[k] = round(float(np.mean(vals)), 2 if k == "sharpe" else 4)
    return out
