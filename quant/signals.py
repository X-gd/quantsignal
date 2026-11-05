"""加权投票 + ML 择时增强。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .factors import compute_factors, sma

DEFAULT_WEIGHTS = {1: 1.2, 2: 1.0, 3: 1.2, 4: 0.8, 5: 1.0, 6: 0.8, 7: 1.2, 8: 1.0, 9: 1.0, 10: 2.0}
DEFAULT_PARAMS = {
    "profile": "balanced",
    "weights": DEFAULT_WEIGHTS,
    "buy_threshold": 2.0,
    "sell_threshold": -2.0,
    "timing_enabled": True,
}


def total_score(factor_list: list[dict], params: dict, timing: float | None = None) -> float:
    w = params.get("weights", DEFAULT_WEIGHTS)
    keys = {str(k): v for k, v in w.items()}
    t = sum(keys.get(str(f["id"]), 1.0) * f["score"] for f in factor_list)
    if timing is not None:
        t += (timing - 50.0) / 25.0
    return round(float(t), 2)


def vote(total: float, params: dict) -> str:
    if total >= params["buy_threshold"]:
        return "BUY"
    if total <= params["sell_threshold"]:
        return "SELL"
    return "HOLD"


def factor_score_history(df: pd.DataFrame, valuation: pd.Series | None = None,
                         min_len: int = 70) -> pd.DataFrame:
    """滚动计算历史每天的因子分，返回以日期为索引、列 1..10 的 DataFrame。"""
    rows = []
    dates = df["date"].tolist()
    for i in range(min_len, len(df) + 1):
        fl = compute_factors(df.iloc[:i], valuation)
        if not fl:
            continue
        row = {str(f["id"]): f["score"] for f in fl}
        row["date"] = dates[i - 1]
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("date")


def train_timing_model(score_hist: pd.DataFrame, df: pd.DataFrame, horizon: int = 10):
    """用因子历史分训练梯度提升择时模型（可省略，失败时返回 None）。"""
    if score_hist.empty or len(score_hist) < 200:
        return None, {}
    try:
        from sklearn.ensemble import GradientBoostingClassifier
    except ImportError:
        return None, {}
    close = df.set_index("date")["close"].astype(float)
    fwd = close.shift(-horizon) / close - 1
    feats = score_hist.copy()
    feats["ret5"] = close.pct_change(5).reindex(feats.index)
    feats["vol10"] = close.pct_change().rolling(10).std().reindex(feats.index)
    y = (fwd.reindex(feats.index) > 0).astype(int)
    mask = fwd.reindex(feats.index).notna()
    X, yy = feats[mask], y[mask]
    if len(X) < 200 or yy.nunique() < 2:
        return None, {}
    model = GradientBoostingClassifier(n_estimators=60, max_depth=3,
                                       learning_rate=0.08, random_state=42)
    model.fit(X, yy)
    acc = float((model.predict(X) == yy).mean())
    return model, {"train_acc": round(acc, 3), "n_samples": int(len(X))}


def timing_score(model, score_hist: pd.DataFrame, df: pd.DataFrame) -> float | None:
    """最新一天的 ML 择时分（0-100，50 为中性）。"""
    if model is None or score_hist.empty:
        return None
    close = df.set_index("date")["close"].astype(float)
    row = score_hist.iloc[[-1]].copy()
    row["ret5"] = [float(close.pct_change(5).iloc[-1])]
    row["vol10"] = [float(close.pct_change().rolling(10).std().iloc[-1])]
    cols = list(model.feature_names_in_)
    return round(float(model.predict_proba(row[cols])[0, 1]) * 100, 1)
