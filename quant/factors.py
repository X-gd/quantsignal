"""10因子计算模块：每个因子输出 -2~+2 分数与可解释说明。

因子清单（五维）：
 1 趋势  MA20/60 双均线金叉死叉
 2 趋势  价格 vs MA60 及均线斜率
 3 动量  MACD 金叉死叉 / 顶底背离
 4 动量  20日动量（ROC）
 5 情绪  RSI(14) 超买超卖
 6 情绪  布林带上下轨触碰/企稳
 7 量价  放量突破 / 放量滞涨破位
 8 量价  缩量回踩 MA20
 9 估值  PE 分位（缺失时用 3 年价格分位代理）
10 风控  ATR 跟踪止损（最高优先级卖点）
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    diff = close.diff()
    up = diff.clip(lower=0).rolling(n, min_periods=n).mean()
    dn = (-diff.clip(upper=0)).rolling(n, min_periods=n).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def macd(close: pd.Series, fast=12, slow=26, sig=9):
    dif = close.ewm(span=fast, adjust=False).mean() - close.ewm(span=slow, adjust=False).mean()
    dea = dif.ewm(span=sig, adjust=False).mean()
    return dif, dea


def boll(close: pd.Series, n=20, k=2.0):
    mid = sma(close, n)
    sd = close.rolling(n, min_periods=n).std()
    return mid, mid + k * sd, mid - k * sd


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    prev = df["close"].shift(1)
    tr = pd.concat(
        [df["high"] - df["low"], (df["high"] - prev).abs(), (df["low"] - prev).abs()],
        axis=1,
    ).max(axis=1)
    return tr.rolling(n, min_periods=n).mean()


def _crossed(a: pd.Series, b: pd.Series, lookback: int = 3):
    d = (a - b).dropna()
    if len(d) < lookback + 1:
        return False, False
    recent = d.iloc[-(lookback + 1):]
    up = any(recent.iloc[i - 1] <= 0 and recent.iloc[i] > 0 for i in range(1, len(recent)))
    down = any(recent.iloc[i - 1] >= 0 and recent.iloc[i] < 0 for i in range(1, len(recent)))
    return up, down


def compute_factors(df: pd.DataFrame, valuation: pd.Series | None = None) -> list[dict]:
    """对一段日线数据计算最新一天的 10 因子打分。

    df: 列 [date, open, high, low, close, volume]，按日期升序。
    valuation: 以日期为索引的 PE 序列（可选，缺失时代理）。
    返回因子列表，或数据不足时返回 []。
    """
    df = df.sort_values("date").reset_index(drop=True)
    if len(df) < 70:
        return []
    c = df["close"]
    v = df["volume"].astype(float)
    last = float(c.iloc[-1])
    out: list[dict] = []

    ma20, ma60 = sma(c, 20), sma(c, 60)

    # 1 双均线
    gu, gd = _crossed(ma20, ma60)
    if gu:
        s, e = 2.0, "MA20 上穿 MA60，出现金叉"
    elif gd:
        s, e = -2.0, "MA20 下穿 MA60，出现死叉"
    else:
        diff = float(ma20.iloc[-1] - ma60.iloc[-1])
        s = 1.0 if diff > 0 else -1.0 if diff < 0 else 0.0
        e = "MA20 高于 MA60（多头排列）" if diff > 0 else "MA20 低于 MA60（空头排列）"
    out.append(dict(id=1, name="MA20/60双均线", dim="趋势", score=s, explain=e))

    # 2 价格 vs MA60
    slope = float(ma60.iloc[-1] - ma60.iloc[-6])
    if last > float(ma60.iloc[-1]) and slope > 0:
        s, e = 1.0, "价格站上 MA60 且均线拐头向上"
    elif last < float(ma60.iloc[-1]) and slope < 0:
        s, e = -1.0, "价格跌破 MA60 且均线向下"
    elif last > float(ma60.iloc[-1]):
        s, e = 0.5, "价格在 MA60 上方但均线走平"
    else:
        s, e = -0.5, "价格在 MA60 下方"
    out.append(dict(id=2, name="价格vs MA60", dim="趋势", score=s, explain=e))

    # 3 MACD
    dif, dea = macd(c)
    d30 = dif.iloc[-30:]
    low_win = df.iloc[-30:]
    bottom_div = last <= float(low_win["low"].min()) and float(dif.iloc[-1]) > float(d30.min())
    top_div = last >= float(low_win["high"].max()) and float(dif.iloc[-1]) < float(d30.max())
    mcu, mcd = _crossed(dif, dea)
    if bottom_div:
        s, e = 2.0, "价格创新低但 MACD 未创新低，底背离"
    elif top_div:
        s, e = -2.0, "价格创新高但 MACD 走弱，顶背离"
    elif mcu:
        s, e = 2.0, "MACD 金叉"
    elif mcd:
        s, e = -2.0, "MACD 死叉"
    else:
        s = 1.0 if float(dif.iloc[-1]) > float(dea.iloc[-1]) else -1.0
        e = "MACD 在零轴上方运行" if s > 0 else "MACD 在零轴下方运行"
    out.append(dict(id=3, name="MACD动量", dim="动量", score=s, explain=e))

    # 4 20日动量
    roc20 = float(c.iloc[-1] / c.iloc[-21] - 1) if len(c) > 21 else 0.0
    roc5 = float(c.iloc[-1] / c.iloc[-6] - 1)
    if roc20 > 0.05:
        s, e = 2.0, f"20日涨幅 {roc20:.1%}，动量强劲"
    elif roc20 > 0 and roc5 > 0:
        s, e = 1.0, f"20日涨幅 {roc20:.1%} 且短线延续"
    elif roc20 < -0.05:
        s, e = -2.0, f"20日跌幅 {roc20:.1%}，动量转弱"
    elif roc20 < 0 and roc5 < 0:
        s, e = -1.0, f"20日跌幅 {roc20:.1%} 且短线仍弱"
    else:
        s, e = 0.0, "动量中性"
    out.append(dict(id=4, name="20日动量", dim="动量", score=s, explain=e))

    # 5 RSI
    r = float(rsi(c).iloc[-1])
    if np.isnan(r):
        s, e = 0.0, "RSI 数据不足"
    elif r < 30:
        s, e = 2.0, f"RSI={r:.0f}，超卖区"
    elif r < 40:
        s, e = 1.0, f"RSI={r:.0f}，接近超卖"
    elif r > 70:
        s, e = -2.0, f"RSI={r:.0f}，超买区"
    elif r > 60:
        s, e = -1.0, f"RSI={r:.0f}，接近超买"
    else:
        s, e = 0.0, f"RSI={r:.0f}，中性"
    out.append(dict(id=5, name="RSI超买超卖", dim="情绪", score=s, explain=e))

    # 6 布林带
    mid, up_b, low_b = boll(c)
    prev_close = float(c.iloc[-2]) if len(c) > 1 else last
    if prev_close < float(low_b.iloc[-2]) and last > float(low_b.iloc[-1]):
        s, e = 2.0, "昨日跌破布林下轨、今日收复，下轨企稳"
    elif last < float(low_b.iloc[-1]):
        s, e = 1.0, "价格触及布林下轨"
    elif last > float(up_b.iloc[-1]) and df["close"].iloc[-1] < df["open"].iloc[-1]:
        s, e = -2.0, "触及布林上轨且收阴，冲高回落"
    elif last > float(up_b.iloc[-1]):
        s, e = -1.0, "价格触及布林上轨"
    else:
        s, e = 0.0, "价格处于布林带内"
    out.append(dict(id=6, name="布林带位置", dim="情绪", score=s, explain=e))

    # 7 量价突破
    vma20 = sma(v, 20)
    high20 = float(df["high"].iloc[-21:-1].max())
    low20 = float(df["low"].iloc[-21:-1].min())
    vr = float(v.iloc[-1] / vma20.iloc[-1]) if vma20.iloc[-1] and not np.isnan(vma20.iloc[-1]) else 1.0
    if df["volume"].iloc[-1] <= 0:
        s, e = 0.0, "无成交量数据（基金净值）"
    elif vr > 1.5 and last > high20:
        s, e = 2.0, f"放量 {vr:.1f} 倍突破 20 日新高"
    elif vr > 2.0 and last < float(df["open"].iloc[-1]):
        s, e = -2.0, f"放量 {vr:.1f} 倍但收阴，疑似出货"
    elif vr > 1.5 and last < low20:
        s, e = -1.0, f"放量 {vr:.1f} 倍跌破 20 日低点"
    else:
        s, e = 0.0, "量价关系中性"
    out.append(dict(id=7, name="量价突破", dim="量价", score=s, explain=e))

    # 8 缩量回踩
    ma20v = float(ma20.iloc[-1])
    slope20 = float(ma20.iloc[-1] - ma20.iloc[-6])
    near = abs(last - ma20v) / ma20v < 0.02
    if df["volume"].iloc[-1] > 0 and near and vr < 0.7 and slope20 > 0:
        s, e = 2.0, "缩量回踩 MA20 且均线向上，回踩企稳"
    elif df["volume"].iloc[-1] > 0 and vr > 1.2 and last < ma20v:
        s, e = -1.0, "放量跌破 MA20"
    else:
        s, e = 0.0, "未出现回踩结构"
    out.append(dict(id=8, name="缩量回踩", dim="量价", score=s, explain=e))

    # 9 估值/分位
    if valuation is not None:
        pe = pd.to_numeric(pd.Series(valuation), errors="coerce").dropna()
    else:
        pe = pd.Series(dtype=float)
    if len(pe) >= 250:
        pe = pe.iloc[-750:]
        pct = float((pe < pe.iloc[-1]).mean())
        label = "PE 历史分位"
    else:
        win = c.iloc[-750:]
        pct = float((win < last).mean())
        label = "价格3年分位（PE缺失代理）"
    if pct < 0.3:
        s, e = 2.0, f"{label} {pct:.0%}，处于低位"
    elif pct < 0.4:
        s, e = 1.0, f"{label} {pct:.0%}，偏低"
    elif pct > 0.7:
        s, e = -2.0, f"{label} {pct:.0%}，处于高位，注意风险"
    elif pct > 0.6:
        s, e = -1.0, f"{label} {pct:.0%}，偏高"
    else:
        s, e = 0.0, f"{label} {pct:.0%}，中性"
    out.append(dict(id=9, name="估值/分位", dim="估值", score=s, explain=e))

    # 10 ATR 跟踪止损
    a = atr(df)
    hh = float(df["high"].iloc[-20:].max())
    stop_line = hh - 3 * float(a.iloc[-1])
    if last < stop_line:
        s, e = -2.0, f"跌破 ATR 跟踪止损线 {stop_line:.2f}，强制风控卖出信号"
    else:
        s, e = 0.0, f"未触发止损（跟踪止损线 {stop_line:.2f}）"
    out.append(dict(id=10, name="ATR跟踪止损", dim="风控", score=s, explain=e))

    return [dict(f, score=round(float(f["score"]), 2)) for f in out]
