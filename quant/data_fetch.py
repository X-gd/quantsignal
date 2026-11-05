"""免费数据源：东方财富公开 HTTP 接口（无需密钥），覆盖 A股/指数/ETF/场外基金净值。

akshare 作为可选后备（requirements 里可注释掉以精简安装）。
"""
from __future__ import annotations

import time

import pandas as pd
import requests

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"}


def _secid(code: str, itype: str) -> str:
    code = code.strip()
    if itype == "fund":
        return code
    if itype == "index":
        return ("0." if code.startswith(("39", "8")) else "1.") + code
    if itype == "etf":
        return ("0." if code.startswith(("15", "16")) else "1.") + code
    # stock：6/9/68 开头为沪市，0/3/2 开头为深市
    return ("1." if code.startswith(("6", "9", "5")) else "0.") + code


def fetch_kline(code: str, itype: str, size: int = 800):
    """东方财富日线（前复权）。返回 (name, DataFrame[date,open,high,low,close,volume]) 或 None。"""
    url = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
    params = {
        "secid": _secid(code, itype), "klt": 101, "fqt": 1, "lmt": size,
        "end": "20500101", "fields1": "f1,f2,f3,f4,f5,f6",
        "fields2": "f51,f52,f53,f54,f55,f56,f57",
    }
    try:
        r = requests.get(url, params=params, headers=UA, timeout=15)
        data = (r.json() or {}).get("data") or {}
        kl = data.get("klines") or []
        if not kl:
            return None, None
        rows = []
        for line in kl:
            p = line.split(",")
            rows.append((p[0], float(p[1]), float(p[4]), float(p[5]),
                         float(p[2]), float(p[6])))  # date, open, high, low, close, vol
        df = pd.DataFrame(rows, columns=["date", "open", "high", "low", "close", "volume"])
        return data.get("name", code), df
    except Exception:
        return None, None


def fetch_fund_nav(code: str, size: int = 800):
    """场外基金历史净值 → 伪日线（open=high=low=close=单位净值）。"""
    url = "https://api.fund.eastmoney.com/f10/lsjz"
    headers = dict(UA, Referer=f"https://fundf10.eastmoney.com/jjjz_{code}.html")
    rows = []
    page, per = 1, 20  # 该接口 pageSize 实测上限为 20，翻页抓取
    while len(rows) < size and page <= 45:
        try:
            r = requests.get(url, headers=headers, timeout=15,
                             params={"fundCode": code, "pageIndex": page, "pageSize": per})
            lst = ((r.json() or {}).get("Data") or {}).get("LSJZList") or []
        except Exception:
            break
        if not lst:
            break
        for it in lst:
            nav = it.get("DWJZ")
            if nav in (None, ""):
                continue
            d = it["FSRQ"]
            rows.append((d, float(nav)))
        page += 1
        time.sleep(0.3)
    if not rows:
        return None, None
    rows.sort(key=lambda x: x[0])
    rows = rows[-size:]
    df = pd.DataFrame(rows, columns=["date", "nav"])
    df["open"] = df["high"] = df["low"] = df["close"] = df["nav"]
    df["volume"] = 0.0
    df = df[["date", "open", "high", "low", "close", "volume"]]
    return code, df


def fetch_daily(code: str, itype: str, size: int = 800):
    """统一入口。itype ∈ stock/index/etf/fund。"""
    if itype == "fund":
        name, df = fetch_fund_nav(code, size)
        if df is not None:
            return name, df
    name, df = fetch_kline(code, itype, size)
    if df is None and itype != "fund":
        # akshare 后备
        try:
            import akshare as ak
            if itype == "index":
                raw = ak.index_zh_a_hist(symbol=code, period="daily")
            elif itype == "etf":
                raw = ak.fund_etf_hist_em(symbol=code, period="daily", adjust="qfq")
            else:
                raw = ak.stock_zh_a_hist(symbol=code, period="daily", adjust="qfq")
            raw = raw.rename(columns={"日期": "date", "开盘": "open", "收盘": "close",
                                      "最高": "high", "最低": "low", "成交量": "volume"})
            df = raw[["date", "open", "high", "low", "close", "volume"]].tail(size)
            df["date"] = df["date"].astype(str)
            return code, df
        except Exception:
            return None, None
    return name, df


def search_instrument(q: str) -> list[dict]:
    """东财联想搜索：股票/指数/ETF + 场外基金。"""
    out: list[dict] = []
    try:
        r = requests.get("https://searchapi.eastmoney.com/api/suggest/get",
                         params={"input": q, "type": "14", "count": 8,
                                 "token": "D43BF722C8E33BDC906FB84D85E326E8"},
                         headers=UA, timeout=10)
        data = ((r.json() or {}).get("QuotationCodeTable") or {}).get("Data") or []
        for it in data or []:
            if not it.get("Code"):
                continue
            tn = it.get("SecurityTypeName", "") or ""
            if "ETF" in tn or "LOF" in tn:
                t = "etf"
            elif "指数" in tn:
                t = "index"
            else:
                t = "stock"
            out.append(dict(code=it["Code"], name=it["Name"], type=t, source="东财"))
    except Exception:
        pass
    try:
        r = requests.get("http://fundsuggest.eastmoney.com/FundSearch/api/FundSearchAPI.ashx",
                         params={"m": 1, "key": q, "pageindex": 0, "pagesize": 5},
                         headers=UA, timeout=10)
        for it in ((r.json() or {}).get("Datas") or [])[:5]:
            code = it.get("CODE") or ""
            if code and it.get("CATEGORYDESC") in ("场外基金", "基金"):
                out.append(dict(code=code, name=it.get("NAME", code), type="fund", source="东财基金"))
    except Exception:
        pass
    seen, uniq = set(), []
    for x in out:
        if x["code"] not in seen:
            seen.add(x["code"])
            uniq.append(x)
    return uniq[:10]
