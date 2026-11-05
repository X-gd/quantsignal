"""SQLite 存储：行情 / 信号 / 通知 / 参数版本 / 优化日志 / 回测缓存。"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS instruments(
    code TEXT PRIMARY KEY, name TEXT, type TEXT);
CREATE TABLE IF NOT EXISTS daily_bars(
    code TEXT, date TEXT, open REAL, high REAL, low REAL, close REAL,
    volume REAL, pe REAL, PRIMARY KEY(code, date));
CREATE TABLE IF NOT EXISTS signals(
    code TEXT, date TEXT, action TEXT, score REAL, params_version INTEGER,
    factors TEXT, PRIMARY KEY(code, date));
CREATE TABLE IF NOT EXISTS notifications(
    id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, kind TEXT, code TEXT,
    name TEXT, action TEXT, message TEXT, priority INTEGER);
CREATE TABLE IF NOT EXISTS param_versions(
    id INTEGER PRIMARY KEY AUTOINCREMENT, created TEXT, code TEXT, params TEXT,
    oos_sharpe REAL, oos_maxdd REAL, oos_metrics TEXT, fold_reports TEXT);
CREATE TABLE IF NOT EXISTS optimize_logs(
    id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, message TEXT);
CREATE TABLE IF NOT EXISTS watchlist(
    code TEXT PRIMARY KEY, name TEXT, type TEXT, added TEXT);
CREATE INDEX IF NOT EXISTS idx_bars_code ON daily_bars(code);
"""


def connect(db_path: str) -> sqlite3.Connection:
    p = Path(db_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(p))
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


def upsert_bars(conn, code: str, df):
    rows = [(code, str(r["date"]), float(r["open"]), float(r["high"]), float(r["low"]),
             float(r["close"]), float(r["volume"]), None)
            for _, r in df.iterrows()]
    conn.executemany(
        "INSERT OR REPLACE INTO daily_bars(code,date,open,high,low,close,volume,pe) "
        "VALUES(?,?,?,?,?,?,?,?)", rows)
    conn.commit()


def load_bars(conn, code: str, limit: int = 800):
    rows = conn.execute(
        "SELECT date,open,high,low,close,volume FROM daily_bars WHERE code=? "
        "ORDER BY date DESC LIMIT ?", (code, limit)).fetchall()
    if not rows:
        return None
    import pandas as pd
    df = pd.DataFrame([dict(r) for r in rows])[::-1].reset_index(drop=True)
    return df


def save_signal(conn, code, date, action, score, version, factors):
    conn.execute(
        "INSERT OR REPLACE INTO signals(code,date,action,score,params_version,factors) "
        "VALUES(?,?,?,?,?,?)",
        (code, str(date), action, score, version, json.dumps(factors, ensure_ascii=False)))
    conn.commit()


def last_signal(conn, code: str):
    r = conn.execute("SELECT action FROM signals WHERE code=? ORDER BY date DESC LIMIT 1",
                     (code,)).fetchone()
    return r["action"] if r else None


def add_notification(conn, ts, kind, code, name, action, message, priority=3):
    conn.execute(
        "INSERT INTO notifications(ts,kind,code,name,action,message,priority) "
        "VALUES(?,?,?,?,?,?,?)", (ts, kind, code, name, action, message, priority))
    conn.commit()


def latest_params(conn, code: str | None = None):
    q = ("SELECT id, created, params, oos_metrics FROM param_versions "
         + ("WHERE code=?" if code else "WHERE code IS NULL") + " ORDER BY id DESC LIMIT 1")
    r = conn.execute(q, (code,) if code else ()).fetchone()
    if not r:
        return None
    return dict(version=r["id"], created=r["created"], params=json.loads(r["params"]),
                oos_metrics=json.loads(r["oos_metrics"] or "{}"))


def save_param_version(conn, created, code, params, oos_metrics, fold_reports):
    conn.execute(
        "INSERT INTO param_versions(created,code,params,oos_sharpe,oos_maxdd,oos_metrics,"
        "fold_reports) VALUES(?,?,?,?,?,?,?)",
        (created, code, json.dumps(params, ensure_ascii=False),
         oos_metrics.get("sharpe"), oos_metrics.get("max_drawdown"),
         json.dumps(oos_metrics, ensure_ascii=False),
         json.dumps(fold_reports, ensure_ascii=False)))
    conn.commit()


def add_optimize_log(conn, ts, message):
    conn.execute("INSERT INTO optimize_logs(ts,message) VALUES(?,?)", (ts, message))
    conn.commit()


def get_watchlist(conn):
    return conn.execute("SELECT code,name,type FROM watchlist ORDER BY added").fetchall()


def add_watch(conn, code, name, itype, added):
    conn.execute("INSERT OR REPLACE INTO watchlist(code,name,type,added) VALUES(?,?,?,?)",
                 (code, name, itype, added))
    conn.execute("INSERT OR REPLACE INTO instruments(code,name,type) VALUES(?,?,?)",
                 (code, name, itype))
    conn.commit()
