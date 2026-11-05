"""FastAPI 后端（Render 免费层部署）。启动：

    uvicorn backend.main:app --host 0.0.0.0 --port 8000

接口：
    GET /api/health
    GET /api/export                  前端一次性全量数据
    GET /api/search?q=               标的搜索（东财联想，免费无密钥）
    GET /api/notifications           通知中心
    GET /api/optimizations           优化日志
    POST /api/watchlist              添加自选（服务端持久化，可选）
"""
from __future__ import annotations

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from quant import data_fetch, db
from quant.export import build_export

DB_PATH = os.environ.get("QUANT_DB", "data/quant.db")

app = FastAPI(title="QuantSignal API", version="1.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def _conn():
    return db.connect(DB_PATH)


@app.get("/api/health")
def health():
    return {"ok": True}


@app.get("/api/export")
def export_all():
    conn = _conn()
    try:
        return build_export(conn)
    finally:
        conn.close()


@app.get("/api/search")
def search(q: str):
    if not q.strip():
        return {"results": []}
    return {"results": data_fetch.search_instrument(q.strip())}


@app.get("/api/notifications")
def notifications(limit: int = 100):
    conn = _conn()
    try:
        rows = conn.execute(
            "SELECT * FROM notifications ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return {"notifications": [dict(r) for r in rows]}
    finally:
        conn.close()


@app.get("/api/optimizations")
def optimizations(limit: int = 50):
    conn = _conn()
    try:
        rows = conn.execute(
            "SELECT * FROM optimize_logs ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return {"optimizations": [dict(r) for r in rows]}
    finally:
        conn.close()


@app.post("/api/watchlist")
def add_watch(code: str, name: str, type: str = "stock"):
    import datetime
    conn = _conn()
    try:
        db.add_watch(conn, code, name, type, datetime.date.today().isoformat())
        return {"ok": True}
    finally:
        conn.close()
