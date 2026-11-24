"""FastAPI app. Run with `uv run uvicorn citewise.api.main:app --reload`."""

from __future__ import annotations

from fastapi import Depends, FastAPI
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from citewise import __version__
from citewise.api import brands, panel, runs
from citewise.core.db import get_db, ping

app = FastAPI(title="Citewise", version=__version__)
app.include_router(brands.router)
app.include_router(panel.router)
app.include_router(runs.router)


@app.get("/health")
def health(db: Session = Depends(get_db)) -> JSONResponse:
    db_ok = ping(db)
    return JSONResponse(
        status_code=200 if db_ok else 503,
        content={"status": "ok" if db_ok else "degraded", "db": "ok" if db_ok else "error"},
    )
