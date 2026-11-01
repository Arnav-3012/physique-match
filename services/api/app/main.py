import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routes import health

app = FastAPI(title="PhysiqueGap API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("WEB_ORIGINS", "http://localhost:5173").split(","),
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)
app.include_router(health.router)
