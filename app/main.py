from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import auth, health

app = FastAPI(title="Voice Authentication API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router)
app.include_router(health.router)

