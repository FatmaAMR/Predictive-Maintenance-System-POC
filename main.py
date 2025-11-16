from fastapi import FastAPI

from presentation.routes import base

app = FastAPI(title="PdM POC")
app.include_router(base.base_router)