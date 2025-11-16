from fastapi import FastAPI

from presentation.routes import base, predict_api

app = FastAPI(title="PdM POC")
app.include_router(base.base_router)
app.include_router(predict_api.data_router)