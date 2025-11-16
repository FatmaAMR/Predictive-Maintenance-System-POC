from fastapi import APIRouter, Depends
from utils.config import get_settings, Settings  

base_router = APIRouter()

@base_router.get("/")
async def root(settings: Settings = Depends(get_settings)):
    return {
        "message": f"Welcome to our {settings.APP_NAME}!"
    }
