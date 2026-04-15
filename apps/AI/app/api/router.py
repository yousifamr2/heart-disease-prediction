from fastapi import APIRouter
from api.endpoints import users, predict, shap, report

api_router = APIRouter()

api_router.include_router(users.router)
api_router.include_router(predict.router)
api_router.include_router(shap.router)
api_router.include_router(report.router)
