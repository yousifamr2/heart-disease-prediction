from fastapi import FastAPI
from api.router import api_router
from core.database import engine, Base
import uvicorn

app = FastAPI(title="Heart Disease Prediction API")

# Start Application Routing
app.include_router(api_router)

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=5000, reload=True)
