import os

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from app.database import Base, engine
from app.router import auth, order_forecast, user, visitor_forecast

# Initialize FastAPI
app = FastAPI(
    title="MKSG Clothing System API",
    description="API for predicting order status and projecting visitors hourly for MKSG Clothing",
    version="0.1.0",
)


# Get client origins
load_dotenv()
CLIENT_URL_PROD = str(os.getenv("CLIENT_URL_PROD"))
CLIENT_URL_DEV = str(os.getenv("CLIENT_URL_DEV"))


# Configure the CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[CLIENT_URL_DEV, CLIENT_URL_PROD],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Create the database tables
Base.metadata.create_all(bind=engine)
app.include_router(auth.router)
app.include_router(order_forecast.router)
app.include_router(user.router)
app.include_router(visitor_forecast.router)


# Endpoint to redirect to API documentation
@app.get("/")
async def redirect_to_docs():
    """
    Redirect to API documentation.
    """
    return RedirectResponse(url="/docs")
