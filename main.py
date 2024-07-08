import pickle

import pandas as pd
from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

import predictors_mapping
from database import Base, SessionLocal, engine
from models import OrderStatus

# Initialize FastAPI
app = FastAPI(
    title="MKSG Clothing System API",
    description="API for predicting order status and projecting visitors hourly for MKSG Clothing",
    version="1.0",
)


# Configure the CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Load the pre-trained model
with open("model/random_forest.pkl", "rb") as f:
    clf = pickle.load(f)


# Pydantic model for prediction input
class OrderStatusBase(BaseModel):
    price_bin: str
    discount_bin: str
    month: str
    week: int
    distance_bin: str
    cancel_rate: int | None = None
    order_status: str | None = None


# Pydantic model for database response
class OrderStatusModel(OrderStatusBase):
    id: int

    class Config:
        from_attributes = True


# Database dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Create the database tables
Base.metadata.create_all(bind=engine)


# Helper function to encode predictors
def encode_predictors(data: OrderStatusBase) -> pd.DataFrame:
    predictors = {
        "price_bin": [data.price_bin],
        "discount_bin": [data.discount_bin],
        "month": [data.month],
        "week": [data.week],
        "distance_bin": [data.distance_bin],
    }

    df = pd.DataFrame(predictors)

    # Convert non-numerical predictors to numerical
    df.price_bin = df.price_bin.map(predictors_mapping.price_bin)
    df.discount_bin = df.discount_bin.map(predictors_mapping.discount_bin)
    df.month = df.month.map(predictors_mapping.month)
    df.distance_bin = df.distance_bin.map(predictors_mapping.distance_bin)
    return df


# Helper function to make classify the order stauts using the loaded model
def classify_order_status(data: OrderStatusBase) -> str:
    df = encode_predictors(data)
    prediction = clf.predict(df)
    order_status = predictors_mapping.order_status[int(prediction[0])]
    return str(order_status)


# Helper function to calculate the cancel rate
def calculate_cancel_rate(data: OrderStatusBase) -> int:
    df = encode_predictors(data)
    prediction = clf.predict_proba(df)
    cancel_rate = prediction[0][1]
    return int(cancel_rate * 100)


# Endpoint to create a new prediction of the order status
@app.post("/api/order_status/", response_model=OrderStatusModel)
async def create_order_status(data: OrderStatusBase, db: Session = Depends(get_db)):
    data.order_status = classify_order_status(data)
    data.cancel_rate = calculate_cancel_rate(data)
    db_data = OrderStatus(**data.model_dump())
    db.add(db_data)
    db.commit()
    db.refresh(db_data)
    return db_data


# Endpoint to read all classified order status
@app.get("/api/order_status/", response_model=list[OrderStatusModel])
async def read_order_status(
    month: str | None = None,
    week: int | None = None,
    order_status: str | None = None,
    db: Session = Depends(get_db),
):
    query = db.query(OrderStatus)

    if month is not None:
        query = query.filter(OrderStatus.month == month)
    if week is not None:
        query = query.filter(OrderStatus.week == week)
    if order_status is not None:
        query = query.filter(OrderStatus.order_status == order_status)

    return query.all()


# Endpoint to delete a predicted order status
@app.delete("/api/order_status/{id}/")
async def delete_order_status(id: int, db: Session = Depends(get_db)):
    db.query(OrderStatus).filter(OrderStatus.id == id).delete()
    db.commit()
    return {"message": "Order status deleted successfully"}


# Endpoint to redirect to API documentation
@app.get("/")
async def redirect_to_docs():
    return RedirectResponse(url="/docs")
