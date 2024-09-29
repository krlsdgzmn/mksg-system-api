import pickle

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

import app.predictors_mapping as predictors_mapping
from app.database import SessionLocal
from app.models import OrderPrediction

router = APIRouter(prefix="/api", tags=["order-prediction"])


# Load the pre-trained model
with open("app/random_forest.pkl", "rb") as f:
    clf = pickle.load(f)


# Pydantic model for prediction input
class OrderPredictionBase(BaseModel):
    price_bin: str
    discount_bin: str
    month: str
    week: int
    distance_bin: str
    cancel_rate: int | None = None
    order_status: str | None = None


# Pydantic model for database response
class OrderPredictionModel(OrderPredictionBase):
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


# Helper function to encode predictors
def encode_predictors(data: OrderPredictionBase) -> pd.DataFrame:
    predictors = {
        "price_bin": [data.price_bin],
        "discount_bin": [data.discount_bin],
        "month": [data.month],
        "week": [data.week],
        "distance_bin": [data.distance_bin],
    }
    df = pd.DataFrame(predictors)
    df.price_bin = df.price_bin.map(predictors_mapping.price_bin)
    df.discount_bin = df.discount_bin.map(predictors_mapping.discount_bin)
    df.month = df.month.map(predictors_mapping.month)
    df.distance_bin = df.distance_bin.map(predictors_mapping.distance_bin)
    return df


# Helper function to classify the order status using the loaded model
def classify_order_status(data: OrderPredictionBase) -> str:
    df = encode_predictors(data)
    prediction = clf.predict(df)
    order_status = predictors_mapping.order_status[int(prediction[0])]
    return str(order_status)


# Helper function to calculate the cancel rate
def calculate_cancel_rate(data: OrderPredictionBase) -> int:
    df = encode_predictors(data)
    prediction = clf.predict_proba(df)
    cancel_rate = prediction[0][1]
    return int(cancel_rate * 100)


# Endpoint to create a new order forecast
@router.post("/order-prediction", response_model=OrderPredictionModel)
async def create_order_forecast(
    data: OrderPredictionBase, db: Session = Depends(get_db)
):
    """
    Create a new order prediction.
    """
    try:
        data.order_status = classify_order_status(data)
        data.cancel_rate = calculate_cancel_rate(data)
        db_data = OrderPrediction(**data.model_dump())
        db.add(db_data)
        db.commit()
        db.refresh(db_data)
        return db_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Endpoint to read all classified order statuses
@router.get("/order-prediction", response_model=list[OrderPredictionModel])
async def read_order_forecast(
    month: list[str] | None = Query(None),
    week: list[int] | None = Query(None),
    order_status: list[str] | None = Query(None),
    db: Session = Depends(get_db),
):
    """
    Retrieve order predictions based on provided filters.
    """
    query = db.query(OrderPrediction)
    if month is not None:
        query = query.filter(OrderPrediction.month.in_(month))
    if week is not None:
        query = query.filter(OrderPrediction.week.in_(week))
    if order_status is not None:
        query = query.filter(OrderPrediction.order_status.in_(order_status))
    try:
        return query.all()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Endpoint to delete a predicted order status
@router.delete("/order-prediction/{id}")
async def delete_order_forecast(id: int, db: Session = Depends(get_db)):
    """
    Delete an order prediction by ID.
    """
    try:
        db.query(OrderPrediction).filter(OrderPrediction.id == id).delete()
        db.commit()
        return {"message": "Order forecast deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
