from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from typing import Optional, List
import pickle
from sqlalchemy.orm import Session
import models
from database import SessionLocal, engine
import pandas as pd
import predictors_mapping


# Initialize the app
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
    cancel_rate: Optional[int] = None
    order_status: Optional[str] = None


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
models.Base.metadata.create_all(bind=engine)


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


# Helper function to make a prediction using the loaded model
def make_prediction(prediction_data: OrderStatusBase) -> str:
    df = encode_predictors(prediction_data)
    prediction = clf.predict(df)
    order_status = predictors_mapping.order_status[int(prediction[0])]
    return str(order_status)


# Helper function to calculate the cancel rate
def calculate_cancel_rate(prediction_data: OrderStatusBase) -> int:
    df = encode_predictors(prediction_data)
    prediction = clf.predict_proba(df)
    cancel_rate = prediction[0][1]
    return int(cancel_rate * 100)


# Endpoint to create a new prediction of the order status
@app.post("/api/order_status/", response_model=OrderStatusModel)
async def predict_order_status(
    prediction_data: OrderStatusBase, db: Session = Depends(get_db)
):
    prediction_data.order_status = make_prediction(prediction_data)
    prediction_data.cancel_rate = calculate_cancel_rate(prediction_data)
    db_prediction = models.OrderStatus(**prediction_data.model_dump())
    db.add(db_prediction)
    db.commit()
    db.refresh(db_prediction)
    return db_prediction


# Endpoint to read all predicted order status
@app.get("/api/order_status/", response_model=List[OrderStatusModel])
async def read_order_status(db: Session = Depends(get_db)):
    return db.query(models.OrderStatus).all()


# Endpoint to delete a predicted order status
@app.delete("/api/order_status/{id}/")
async def delete_order_status(id: int, db: Session = Depends(get_db)):
    db.query(models.OrderStatus).filter(models.OrderStatus.id == id).delete()
    db.commit()
    return {"message": "Order status deleted successfully"}


# Endpoint to redirect to API documentation
@app.get("/")
async def redirect_to_docs():
    return RedirectResponse(url="/docs")
