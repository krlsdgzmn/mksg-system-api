from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import and_
from sqlalchemy.orm import Session

from app.database import SessionLocal
from app.models import VisitorForecast

router = APIRouter(prefix="/api", tags=["visitor-forecast"])


# Database dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


class VisitorForecastBase(BaseModel):
    datetime: datetime
    value: int


# Endpoint to get the visitor forecast for seven days (including today)
@router.get("/visitor-forecast/", response_model=list[VisitorForecastBase])
def get_seven_day_forecast(db: Session = Depends(get_db)):
    today = datetime.now(timezone.utc)
    seven_days_later = today + timedelta(days=6)

    forecast_data = (
        db.query(VisitorForecast)
        .filter(
            and_(
                VisitorForecast.datetime >= today,
                VisitorForecast.datetime <= seven_days_later,
            )
        )
        .order_by(VisitorForecast.datetime)
        .all()
    )

    if not forecast_data:
        raise HTTPException(
            status_code=404, detail="No forecast data found for the next seven days"
        )

    return forecast_data


sample_forecast = [
    {"datetime": "2024-08-30 00:00:00", "value": 5},
    {"datetime": "2024-08-30 01:00:00", "value": 10},
    {"datetime": "2024-08-30 02:00:00", "value": 8},
    {"datetime": "2024-08-30 03:00:00", "value": 8},
    {"datetime": "2024-08-30 04:00:00", "value": 9},
    {"datetime": "2024-08-30 05:00:00", "value": 7},
    {"datetime": "2024-08-30 06:00:00", "value": 9},
    {"datetime": "2024-08-30 07:00:00", "value": 10},
    {"datetime": "2024-08-30 08:00:00", "value": 15},
    {"datetime": "2024-08-30 09:00:00", "value": 20},
    {"datetime": "2024-08-30 10:00:00", "value": 35},
    {"datetime": "2024-08-30 11:00:00", "value": 45},
    {"datetime": "2024-08-30 12:00:00", "value": 50},
    {"datetime": "2024-08-30 13:00:00", "value": 48},
    {"datetime": "2024-08-30 14:00:00", "value": 40},
    {"datetime": "2024-08-30 15:00:00", "value": 38},
    {"datetime": "2024-08-30 16:00:00", "value": 42},
    {"datetime": "2024-08-30 17:00:00", "value": 50},
    {"datetime": "2024-08-30 18:00:00", "value": 60},
    {"datetime": "2024-08-30 19:00:00", "value": 65},
    {"datetime": "2024-08-30 20:00:00", "value": 68},
    {"datetime": "2024-08-30 21:00:00", "value": 55},
    {"datetime": "2024-08-30 22:00:00", "value": 30},
    {"datetime": "2024-08-30 23:00:00", "value": 15},
    {"datetime": "2024-08-31 00:00:00", "value": 5},
    {"datetime": "2024-08-31 01:00:00", "value": 10},
    {"datetime": "2024-08-31 02:00:00", "value": 8},
    {"datetime": "2024-08-31 03:00:00", "value": 8},
    {"datetime": "2024-08-31 04:00:00", "value": 9},
    {"datetime": "2024-08-31 05:00:00", "value": 7},
    {"datetime": "2024-08-31 06:00:00", "value": 9},
    {"datetime": "2024-08-31 07:00:00", "value": 10},
    {"datetime": "2024-08-31 08:00:00", "value": 15},
    {"datetime": "2024-08-31 09:00:00", "value": 20},
    {"datetime": "2024-08-31 10:00:00", "value": 35},
    {"datetime": "2024-08-31 11:00:00", "value": 45},
    {"datetime": "2024-08-31 12:00:00", "value": 50},
    {"datetime": "2024-08-31 13:00:00", "value": 48},
    {"datetime": "2024-08-31 14:00:00", "value": 40},
    {"datetime": "2024-08-31 15:00:00", "value": 38},
    {"datetime": "2024-08-31 16:00:00", "value": 42},
    {"datetime": "2024-08-31 17:00:00", "value": 50},
    {"datetime": "2024-08-31 18:00:00", "value": 60},
    {"datetime": "2024-08-31 19:00:00", "value": 65},
    {"datetime": "2024-08-31 20:00:00", "value": 68},
    {"datetime": "2024-08-31 21:00:00", "value": 55},
    {"datetime": "2024-08-31 22:00:00", "value": 30},
    {"datetime": "2024-08-31 23:00:00", "value": 15},
    {"datetime": "2024-09-01 00:00:00", "value": 5},
    {"datetime": "2024-09-01 01:00:00", "value": 10},
    {"datetime": "2024-09-01 02:00:00", "value": 8},
    {"datetime": "2024-09-01 03:00:00", "value": 8},
    {"datetime": "2024-09-01 04:00:00", "value": 9},
    {"datetime": "2024-09-01 05:00:00", "value": 7},
    {"datetime": "2024-09-01 06:00:00", "value": 9},
    {"datetime": "2024-09-01 07:00:00", "value": 10},
    {"datetime": "2024-09-01 08:00:00", "value": 15},
    {"datetime": "2024-09-01 09:00:00", "value": 20},
    {"datetime": "2024-09-01 10:00:00", "value": 35},
    {"datetime": "2024-09-01 11:00:00", "value": 45},
    {"datetime": "2024-09-01 12:00:00", "value": 50},
    {"datetime": "2024-09-01 13:00:00", "value": 48},
    {"datetime": "2024-09-01 14:00:00", "value": 40},
    {"datetime": "2024-09-01 15:00:00", "value": 38},
    {"datetime": "2024-09-01 16:00:00", "value": 42},
    {"datetime": "2024-09-01 17:00:00", "value": 50},
    {"datetime": "2024-09-01 18:00:00", "value": 60},
    {"datetime": "2024-09-01 19:00:00", "value": 65},
    {"datetime": "2024-09-01 20:00:00", "value": 68},
    {"datetime": "2024-09-01 21:00:00", "value": 55},
    {"datetime": "2024-09-01 22:00:00", "value": 30},
    {"datetime": "2024-09-01 23:00:00", "value": 15},
    {"datetime": "2024-09-02 00:00:00", "value": 5},
    {"datetime": "2024-09-02 01:00:00", "value": 10},
    {"datetime": "2024-09-02 02:00:00", "value": 8},
    {"datetime": "2024-09-02 03:00:00", "value": 8},
    {"datetime": "2024-09-02 04:00:00", "value": 9},
    {"datetime": "2024-09-02 05:00:00", "value": 7},
    {"datetime": "2024-09-02 06:00:00", "value": 9},
    {"datetime": "2024-09-02 07:00:00", "value": 10},
    {"datetime": "2024-09-02 08:00:00", "value": 15},
    {"datetime": "2024-09-02 09:00:00", "value": 20},
    {"datetime": "2024-09-02 10:00:00", "value": 35},
    {"datetime": "2024-09-02 11:00:00", "value": 45},
    {"datetime": "2024-09-02 12:00:00", "value": 50},
    {"datetime": "2024-09-02 13:00:00", "value": 48},
    {"datetime": "2024-09-02 14:00:00", "value": 40},
    {"datetime": "2024-09-02 15:00:00", "value": 38},
    {"datetime": "2024-09-02 16:00:00", "value": 42},
    {"datetime": "2024-09-02 17:00:00", "value": 50},
    {"datetime": "2024-09-02 18:00:00", "value": 60},
    {"datetime": "2024-09-02 19:00:00", "value": 65},
    {"datetime": "2024-09-02 20:00:00", "value": 68},
    {"datetime": "2024-09-02 21:00:00", "value": 55},
    {"datetime": "2024-09-02 22:00:00", "value": 30},
    {"datetime": "2024-09-02 23:00:00", "value": 15},
    {"datetime": "2024-09-03 00:00:00", "value": 5},
    {"datetime": "2024-09-03 01:00:00", "value": 10},
    {"datetime": "2024-09-03 02:00:00", "value": 8},
    {"datetime": "2024-09-03 03:00:00", "value": 8},
    {"datetime": "2024-09-03 04:00:00", "value": 9},
    {"datetime": "2024-09-03 05:00:00", "value": 7},
    {"datetime": "2024-09-03 06:00:00", "value": 9},
    {"datetime": "2024-09-03 07:00:00", "value": 10},
    {"datetime": "2024-09-03 08:00:00", "value": 15},
    {"datetime": "2024-09-03 09:00:00", "value": 20},
    {"datetime": "2024-09-03 10:00:00", "value": 35},
    {"datetime": "2024-09-03 11:00:00", "value": 45},
    {"datetime": "2024-09-03 12:00:00", "value": 50},
    {"datetime": "2024-09-03 13:00:00", "value": 48},
    {"datetime": "2024-09-03 14:00:00", "value": 40},
    {"datetime": "2024-09-03 15:00:00", "value": 38},
    {"datetime": "2024-09-03 16:00:00", "value": 42},
    {"datetime": "2024-09-03 17:00:00", "value": 50},
    {"datetime": "2024-09-03 18:00:00", "value": 60},
    {"datetime": "2024-09-03 19:00:00", "value": 65},
    {"datetime": "2024-09-03 20:00:00", "value": 68},
    {"datetime": "2024-09-03 21:00:00", "value": 55},
    {"datetime": "2024-09-03 22:00:00", "value": 30},
    {"datetime": "2024-09-03 23:00:00", "value": 15},
    {"datetime": "2024-09-04 00:00:00", "value": 5},
    {"datetime": "2024-09-04 01:00:00", "value": 10},
    {"datetime": "2024-09-04 02:00:00", "value": 8},
    {"datetime": "2024-09-04 03:00:00", "value": 8},
    {"datetime": "2024-09-04 04:00:00", "value": 9},
    {"datetime": "2024-09-04 05:00:00", "value": 7},
    {"datetime": "2024-09-04 06:00:00", "value": 9},
    {"datetime": "2024-09-04 07:00:00", "value": 10},
    {"datetime": "2024-09-04 08:00:00", "value": 15},
    {"datetime": "2024-09-04 09:00:00", "value": 20},
    {"datetime": "2024-09-04 10:00:00", "value": 35},
    {"datetime": "2024-09-04 11:00:00", "value": 45},
    {"datetime": "2024-09-04 12:00:00", "value": 50},
    {"datetime": "2024-09-04 13:00:00", "value": 48},
    {"datetime": "2024-09-04 14:00:00", "value": 40},
    {"datetime": "2024-09-04 15:00:00", "value": 38},
    {"datetime": "2024-09-04 16:00:00", "value": 42},
    {"datetime": "2024-09-04 17:00:00", "value": 50},
    {"datetime": "2024-09-04 18:00:00", "value": 60},
    {"datetime": "2024-09-04 19:00:00", "value": 65},
    {"datetime": "2024-09-04 20:00:00", "value": 68},
    {"datetime": "2024-09-04 21:00:00", "value": 55},
    {"datetime": "2024-09-04 22:00:00", "value": 30},
    {"datetime": "2024-09-04 23:00:00", "value": 15},
    {"datetime": "2024-09-05 00:00:00", "value": 5},
    {"datetime": "2024-09-05 01:00:00", "value": 10},
    {"datetime": "2024-09-05 02:00:00", "value": 8},
    {"datetime": "2024-09-05 03:00:00", "value": 8},
    {"datetime": "2024-09-05 04:00:00", "value": 9},
    {"datetime": "2024-09-05 05:00:00", "value": 7},
    {"datetime": "2024-09-05 06:00:00", "value": 9},
    {"datetime": "2024-09-05 07:00:00", "value": 10},
    {"datetime": "2024-09-05 08:00:00", "value": 15},
    {"datetime": "2024-09-05 09:00:00", "value": 20},
    {"datetime": "2024-09-05 10:00:00", "value": 35},
    {"datetime": "2024-09-05 11:00:00", "value": 45},
    {"datetime": "2024-09-05 12:00:00", "value": 50},
    {"datetime": "2024-09-05 13:00:00", "value": 48},
    {"datetime": "2024-09-05 14:00:00", "value": 40},
    {"datetime": "2024-09-05 15:00:00", "value": 38},
    {"datetime": "2024-09-05 16:00:00", "value": 42},
    {"datetime": "2024-09-05 17:00:00", "value": 50},
    {"datetime": "2024-09-05 18:00:00", "value": 60},
    {"datetime": "2024-09-05 19:00:00", "value": 65},
    {"datetime": "2024-09-05 20:00:00", "value": 68},
    {"datetime": "2024-09-05 21:00:00", "value": 55},
    {"datetime": "2024-09-05 22:00:00", "value": 30},
    {"datetime": "2024-09-05 23:00:00", "value": 15},
]


# Endpoint to return the sample forecast
@router.get("/sample-forecast/", response_model=list[VisitorForecastBase])
def get_sample_forecast():
    return sample_forecast
