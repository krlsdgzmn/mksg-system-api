import re
from datetime import datetime, timezone
from io import BytesIO, StringIO

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from prophet import Prophet
from pydantic import BaseModel
from pytz import timezone
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.database import SessionLocal
from app.models import VisitorActual, VisitorForecast

router = APIRouter(prefix="/api", tags=["visitor-forecast"])


class VisitorForecastBase(BaseModel):
    ds: datetime
    yhat: int
    yhat_upper: int
    yhat_lower: int


class VisitorActualBase(BaseModel):
    date: datetime
    page_views: int
    log: float


# Database dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Helper function to remove comma and quotation
def remove_comma_and_quotation(df, column_name):
    pattern = r'[",]'
    df[column_name] = df[column_name].apply(lambda x: re.sub(pattern, "", x))
    return df


# Helper fucntion to preprocess yesterday's data
def preprocess(yesterday_df) -> pd.DataFrame:
    yesterday_df = yesterday_df[["Date", "Page Views"]]
    yesterday_df.columns = ["date", "page_views"]
    yesterday_df["date"] = pd.to_datetime(yesterday_df["date"], format="%d/%m/%Y %H:%M")
    yesterday_df["page_views"] = yesterday_df["page_views"].astype(str)
    yesterday_df = remove_comma_and_quotation(yesterday_df, "page_views")
    yesterday_df["page_views"] = yesterday_df["page_views"].astype(int)
    yesterday_df = yesterday_df.set_index("date")
    yesterday_df["log"] = np.log1p(yesterday_df["page_views"])
    return yesterday_df


# Helper function to add features for training
def add_features(df) -> pd.DataFrame:
    # Add lag_1, lag_2, lag_24
    df["lag_1"] = df["y"].shift(1)
    df["lag_2"] = df["y"].shift(2)
    df["lag_24"] = df["y"].shift(24)

    # Apply backward fill to remove NaN values
    df["lag_1"] = df["lag_1"].bfill()
    df["lag_2"] = df["lag_2"].bfill()
    df["lag_24"] = df["lag_24"].bfill()

    # Add other regressors: hour, day_of_week, month
    df["hour"] = df["ds"].dt.hour
    df["day_of_week"] = df["ds"].dt.day_of_week
    df["month"] = df["ds"].dt.month

    return df


# Helper function add feature lags in future DataFrame
def populate_lags(model, future, initial_data, periods=24, freq="h"):
    # Create future dataframe
    future = model.make_future_dataframe(
        periods=periods, freq=freq, include_history=False
    )

    # Initialize lag columns with the last known values
    future["lag_1"] = initial_data["y"].iloc[-1]
    future["lag_2"] = initial_data["y"].iloc[-2]
    future["lag_24"] = initial_data["y"].iloc[-24]

    # Add other regressors: hour, day_of_week, month
    future["hour"] = future["ds"].dt.hour
    future["day_of_week"] = future["ds"].dt.day_of_week
    future["month"] = future["ds"].dt.month

    # Iteratively predict and update lag values
    for i in range(len(future)):
        # Make prediction for current row
        forecast = model.predict(future.iloc[[i]])
        yhat = forecast["yhat"].values[0]

        # Update lag values for next iterations
        if i + 1 < len(future):
            future.loc[future.index[i + 1], "lag_1"] = yhat
        if i + 2 < len(future):
            future.loc[future.index[i + 2], "lag_2"] = yhat
        if i + 24 < len(future):
            future.loc[future.index[i + 24], "lag_24"] = yhat

    return future


# Endpoint to import and retrain the model forecast
@router.post("/visitor-forecast")
async def import_and_retrain_forecast(
    file: UploadFile = File(...), db: Session = Depends(get_db)
):
    try:
        # Read the uploaded Excel file directly as a binary file
        xlsx_file = await file.read()
        yesterday_df = pd.read_excel(
            BytesIO(xlsx_file),
            skiprows=3,
            sheet_name="All",
            engine="openpyxl",
        )

        clean_df = preprocess(yesterday_df)
        clean_df = clean_df.reset_index()

        # Check the latest date in VisitorActual
        latest_entry = (
            db.query(VisitorActual).order_by(VisitorActual.date.desc()).first()
        )

        if latest_entry:
            latest_date = pd.to_datetime(
                latest_entry.date
            )  # Keep the timestamp (date + time)
            uploaded_dates = pd.to_datetime(
                clean_df["date"]
            )  # Ensure timestamps include hours

            # Check if the earliest uploaded date is directly after the latest date in the DB
            expected_next_date = latest_date + pd.Timedelta(hours=1)
            if uploaded_dates.min() != expected_next_date:
                raise HTTPException(
                    status_code=400,
                    detail=f"Uploaded date {uploaded_dates.min()} is not sequential. The next expected date is {expected_next_date}.",
                )

        # Proceed to upload the data (now with the full timestamp)
        for _, row in clean_df.iterrows():
            date = pd.to_datetime(row["date"])  # Store full timestamp
            page_views = row["page_views"]
            log = row["log"]

            visitor_actual = VisitorActual(date=date, page_views=page_views, log=log)
            db.add(visitor_actual)

        db.commit()

        # Retrieve data from VisitorActual to create the training DataFrame
        visitor_data = db.query(VisitorActual).all()
        traffic_df = pd.DataFrame(
            [(v.date, v.page_views, v.log) for v in visitor_data],
            columns=["ds", "page_views", "y"],
        )

        traffic_df = add_features(traffic_df)

        # Initialize the Prophet model
        prophet = Prophet()

        # Add the regressors
        prophet.add_regressor("lag_1")
        prophet.add_regressor("lag_2")
        prophet.add_regressor("lag_24")
        prophet.add_regressor("hour")
        prophet.add_regressor("day_of_week")
        prophet.add_regressor("month")

        # Fit the model
        prophet.fit(traffic_df)

        future_df = prophet.make_future_dataframe(
            periods=24, include_history=False, freq="h"
        )
        future_df = populate_lags(prophet, future_df, traffic_df, periods=24, freq="h")
        forecast_df = prophet.predict(future_df)
        forecast_df["yhat"] = np.expm1(forecast_df["yhat"]).astype(int)
        forecast_df["yhat_upper"] = np.expm1(forecast_df["yhat_upper"]).astype(int)
        forecast_df["yhat_lower"] = np.expm1(forecast_df["yhat_lower"]).astype(int)
        forecast_df = forecast_df[["ds", "yhat", "yhat_upper", "yhat_lower"]]

        for _, row in forecast_df.iterrows():
            ds = pd.to_datetime(
                row["ds"]
            )  # Ensure the full timestamp (with hours) is saved
            yhat = row["yhat"]
            yhat_upper = row["yhat_upper"]
            yhat_lower = row["yhat_lower"]

            visitor_forecast = VisitorForecast(
                ds=ds, yhat=yhat, yhat_upper=yhat_upper, yhat_lower=yhat_lower
            )

            db.add(visitor_forecast)

        db.commit()

        return {"data": "Data imported and retrained the model successfully."}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))


# Endpoint to import existing merged csv file
@router.post("/visitor-actual")
async def import_merged_data(
    file: UploadFile = File(...), db: Session = Depends(get_db)
):
    try:
        csv_file = await file.read()
        df = pd.read_csv(StringIO(csv_file.decode("utf-8")))

        if not all(col in df.columns for col in ["date", "page_views", "log"]):
            raise HTTPException(
                status_code=400,
                detail="CSV must contain date, page_views, and log columns.",
            )

        for _, row in df.iterrows():
            date = pd.to_datetime(row["date"])
            page_views = row["page_views"]
            log = row["log"]

            visitor_actual = VisitorActual(date=date, page_views=page_views, log=log)
            db.add(visitor_actual)

        db.commit()
        return {"detail": "Data imported successfully."}

    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))


# Endpoint to get the visitor
@router.get("/visitor-forecast", response_model=list[VisitorForecastBase])
def get_today_forecast(db: Session = Depends(get_db)):
    philippine_timezone = timezone("Asia/Manila")
    today = datetime.now(philippine_timezone).date()
    print(today)

    forecast_data = (
        db.query(VisitorForecast)
        .filter(
            func.date(VisitorForecast.ds) == today,
        )
        .order_by(VisitorForecast.ds)
        .all()
    )

    if not forecast_data:
        raise HTTPException(status_code=404, detail="No forecast data found for today")

    return forecast_data


# Endpoint to read the actual page_views
@router.get("/visitor-actual")
async def read_visitor_actual(db: Session = Depends(get_db)):
    try:
        query = db.query(VisitorActual).order_by(VisitorActual.date.desc()).first()

        return {"date": query.date.date()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
