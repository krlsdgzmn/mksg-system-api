from app.database import Base
from sqlalchemy import Column, Integer, String, DateTime, Float


class OrderPrediction(Base):
    __tablename__ = "order_prediction"

    id = Column(Integer, primary_key=True)
    price_bin = Column(String)
    discount_bin = Column(String)
    month = Column(String)
    week = Column(Integer)
    distance_bin = Column(String)
    cancel_rate = Column(Integer)
    order_status = Column(String)


class VisitorForecast(Base):
    __tablename__ = "visitor_forecast"

    ds = Column(DateTime, primary_key=True, unique=True)
    yhat = Column(Integer)
    yhat_upper = Column(Integer)
    yhat_lower = Column(Integer)


class VisitorActual(Base):
    __tablename__ = "visitor_actual"

    date = Column(DateTime, primary_key=True, unique=True)
    page_views = Column(Integer)
    log = Column(Float)
