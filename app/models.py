from app.database import Base
from sqlalchemy import Column, Integer, String


class OrderForecast(Base):
    __tablename__ = "order_forecast"

    id = Column(Integer, primary_key=True)
    price_bin = Column(String)
    discount_bin = Column(String)
    month = Column(String)
    week = Column(Integer)
    distance_bin = Column(String)
    cancel_rate = Column(Integer)
    order_status = Column(String)


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True)
    username = Column(String, unique=True)
    password = Column(String)
    first_name = Column(String)
    last_name = Column(String)
    role = Column(String)
