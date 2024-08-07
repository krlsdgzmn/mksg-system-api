from database import Base
from sqlalchemy import Column, Integer, String


class OrderStatus(Base):
    __tablename__ = "order_status"

    id = Column(Integer, primary_key=True)
    price_bin = Column(String)
    discount_bin = Column(String)
    month = Column(String)
    week = Column(Integer)
    distance_bin = Column(String)
    cancel_rate = Column(Integer)
    order_status = Column(String)
