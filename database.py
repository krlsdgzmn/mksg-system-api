import os

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Production
load_dotenv(dotenv_path=".env.local")
DATABASE_URL = str(os.environ.get("POSTGRES_URL"))
engine = create_engine(DATABASE_URL)

# Dev Mode
# URL_DATABASE = "sqlite:///database.db"
# engine = create_engine(URL_DATABASE, connect_args={"check_same_thread": False})

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()
