from datetime import datetime, timedelta, timezone
import os

from dotenv import load_dotenv
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel
from sqlalchemy.orm import Session
from starlette.status import HTTP_200_OK

from app.database import SessionLocal
from app.models import User

router = APIRouter(prefix="/api", tags=["auth"])

# Load environment variables
load_dotenv()
SECRET_KEY = str(os.getenv("AUTH_KEY"))
ALGORITHM = str(os.getenv("AUTH_ALGO"))
ACCESS_TOKEN_EXP_MINS = 30

# Initialize the password hashing context
bcrypt_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# OAuth2 bearer token scheme
oauth2_bearer = OAuth2PasswordBearer(tokenUrl="auth/token")


# Pydantic model for user registration and response
class UserBase(BaseModel):
    username: str
    password: str
    role: str


class UserResponse(BaseModel):
    id: int
    username: str
    role: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str
    user: UserResponse


# Database dependency
def get_db():
    """
    Provide a database session to endpoints.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Helper function to retrieve user by username
def get_user(db: Session, username: str) -> User | None:
    """
    Retrieve a user from the database by username.
    """
    return db.query(User).filter(User.username == username).first()


@router.post("/auth/register/")
async def register_user(user: UserBase, db: Session = Depends(get_db)):
    """
    Register a new user.
    """
    # Check if the user already exists
    db_user = get_user(db=db, username=user.username)
    if db_user:
        raise HTTPException(status_code=400, detail="User already exists.")

    # Hash the user's password and save the new user to the database
    hashed_password = bcrypt_context.hash(user.password)
    db_user = User(username=user.username, password=hashed_password, role=user.role)
    db.add(db_user)
    db.commit()
    return {"message": f"User {user.username} has been created."}


# Helper function to authenticate user credentials
def authenticate_user(username: str, password: str, db: Session) -> User | bool:
    """
    Authenticate the user by verifying the username and password.
    """
    user = get_user(db=db, username=username)
    if not user or not bcrypt_context.verify(password, user.password):
        return False
    return user


# Helper function to create access token
def create_access_token(data: dict, expires_delta: timedelta | None = None) -> str:
    """
    Create a JWT access token.
    """
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (
        expires_delta or timedelta(minutes=ACCESS_TOKEN_EXP_MINS)
    )
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


@router.post("/auth/token", response_model=TokenResponse)
def login_for_access_token(
    form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)
):
    """
    Authenticate the user and return the access token along with user data.
    """
    user = authenticate_user(form_data.username, form_data.password, db)
    if not isinstance(user, User):  # Ensure user is a User instance
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXP_MINS)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": {"id": user.id, "username": user.username, "role": user.role},
    }


# Helper function to verify and decode JWT token
def verify_token(token: str = Depends(oauth2_bearer)) -> dict:
    """
    Verify the JWT token and decode its payload.
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        if not username:
            raise HTTPException(status_code=403, detail="Token is invalid or expired")
        return payload
    except JWTError:
        raise HTTPException(status_code=403, detail="Token is invalid or expired")


@router.get("/auth/verify-token/{token}")
async def verify_user_token(token: str):
    verify_token(token=token)
    return HTTP_200_OK