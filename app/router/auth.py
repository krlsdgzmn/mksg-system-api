from datetime import datetime, timedelta, timezone
import os

from dotenv import load_dotenv
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel
from sqlalchemy.orm import Session

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
    first_name: str
    last_name: str
    role: str


class UserResponse(BaseModel):
    id: int
    username: str
    first_name: str
    last_name: str
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
    db_user = User(
        username=user.username,
        password=hashed_password,
        first_name=user.first_name,
        last_name=user.last_name,
        role=user.role,
    )
    db.add(db_user)
    db.commit()
    return {"message": f"User {user.username} has been created."}


# Helper function to authenticate user credentials
def authenticate_user(username: str, password: str, db: Session) -> UserBase | bool:
    """
    Authenticate the user by verifying the username and password.
    """
    user = get_user(db=db, username=username)
    if not user or not bcrypt_context.verify(password, str(user.password)):
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


@router.post("/auth/token/", response_model=TokenResponse)
def login_for_access_token(
    form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)
):
    """
    Authenticate the user and return the access token along with user data.
    """
    user = authenticate_user(form_data.username, form_data.password, db)
    if not isinstance(user, User):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXP_MINS)
    access_token = create_access_token(
        data={
            "user": {
                "id": user.id,
                "username": user.username,
                "first_name": user.first_name,
                "last_name": user.last_name,
                "role": user.role,
            },
        },
        expires_delta=access_token_expires,
    )
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": {
            "id": user.id,
            "username": user.username,
            "first_name": user.first_name,
            "last_name": user.last_name,
            "role": user.role,
        },
    }


# Helper function to verify and decode JWT token
def verify_token(token: str = Depends(oauth2_bearer)) -> dict:
    """
    Verify the JWT token and decode its payload.
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("user")
        if not username:
            raise HTTPException(status_code=403, detail="Token is invalid or expired")
        return payload
    except JWTError:
        raise HTTPException(status_code=403, detail="Token is invalid or expired")


@router.get("/auth/verify-token/{token}")
async def verify_user_token(token: str):
    return verify_token(token=token)


# Endpoint to update user by id
@router.put("/auth/user/{id}/", response_model=UserResponse)
async def update_user(id: int, updated_user: UserBase, db: Session = Depends(get_db)):
    """
    Update a user's details by their ID.
    """
    try:
        # Retrieve the existing user by ID
        prev_user = db.query(User).filter(User.id == id).first()
        if not prev_user:
            raise HTTPException(status_code=404, detail=f"User {id} not found")

        # Hash the password if it was updated
        if updated_user.password:
            hashed_password = bcrypt_context.hash(updated_user.password)
        else:
            hashed_password = prev_user.password

        # Update the user's details
        # prev_user.username = updated_user.username
        prev_user.password = hashed_password
        prev_user.first_name = updated_user.first_name
        prev_user.last_name = updated_user.last_name
        prev_user.role = updated_user.role

        # Commit the changes to the database
        db.commit()
        db.refresh(prev_user)

        # Return the updated user data
        return UserResponse(
            id=prev_user.id,
            username=prev_user.username,
            first_name=prev_user.first_name,
            last_name=prev_user.last_name,
            role=prev_user.role,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
