from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.database import SessionLocal
from app.models import User

router = APIRouter(prefix="/api", tags=["user"])


class UserResponse(BaseModel):
    id: int
    username: str
    first_name: str
    last_name: str
    role: str


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


# Helper function to get user by id
def get_user_by_id(id: int, db: Session):
    return db.query(User).filter(User.id == id).first()


# Endpoint to get all users
@router.get("/user/", response_model=list[UserResponse])
async def read_users(db: Session = Depends(get_db)):
    try:
        return db.query(User).all()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Endpoint to get user by id
@router.get("/user/{id}/", response_model=UserResponse)
async def read_user_by_id(id: int, db: Session = Depends(get_db)):
    try:
        user = db.query(User).filter(User.id == id).first()
        if user is None:
            raise HTTPException(status_code=404, detail="User not found")
        return user
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Endpoint to delete user by id
@router.delete("/user/{id}/")
async def delete_user(id: int, db: Session = Depends(get_db)):
    try:
        user = get_user_by_id(id, db)
        if user == 0:
            raise HTTPException(status_code=404, detail=f"User {id} not found")

        db.delete(user)
        db.commit()

        return {"message": f"User {id} has been successfully deleted"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
