"""
Authentication endpoints for Nautilus API.
"""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from jules_api.app.schemas.user import UserCreate, UserLogin, Token
from jules_api.app.db.session import get_db
from jules_api.app.db.models.user import User as DBUser
from jules_api.app.core.security import verify_password, get_password_hash, create_access_token
from jules_api.app.core.config import settings
from datetime import timedelta
import logging

router = APIRouter()
logger = logging.getLogger("auth")

@router.post("/signup", response_model=Token, summary="Register a new user", tags=["auth"])
def signup(user_in: UserCreate, db: Session = Depends(get_db)):
    """
    Register a new user.
    """
    user = db.query(DBUser).filter(DBUser.email == user_in.email).first()
    if user:
        logger.warning(f"Signup attempt with existing email: {user_in.email}")
        raise HTTPException(status_code=400, detail="Email already registered")
    hashed_password = get_password_hash(user_in.password)
    db_user = DBUser(email=user_in.email, hashed_password=hashed_password, full_name=user_in.full_name)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    access_token = create_access_token(subject=db_user.email)
    logger.info(f"User registered: {user_in.email}")
    return {"access_token": access_token, "token_type": "bearer"}

@router.post("/login", response_model=Token, summary="User login", tags=["auth"])
def login(user_in: UserLogin, db: Session = Depends(get_db)):
    """
    Authenticate user and return JWT token.
    """
    user = db.query(DBUser).filter(DBUser.email == user_in.email).first()
    if not user or not verify_password(user_in.password, user.hashed_password):
        logger.warning(f"Failed login attempt for: {user_in.email}")
        raise HTTPException(status_code=401, detail="Incorrect email or password")
    access_token = create_access_token(subject=user.email, expires_delta=timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES))
    logger.info(f"User logged in: {user_in.email}")
    return {"access_token": access_token, "token_type": "bearer"}

# Additional endpoints for email verification, password change, etc. would be added here.
