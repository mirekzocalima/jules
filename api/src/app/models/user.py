"""
Pydantic user model for dependency injection and token payload.
"""
from pydantic import BaseModel, EmailStr
from typing import Optional

class TokenPayload(BaseModel):
    sub: Optional[str] = None
    exp: Optional[int] = None

class User(BaseModel):
    id: int
    email: EmailStr
    full_name: str
    is_active: bool
    is_verified: bool

    class Config:
        orm_mode = True
