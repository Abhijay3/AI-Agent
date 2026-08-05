from pydantic import BaseModel, EmailStr, Field


class ChatRequest(BaseModel):
    session_id: str = Field(..., min_length=1, max_length=100, pattern=r"^[A-Za-z0-9_-]+$")
    message: str = Field(..., min_length=1, max_length=2000)


class SignupRequest(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=8, max_length=128)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=1, max_length=128)


class AuthResponse(BaseModel):
    token: str
    user_id: str
    email: str


class MeResponse(BaseModel):
    user_id: str
    email: str


class ChatResponse(BaseModel):
    reply: str


class UploadResponse(BaseModel):
    filename: str


class HistoryMessage(BaseModel):
    role: str
    content: str


class HistoryResponse(BaseModel):
    messages: list[HistoryMessage]


class MemoryItem(BaseModel):
    key: str
    value: str


class MemoriesResponse(BaseModel):
    memories: list[MemoryItem]


class TicketItem(BaseModel):
    id: int
    name: str
    email: str
    subject: str
    description: str
    status: str
    created_at: str


class TicketsResponse(BaseModel):
    tickets: list[TicketItem]


class TicketStatusUpdate(BaseModel):
    status: str = Field(..., pattern=r"^(open|resolved)$")
