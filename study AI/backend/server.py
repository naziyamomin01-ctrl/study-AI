from fastapi import FastAPI, APIRouter, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict, EmailStr
from typing import List, Optional
import uuid
from datetime import datetime, timezone, timedelta
import bcrypt
import jwt
from emergentintegrations.llm.chat import LlmChat, UserMessage

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# JWT Settings
JWT_SECRET = os.environ.get('JWT_SECRET', 'your-secret-key-change-in-production')
JWT_ALGORITHM = 'HS256'
JWT_EXPIRATION_DAYS = 30

# AI Settings
EMERGENT_LLM_KEY = os.environ.get('EMERGENT_LLM_KEY')

# Create the main app without a prefix
app = FastAPI()

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")

# Security
security = HTTPBearer()

# Models
class UserRegister(BaseModel):
    email: EmailStr
    password: str
    name: str

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class User(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    email: EmailStr
    name: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class StudyMaterial(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    title: str
    content: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class Flashcard(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    material_id: Optional[str] = None
    front: str
    back: str
    next_review: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    interval: int = 1
    ease_factor: float = 2.5
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class Quiz(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    material_id: Optional[str] = None
    title: str
    questions: List[dict]
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class QuizResult(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    quiz_id: str
    answers: List[dict]
    score: float
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class Note(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    material_id: Optional[str] = None
    title: str
    content: str
    summary: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    role: str
    content: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class StudySession(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    duration: int
    focus_mode: bool = False
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class Progress(BaseModel):
    model_config = ConfigDict(extra="ignore")
    user_id: str
    total_study_time: int = 0
    flashcards_reviewed: int = 0
    quizzes_completed: int = 0
    current_streak: int = 0
    longest_streak: int = 0
    last_study_date: Optional[datetime] = None
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

# Helper Functions
def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def verify_password(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))

def create_token(user_id: str) -> str:
    expiration = datetime.now(timezone.utc) + timedelta(days=JWT_EXPIRATION_DAYS)
    return jwt.encode({'user_id': user_id, 'exp': expiration}, JWT_SECRET, algorithm=JWT_ALGORITHM)

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        token = credentials.credentials
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        user_id = payload.get('user_id')
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid token")
        return user_id
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

# Auth Routes
@api_router.post("/auth/register")
async def register(user_data: UserRegister):
    existing_user = await db.users.find_one({"email": user_data.email}, {"_id": 0})
    if existing_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    user = User(email=user_data.email, name=user_data.name)
    user_dict = user.model_dump()
    user_dict['created_at'] = user_dict['created_at'].isoformat()
    user_dict['password'] = hash_password(user_data.password)
    
    await db.users.insert_one(user_dict)
    
    progress = Progress(user_id=user.id)
    progress_dict = progress.model_dump()
    progress_dict['updated_at'] = progress_dict['updated_at'].isoformat()
    if progress_dict.get('last_study_date'):
        progress_dict['last_study_date'] = progress_dict['last_study_date'].isoformat()
    await db.progress.insert_one(progress_dict)
    
    token = create_token(user.id)
    return {"token": token, "user": {"id": user.id, "email": user.email, "name": user.name}}

@api_router.post("/auth/login")
async def login(credentials: UserLogin):
    user = await db.users.find_one({"email": credentials.email}, {"_id": 0})
    if not user or not verify_password(credentials.password, user['password']):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    token = create_token(user['id'])
    return {"token": token, "user": {"id": user['id'], "email": user['email'], "name": user['name']}}

# Study Materials Routes
@api_router.post("/materials")
async def create_material(title: str, content: str, user_id: str = Depends(get_current_user)):
    material = StudyMaterial(user_id=user_id, title=title, content=content)
    material_dict = material.model_dump()
    material_dict['created_at'] = material_dict['created_at'].isoformat()
    await db.materials.insert_one(material_dict)
    return material

@api_router.get("/materials")
async def get_materials(user_id: str = Depends(get_current_user)):
    materials = await db.materials.find({"user_id": user_id}, {"_id": 0}).to_list(1000)
    for mat in materials:
        if isinstance(mat['created_at'], str):
            mat['created_at'] = datetime.fromisoformat(mat['created_at'])
    return materials

@api_router.get("/materials/{material_id}")
async def get_material(material_id: str, user_id: str = Depends(get_current_user)):
    material = await db.materials.find_one({"id": material_id, "user_id": user_id}, {"_id": 0})
    if not material:
        raise HTTPException(status_code=404, detail="Material not found")
    return material

# Flashcards Routes
@api_router.post("/flashcards/generate")
async def generate_flashcards(material_id: str, user_id: str = Depends(get_current_user)):
    material = await db.materials.find_one({"id": material_id, "user_id": user_id}, {"_id": 0})
    if not material:
        raise HTTPException(status_code=404, detail="Material not found")
    
    chat = LlmChat(
        api_key=EMERGENT_LLM_KEY,
        session_id=f"flashcards_{user_id}_{material_id}",
        system_message="You are a helpful study assistant. Generate flashcards from the provided content."
    ).with_model("openai", "gpt-5.1")
    
    prompt = f"""Generate 5-10 flashcards from this content. Return ONLY a JSON array with objects containing 'front' and 'back' fields.

Content:
{material['content']}

Example format:
[
  {{"front": "Question or concept", "back": "Answer or explanation"}},
  {{"front": "Question 2", "back": "Answer 2"}}
]"""
    
    message = UserMessage(text=prompt)
    response = await chat.send_message(message)
    
    import json
    try:
        response_text = response.strip()
        if response_text.startswith('```json'):
            response_text = response_text[7:]
        if response_text.startswith('```'):
            response_text = response_text[3:]
        if response_text.endswith('```'):
            response_text = response_text[:-3]
        
        flashcard_data = json.loads(response_text.strip())
        flashcards = []
        
        for card_data in flashcard_data:
            flashcard = Flashcard(
                user_id=user_id,
                material_id=material_id,
                front=card_data['front'],
                back=card_data['back']
            )
            flashcard_dict = flashcard.model_dump()
            flashcard_dict['created_at'] = flashcard_dict['created_at'].isoformat()
            flashcard_dict['next_review'] = flashcard_dict['next_review'].isoformat()
            await db.flashcards.insert_one(flashcard_dict)
            flashcards.append(flashcard)
        
        return flashcards
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating flashcards: {str(e)}")

@api_router.get("/flashcards")
async def get_flashcards(user_id: str = Depends(get_current_user)):
    flashcards = await db.flashcards.find({"user_id": user_id}, {"_id": 0}).to_list(1000)
    for card in flashcards:
        if isinstance(card['created_at'], str):
            card['created_at'] = datetime.fromisoformat(card['created_at'])
        if isinstance(card['next_review'], str):
            card['next_review'] = datetime.fromisoformat(card['next_review'])
    return flashcards

@api_router.post("/flashcards/{flashcard_id}/review")
async def review_flashcard(flashcard_id: str, quality: int, user_id: str = Depends(get_current_user)):
    flashcard = await db.flashcards.find_one({"id": flashcard_id, "user_id": user_id}, {"_id": 0})
    if not flashcard:
        raise HTTPException(status_code=404, detail="Flashcard not found")
    
    # Spaced repetition algorithm (SM-2)
    if quality >= 3:
        if flashcard['interval'] == 1:
            interval = 6
        else:
            interval = flashcard['interval'] * flashcard['ease_factor']
    else:
        interval = 1
    
    ease_factor = flashcard['ease_factor'] + (0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02))
    ease_factor = max(1.3, ease_factor)
    
    next_review = datetime.now(timezone.utc) + timedelta(days=int(interval))
    
    await db.flashcards.update_one(
        {"id": flashcard_id},
        {"$set": {
            "interval": int(interval),
            "ease_factor": ease_factor,
            "next_review": next_review.isoformat()
        }}
    )
    
    await update_progress(user_id, flashcards_reviewed=1)
    
    return {"message": "Review recorded", "next_review": next_review}

# Quiz Routes
@api_router.post("/quizzes/generate")
async def generate_quiz(material_id: str, user_id: str = Depends(get_current_user)):
    material = await db.materials.find_one({"id": material_id, "user_id": user_id}, {"_id": 0})
    if not material:
        raise HTTPException(status_code=404, detail="Material not found")
    
    chat = LlmChat(
        api_key=EMERGENT_LLM_KEY,
        session_id=f"quiz_{user_id}_{material_id}",
        system_message="You are a helpful study assistant. Generate quiz questions from the provided content."
    ).with_model("openai", "gpt-5.1")
    
    prompt = f"""Generate 5 multiple choice questions from this content. Return ONLY a JSON array with objects containing 'question', 'options' (array of 4 strings), and 'correct_index' (0-3).

Content:
{material['content']}

Example format:
[
  {{"question": "What is...", "options": ["A", "B", "C", "D"], "correct_index": 0}}
]"""
    
    message = UserMessage(text=prompt)
    response = await chat.send_message(message)
    
    import json
    try:
        response_text = response.strip()
        if response_text.startswith('```json'):
            response_text = response_text[7:]
        if response_text.startswith('```'):
            response_text = response_text[3:]
        if response_text.endswith('```'):
            response_text = response_text[:-3]
        
        questions = json.loads(response_text.strip())
        
        quiz = Quiz(
            user_id=user_id,
            material_id=material_id,
            title=f"Quiz for {material['title']}",
            questions=questions
        )
        quiz_dict = quiz.model_dump()
        quiz_dict['created_at'] = quiz_dict['created_at'].isoformat()
        await db.quizzes.insert_one(quiz_dict)
        
        return quiz
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating quiz: {str(e)}")

@api_router.get("/quizzes")
async def get_quizzes(user_id: str = Depends(get_current_user)):
    quizzes = await db.quizzes.find({"user_id": user_id}, {"_id": 0}).to_list(1000)
    for quiz in quizzes:
        if isinstance(quiz['created_at'], str):
            quiz['created_at'] = datetime.fromisoformat(quiz['created_at'])
    return quizzes

@api_router.post("/quizzes/{quiz_id}/submit")
async def submit_quiz(quiz_id: str, answers: List[int], user_id: str = Depends(get_current_user)):
    quiz = await db.quizzes.find_one({"id": quiz_id, "user_id": user_id}, {"_id": 0})
    if not quiz:
        raise HTTPException(status_code=404, detail="Quiz not found")
    
    correct = 0
    results = []
    for i, answer in enumerate(answers):
        is_correct = answer == quiz['questions'][i]['correct_index']
        if is_correct:
            correct += 1
        results.append({"question_index": i, "user_answer": answer, "correct": is_correct})
    
    score = (correct / len(quiz['questions'])) * 100
    
    quiz_result = QuizResult(
        user_id=user_id,
        quiz_id=quiz_id,
        answers=results,
        score=score
    )
    result_dict = quiz_result.model_dump()
    result_dict['created_at'] = result_dict['created_at'].isoformat()
    await db.quiz_results.insert_one(result_dict)
    
    await update_progress(user_id, quizzes_completed=1)
    
    return {"score": score, "results": results}

# Notes Routes
@api_router.post("/notes")
async def create_note(title: str, content: str, user_id: str = Depends(get_current_user)):
    note = Note(user_id=user_id, title=title, content=content)
    note_dict = note.model_dump()
    note_dict['created_at'] = note_dict['created_at'].isoformat()
    await db.notes.insert_one(note_dict)
    return note

@api_router.post("/notes/summarize")
async def summarize_note(material_id: str, user_id: str = Depends(get_current_user)):
    material = await db.materials.find_one({"id": material_id, "user_id": user_id}, {"_id": 0})
    if not material:
        raise HTTPException(status_code=404, detail="Material not found")
    
    chat = LlmChat(
        api_key=EMERGENT_LLM_KEY,
        session_id=f"summarize_{user_id}_{material_id}",
        system_message="You are a helpful study assistant. Summarize content concisely."
    ).with_model("openai", "gpt-5.1")
    
    prompt = f"""Summarize the following study material in a clear, concise way. Focus on key concepts and important points.

Content:
{material['content']}"""
    
    message = UserMessage(text=prompt)
    response = await chat.send_message(message)
    
    note = Note(
        user_id=user_id,
        material_id=material_id,
        title=f"Summary: {material['title']}",
        content=material['content'],
        summary=response
    )
    note_dict = note.model_dump()
    note_dict['created_at'] = note_dict['created_at'].isoformat()
    await db.notes.insert_one(note_dict)
    
    return note

@api_router.get("/notes")
async def get_notes(user_id: str = Depends(get_current_user)):
    notes = await db.notes.find({"user_id": user_id}, {"_id": 0}).to_list(1000)
    for note in notes:
        if isinstance(note['created_at'], str):
            note['created_at'] = datetime.fromisoformat(note['created_at'])
    return notes

# Chat Routes
@api_router.post("/chat")
async def chat(message: str, user_id: str = Depends(get_current_user)):
    user_message = ChatMessage(user_id=user_id, role="user", content=message)
    user_msg_dict = user_message.model_dump()
    user_msg_dict['created_at'] = user_msg_dict['created_at'].isoformat()
    await db.chat_messages.insert_one(user_msg_dict)
    
    history = await db.chat_messages.find({"user_id": user_id}, {"_id": 0}).sort("created_at", -1).limit(10).to_list(10)
    history.reverse()
    
    chat = LlmChat(
        api_key=EMERGENT_LLM_KEY,
        session_id=f"tutor_{user_id}",
        system_message="You are a helpful AI study tutor. Help students understand concepts, answer questions, and provide explanations."
    ).with_model("openai", "gpt-5.1")
    
    user_msg = UserMessage(text=message)
    response = await chat.send_message(user_msg)
    
    ai_message = ChatMessage(user_id=user_id, role="assistant", content=response)
    ai_msg_dict = ai_message.model_dump()
    ai_msg_dict['created_at'] = ai_msg_dict['created_at'].isoformat()
    await db.chat_messages.insert_one(ai_msg_dict)
    
    return {"message": response}

@api_router.get("/chat/history")
async def get_chat_history(user_id: str = Depends(get_current_user)):
    messages = await db.chat_messages.find({"user_id": user_id}, {"_id": 0}).sort("created_at", 1).to_list(100)
    for msg in messages:
        if isinstance(msg['created_at'], str):
            msg['created_at'] = datetime.fromisoformat(msg['created_at'])
    return messages

# Study Sessions Routes
@api_router.post("/sessions")
async def create_session(duration: int, focus_mode: bool = False, user_id: str = Depends(get_current_user)):
    session = StudySession(user_id=user_id, duration=duration, focus_mode=focus_mode)
    session_dict = session.model_dump()
    session_dict['created_at'] = session_dict['created_at'].isoformat()
    await db.sessions.insert_one(session_dict)
    
    await update_progress(user_id, total_study_time=duration)
    
    return session

@api_router.get("/sessions")
async def get_sessions(user_id: str = Depends(get_current_user)):
    sessions = await db.sessions.find({"user_id": user_id}, {"_id": 0}).sort("created_at", -1).limit(20).to_list(20)
    for session in sessions:
        if isinstance(session['created_at'], str):
            session['created_at'] = datetime.fromisoformat(session['created_at'])
    return sessions

# Progress Routes
@api_router.get("/progress")
async def get_progress(user_id: str = Depends(get_current_user)):
    progress = await db.progress.find_one({"user_id": user_id}, {"_id": 0})
    if not progress:
        progress = Progress(user_id=user_id).model_dump()
        progress['updated_at'] = progress['updated_at'].isoformat()
        if progress.get('last_study_date'):
            progress['last_study_date'] = progress['last_study_date'].isoformat()
        await db.progress.insert_one(progress)
    return progress

async def update_progress(user_id: str, total_study_time: int = 0, flashcards_reviewed: int = 0, quizzes_completed: int = 0):
    progress = await db.progress.find_one({"user_id": user_id}, {"_id": 0})
    if not progress:
        progress = Progress(user_id=user_id).model_dump()
        progress['updated_at'] = progress['updated_at'].isoformat()
        if progress.get('last_study_date'):
            progress['last_study_date'] = progress['last_study_date'].isoformat()
        await db.progress.insert_one(progress)
        progress = await db.progress.find_one({"user_id": user_id}, {"_id": 0})
    
    last_study_date = progress.get('last_study_date')
    if isinstance(last_study_date, str):
        last_study_date = datetime.fromisoformat(last_study_date)
    
    current_date = datetime.now(timezone.utc).date()
    
    if last_study_date:
        last_date = last_study_date.date() if isinstance(last_study_date, datetime) else last_study_date
        days_diff = (current_date - last_date).days
        
        if days_diff == 1:
            current_streak = progress.get('current_streak', 0) + 1
        elif days_diff > 1:
            current_streak = 1
        else:
            current_streak = progress.get('current_streak', 1)
    else:
        current_streak = 1
    
    longest_streak = max(progress.get('longest_streak', 0), current_streak)
    
    await db.progress.update_one(
        {"user_id": user_id},
        {"$inc": {
            "total_study_time": total_study_time,
            "flashcards_reviewed": flashcards_reviewed,
            "quizzes_completed": quizzes_completed
        },
         "$set": {
            "current_streak": current_streak,
            "longest_streak": longest_streak,
            "last_study_date": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat()
        }}
    )

# Include the router in the main app
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()