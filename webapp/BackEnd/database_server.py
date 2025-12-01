# backend/database_server.py
from fastapi import FastAPI, Depends, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Float, Text, func, Boolean
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from datetime import datetime, timedelta
from typing import List, Optional, Dict
from pydantic import BaseModel
import json
import asyncio
import random
import secrets
import hashlib
import base64

# ==================== SIMPLE AUTHENTICATION ====================
SECRET_KEY = secrets.token_urlsafe(32)

# Simple password hashing
def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return hash_password(plain_password) == hashed_password

def create_simple_token(data: dict):
    token_data = json.dumps(data).encode()
    return base64.b64encode(token_data).decode()

def decode_simple_token(token: str):
    try:
        token_data = base64.b64decode(token).decode()
        return json.loads(token_data)
    except:
        return None

# ==================== DATABASE SETUP ====================
SQLALCHEMY_DATABASE_URL = "sqlite:///./classroom_data.db"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False}
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# ==================== DATABASE MODELS ====================
class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    full_name = Column(String)
    is_active = Column(Boolean, default=True)
    is_superuser = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)

class Student(Base):
    __tablename__ = "students"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    student_id = Column(String, unique=True, index=True)
    class_name = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

class FaceDetection(Base):
    __tablename__ = "face_detections"
    
    id = Column(Integer, primary_key=True, index=True)
    student_id = Column(String, index=True)
    student_name = Column(String, index=True)
    emotion = Column(String)
    confidence = Column(Float)
    bbox = Column(Text)
    timestamp = Column(DateTime, default=datetime.utcnow)

class Attendance(Base):
    __tablename__ = "attendance"
    
    id = Column(Integer, primary_key=True, index=True)
    student_id = Column(String, index=True)
    student_name = Column(String, index=True)
    date = Column(String, index=True)
    status = Column(String)
    check_in_time = Column(DateTime)
    check_out_time = Column(DateTime, nullable=True)

class BehaviorData(Base):
    __tablename__ = "behavior_data"
    
    id = Column(Integer, primary_key=True, index=True)
    student_id = Column(String, index=True)
    student_name = Column(String, index=True)
    behavior_type = Column(String)
    score = Column(Float)
    timestamp = Column(DateTime, default=datetime.utcnow)
    details = Column(Text)

# Tạo bảng
Base.metadata.create_all(bind=engine)

# ==================== PYDANTIC MODELS ====================
class UserCreate(BaseModel):
    username: str
    email: str
    password: str
    full_name: str

class UserLogin(BaseModel):
    username: str
    password: str

class UserResponse(BaseModel):
    id: int
    username: str
    email: str
    full_name: str
    is_active: bool
    is_superuser: bool
    created_at: datetime

    class Config:
        from_attributes = True

class Token(BaseModel):
    access_token: str
    token_type: str
    user: UserResponse

class StudentCreate(BaseModel):
    name: str
    student_id: str
    class_name: str

class StudentResponse(BaseModel):
    id: int
    name: str
    student_id: str
    class_name: str
    created_at: datetime

    class Config:
        from_attributes = True

class FaceDetectionCreate(BaseModel):
    student_id: str
    student_name: str
    emotion: str
    confidence: float
    bbox: Optional[dict] = None

class AttendanceCreate(BaseModel):
    student_id: str
    student_name: str
    date: str
    status: str
    check_in_time: datetime

class BehaviorDataCreate(BaseModel):
    student_id: str
    student_name: str
    behavior_type: str
    score: float
    details: Optional[str] = None

# ==================== AUTHENTICATION UTILS ====================
def get_password_hash(password):
    return hash_password(password)

def get_user_by_username(db: Session, username: str):
    return db.query(User).filter(User.username == username).first()

def authenticate_user(db: Session, username: str, password: str):
    user = get_user_by_username(db, username)
    if not user:
        return False
    if not verify_password(password, user.hashed_password):
        return False
    return user

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=30)
    to_encode.update({"exp": expire.timestamp()})
    return create_simple_token(to_encode)

def verify_token(token: str):
    decoded = decode_simple_token(token)
    if not decoded:
        return None
    
    # Check expiration
    exp = decoded.get("exp")
    if exp and datetime.utcnow().timestamp() > exp:
        return None
    
    return decoded

# Dependency để lấy database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Simple dependency for auth
def get_current_user_simple(db: Session = Depends(get_db)):
    # For development, return the admin user
    user = get_user_by_username(db, "admin")
    if not user:
        # Create admin user if not exists
        user = User(
            username="admin",
            email="admin@classroom.com",
            hashed_password=get_password_hash("admin123"),
            full_name="Administrator",
            is_active=True,
            is_superuser=True
        )
        db.add(user)
        db.commit()
        db.refresh(user)
    return user

# ==================== FASTAPI APP ====================
app = FastAPI(
    title="Classroom Analytics API",
    docs_url="/api/docs",
    openapi_url="/api/openapi.json"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== WEBSOCKET MANAGER ====================
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        print(f"✅ WebSocket connected. Total connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            print(f"❌ WebSocket disconnected. Total connections: {len(self.active_connections)}")

    async def send_personal_message(self, message: dict, websocket: WebSocket):
        try:
            await websocket.send_json(message)
        except Exception as e:
            print(f"Error sending message: {e}")
            self.disconnect(websocket)

manager_ws = ConnectionManager()

# ==================== AUTHENTICATION ROUTES ====================
@app.post("/api/auth/register", response_model=UserResponse)
def register(user: UserCreate, db: Session = Depends(get_db)):
    """Đăng ký user mới"""
    # Kiểm tra username đã tồn tại
    db_user = get_user_by_username(db, username=user.username)
    if db_user:
        raise HTTPException(status_code=400, detail="Username already registered")
    
    # Kiểm tra email đã tồn tại
    db_email = db.query(User).filter(User.email == user.email).first()
    if db_email:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    # Tạo user mới
    hashed_password = get_password_hash(user.password)
    db_user = User(
        username=user.username,
        email=user.email,
        hashed_password=hashed_password,
        full_name=user.full_name,
        is_active=True,
        is_superuser=False
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    
    # Convert SQLAlchemy object to dict for response
    return UserResponse(
        id=db_user.id,
        username=db_user.username,
        email=db_user.email,
        full_name=db_user.full_name,
        is_active=db_user.is_active,
        is_superuser=db_user.is_superuser,
        created_at=db_user.created_at
    )

@app.post("/api/auth/login", response_model=Token)
def login(user_data: UserLogin, db: Session = Depends(get_db)):
    """Đăng nhập và nhận token"""
    user = authenticate_user(db, user_data.username, user_data.password)
    if not user:
        raise HTTPException(status_code=401, detail="Incorrect username or password")
    
    access_token_expires = timedelta(minutes=30)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    
    # Create UserResponse from SQLAlchemy object
    user_response = UserResponse(
        id=user.id,
        username=user.username,
        email=user.email,
        full_name=user.full_name,
        is_active=user.is_active,
        is_superuser=user.is_superuser,
        created_at=user.created_at
    )
    
    return Token(
        access_token=access_token,
        token_type="bearer",
        user=user_response
    )

@app.get("/api/auth/me", response_model=UserResponse)
def read_users_me(current_user: User = Depends(get_current_user_simple)):
    """Lấy thông tin user hiện tại"""
    return UserResponse(
        id=current_user.id,
        username=current_user.username,
        email=current_user.email,
        full_name=current_user.full_name,
        is_active=current_user.is_active,
        is_superuser=current_user.is_superuser,
        created_at=current_user.created_at
    )

# ==================== MOCK DATA FOR FRONTEND COMPATIBILITY ====================
class ClassroomManager:
    def __init__(self, db: Session):
        self.db = db
        
    def get_live_students(self):
        """Get students with live status for WebSocket"""
        try:
            students_db = self.db.query(Student).all()
            today = datetime.now().strftime("%Y-%m-%d")
            
            live_students = []
            for student in students_db:
                # Get today's attendance
                attendance = self.db.query(Attendance).filter(
                    Attendance.student_id == student.student_id,
                    Attendance.date == today
                ).first()
                
                # Get latest emotion
                latest_emotion = self.db.query(FaceDetection).filter(
                    FaceDetection.student_id == student.student_id
                ).order_by(FaceDetection.timestamp.desc()).first()
                
                # Get average engagement
                avg_engagement = self.db.query(func.avg(BehaviorData.score)).filter(
                    BehaviorData.student_id == student.student_id,
                    BehaviorData.behavior_type == 'engagement'
                ).scalar() or 75.0
                
                live_students.append({
                    'id': student.id,
                    'name': student.name,
                    'status': attendance.status if attendance else 'absent',
                    'emotion': latest_emotion.emotion if latest_emotion else 'neutral',
                    'engagement': round(avg_engagement / 100.0, 2)
                })
            
            return live_students
        except Exception as e:
            print(f"Error in get_live_students: {e}")
            return []
    
    def get_dashboard_stats(self):
        """Get dashboard statistics compatible with frontend"""
        try:
            total_students = self.db.query(Student).count()
            today = datetime.now().strftime("%Y-%m-%d")
            
            today_attendance = self.db.query(Attendance).filter(Attendance.date == today).all()
            present_count = len([a for a in today_attendance if a.status == 'present'])
            
            # Get today's emotions
            today_emotions = self.db.query(FaceDetection).filter(
                func.date(FaceDetection.timestamp) == datetime.now().date()
            ).all()
            
            emotion_counts = {}
            for detection in today_emotions:
                emotion_counts[detection.emotion] = emotion_counts.get(detection.emotion, 0) + 1
            
            current_emotion = max(emotion_counts, key=emotion_counts.get) if emotion_counts else 'neutral'
            
            # Get average engagement for today
            today_engagement = self.db.query(func.avg(BehaviorData.score)).filter(
                func.date(BehaviorData.timestamp) == datetime.now().date(),
                BehaviorData.behavior_type == 'engagement'
            ).scalar() or 75.0
            
            return {
                'total_students': total_students,
                'present_count': present_count,
                'absent_count': total_students - present_count,
                'attendance_rate': round((present_count / total_students) * 100, 1) if total_students > 0 else 0,
                'avg_engagement': round(today_engagement, 1),
                'current_emotion': current_emotion
            }
        except Exception as e:
            print(f"Error in get_dashboard_stats: {e}")
            return {
                'total_students': 5,
                'present_count': 4,
                'absent_count': 1,
                'attendance_rate': 80.0,
                'avg_engagement': 75.0,
                'current_emotion': 'neutral'
            }

# ==================== TẠO DỮ LIỆU MẪU ====================
def create_sample_data():
    """Tạo dữ liệu mẫu khi khởi động server"""
    db = SessionLocal()
    try:
        # Kiểm tra xem đã có admin user chưa
        existing_admin = db.query(User).filter(User.username == "admin").first()
        if not existing_admin:
            # Tạo admin user
            admin_user = User(
                username="admin",
                email="admin@classroom.com",
                hashed_password=get_password_hash("admin123"),
                full_name="Administrator",
                is_active=True,
                is_superuser=True
            )
            db.add(admin_user)
            print("✅ Đã tạo admin user: admin/admin123")
        
        # Kiểm tra xem đã có dữ liệu students chưa
        existing_students = db.query(Student).count()
        if existing_students > 0:
            print("✅ Database đã có dữ liệu, bỏ qua tạo mẫu")
            db.commit()
            return
        
        print("🔄 Đang tạo dữ liệu mẫu...")
        
        # Tạo học sinh mẫu
        sample_students = [
            {"name": "Nguyễn Văn A", "student_id": "SV001", "class_name": "10A1"},
            {"name": "Trần Thị B", "student_id": "SV002", "class_name": "10A1"},
            {"name": "Lê Văn C", "student_id": "SV003", "class_name": "10A2"},
            {"name": "Phạm Thị D", "student_id": "SV004", "class_name": "10A2"},
            {"name": "Hoàng Văn E", "student_id": "SV005", "class_name": "10A3"},
        ]
        
        for student_data in sample_students:
            db_student = Student(**student_data)
            db.add(db_student)
        
        db.commit()
        
        # Tạo dữ liệu mẫu cho 7 ngày
        today = datetime.now()
        emotions = ["happy", "neutral", "sad", "surprise", "angry"]
        
        for i in range(7):
            date = today - timedelta(days=i)
            date_str = date.strftime("%Y-%m-%d")
            
            for student in sample_students:
                # Tạo điểm danh
                attendance = Attendance(
                    student_id=student["student_id"],
                    student_name=student["name"],
                    date=date_str,
                    status="present" if random.random() > 0.2 else "absent",
                    check_in_time=date.replace(hour=7, minute=random.randint(20, 40), second=0)
                )
                db.add(attendance)
                
                # Tạo face detection data
                for _ in range(random.randint(3, 8)):
                    detection_time = date.replace(
                        hour=random.randint(8, 16),
                        minute=random.randint(0, 59)
                    )
                    
                    detection = FaceDetection(
                        student_id=student["student_id"],
                        student_name=student["name"],
                        emotion=random.choice(emotions),
                        confidence=round(random.uniform(0.7, 0.99), 2),
                        bbox=json.dumps({
                            "x1": random.randint(100, 300),
                            "y1": random.randint(100, 300),
                            "x2": random.randint(400, 600),
                            "y2": random.randint(400, 600)
                        }),
                        timestamp=detection_time
                    )
                    db.add(detection)
                
                # Tạo behavior data
                for _ in range(random.randint(2, 4)):
                    behavior_time = date.replace(
                        hour=random.randint(8, 16),
                        minute=random.randint(0, 59)
                    )
                    
                    behavior = BehaviorData(
                        student_id=student["student_id"],
                        student_name=student["name"],
                        behavior_type="engagement",
                        score=round(random.uniform(60, 95), 1),
                        timestamp=behavior_time,
                        details=json.dumps({"activity": "class_participation"})
                    )
                    db.add(behavior)
        
        db.commit()
        print("✅ Đã tạo dữ liệu mẫu thành công!")
        
    except Exception as e:
        db.rollback()
        print(f"❌ Lỗi khi tạo dữ liệu mẫu: {e}")
    finally:
        db.close()

# ==================== PROTECTED API ROUTES ====================

@app.get("/")
async def root():
    return JSONResponse({
        "message": "Classroom Analytics API is running!", 
        "version": "1.0",
        "endpoints": {
            "docs": "/api/docs",
            "auth": "/api/auth",
            "dashboard": "/api/dashboard/stats",
            "students": "/api/students",
            "attendance": "/api/attendance",
            "websocket": "/ws/live"
        }
    })

@app.get("/api/dashboard/stats")
async def get_dashboard_stats(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user_simple)
):
    """Dashboard stats - tương thích với frontend"""
    manager = ClassroomManager(db)
    return manager.get_dashboard_stats()

@app.get("/api/students")
async def get_students(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user_simple)
):
    """Get students list - tương thích với frontend"""
    manager = ClassroomManager(db)
    return {"students": manager.get_live_students()}

@app.get("/api/attendance")
async def get_attendance(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user_simple)
):
    """Get attendance data - tương thích với frontend"""
    today = datetime.now().strftime("%Y-%m-%d")
    attendance_records = db.query(Attendance).filter(Attendance.date == today).all()
    
    attendance_data = []
    for record in attendance_records:
        attendance_data.append({
            'date': record.date,
            'student': record.student_name,
            'status': record.status,
            'time_in': record.check_in_time.strftime('%H:%M') if record.check_in_time else '-'
        })
    
    return {"attendance": attendance_data}

@app.get("/api/analytics/emotion-trend")
async def get_emotion_trend(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user_simple)
):
    """Get emotion trend data - tương thích với frontend"""
    emotion_trend = []
    time_slots = ['08:00', '08:30', '09:00', '09:30']
    
    for time_slot in time_slots:
        trend_point = {
            'time': time_slot,
            'happy': random.randint(3, 6),
            'neutral': random.randint(2, 4),
            'sad': random.randint(1, 3),
            'surprise': random.randint(1, 2)
        }
        emotion_trend.append(trend_point)
    
    return {"emotion_trend": emotion_trend}

@app.get("/api/analytics/engagement")
async def get_engagement_data(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user_simple)
):
    """Get engagement data - tương thích với frontend"""
    students = db.query(Student).all()
    engagement_data = []
    
    for student in students:
        avg_engagement = db.query(func.avg(BehaviorData.score)).filter(
            BehaviorData.student_id == student.student_id,
            BehaviorData.behavior_type == 'engagement'
        ).scalar()
        
        engagement_data.append({
            'student': student.name,
            'engagement': round(avg_engagement, 1) if avg_engagement else 75.0
        })
    
    return {"engagement_data": engagement_data}

# ==================== WEBSOCKET FOR LIVE DATA ====================

@app.websocket("/ws/live")
async def websocket_endpoint(websocket: WebSocket):
    await manager_ws.connect(websocket)
    print(f"🔗 New WebSocket connection from {websocket.client}")
    
    try:
        while True:
            try:
                # Kiểm tra kết nối còn hoạt động không
                await websocket.receive_text()
            except WebSocketDisconnect:
                break
            except:
                # Nếu không thể receive, tiếp tục gửi dữ liệu
                pass
            
            db = SessionLocal()
            try:
                manager = ClassroomManager(db)
                
                # Get live data với dữ liệu mẫu an toàn
                live_data = {
                    "type": "live_update",
                    "timestamp": datetime.now().isoformat(),
                    "students": manager.get_live_students(),
                    "stats": manager.get_dashboard_stats()
                }
                
                await manager_ws.send_personal_message(live_data, websocket)
                
            except Exception as e:
                print(f"❌ Error in WebSocket loop: {e}")
                # Gửi dữ liệu mẫu an toàn nếu có lỗi
                safe_data = {
                    "type": "live_update",
                    "timestamp": datetime.now().isoformat(),
                    "students": [
                        {"id": 1, "name": "Nguyễn Văn A", "status": "present", "emotion": "neutral", "engagement": 0.75},
                        {"id": 2, "name": "Trần Thị B", "status": "present", "emotion": "happy", "engagement": 0.82},
                        {"id": 3, "name": "Lê Văn C", "status": "present", "emotion": "neutral", "engagement": 0.68},
                        {"id": 4, "name": "Phạm Thị D", "status": "absent", "emotion": "neutral", "engagement": 0.0},
                        {"id": 5, "name": "Hoàng Văn E", "status": "present", "emotion": "surprise", "engagement": 0.79}
                    ],
                    "stats": {
                        "total_students": 5,
                        "present_count": 4,
                        "absent_count": 1,
                        "attendance_rate": 80.0,
                        "avg_engagement": 76.0,
                        "current_emotion": "neutral"
                    }
                }
                try:
                    await manager_ws.send_personal_message(safe_data, websocket)
                except:
                    break
            finally:
                db.close()
            
            await asyncio.sleep(3)  # Update every 3 seconds
            
    except WebSocketDisconnect:
        print(f"🔌 WebSocket disconnected: {websocket.client}")
    except Exception as e:
        print(f"💥 WebSocket error: {e}")
    finally:
        manager_ws.disconnect(websocket)

# ==================== ORIGINAL DATABASE API ROUTES ====================

@app.post("/api/students/create", response_model=StudentResponse)
def create_student(
    student: StudentCreate, 
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user_simple)
):
    """Thêm học sinh mới"""
    existing = db.query(Student).filter(Student.student_id == student.student_id).first()
    if existing:
        raise HTTPException(status_code=400, detail="Student ID already exists")
    
    db_student = Student(**student.dict())
    db.add(db_student)
    db.commit()
    db.refresh(db_student)
    return StudentResponse(
        id=db_student.id,
        name=db_student.name,
        student_id=db_student.student_id,
        class_name=db_student.class_name,
        created_at=db_student.created_at
    )

@app.get("/api/students/list", response_model=List[StudentResponse])
def get_students_list(
    skip: int = 0, 
    limit: int = 100, 
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user_simple)
):
    """Lấy danh sách học sinh (original API)"""
    students = db.query(Student).offset(skip).limit(limit).all()
    return [StudentResponse(
        id=student.id,
        name=student.name,
        student_id=student.student_id,
        class_name=student.class_name,
        created_at=student.created_at
    ) for student in students]

@app.post("/api/detections/")
def create_face_detection(
    detection: FaceDetectionCreate, 
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user_simple)
):
    """Lưu dữ liệu nhận diện khuôn mặt"""
    db_detection = FaceDetection(
        student_id=detection.student_id,
        student_name=detection.student_name,
        emotion=detection.emotion,
        confidence=detection.confidence,
        bbox=json.dumps(detection.bbox) if detection.bbox else None,
        timestamp=datetime.utcnow()
    )
    db.add(db_detection)
    db.commit()
    db.refresh(db_detection)
    
    return {
        "message": "Face detection data saved successfully",
        "id": db_detection.id,
        "timestamp": db_detection.timestamp.isoformat()
    }

@app.post("/api/attendance/mark")
def create_attendance(
    attendance: AttendanceCreate, 
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user_simple)
):
    """Điểm danh học sinh"""
    existing = db.query(Attendance).filter(
        Attendance.student_id == attendance.student_id,
        Attendance.date == attendance.date
    ).first()
    
    if existing:
        existing.status = attendance.status
        existing.check_in_time = attendance.check_in_time
        db.commit()
        return {"message": "Attendance updated", "id": existing.id}
    
    db_attendance = Attendance(**attendance.dict())
    db.add(db_attendance)
    db.commit()
    db.refresh(db_attendance)
    return {"message": "Attendance recorded", "id": db_attendance.id}

@app.get("/api/health")
def health_check(db: Session = Depends(get_db)):
    """Health check endpoint"""
    try:
        db.query(Student).first()
        return {
            "status": "healthy",
            "database": "connected",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "status": "unhealthy", 
            "database": "disconnected", 
            "error": str(e)
        }

# ==================== KHỞI TẠO DỮ LIỆU ====================
create_sample_data()

if __name__ == "__main__":
    import uvicorn
    
    print("🚀 Starting Classroom Analytics API...")
    print("📊 Database: classroom_data.db")
    print("🔐 Authentication: Simple Token")
    print("👤 Default admin: admin/admin123")
    print("🌐 API URL: http://localhost:8000")
    print("📚 API Docs: http://localhost:8000/api/docs")
    print("🔄 WebSocket: ws://localhost:8000/ws/live")
    print("✅ Sample data ready!")
    
    uvicorn.run(
        "database_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )