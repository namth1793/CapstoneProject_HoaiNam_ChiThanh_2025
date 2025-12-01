# backend/main.py
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
import json
import numpy as np
from datetime import datetime, timedelta
import pandas as pd
from typing import List, Dict
import asyncio
import random

app = FastAPI(
    title="Classroom Analytics API", 
    docs_url="/api/docs", 
    openapi_url="/api/openapi.json"
)

# CORS - Cho phép tất cả origins để development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mock data
class ClassroomManager:
    def __init__(self):
        self.students = [
            {'id': 1, 'name': 'Nguyễn Văn A', 'status': 'present', 'emotion': 'happy', 'engagement': 0.8},
            {'id': 2, 'name': 'Trần Thị B', 'status': 'present', 'emotion': 'neutral', 'engagement': 0.6},
            {'id': 3, 'name': 'Lê Văn C', 'status': 'absent', 'emotion': 'sad', 'engagement': 0.3},
            {'id': 4, 'name': 'Phạm Thị D', 'status': 'present', 'emotion': 'happy', 'engagement': 0.9},
            {'id': 5, 'name': 'Hoàng Văn E', 'status': 'present', 'emotion': 'surprise', 'engagement': 0.7},
        ]
        self.attendance_data = []
        self.emotion_history = []

    def get_dashboard_stats(self):
        present_count = len([s for s in self.students if s['status'] == 'present'])
        avg_engagement = np.mean([s['engagement'] for s in self.students if s['status'] == 'present'])
        
        emotions = [s['emotion'] for s in self.students if s['status'] == 'present']
        if emotions:
            current_emotion = max(set(emotions), key=emotions.count)
        else:
            current_emotion = 'neutral'
        
        return {
            'total_students': len(self.students),
            'present_count': present_count,
            'absent_count': len(self.students) - present_count,
            'attendance_rate': round((present_count / len(self.students)) * 100, 1),
            'avg_engagement': round(avg_engagement * 100, 1),
            'current_emotion': current_emotion
        }

manager = ClassroomManager()

# WebSocket connections
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except:
                self.disconnect(connection)

manager_ws = ConnectionManager()

@app.get("/")
async def root():
    return JSONResponse({
        "message": "Classroom Analytics API is running!", 
        "version": "1.0",
        "endpoints": {
            "docs": "/api/docs",
            "dashboard": "/api/dashboard/stats",
            "students": "/api/students",
            "websocket": "/ws/live"
        }
    })

@app.get("/api/dashboard/stats")
async def get_dashboard_stats():
    return manager.get_dashboard_stats()

@app.get("/api/students")
async def get_students():
    return {"students": manager.students}

@app.get("/api/attendance")
async def get_attendance():
    today = datetime.now().date()
    attendance_data = [
        {'date': today.isoformat(), 'student': 'Nguyễn Văn A', 'status': 'present', 'time_in': '07:30'},
        {'date': today.isoformat(), 'student': 'Trần Thị B', 'status': 'present', 'time_in': '07:35'},
        {'date': today.isoformat(), 'student': 'Lê Văn C', 'status': 'absent', 'time_in': '-'},
        {'date': today.isoformat(), 'student': 'Phạm Thị D', 'status': 'present', 'time_in': '07:28'},
        {'date': today.isoformat(), 'student': 'Hoàng Văn E', 'status': 'present', 'time_in': '07:40'},
    ]
    return {"attendance": attendance_data}

@app.get("/api/analytics/emotion-trend")
async def get_emotion_trend():
    emotion_trend = [
        {'time': '08:00', 'happy': 5, 'neutral': 3, 'sad': 2, 'surprise': 1},
        {'time': '08:30', 'happy': 6, 'neutral': 2, 'sad': 2, 'surprise': 1},
        {'time': '09:00', 'happy': 4, 'neutral': 4, 'sad': 2, 'surprise': 1},
        {'time': '09:30', 'happy': 5, 'neutral': 3, 'sad': 1, 'surprise': 2},
    ]
    return {"emotion_trend": emotion_trend}

@app.get("/api/analytics/engagement")
async def get_engagement_data():
    engagement_data = [
        {'student': 'Nguyễn Văn A', 'engagement': 85},
        {'student': 'Trần Thị B', 'engagement': 65},
        {'student': 'Lê Văn C', 'engagement': 45},
        {'student': 'Phạm Thị D', 'engagement': 92},
        {'student': 'Hoàng Văn E', 'engagement': 78},
    ]
    return {"engagement_data": engagement_data}

@app.get("/api/reports/export")
async def export_report():
    # Create sample CSV
    df = pd.DataFrame([
        {'Ngày': '2024-01-15', 'Học sinh': 'Nguyễn Văn A', 'Trạng thái': 'Có mặt', 'Giờ vào': '07:30', 'Cảm xúc': 'Vui vẻ'},
        {'Ngày': '2024-01-15', 'Học sinh': 'Trần Thị B', 'Trạng thái': 'Có mặt', 'Giờ vào': '07:35', 'Cảm xúc': 'Bình thường'},
        {'Ngày': '2024-01-15', 'Học sinh': 'Lê Văn C', 'Trạng thái': 'Vắng', 'Giờ vào': '', 'Cảm xúc': ''},
        {'Ngày': '2024-01-15', 'Học sinh': 'Phạm Thị D', 'Trạng thái': 'Có mặt', 'Giờ vào': '07:28', 'Cảm xúc': 'Vui vẻ'},
        {'Ngày': '2024-01-15', 'Học sinh': 'Hoàng Văn E', 'Trạng thái': 'Có mặt', 'Giờ vào': '07:40', 'Cảm xúc': 'Ngạc nhiên'},
    ])
    
    filename = f"diem_danh_{datetime.now().strftime('%Y%m%d')}.csv"
    df.to_csv(filename, index=False, encoding='utf-8-sig')
    
    return FileResponse(filename, filename=filename, media_type='text/csv')

@app.websocket("/ws/live")
async def websocket_endpoint(websocket: WebSocket):
    await manager_ws.connect(websocket)
    try:
        while True:
            # Simulate real-time data with some randomness
            for student in manager.students:
                if student['status'] == 'present':
                    # Randomly change emotions for demo
                    if random.random() < 0.1:  # 10% chance to change emotion
                        emotions = ['happy', 'neutral', 'sad', 'surprise', 'angry']
                        student['emotion'] = random.choice(emotions)
                    # Slightly fluctuate engagement
                    student['engagement'] = max(0.1, min(0.99, student['engagement'] + random.uniform(-0.05, 0.05)))
            
            live_data = {
                "type": "live_update",
                "timestamp": datetime.now().isoformat(),
                "students": manager.students,
                "stats": manager.get_dashboard_stats()
            }
            await websocket.send_json(live_data)
            await asyncio.sleep(3)  # Update every 3 seconds
    except WebSocketDisconnect:
        manager_ws.disconnect(websocket)

# Chỉ chạy khi được execute trực tiếp
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)