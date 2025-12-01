#!/usr/bin/env python3
"""
FACE RECOGNITION SYSTEM - INSIGHTFACE + DEEPFACE + YOLOv11-POSE + ATTENDANCE + REAL-TIME BACKEND
Gửi dữ liệu real-time cho backend mỗi giây - ĐÃ SỬA LỖI UnboundLocalError
"""

import os
import cv2
import numpy as np
import pandas as pd
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import pickle
import time
import subprocess
import sys
from datetime import datetime
import requests
import json

# ==================== CÀI ĐẶT DEPENDENCIES ====================
def install_dependencies():
    """Tự động cài đặt dependencies nếu chưa có"""
    packages = [
        "torch",
        "torchvision", 
        "opencv-python", 
        "matplotlib",
        "scikit-learn",
        "pillow",
        "numpy",
        "insightface",
        "onnxruntime",
        "deepface",
        "pandas",
        "tf-keras",
        "tensorflow",
        "ultralytics",
        "requests"
    ]
    
    print("🔧 Kiểm tra và cài đặt dependencies...")
    
    for package in packages:
        try:
            if package == "torch":
                __import__("torch")
            elif package == "torchvision":
                __import__("torchvision")
            elif package == "insightface":
                __import__("insightface")
            elif package == "deepface":
                __import__("deepface")
            elif package == "ultralytics":
                __import__("ultralytics")
            elif package == "requests":
                __import__("requests")
            else:
                __import__(package.replace('-', '_'))
            print(f"✅ {package} đã được cài đặt")
        except ImportError:
            print(f"📥 Đang cài đặt {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✅ Đã cài đặt {package}")

# ==================== BACKEND DATA SENDER ====================
class BackendDataSender:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.is_connected = False
        self.last_sent_time = 0
        self.send_interval = 1.0  # Gửi mỗi 1 giây
        self.test_connection()
    
    def test_connection(self):
        """Kiểm tra kết nối đến backend"""
        try:
            response = requests.get(f"{self.base_url}/api/health", timeout=3)
            if response.status_code == 200:
                self.is_connected = True
                print("✅ Đã kết nối đến backend thành công!")
            else:
                print(f"⚠️ Backend trả về mã lỗi: {response.status_code}")
                self.is_connected = False
        except Exception as e:
            print(f"❌ Không thể kết nối đến backend: {str(e)}")
            self.is_connected = False
    
    def can_send_realtime(self):
        """Kiểm tra xem có thể gửi real-time data không"""
        current_time = time.time()
        if current_time - self.last_sent_time >= self.send_interval:
            self.last_sent_time = current_time
            return True
        return False
    
    def send_realtime_data(self, student_data_list):
        """Gửi dữ liệu real-time cho tất cả học sinh được phát hiện"""
        if not self.is_connected or not self.can_send_realtime():
            return False
        
        try:
            # Tính toán thống kê
            present_count = len([s for s in student_data_list if s.get('status') == 'present'])
            total_count = len(student_data_list)
            
            # Tính emotion distribution
            emotion_count = {}
            engagement_scores = []
            
            for student in student_data_list:
                emotion = student.get('emotion', 'neutral')
                emotion_count[emotion] = emotion_count.get(emotion, 0) + 1
                engagement_scores.append(student.get('engagement', 0))
            
            avg_engagement = np.mean(engagement_scores) * 100 if engagement_scores else 75.0
            dominant_emotion = max(emotion_count.items(), key=lambda x: x[1])[0] if emotion_count else 'neutral'
            
            data = {
                "type": "live_update",
                "timestamp": datetime.now().isoformat(),
                "students": student_data_list,
                "stats": {
                    "total_students": total_count,
                    "present_count": present_count,
                    "absent_count": max(5 - present_count, 0),  # Giả định lớp có 5 học sinh
                    "attendance_rate": round((present_count / max(total_count, 1)) * 100, 1),
                    "avg_engagement": round(avg_engagement, 1),
                    "current_emotion": dominant_emotion
                }
            }
            
            # Gửi qua endpoint chính của backend
            response = requests.post(
                f"{self.base_url}/api/realtime/update",
                json=data,
                timeout=2
            )
            
            if response.status_code == 200:
                print(f"📤 Real-time: {len(student_data_list)} students, {avg_engagement:.1f}% engagement")
                return True
            else:
                # Thử gửi qua WebSocket endpoint
                try:
                    ws_response = requests.post(
                        f"{self.base_url}/api/websocket/broadcast",
                        json=data,
                        timeout=2
                    )
                    return ws_response.status_code == 200
                except:
                    return False
                
        except Exception as e:
            # print(f"❌ Lỗi gửi real-time data: {str(e)}")
            return False

    def send_face_detection(self, student_id, student_name, emotion, confidence, bbox):
        """Gửi dữ liệu nhận diện khuôn mặt đến backend"""
        if not self.is_connected:
            return False
        
        try:
            data = {
                "student_id": student_id,
                "student_name": student_name,
                "emotion": emotion,
                "confidence": confidence,
                "bbox": bbox
            }
            
            response = requests.post(
                f"{self.base_url}/api/detections/",
                json=data,
                timeout=3
            )
            
            return response.status_code == 200
                
        except Exception as e:
            print(f"❌ Lỗi gửi face detection: {str(e)}")
            return False
    
    def send_behavior_data(self, student_id, student_name, behavior_type, score, details=None):
        """Gửi dữ liệu hành vi đến backend"""
        if not self.is_connected:
            return False
        
        try:
            data = {
                "student_id": student_id,
                "student_name": student_name,
                "behavior_type": behavior_type,
                "score": score,
                "details": details or "{}"
            }
            
            response = requests.post(
                f"{self.base_url}/api/behavior/",
                json=data,
                timeout=3
            )
            
            return response.status_code == 200
                
        except Exception as e:
            print(f"❌ Lỗi gửi behavior data: {str(e)}")
            return False
    
    def mark_attendance(self, student_id, student_name, status="present"):
        """Điểm danh học sinh trên backend"""
        if not self.is_connected:
            return False
        
        try:
            now = datetime.now()
            data = {
                "student_id": student_id,
                "student_name": student_name,
                "date": now.strftime("%Y-%m-%d"),
                "status": status,
                "check_in_time": now.isoformat()
            }
            
            response = requests.post(
                f"{self.base_url}/api/attendance/mark",
                json=data,
                timeout=3
            )
            
            if response.status_code == 200:
                print(f"✅ Đã điểm danh trên backend: {student_name}")
                return True
            else:
                print(f"⚠️ Lỗi điểm danh trên backend: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Lỗi điểm danh trên backend: {str(e)}")
            return False

# ==================== BEHAVIOR DETECTION WITH YOLOv11-POSE ====================
class BehaviorDetector:
    def __init__(self):
        self.pose_model = None
        self.initialize_pose_detector()
    
    def initialize_pose_detector(self):
        """Khởi tạo YOLOv11 pose detector"""
        try:
            from ultralytics import YOLO
            
            # Load YOLOv11 pose model
            self.pose_model = YOLO('yolo11n-pose.pt')  # Sẽ tự động download
            print("✅ YOLOv11 Pose detector đã sẵn sàng")
            return True
            
        except Exception as e:
            print(f"❌ Lỗi khởi tạo YOLOv11 Pose: {str(e)}")
            return False
    
    def detect_behavior(self, image):
        """Nhận diện hành vi từ pose estimation"""
        try:
            # Run pose detection
            results = self.pose_model(image, verbose=False)
            
            behaviors = []
            
            for result in results:
                if hasattr(result, 'keypoints') and result.keypoints is not None and len(result.keypoints) > 0:
                    for person_idx, keypoints in enumerate(result.keypoints.data):
                        # Lấy keypoints (17 points, mỗi point có [x, y, confidence])
                        kpts = keypoints.cpu().numpy()
                        
                        # Phân tích hành vi dựa trên keypoints
                        behavior = self._analyze_pose_behavior(kpts)
                        
                        # Lấy bounding box nếu có
                        bbox = None
                        if hasattr(result, 'boxes') and result.boxes is not None and len(result.boxes) > person_idx:
                            bbox = result.boxes[person_idx].xyxy[0].cpu().numpy()
                        
                        behaviors.append({
                            'behavior': behavior,
                            'keypoints': kpts,
                            'bbox': bbox,
                            'person_idx': person_idx
                        })
            
            return behaviors
            
        except Exception as e:
            print(f"❌ Lỗi nhận diện hành vi: {str(e)}")
            return []
    
    def _analyze_pose_behavior(self, keypoints):
        """Phân tích hành vi dựa trên keypoints pose"""
        try:
            # Chỉ số keypoints theo COCO format
            NOSE = 0
            LEFT_EYE = 1
            RIGHT_EYE = 2
            LEFT_EAR = 3
            RIGHT_EAR = 4
            LEFT_SHOULDER = 5
            RIGHT_SHOULDER = 6
            LEFT_ELBOW = 7
            RIGHT_ELBOW = 8
            LEFT_WRIST = 9
            RIGHT_WRIST = 10
            LEFT_HIP = 11
            RIGHT_HIP = 12
            LEFT_KNEE = 13
            RIGHT_KNEE = 14
            LEFT_ANKLE = 15
            RIGHT_ANKLE = 16
            
            # Lấy tọa độ keypoints
            def get_point(idx):
                if keypoints[idx][2] > 0.3:  # Confidence threshold
                    return keypoints[idx][:2]
                return None
            
            # Lấy các điểm quan trọng
            left_shoulder = get_point(LEFT_SHOULDER)
            right_shoulder = get_point(RIGHT_SHOULDER)
            left_elbow = get_point(LEFT_ELBOW)
            right_elbow = get_point(RIGHT_ELBOW)
            left_wrist = get_point(LEFT_WRIST)
            right_wrist = get_point(RIGHT_WRIST)
            left_hip = get_point(LEFT_HIP)
            right_hip = get_point(RIGHT_HIP)
            left_knee = get_point(LEFT_KNEE)
            right_knee = get_point(RIGHT_KNEE)
            
            # Phân tích các hành vi
            behaviors = []
            
            # 1. Kiểm tra giơ tay
            if left_wrist is not None and left_shoulder is not None:
                if left_wrist[1] < left_shoulder[1]:
                    behaviors.append("raising_hand")
            if right_wrist is not None and right_shoulder is not None:
                if right_wrist[1] < right_shoulder[1]:
                    behaviors.append("raising_hand")
            
            # 2. Kiểm tra đứng lên/ngồi xuống
            if (left_hip is not None and left_knee is not None and 
                right_hip is not None and right_knee is not None):
                hip_height = (left_hip[1] + right_hip[1]) / 2
                knee_height = (left_knee[1] + right_knee[1]) / 2
                if abs(hip_height - knee_height) < 50:  # Đứng
                    behaviors.append("standing")
                else:  # Ngồi
                    behaviors.append("sitting")
            
            # 3. Kiểm tra vỗ tay
            if left_wrist is not None and right_wrist is not None:
                distance = np.sqrt(np.sum((left_wrist - right_wrist) ** 2))
                if distance < 50:
                    behaviors.append("clapping")
            
            # 4. Kiểm tra đi bộ/chạy
            if (left_hip is not None and right_hip is not None and 
                left_knee is not None and right_knee is not None):
                hip_distance = abs(left_hip[0] - right_hip[0])
                if hip_distance > 30:
                    behaviors.append("walking")
            
            # 5. Mặc định
            if not behaviors:
                behaviors.append("normal")
            
            return ", ".join(behaviors)
            
        except Exception as e:
            print(f"❌ Lỗi phân tích hành vi: {str(e)}")
            return "unknown"

# ==================== ATTENDANCE SYSTEM ====================
class AttendanceSystem:
    def __init__(self, csv_file="attendance.csv"):
        self.csv_file = csv_file
        self.backend_sender = BackendDataSender()
        self.initialize_attendance_file()
    
    def initialize_attendance_file(self):
        """Khởi tạo file điểm danh"""
        try:
            if not os.path.exists(self.csv_file):
                df = pd.DataFrame(columns=[
                    'Name', 'Date', 'Time', 'Emotion', 'Behavior', 'Confidence'
                ])
                df.to_csv(self.csv_file, index=False)
                print(f"✅ Đã tạo file điểm danh: {self.csv_file}")
            else:
                df = pd.read_csv(self.csv_file)
                print(f"✅ File điểm danh đã tồn tại: {len(df)} records")
        except Exception as e:
            print(f"❌ Lỗi khởi tạo file điểm danh: {str(e)}")
            df = pd.DataFrame(columns=[
                'Name', 'Date', 'Time', 'Emotion', 'Behavior', 'Confidence'
            ])
            df.to_csv(self.csv_file, index=False)
    
    def mark_attendance(self, name, emotion, behavior, confidence, bbox=None):
        """Điểm danh vào file CSV và gửi lên backend"""
        try:
            now = datetime.now()
            date_str = now.strftime("%Y-%m-%d")
            time_str = now.strftime("%H:%M:%S")
            
            # Tạo student_id từ tên (cho demo)
            student_id = f"SV{hash(name) % 10000:04d}"
            
            # Gửi dữ liệu lên backend
            if self.backend_sender.is_connected:
                # Gửi face detection data
                self.backend_sender.send_face_detection(
                    student_id=student_id,
                    student_name=name,
                    emotion=emotion,
                    confidence=confidence,
                    bbox=bbox or {"x1": 0, "y1": 0, "x2": 100, "y2": 100}
                )
                
                # Gửi behavior data
                engagement_score = confidence * 100
                self.backend_sender.send_behavior_data(
                    student_id=student_id,
                    student_name=name,
                    behavior_type="engagement",
                    score=engagement_score,
                    details=json.dumps({"behavior": behavior, "emotion": emotion})
                )
                
                # Điểm danh
                self.backend_sender.mark_attendance(student_id, name, "present")
            
            # Lưu vào file local
            try:
                df = pd.read_csv(self.csv_file)
            except:
                df = pd.DataFrame(columns=[
                    'Name', 'Date', 'Time', 'Emotion', 'Behavior', 'Confidence'
                ])
            
            # Kiểm tra điểm danh trong vòng 2 phút
            two_minutes_ago = (datetime.now() - pd.Timedelta(minutes=2)).strftime("%H:%M:%S")
            recent_entries = df[
                (df['Name'] == name) & 
                (df['Date'] == date_str) & 
                (df['Time'] > two_minutes_ago)
            ]
            
            if len(recent_entries) == 0:
                new_entry = {
                    'Name': name,
                    'Date': date_str,
                    'Time': time_str,
                    'Emotion': emotion,
                    'Behavior': behavior,
                    'Confidence': f"{confidence:.4f}"
                }
                
                df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)
                df.to_csv(self.csv_file, index=False)
                print(f"✅ Đã điểm danh: {name} | 😊 {emotion} | 🎯 {behavior}")
                return True
            else:
                return False
                
        except Exception as e:
            print(f"❌ Lỗi điểm danh: {str(e)}")
            return False
    
    def view_attendance(self):
        """Xem lịch sử điểm danh"""
        try:
            if not os.path.exists(self.csv_file):
                print("📭 Chưa có file điểm danh")
                return
                
            df = pd.read_csv(self.csv_file)
            if len(df) > 0:
                print("\n📊 LỊCH SỬ ĐIỂM DANH:")
                print("=" * 80)
                for _, row in df.iterrows():
                    print(f"👤 {row['Name']} | 📅 {row['Date']} | 🕒 {row['Time']} | 😊 {row['Emotion']} | 🎯 {row['Behavior']}")
                print("=" * 80)
                print(f"📈 Tổng số lượt điểm danh: {len(df)}")
            else:
                print("📭 Chưa có dữ liệu điểm danh")
        except Exception as e:
            print(f"❌ Lỗi đọc file điểm danh: {str(e)}")

# ==================== EMOTION DETECTION ====================
class EmotionDetector:
    def __init__(self):
        self.emotion_model = None
    
    def detect_emotion(self, face_image):
        """Nhận diện cảm xúc từ khuôn mặt"""
        try:
            from deepface import DeepFace
            
            # Chuyển đổi ảnh sang RGB
            face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            
            # Phân tích cảm xúc
            analysis = DeepFace.analyze(
                face_rgb, 
                actions=['emotion'],
                enforce_detection=False,
                silent=True
            )
            
            if isinstance(analysis, list):
                analysis = analysis[0]
            
            emotion = analysis['dominant_emotion']
            emotion_confidence = analysis['emotion'][emotion]
            
            return emotion, emotion_confidence
            
        except Exception as e:
            print(f"❌ Lỗi nhận diện cảm xúc: {str(e)}")
            return "unknown", 0.0

# ==================== FACE RECOGNITION SYSTEM ====================
class CompleteRecognitionSystem:
    def __init__(self, model_name='buffalo_l'):
        self.model_name = model_name
        self.face_analyzer = None
        self.l2_normalizer = Normalizer('l2')
        self.emotion_detector = EmotionDetector()
        self.behavior_detector = BehaviorDetector()
        self.attendance_system = AttendanceSystem()
        self.backend_sender = BackendDataSender()
        
    def initialize_system(self):
        """Khởi tạo toàn bộ hệ thống"""
        print("🚀 Đang khởi tạo hệ thống hoàn chỉnh...")
        
        # Khởi tạo InsightFace
        try:
            import insightface
            from insightface.app import FaceAnalysis
            
            self.face_analyzer = FaceAnalysis(
                name=self.model_name,
                providers=['CPUExecutionProvider']
            )
            self.face_analyzer.prepare(ctx_id=0, det_size=(640, 640))
            print("✅ InsightFace đã khởi tạo thành công!")
            
        except Exception as e:
            print(f"❌ Lỗi khởi tạo InsightFace: {str(e)}")
            return False
        
        # Khởi tạo Behavior Detector
        if not self.behavior_detector.initialize_pose_detector():
            print("⚠️ Không thể khởi tạo Behavior Detector")
        
        print("✅ Hệ thống hoàn chỉnh đã sẵn sàng!")
        return True

    def detect_faces(self, image):
        """Phát hiện khuôn mặt với InsightFace"""
        try:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            faces = self.face_analyzer.get(image_rgb)
            
            face_results = []
            for face in faces:
                bbox = face.bbox.astype(int)
                x1, y1, x2, y2 = bbox
                w = x2 - x1
                h = y2 - y1
                
                face_roi = image[y1:y2, x1:x2]
                if face_roi.size == 0:
                    continue
                
                embedding = face.normed_embedding
                
                # Nhận diện cảm xúc
                emotion, emotion_conf = self.emotion_detector.detect_emotion(face_roi)
                
                face_results.append({
                    'face_image': face_roi,
                    'bbox': (x1, y1, w, h),
                    'embedding': embedding,
                    'det_score': face.det_score,
                    'landmarks': face.kps if hasattr(face, 'kps') else None,
                    'emotion': emotion,
                    'emotion_confidence': emotion_conf
                })
            
            return face_results
            
        except Exception as e:
            print(f"❌ Lỗi detect faces: {str(e)}")
            return []

    def extract_features(self, face_data):
        """Trích xuất features từ khuôn mặt"""
        try:
            embedding = face_data['embedding']
            embedding = embedding.reshape(1, -1)
            features_normalized = self.l2_normalizer.transform(embedding)
            return features_normalized[0]
        except Exception as e:
            print(f"❌ Lỗi extract features: {str(e)}")
            return None

    def train_face_recognition(self, database_path="database"):
        """Train hệ thống nhận diện khuôn mặt"""
        if not os.path.exists(database_path):
            print(f"❌ Thư mục database không tồn tại: {database_path}")
            return False
        
        database = {}
        features_list = []
        labels_list = []
        
        print("📁 Đang xử lý database...")
        
        persons = [p for p in os.listdir(database_path) if os.path.isdir(os.path.join(database_path, p))]
        if len(persons) < 1:
            print("❌ Không có người nào trong database!")
            return False
        
        for person_name in persons:
            person_path = os.path.join(database_path, person_name)
            print(f"👤 Đang xử lý: {person_name}")
            person_features = []
            
            image_files = [f for f in os.listdir(person_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            for image_file in image_files:
                image_path = os.path.join(person_path, image_file)
                image = cv2.imread(image_path)
                if image is None:
                    continue
                
                face_results = self.detect_faces(image)
                for face_data in face_results:
                    features = self.extract_features(face_data)
                    if features is not None:
                        person_features.append(features)
                        features_list.append(features)
                        labels_list.append(person_name)
            
            if person_features:
                database[person_name] = person_features
                print(f"  ➕ {person_name}: {len(person_features)} khuôn mặt")
        
        if len(features_list) == 0:
            print("❌ Không có dữ liệu để train!")
            return False
        
        print(f"\n📊 Thống kê database:")
        print(f"👥 Số người: {len(database)}")
        print(f"🖼️ Tổng khuôn mặt: {len(features_list)}")
        
        # Train SVM model
        print("\n🎯 Đang train SVM model...")
        self.svm_model = SVC(kernel='linear', probability=True, random_state=42)
        self.svm_model.fit(features_list, labels_list)
        
        accuracy = accuracy_score(labels_list, self.svm_model.predict(features_list))
        print(f"✅ Training hoàn tất! Accuracy: {accuracy:.4f}")
        
        # Lưu model
        with open("face_recognition_model.pkl", 'wb') as f:
            pickle.dump(self.svm_model, f)
        
        with open("face_database.pkl", 'wb') as f:
            pickle.dump({
                'database': database,
                'features': features_list,
                'labels': labels_list
            }, f)
        
        print("💾 Đã lưu model và database")
        return True

    def load_trained_model(self):
        """Load model đã train"""
        try:
            with open("face_recognition_model.pkl", 'rb') as f:
                self.svm_model = pickle.load(f)
            
            with open("face_database.pkl", 'rb') as f:
                db_info = pickle.load(f)
            
            print(f"✅ Đã load trained model - {len(self.svm_model.classes_)} classes")
            return True
            
        except FileNotFoundError:
            print("❌ Không tìm thấy file model. Vui lòng train model trước.")
            return False

    def recognize_face(self, face_data, threshold=0.6):
        """Nhận diện khuôn mặt"""
        if not hasattr(self, 'svm_model') or self.svm_model is None:
            return "Unknown", 0.0
        
        features = self.extract_features(face_data)
        if features is None:
            return "Unknown", 0.0
        
        try:
            probabilities = self.svm_model.predict_proba([features])[0]
            max_prob = np.max(probabilities)
            predicted_class = self.svm_model.classes_[np.argmax(probabilities)]
            
            if max_prob < threshold:
                return "Unknown", max_prob
            else:
                return predicted_class, max_prob
        except:
            return "Unknown", 0.0

# ==================== REAL-TIME RECOGNITION ====================
def real_time_recognition():
    """Chạy real-time recognition với gửi dữ liệu real-time - ĐÃ SỬA LỖI"""
    system = CompleteRecognitionSystem()
    
    if not system.initialize_system():
        print("❌ Không thể khởi tạo hệ thống")
        return
    
    model_loaded = system.load_trained_model()
    if not model_loaded:
        print("⚠️ Chạy ở chế độ chỉ detect cảm xúc và hành vi")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Không thể mở webcam!")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("🎥 Hệ thống hoàn chỉnh đã bắt đầu!")
    print("📊 Tính năng: Nhận diện khuôn mặt + Cảm xúc + Hành vi + Điểm danh + Real-time Backend")
    print("🎮 Nhấn 'q' để thoát, 's' để chụp ảnh, 'v' để xem điểm danh")
    
    attendance_status = {}
    frame_count = 0
    
    # KHỞI TẠO BIẾN TRƯỚC - SỬA LỖI UnboundLocalError
    face_results = []
    behavior_results = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Giảm tần suất detection để tăng performance
        if frame_count % 2 == 0:
            face_results = system.detect_faces(frame)
            behavior_results = system.behavior_detector.detect_behavior(frame)
            
            # 🔥 GỬI DỮ LIỆU REAL-TIME CHO TẤT CẢ HỌC SINH
            student_data_list = []
            
            for i, face_data in enumerate(face_results):
                bbox = face_data['bbox']
                x, y, w, h = bbox
                emotion = face_data['emotion']
                emotion_conf = face_data['emotion_confidence']
                
                if model_loaded:
                    name, confidence = system.recognize_face(face_data)
                else:
                    name, confidence = "Unknown", 0.0
                
                # Tìm hành vi tương ứng
                behavior = "normal"
                for behav in behavior_results:
                    if behav['bbox'] is not None:
                        try:
                            bx1, by1, bx2, by2 = behav['bbox'].astype(int)
                            if (x < bx2 and x + w > bx1 and y < by2 and y + h > by1):
                                behavior = behav['behavior']
                                break
                        except:
                            continue
                
                # Tạo student data để gửi real-time
                student_data = {
                    'id': i + 1,
                    'name': name,
                    'status': 'present' if name != "Unknown" else 'unknown',
                    'emotion': emotion,
                    'engagement': confidence,
                    'behavior': behavior,
                    'bbox': {
                        'x': int(x), 'y': int(y), 
                        'width': int(w), 'height': int(h)
                    },
                    'confidence': confidence,
                    'timestamp': datetime.now().isoformat()
                }
                
                student_data_list.append(student_data)
                
                # Điểm danh nếu nhận diện được
                if name != "Unknown" and confidence > 0.6:
                    if name not in attendance_status:
                        bbox_dict = {"x1": x, "y1": y, "x2": x+w, "y2": y+h}
                        system.attendance_system.mark_attendance(
                            name, emotion, behavior, confidence, bbox_dict
                        )
                        attendance_status[name] = True
            
            # GỬI REAL-TIME DATA
            if student_data_list and system.backend_sender.is_connected:
                system.backend_sender.send_realtime_data(student_data_list)
        
        # Hiển thị kết quả - SỬA: LUÔN sử dụng biến đã được khởi tạo
        for i, face_data in enumerate(face_results):
            bbox = face_data['bbox']
            x, y, w, h = bbox
            emotion = face_data['emotion']
            emotion_conf = face_data['emotion_confidence']
            
            if model_loaded:
                name, confidence = system.recognize_face(face_data)
                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            else:
                name, confidence = "Unknown", 0.0
                color = (255, 255, 0)
            
            # Vẽ bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            
            # Tìm hành vi
            behavior_text = "normal"
            for behav in behavior_results:
                if behav['bbox'] is not None:
                    try:
                        bx1, by1, bx2, by2 = behav['bbox'].astype(int)
                        if (x < bx2 and x + w > bx1 and y < by2 and y + h > by1):
                            behavior_text = behav['behavior']
                            break
                    except:
                        continue
            
            # Hiển thị thông tin
            behavior_display = f"{behavior_text}"
            cv2.putText(frame, behavior_display, (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            emotion_text = f"{emotion} ({emotion_conf:.1f})"
            cv2.putText(frame, emotion_text, (x, y + h + 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # Vẽ pose keypoints
        for behavior in behavior_results:
            if behavior['keypoints'] is not None:
                for kpt in behavior['keypoints']:
                    x, y, conf = kpt
                    if conf > 0.3:
                        cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 0), -1)
        
        # Hiển thị trạng thái real-time
        backend_status = "🟢 REAL-TIME" if system.backend_sender.is_connected else "🔴 OFFLINE"
        info_text = f"Faces: {len(face_results)} | Backend: {backend_status}"
        cv2.putText(frame, info_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow('Real-time Face Recognition + Backend', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            filename = f"capture_{int(time.time())}.jpg"
            cv2.imwrite(filename, frame)
            print(f"✅ Đã lưu ảnh: {filename}")
        elif key == ord('v'):
            system.attendance_system.view_attendance()
    
    cap.release()
    cv2.destroyAllWindows()
    print("👋 Đã thoát!")

# ==================== CÁC HÀM PHỤ TRỢ ====================
def create_folder_structure():
    """Tạo cấu trúc thư mục"""
    folders = [
        "database",
        "database/person1",
        "database/person2", 
        "database/person3",
        "test_images"
    ]
    
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
        print(f"✅ Đã tạo: {folder}/")
    
    print("\n📁 Cấu trúc thư mục đã được tạo!")

def train_model():
    """Train model từ database"""
    system = CompleteRecognitionSystem()
    
    if not system.initialize_system():
        return
    
    if not os.path.exists("database"):
        os.makedirs("database")
        print("📁 Đã tạo thư mục 'database'")
        return
    
    success = system.train_face_recognition()
    if success:
        print("🎉 Train model thành công!")
    else:
        print("❌ Train model thất bại!")

def view_attendance():
    """Xem lịch sử điểm danh"""
    attendance_system = AttendanceSystem()
    attendance_system.view_attendance()

def test_backend_connection():
    """Kiểm tra kết nối backend"""
    sender = BackendDataSender()
    if sender.is_connected:
        print("✅ Kết nối backend: THÀNH CÔNG")
    else:
        print("❌ Kết nối backend: THẤT BẠI")

# ==================== MAIN MENU ====================
def main_menu():
    """Hiển thị menu chính"""
    while True:
        print("\n" + "="*70)
        print("🎭 COMPLETE RECOGNITION SYSTEM - FACE + EMOTION + BEHAVIOR + ATTENDANCE + BACKEND")
        print("="*70)
        print("1. 📁 Tạo cấu trúc thư mục")
        print("2. 🎯 Train face recognition model")
        print("3. 🎥 Real-time (Face + Emotion + Behavior + Attendance + Backend)")
        print("4. 📊 Xem lịch sử điểm danh")
        print("5. 🔗 Kiểm tra kết nối backend")
        print("6. 🚪 Thoát")
        print("="*70)
        
        choice = input("👉 Chọn chức năng (1-6): ").strip()
        
        if choice == "1":
            create_folder_structure()
        elif choice == "2":
            train_model()
        elif choice == "3":
            real_time_recognition()
        elif choice == "4":
            view_attendance()
        elif choice == "5":
            test_backend_connection()
        elif choice == "6":
            print("👋 Tạm biệt!")
            break
        else:
            print("❌ Lựa chọn không hợp lệ!")
        
        input("\n👉 Nhấn Enter để tiếp tục...")

# ==================== MAIN ====================
if __name__ == "__main__":
    print("🔧 Đang kiểm tra hệ thống...")
    install_dependencies()
    
    print("\n🎯 Khởi động Hệ thống Nhận diện Hoàn chỉnh...")
    print("📊 Tính năng: Nhận diện khuôn mặt + Cảm xúc + Hành vi + Điểm danh + Real-time Backend")
    
    main_menu()