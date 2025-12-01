#!/usr/bin/env python3
"""
CLASSROOM BEHAVIOR DETECTION SYSTEM - GPU SUPPORT
Tối ưu hóa để chạy trên GPU
"""

import os
import cv2
import numpy as np
import pandas as pd
import torch
import time
import subprocess
import sys
from datetime import datetime

# ==================== KIỂM TRA VÀ CÀI ĐẶT GPU ====================
def setup_gpu():
    """Thiết lập và kiểm tra GPU"""
    print("🔍 Đang kiểm tra GPU...")
    
    # Kiểm tra xem GPU có khả dụng không
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        current_device = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(current_device)
        
        print(f"✅ GPU khả dụng: {gpu_name}")
        print(f"📊 Số GPU: {gpu_count}")
        print(f"🔧 Đang sử dụng GPU: {current_device}")
        
        # Thiết lập device mặc định
        device = torch.device('cuda')
        torch.cuda.set_device(current_device)
        
        # Hiển thị thông tin bộ nhớ GPU
        gpu_memory = torch.cuda.get_device_properties(current_device).total_memory / (1024**3)  # GB
        print(f"💾 Bộ nhớ GPU: {gpu_memory:.1f} GB")
        
        return device, True
    else:
        print("❌ GPU không khả dụng, sử dụng CPU")
        return torch.device('cpu'), False

def install_dependencies():
    """Tự động cài đặt dependencies với hỗ trợ GPU"""
    packages = [
        "torch",
        "torchvision", 
        "opencv-python", 
        "matplotlib",
        "scikit-learn",
        "pillow",
        "numpy",
        "ultralytics",
        "pandas"
    ]
    
    print("🔧 Kiểm tra và cài đặt dependencies...")
    
    # Kiểm tra phiên bản PyTorch có hỗ trợ GPU không
    try:
        import torch
        if not torch.cuda.is_available():
            print("⚠️ PyTorch không tìm thấy GPU, kiểm tra driver CUDA")
    except ImportError:
        print("📥 PyTorch chưa được cài đặt")
    
    for package in packages:
        try:
            if package == "torch":
                __import__("torch")
            elif package == "torchvision":
                __import__("torchvision")
            elif package == "ultralytics":
                __import__("ultralytics")
            else:
                __import__(package.replace('-', '_'))
            print(f"✅ {package} đã được cài đặt")
        except ImportError:
            print(f"📥 Đang cài đặt {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✅ Đã cài đặt {package}")

# ==================== CLASSROOM BEHAVIOR DETECTION WITH GPU ====================
class ClassroomBehaviorDetectorGPU:
    def __init__(self):
        self.pose_model = None
        self.device = None
        self.use_gpu = False
        self.behavior_history = {}
        
    def initialize_pose_detector(self):
        """Khởi tạo YOLOv11 pose detector với GPU"""
        try:
            from ultralytics import YOLO
            
            # Thiết lập GPU
            self.device, self.use_gpu = setup_gpu()
            
            print("🚀 Đang khởi tạo YOLOv11 Pose detector với GPU...")
            
            # Load model với device specification
            if self.use_gpu:
                # Sử dụng GPU
                self.pose_model = YOLO('yolo11n-pose.pt')
                # Chuyển model sang GPU
                self.pose_model.to(self.device)
                print("✅ YOLOv11 Pose detector đã được tải lên GPU")
            else:
                # Sử dụng CPU
                self.pose_model = YOLO('yolo11n-pose.pt')
                print("✅ YOLOv11 Pose detector đã sẵn sàng (CPU)")
            
            # Test inference để kiểm tra tốc độ
            self._test_inference_speed()
            
            return True
            
        except Exception as e:
            print(f"❌ Lỗi khởi tạo YOLOv11 Pose: {str(e)}")
            return False
    
    def _test_inference_speed(self):
        """Kiểm tra tốc độ inference"""
        print("⏱️ Đang kiểm tra tốc độ inference...")
        
        # Tạo ảnh test
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Warm-up
        for _ in range(3):
            _ = self.pose_model(test_image, verbose=False)
        
        # Đo tốc độ
        start_time = time.time()
        for _ in range(10):
            results = self.pose_model(test_image, verbose=False)
        end_time = time.time()
        
        fps = 10 / (end_time - start_time)
        device_type = "GPU" if self.use_gpu else "CPU"
        print(f"🎯 Tốc độ inference ({device_type}): {fps:.1f} FPS")
    
    def detect_classroom_behaviors(self, image):
        """Nhận diện hành vi học sinh với tối ưu GPU"""
        try:
            # Run pose detection với cài đặt tối ưu cho GPU
            if self.use_gpu:
                # Sử dụng half precision (FP16) để tăng tốc độ trên GPU
                results = self.pose_model(image, verbose=False, half=True)
            else:
                results = self.pose_model(image, verbose=False)
            
            behaviors = []
            
            for result in results:
                if hasattr(result, 'keypoints') and result.keypoints is not None and len(result.keypoints) > 0:
                    for person_idx, keypoints in enumerate(result.keypoints.data):
                        # Chuyển keypoints sang numpy (tự động xử lý device)
                        kpts = keypoints.cpu().numpy()
                        
                        # Phân tích hành vi lớp học
                        behavior_info = self._analyze_classroom_behavior(kpts)
                        
                        # Lấy bounding box
                        bbox = None
                        if hasattr(result, 'boxes') and result.boxes is not None and len(result.boxes) > person_idx:
                            bbox = result.boxes[person_idx].xyxy[0].cpu().numpy()
                        
                        # Gán ID cho mỗi người
                        person_id = f"person_{person_idx}"
                        
                        behaviors.append({
                            'person_id': person_id,
                            'behavior': behavior_info['behavior'],
                            'behavior_score': behavior_info['score'],
                            'keypoints': kpts,
                            'bbox': bbox,
                            'person_idx': person_idx,
                            'details': behavior_info['details']
                        })
            
            return behaviors
            
        except Exception as e:
            print(f"❌ Lỗi nhận diện hành vi: {str(e)}")
            return []
    
    def _analyze_classroom_behavior(self, keypoints):
        """Phân tích hành vi học sinh trong lớp học"""
        try:
            # Chỉ số keypoints theo COCO format
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
            
            # Lấy tọa độ keypoints
            def get_point(idx):
                if keypoints[idx][2] > 0.3:
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
            
            # Tính toán các chỉ số hành vi
            behavior_scores = {
                'ngoi_nghiêm_chỉnh': 0,
                'giơ_tay_phát_biểu': 0,
                'quay_sau_quay_truoc': 0,
                'dung_len': 0,
                'cum_khai': 0,
                'viet_bai': 0,
                'doc_sach': 0
            }
            
            details = []
            
            # 1. Kiểm tra tư thế ngồi
            if (left_hip and right_hip and left_knee and right_knee):
                hip_height = (left_hip[1] + right_hip[1]) / 2
                knee_height = (left_knee[1] + right_knee[1]) / 2
                sitting_ratio = abs(hip_height - knee_height)
                
                if 30 < sitting_ratio < 100:
                    behavior_scores['ngoi_nghiêm_chỉnh'] += 0.8
                    details.append("Ngồi nghiêm chỉnh")
                elif sitting_ratio < 30:
                    behavior_scores['dung_len'] += 0.9
                    details.append("Đứng lên")
            
            # 2. Kiểm tra giơ tay phát biểu
            if left_wrist and left_shoulder and left_wrist[1] < left_shoulder[1] - 20:
                behavior_scores['giơ_tay_phát_biểu'] += 0.9
                details.append("Giơ tay trái")
            if right_wrist and right_shoulder and right_wrist[1] < right_shoulder[1] - 20:
                behavior_scores['giơ_tay_phát_biểu'] += 0.9
                details.append("Giơ tay phải")
            
            # 3. Kiểm tra quay người
            if left_shoulder and right_shoulder:
                shoulder_angle = abs(left_shoulder[0] - right_shoulder[0])
                if shoulder_angle < 30:
                    behavior_scores['quay_sau_quay_truoc'] += 0.7
                    details.append("Quay người")
            
            # 4. Kiểm tra tư thế cúi đầu
            if left_shoulder and right_shoulder and left_hip and right_hip:
                upper_body_angle = abs((left_shoulder[1] + right_shoulder[1])/2 - (left_hip[1] + right_hip[1])/2)
                if upper_body_angle > 50:
                    behavior_scores['viet_bai'] += 0.6
                    behavior_scores['doc_sach'] += 0.6
                    details.append("Cúi người (viết/đọc)")
            
            # 5. Kiểm tra tư thế tay
            if (left_wrist and right_wrist and left_elbow and right_elbow and
                left_shoulder and right_shoulder):
                avg_wrist_y = (left_wrist[1] + right_wrist[1]) / 2
                avg_shoulder_y = (left_shoulder[1] + right_shoulder[1]) / 2
                
                if abs(avg_wrist_y - avg_shoulder_y) < 50:
                    behavior_scores['cum_khai'] += 0.7
                    details.append("Tay để trước ngực")
            
            # Xác định hành vi chính
            main_behavior = max(behavior_scores, key=behavior_scores.get)
            max_score = behavior_scores[main_behavior]
            
            if max_score < 0.5:
                main_behavior = "ngoi_nghiêm_chỉnh"
                max_score = 0.5
                details = ["Tư thế bình thường"]
            
            return {
                'behavior': main_behavior,
                'score': max_score,
                'details': details,
                'all_scores': behavior_scores
            }
            
        except Exception as e:
            print(f"❌ Lỗi phân tích hành vi: {str(e)}")
            return {
                'behavior': "unknown",
                'score': 0,
                'details': ["Không xác định"],
                'all_scores': {}
            }

# ==================== ATTENDANCE & BEHAVIOR LOGGING ====================
class ClassroomLogger:
    def __init__(self, csv_file="classroom_behavior.csv"):
        self.csv_file = csv_file
        self.initialize_log_file()
    
    def initialize_log_file(self):
        """Khởi tạo file log hành vi"""
        try:
            if not os.path.exists(self.csv_file):
                df = pd.DataFrame(columns=[
                    'Timestamp', 
                    'Person_ID', 
                    'Behavior', 
                    'Behavior_Score',
                    'Details',
                    'Device'
                ])
                df.to_csv(self.csv_file, index=False)
                print(f"✅ Đã tạo file log hành vi: {self.csv_file}")
            else:
                df = pd.read_csv(self.csv_file)
                print(f"✅ File log đã tồn tại: {len(df)} records")
        except Exception as e:
            print(f"❌ Lỗi khởi tạo file log: {str(e)}")
            df = pd.DataFrame(columns=[
                'Timestamp', 'Person_ID', 'Behavior', 'Behavior_Score', 'Details', 'Device'
            ])
            df.to_csv(self.csv_file, index=False)
    
    def log_behavior(self, person_id, behavior, score, details, device_type):
        """Ghi log hành vi"""
        try:
            df = pd.read_csv(self.csv_file)
            
            current_time = datetime.now()
            five_seconds_ago = (current_time - pd.Timedelta(seconds=5)).strftime("%H:%M:%S")
            
            recent_logs = df[
                (df['Person_ID'] == person_id) & 
                (df['Timestamp'] > five_seconds_ago)
            ]
            
            if len(recent_logs) == 0:
                new_entry = {
                    'Timestamp': current_time.strftime("%H:%M:%S"),
                    'Person_ID': person_id,
                    'Behavior': behavior,
                    'Behavior_Score': f"{score:.3f}",
                    'Details': ", ".join(details),
                    'Device': device_type
                }
                
                df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)
                df.to_csv(self.csv_file, index=False)
                
                if score > 0.7:
                    behavior_vn = self._translate_behavior(behavior)
                    print(f"📝 {person_id}: {behavior_vn} (Độ tin cậy: {score:.2f}) - {device_type}")
                
                return True
            return False
                
        except Exception as e:
            print(f"❌ Lỗi ghi log: {str(e)}")
            return False
    
    def _translate_behavior(self, behavior):
        """Dịch hành vi sang tiếng Việt"""
        translations = {
            'ngoi_nghiêm_chỉnh': 'Ngồi nghiêm chỉnh',
            'giơ_tay_phát_biểu': 'Giơ tay phát biểu',
            'quay_sau_quay_truoc': 'Quay sau/quay trước',
            'dung_len': 'Đứng lên',
            'cum_khai': 'Chụm khai (tay để bàn)',
            'viet_bai': 'Viết bài',
            'doc_sach': 'Đọc sách',
            'unknown': 'Không xác định'
        }
        return translations.get(behavior, behavior)
    
    def view_behavior_logs(self):
        """Xem lịch sử hành vi"""
        try:
            if not os.path.exists(self.csv_file):
                print("📭 Chưa có file log hành vi")
                return
                
            df = pd.read_csv(self.csv_file)
            if len(df) > 0:
                print("\n📊 LỊCH SỬ HÀNH VI LỚP HỌC:")
                print("=" * 100)
                for _, row in df.iterrows():
                    behavior_vn = self._translate_behavior(row['Behavior'])
                    print(f"🕒 {row['Timestamp']} | 👤 {row['Person_ID']} | 🎯 {behavior_vn} | 📈 {row['Behavior_Score']} | 💻 {row['Device']}")
                print("=" * 100)
                print(f"📈 Tổng số lượt ghi nhận: {len(df)}")
                
                # Thống kê theo device
                device_stats = df['Device'].value_counts()
                print(f"\n📱 Thống kê theo thiết bị:")
                for device, count in device_stats.items():
                    print(f"  {device}: {count} lượt")
                    
            else:
                print("📭 Chưa có dữ liệu hành vi")
        except Exception as e:
            print(f"❌ Lỗi đọc file log: {str(e)}")

# ==================== REAL-TIME CLASSROOM MONITORING WITH GPU ====================
def real_time_classroom_monitoring_gpu():
    """Giám sát hành vi lớp học real-time với GPU"""
    detector = ClassroomBehaviorDetectorGPU()
    logger = ClassroomLogger()
    
    if not detector.initialize_pose_detector():
        print("❌ Không thể khởi tạo hệ thống")
        return
    
    device_type = "GPU" if detector.use_gpu else "CPU"
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Không thể mở webcam!")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print(f"🎓 HỆ THỐNG GIÁM SÁT LỚP HỌC - CHẠY TRÊN {device_type}")
    print("📊 Đang nhận diện các hành vi...")
    print("🎮 Nhấn 'q' để thoát, 's' để chụp ảnh, 'v' để xem log hành vi")
    
    frame_count = 0
    behavior_results = []
    fps_history = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        start_time = time.time()
        
        # Tăng tần suất detection khi dùng GPU
        detection_interval = 2 if detector.use_gpu else 4
        if frame_count % detection_interval == 0:
            behavior_results = detector.detect_classroom_behaviors(frame)
        
        # Vẽ kết quả lên frame
        for behavior in behavior_results:
            if behavior['bbox'] is not None:
                try:
                    x1, y1, x2, y2 = behavior['bbox'].astype(int)
                    
                    # Màu sắc theo hành vi
                    color_map = {
                        'ngoi_nghiêm_chỉnh': (0, 255, 0),
                        'giơ_tay_phát_biểu': (255, 255, 0),
                        'viet_bai': (255, 165, 0),
                        'doc_sach': (255, 165, 0),
                        'cum_khai': (0, 255, 255),
                        'quay_sau_quay_truoc': (0, 0, 255),
                        'dung_len': (0, 0, 255),
                        'unknown': (128, 128, 128)
                    }
                    
                    behavior_type = behavior['behavior']
                    color = color_map.get(behavior_type, (128, 128, 128))
                    
                    # Vẽ bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Hiển thị hành vi
                    behavior_vn = logger._translate_behavior(behavior_type)
                    behavior_text = f"{behavior_vn} ({behavior['behavior_score']:.1f})"
                    
                    cv2.putText(frame, behavior_text, (x1, y1 - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    
                    # Ghi log hành vi
                    logger.log_behavior(
                        behavior['person_id'], 
                        behavior_type, 
                        behavior['behavior_score'],
                        behavior['details'],
                        device_type
                    )
                    
                except Exception as e:
                    continue
        
        # Tính FPS
        end_time = time.time()
        fps = 1.0 / (end_time - start_time)
        fps_history.append(fps)
        if len(fps_history) > 30:
            fps_history.pop(0)
        avg_fps = sum(fps_history) / len(fps_history)
        
        # Hiển thị thông tin hiệu suất
        active_students = len(behavior_results)
        info_text = f"FPS: {avg_fps:.1f} | Học sinh: {active_students} | Device: {device_type}"
        cv2.putText(frame, info_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Hiển thị trạng thái GPU
        if detector.use_gpu:
            gpu_status = f"GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}"
            cv2.putText(frame, gpu_status, (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        cv2.imshow(f'Classroom Behavior Monitoring - {device_type}', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            filename = f"classroom_{device_type}_{int(time.time())}.jpg"
            cv2.imwrite(filename, frame)
            print(f"✅ Đã lưu ảnh: {filename}")
        elif key == ord('v'):
            logger.view_behavior_logs()
    
    cap.release()
    cv2.destroyAllWindows()
    print(f"👋 Đã thoát hệ thống giám sát ({device_type})!")

# ==================== MAIN MENU ====================
def main_menu():
    """Hiển thị menu chính"""
    while True:
        print("\n" + "="*70)
        print("🎓 HỆ THỐNG GIÁM SÁT HÀNH VI LỚP HỌC - GPU SUPPORT")
        print("="*70)
        print("1. 🎥 Bắt đầu giám sát real-time (GPU/CPU tự động)")
        print("2. 📊 Xem lịch sử hành vi")
        print("3. 🔧 Kiểm tra thông tin GPU")
        print("4. 🚪 Thoát")
        print("="*70)
        
        choice = input("👉 Chọn chức năng (1-4): ").strip()
        
        if choice == "1":
            real_time_classroom_monitoring_gpu()
        elif choice == "2":
            logger = ClassroomLogger()
            logger.view_behavior_logs()
        elif choice == "3":
            setup_gpu()
        elif choice == "4":
            print("👋 Tạm biệt!")
            break
        else:
            print("❌ Lựa chọn không hợp lệ!")
        
        input("\n👉 Nhấn Enter để tiếp tục...")

# ==================== MAIN ====================
if __name__ == "__main__":
    print("🔧 Đang kiểm tra hệ thống và GPU...")
    install_dependencies()
    
    print("\n🎓 KHỞI ĐỘNG HỆ THỐNG GIÁM SÁT LỚP HỌC VỚI GPU")
    print("📊 Nhận diện hành vi học sinh với AI tối ưu hóa GPU")
    
    main_menu()