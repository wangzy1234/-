import cv2
from ultralytics import YOLO
import os
import datetime

# 获取当前文件所在的目录，然后构建模型的绝对路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "yolo_model", "YOLOv8n+SE.pt")
CAPTURES_DIR = os.path.join(BASE_DIR, "static", "fire_captures")

# 确保截图目录存在
if not os.path.exists(CAPTURES_DIR):
    os.makedirs(CAPTURES_DIR)

# 加载YOLOv8+SE模型并强制使用CPU
model = YOLO(MODEL_PATH)
model.to("cpu") # 确保模型在CPU上运行

# 尝试打开默认摄像头
camera = None
current_confidence_threshold = 0.5 # Default confidence threshold

def set_confidence_threshold(threshold):
    global current_confidence_threshold
    try:
        new_threshold = float(threshold)
        if 0.0 <= new_threshold <= 1.0:
            current_confidence_threshold = new_threshold
            print(f"Confidence threshold set to: {current_confidence_threshold}")
            return True
        else:
            print(f"Invalid threshold value: {new_threshold}. Must be between 0.0 and 1.0.")
            return False
    except ValueError:
        print(f"Invalid threshold format: {threshold}")
        return False

def initialize_camera(camera_index=0):
    global camera
    if camera is not None:
        camera.release()
    camera = cv2.VideoCapture(camera_index)
    if not camera.isOpened():
        print(f"错误：无法打开摄像头索引 {camera_index}")
        camera = None
        return False
    print(f"摄像头 {camera_index} 初始化成功")
    return True

# 首次尝试初始化摄像头
initialize_camera()

def get_fire_detection_frame():
    global camera
    global current_confidence_threshold
    fire_detected_this_frame = False
    detection_info = [] # 用于存储检测到的火灾信息
    snapshot_path = None

    if camera is None or not camera.isOpened():
        if not initialize_camera():
            error_frame_path = os.path.join(BASE_DIR, "static", "camera_error.png")
            if not os.path.exists(error_frame_path):
                import numpy as np
                error_img_data = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(error_img_data, "Camera Error", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.imwrite(error_frame_path, error_img_data)
            
            error_frame = cv2.imread(error_frame_path)
            ret, buffer = cv2.imencode(".jpg", error_frame)
            return buffer.tobytes(), fire_detected_this_frame, detection_info, snapshot_path

    success, frame = camera.read()
    if not success:
        error_frame_path = os.path.join(BASE_DIR, "static", "camera_error.png")
        error_frame = cv2.imread(error_frame_path)
        if error_frame is None:
            import numpy as np
            error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(error_frame, "Frame Read Error", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        ret, buffer = cv2.imencode(".jpg", error_frame)
        return buffer.tobytes(), fire_detected_this_frame, detection_info, snapshot_path

    original_frame_for_snapshot = frame.copy()
    results = model(frame, verbose=False, conf=current_confidence_threshold) # Pass confidence to model
    annotated_frame = results[0].plot()

    if results[0].boxes is not None and len(results[0].boxes) > 0:
        for i in range(len(results[0].boxes)):
            cls = int(results[0].boxes.cls[i].item())
            conf = float(results[0].boxes.conf[i].item())

            if cls == 0: # Assuming class 0 is fire
                fire_detected_this_frame = True
                box = results[0].boxes.xyxy[i].cpu().numpy().astype(int)
                
                timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                snapshot_filename = f"fire_capture_{timestamp_str}.jpg"
                relative_snapshot_path = os.path.join("static", "fire_captures", snapshot_filename)
                absolute_snapshot_path = os.path.join(CAPTURES_DIR, snapshot_filename)
                
                cv2.imwrite(absolute_snapshot_path, annotated_frame) 
                snapshot_path = relative_snapshot_path

                detection_info.append({
                    "class_id": cls,
                    "confidence": round(conf, 2),
                    "box": box.tolist(),
                    "snapshot": snapshot_path
                })
                break 

    ret, buffer = cv2.imencode(".jpg", annotated_frame)
    return buffer.tobytes(), fire_detected_this_frame, detection_info, snapshot_path

def release_camera():
    global camera
    if camera is not None:
        camera.release()
        camera = None
        print("摄像头已释放")

if __name__ == "__main__":
    if not initialize_camera():
        print("无法初始化摄像头，退出测试。")
    else:
        set_confidence_threshold(0.3) # Example: Set lower threshold for testing
        while True:
            frame_bytes, fire_detected, detections, snapshot = get_fire_detection_frame()
            if frame_bytes:
                import numpy as np
                nparr = np.frombuffer(frame_bytes, np.uint8)
                img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if img_np is not None:
                    cv2.imshow("Fire Detection Test", img_np)
                    if fire_detected:
                        print(f"火灾检测到! 详情: {detections}")
                        if snapshot:
                            print(f"截图已保存: {snapshot}")
                else:
                    print("无法解码帧")
                    break
            else:
                print("获取帧失败")
                break
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        release_camera()
        cv2.destroyAllWindows()

