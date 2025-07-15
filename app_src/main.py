# main.py
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from flask import Flask, render_template, Response, request, jsonify
from flask_socketio import SocketIO, emit
import time
import datetime
from video_processing import get_fire_detection_frame, release_camera, initialize_camera, set_confidence_threshold

app = Flask(__name__,
            static_folder='static',
            template_folder='static')
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, async_mode='threading')

fire_events = [] # To store fire event timestamps and snapshot paths

@app.route('/')
def index():
    return render_template('index.html')

def gen_frames():
    if not initialize_camera():
        print("Camera initialization failed in gen_frames. Will show error image.")
        while True:
            frame_bytes, _, _, _ = get_fire_detection_frame()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.1)

    last_fire_alert_time = 0
    alert_interval = 10 # seconds

    while True:
        frame_bytes, fire_detected, detection_details, snapshot_path = get_fire_detection_frame()
        if fire_detected and detection_details: # Ensure detection_details is not empty
            current_time = time.time()
            if current_time - last_fire_alert_time > alert_interval:
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                current_snapshot_path = snapshot_path
                
                event_info = {
                    "time": timestamp, 
                    "details": detection_details,
                    "snapshot_path": current_snapshot_path
                }
                fire_events.append(event_info)
                socketio.emit('fire_alert', event_info, namespace='/fire_detection')
                print(f"Fire alert sent: {event_info}")
                last_fire_alert_time = current_time
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        socketio.sleep(0.05)

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_fire_events', methods=['GET'])
def get_fire_events_route():
    return jsonify(fire_events)

@socketio.on('connect', namespace='/fire_detection')
def handle_connect():
    print('Client connected to fire_detection namespace')

@socketio.on('disconnect', namespace='/fire_detection')
def handle_disconnect():
    print('Client disconnected from fire_detection namespace')

@app.route('/change_camera', methods=['POST'])
def change_camera_route():
    data = request.get_json()
    camera_index = data.get('camera_index', 0)
    if isinstance(camera_index, int):
        print(f"Attempting to switch to camera index: {camera_index}")
        if initialize_camera(camera_index):
            return jsonify({"success": True, "message": f"Switched to camera {camera_index}"})
        else:
            initialize_camera(0) 
            return jsonify({"success": False, "message": f"Failed to switch to camera {camera_index}. Reverted to default camera if available."}), 500
    return jsonify({"success": False, "message": "Invalid camera index"}), 400

@app.route('/set_confidence', methods=['POST'])
def set_confidence_route():
    data = request.get_json()
    threshold = data.get('threshold')
    if threshold is not None:
        if set_confidence_threshold(threshold):
            return jsonify({"success": True, "message": f"Confidence threshold set to {threshold}"})
        else:
            return jsonify({"success": False, "message": "Invalid threshold value"}), 400
    return jsonify({"success": False, "message": "Threshold not provided"}), 400

@app.route("/static/fire_captures/<filename>")
def serve_capture(filename):
    return app.send_static_file(os.path.join("fire_captures", filename))

if __name__ == '__main__':
    static_dir = os.path.join(os.path.dirname(__file__), 'static')
    error_image_path = os.path.join(static_dir, 'camera_error.png')
    if not os.path.exists(error_image_path):
        if not os.path.exists(static_dir):
            os.makedirs(static_dir)
        try:
            import numpy as np
            import cv2
            error_img_data = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(error_img_data, "Camera Error", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.imwrite(error_image_path, error_img_data)
            print(f"Created dummy error image at {error_image_path}")
        except Exception as e:
            print(f"Could not create dummy error image: {e}")

    print("Starting Flask-SocketIO server...")
    socketio.run(app, host='0.0.0.0', port=5000, debug=False, allow_unsafe_werkzeug=True)