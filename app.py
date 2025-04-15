from flask import Flask, render_template, request, redirect, url_for, send_from_directory, Response
from ultralytics import YOLO
import os
import cv2
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Folders
UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER

# Load YOLOv10n model
model = YOLO('best.pt')

##############################
# ROUTES
##############################

@app.route('/')
def index():
    return render_template("home.html")

@app.route('/home')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('abstract.html')

@app.route('/camera')
def camera_page():
    return render_template('new.html')


##############################
# IMAGE/VIDEO UPLOAD
##############################

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(url_for('home'))

    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('home'))

    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)

        # VIDEO
        if filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            result_video_path = process_video(file_path, filename)
            return render_template('result.html',
                                   original_file=filename,
                                   result_video=result_video_path)

        # IMAGE
        else:
            results = model(file_path)
            result_img_path = os.path.join(RESULTS_FOLDER, f"result_{filename}")
            annotated_frame = results[0].plot()
            cv2.imwrite(result_img_path, annotated_frame)

            return render_template('result.html',
                                   original_file=filename,
                                   result_file=f"result_{filename}")


##############################
# VIDEO PROCESSING FUNCTION
##############################

def process_video(input_path, filename):
    cap = cv2.VideoCapture(input_path)
    fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264 codec
    output_path = os.path.join(RESULTS_FOLDER, f"result_{filename}")

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        annotated_frame = results[0].plot()
        out.write(annotated_frame)

    cap.release()
    out.release()

    return f"result_{filename}"


##############################
# FILE SERVING
##############################

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/results/<filename>')
def result_file(filename):
    return send_from_directory(RESULTS_FOLDER, filename)

@app.route('/results/videos/<filename>')
def result_video(filename):
    return send_from_directory(RESULTS_FOLDER, filename, mimetype='video/mp4')


##############################
# LIVE CAMERA STREAMING
##############################

camera = None

# def generate_frames():
#     global camera
#     camera = cv2.VideoCapture(0)

#     if not camera.isOpened():
#         print("Error: Could not open camera.")
#         return

#     while True:
#         success, frame = camera.read()
#         if not success:
#             break

#         # YOLO detection
#         results = model.predict(frame, verbose=False)
#         for result in results[0].boxes.data.tolist():
#             x1, y1, x2, y2, conf, cls = result
#             x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
#             label = f"{model.names[int(cls)]} {conf:.2f}"
#             cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#             cv2.putText(frame, label, (x1, y1 - 10),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

#         _, buffer = cv2.imencode('.jpg', frame)
#         frame_bytes = buffer.tobytes()

#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
def generate_frames():
    global camera
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("Error: Could not open camera.")
        return

    while True:
        success, frame = camera.read()
        if not success:
            break

        # Flip the frame horizontally to fix mirrored text
        frame = cv2.flip(frame, 1)

        # YOLO detection
        results = model.predict(frame, verbose=False)
        for result in results[0].boxes.data.tolist():
            x1, y1, x2, y2, conf, cls = result
            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
            label = f"{model.names[int(cls)]} {conf:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/close_camera', methods=['POST'])
def close_camera():
    global camera
    if camera:
        camera.release()
        camera = None
    return '', 204


if __name__ == '__main__':
    app.run(debug=True)
