Eye Blink Detection using LRCN
This project implements real-time eye blink detection using a Long-term Recurrent Convolutional Network (LRCN) model. It takes a video feed, processes eye region frames, and classifies blink events.
Features:
Real-time video processing
Eye detection and frame preprocessing
LRCN model trained for blink classification
Visualization of blink prediction results

Project Structure
app.py                  → Main code to run detection

MODEL_TRAIN.ipynb       → Notebook for training the model

best.pt / yolov8n.pt    → Trained models

requirements.txt        → Dependencies

dataset.yaml / data.yaml → Dataset configuration


Installation

git clone https://github.com/shujath876/EYE-blink-detection_project.git
cd EYE-blink-detection_project
pip install -r requirements.txt
python app.py

Demo

Insert link to demo video or image (optional)

How it Works

1. Capture video input

2. Detect and crop eye region

3. Feed sequence of frames to LRCN model

4. Predict blink / no blink

Applications

Drowsiness detection in drivers

Eye-based human-computer interaction (HCI)

Assistive tech for differently-abled individuals

Author
Shujath Ahmed
Final Year B.Tech Student
 Email: imhussain003@gmail.com


---

Feel free to copy-paste this into your README.md file! Let me know if you'd like help with anything else, such as pushing this to GitHub or setting up a demo.
