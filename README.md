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

How it Works

1. Capture video input

2. Detect and crop eye region

3. Feed sequence of frames to LRCN model

4. Predict blink / no blink

   

   ![WhatsApp Image 2025-04-15 at 23 20 43_e1c855ee](https://github.com/user-attachments/assets/0cce795d-48c3-483c-afb4-4a004336642f)![WhatsApp Image 2025-04-15 at 23 16 30_975f0a92](https://github.com/user-attachments/assets/c95823ce-4b91-440e-b73d-828e9adafceb)
![WhatsApp Image 2025-04-15 at 23 17 37_c275ffcb](https://github.com/user-attachments/assets/b21ffe76-3b8f-449f-a970-cc8cc9172d93)
![WhatsApp Image 2025-04-15 at 23 20 15_76ef1050](https://github.com/user-attachments/assets/cbbc578b-059b-4c62-8ffb-68c3d981bde8)
![WhatsApp Image 2025-04-15 at 23 20 43_e1c855ee](https://github.com/user-attachments/assets/01a96d06-5a0e-44d3-a963-4ea37396d6d5)


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
