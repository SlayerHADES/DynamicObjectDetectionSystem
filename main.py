import cv2
import numpy as np
from keras.models import load_model
import time

# === Config ===
MODEL_PATH = "keras_Model.h5"
LABELS_PATH = "labels.txt"
INPUT_SIZE = (224, 224)
CONFIDENCE_THRESHOLD = 0.9
MOTION_SENSITIVITY = 30
MIN_MOTION_AREA = 500
BOX_COLOR = (0, 255, 0)
TEXT_COLOR = (0, 255, 0)

# === Load Model and Labels ===
model = load_model(MODEL_PATH, compile=False)
class_names = [line.strip() for line in open(LABELS_PATH, "r").readlines()]

# === Initialize Camera ===
camera = cv2.VideoCapture(0)
prev_gray = None
fps_start_time = time.time()
frame_count = 0

while True:
    ret, frame = camera.read()
    if not ret:
        print("Failed to grab frame.")
        break

    display_frame = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    motion_boxes = []

    # === Motion Detection ===
    if prev_gray is not None:
        frame_diff = cv2.absdiff(prev_gray, gray)
        _, thresh = cv2.threshold(frame_diff, MOTION_SENSITIVITY, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w * h > MIN_MOTION_AREA:
                motion_boxes.append((x, y, w, h))

    prev_gray = gray.copy()

    # === Prediction ===
    for (x, y, w, h) in motion_boxes:
        roi = frame[y:y + h, x:x + w]
        if roi.size == 0:
            continue

        resized = cv2.resize(roi, INPUT_SIZE, interpolation=cv2.INTER_AREA)
        input_data = np.asarray(resized, dtype=np.float32).reshape(1, *INPUT_SIZE, 3)
        input_data = (input_data / 127.5) - 1

        prediction = model.predict(input_data, verbose=0)
        index = np.argmax(prediction)
        confidence = prediction[0][index]

        if confidence >= CONFIDENCE_THRESHOLD:
            label = f"{class_names[index]} {int(confidence * 100)}%"
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), BOX_COLOR, 2)
            cv2.putText(display_frame, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 2)

    # === FPS Display ===
    frame_count += 1
    elapsed_time = time.time() - fps_start_time
    if elapsed_time > 1.0:
        fps = frame_count / elapsed_time
        fps_text = f"FPS: {fps:.1f}"
        cv2.putText(display_frame, fps_text, (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        frame_count = 0
        fps_start_time = time.time()

    # === Show Output ===
    cv2.imshow("Dynamic Object Detection", display_frame)

    # ESC to exit
    if cv2.waitKey(1) & 0xFF == 27:
        break

# === Cleanup ===
camera.release()
cv2.destroyAllWindows()
