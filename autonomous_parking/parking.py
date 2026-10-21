import tkinter as tk
from tkinter import ttk
import cv2
from PIL import Image, ImageTk
from ultralytics import YOLO

# ==========================================================
# CONFIGURATION
# ==========================================================

MODEL_PATH = "/home/ayskopi/Downloads/best.pt"
CONFIDENCE = 0.70
CAMERA_INDEX = 0 

# ==========================================================
# LOAD MODEL
# ==========================================================

model = YOLO(MODEL_PATH)

# ==========================================================
# CAMERA
# ==========================================================

cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    raise RuntimeError(
        "Could not open camera. "
        "Check your camera index or permissions."
    )

# ==========================================================
# TKINTER WINDOW
# ==========================================================

root = tk.Tk()
root.title("Autonomous Parking - YOLO Live Test")
root.geometry("1000x750")

video_label = ttk.Label(root)
video_label.pack(padx=10, pady=10)

status_label = ttk.Label(
    root,
    text="Starting camera...",
    font=("Arial", 16)
)
status_label.pack(pady=10)

# ==========================================================
# UPDATE CAMERA
# ==========================================================

def update_frame():

    ret, frame = cap.read()

    if not ret:
        status_label.config(
            text="ERROR: Could not read camera"
        )
        root.after(100, update_frame)
        return

    # ------------------------------------------------------
    # YOLO DETECTION
    # ------------------------------------------------------

    results = model.predict(
        source=frame,
        conf=CONFIDENCE,
        verbose=False
    )

    result = results[0]

    # ------------------------------------------------------
    # DRAW DETECTIONS
    # ------------------------------------------------------

    annotated = result.plot()

    # ------------------------------------------------------
    # DETECTION STATUS
    # ------------------------------------------------------

    if result.boxes is not None and len(result.boxes) > 0:

        confidences = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()

        highest_conf = float(max(confidences))

        detected_names = []

        for cls in classes:
            detected_names.append(
                model.names[int(cls)]
            )

        status_label.config(
            text=(
                f"DETECTED | "
                f"{', '.join(detected_names)} | "
                f"Confidence: {highest_conf:.2%}"
            )
        )

    else:

        status_label.config(
            text="NO PARKING AREA DETECTED"
        )

    # ------------------------------------------------------
    # DISPLAY FRAMES
    # ------------------------------------------------------

    frame_rgb = cv2.cvtColor(
        annotated,
        cv2.COLOR_BGR2RGB
    )

    image = Image.fromarray(frame_rgb)

    # Resize for window
    image.thumbnail((960, 600))

    photo = ImageTk.PhotoImage(image)

    video_label.configure(image=photo)
    video_label.image = photo

    # Continue updating
    root.after(30, update_frame)


# ==========================================================
# CLOSE WINDOW
# ==========================================================

def close_camera():

    cap.release()
    root.destroy()


root.protocol("WM_DELETE_WINDOW", close_camera)

# Start
update_frame()

root.mainloop()