# Autonomous Parking

Detect rectangular parking areas in images and a live camera feed using **YOLOv8**. This project includes a labeled dataset, a training and evaluation notebook, trained weights, and a desktop camera application built with OpenCV and Tkinter.

The current implementation provides parking-area detection; vehicle steering and parking control are outside its scope.

## Features

- One detection class: `parking_area` (class ID `0`).
- YOLOv8 nano (`yolov8n.pt`) training on a custom dataset.
- Validation metrics, training plots, and test-image predictions.
- Live camera preview with bounding boxes, class names, and confidence scores.
- Configurable camera index and detection confidence threshold.

## Requirements

- Python **3.10 or 3.11** recommended, with `pip` and virtual environment support.
- Git to clone the repository.
- A webcam or USB camera and a graphical desktop for `parking.py`.
- CPU support; an NVIDIA GPU with a compatible CUDA-enabled PyTorch installation is optional for faster training.

Python packages: `ultralytics`, `torch`, `torchvision`, `opencv-python`, `pillow`, `numpy`, `pyyaml`, `pandas`, `matplotlib`, and `seaborn`. The notebook also needs `jupyter` and `ipykernel`. Tkinter is supplied by Python or your operating system, rather than installed through `pip`.

## Installation and Setup

Clone the repository and create an isolated Python environment:

```bash
git clone https://github.com/arabyte-src/AUTONOMOUS-PARKING.git
cd AUTONOMOUS-PARKING
python -m venv .venv
```

Activate the environment:

```bash
# Linux / macOS
source .venv/bin/activate
```

```powershell
# Windows PowerShell
.venv\Scripts\Activate.ps1
```

Install dependencies. These commands use CPU-only PyTorch, matching the notebook's default setup:

```bash
python -m pip install --upgrade pip
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install ultralytics opencv-python pillow numpy pyyaml pandas matplotlib seaborn
```

For GPU support, install the appropriate PyTorch build using the [official installation guide](https://pytorch.org/get-started/locally/) instead of the CPU-only command.

Check Tkinter with `python -m tkinter`; a small window should open. If Tkinter is missing on Ubuntu/Debian, install it with `sudo apt install python3-tk`. Use the desktop `opencv-python` package for the camera application.

**Run all remaining terminal commands from the project directory:**

```bash
cd autonomous_parking
```

## Dataset Structure

```text
AUTONOMOUS-PARKING/
├── README.md
└── autonomous_parking/
    ├── parking.py
    ├── autonomous_parking_yolo.ipynb
    ├── yolov8n.pt
    ├── parking_area_yolo_dataset/
    │   ├── data.yaml
    │   ├── train/
    │   │   ├── images/
    │   │   └── labels/
    │   ├── valid/
    │   │   ├── images/
    │   │   └── labels/
    │   └── test/
    │       ├── images/
    │       └── labels/
    └── parking_area_runs/
        ├── data_local.yaml
        └── yolo_training/
            ├── results.csv
            ├── results.png
            └── weights/
                ├── best.pt
                └── last.pt
```

The current dataset contains **291 training, 91 validation, and 49 test images**. Each image has a matching `.txt` label with the same filename stem. Each bounding-box annotation follows normalized YOLO format:

```text
class_id x_center y_center width height
```

Coordinates and dimensions are fractions of the image size, between `0` and `1`. The dataset's original synthetic examples are supplemented by additional images. Evaluate on real camera footage to assess performance in your intended setting. The dataset's bundled `README.txt` lists the original split sizes; the counts above reflect the current image folders. The test labels folder also contains one label without a matching image (`new_IMG_20261020_223044.txt`); it is not a test image.

### Set Your Dataset Path

The bundled YAML files contain paths from other machines. Before training or evaluation, edit `parking_area_yolo_dataset/data.yaml` to use the **absolute path** to your local dataset:

```yaml
path: /absolute/path/to/AUTONOMOUS-PARKING/autonomous_parking/parking_area_yolo_dataset
train: train/images
val: valid/images
test: test/images
nc: 1
names:
  0: parking_area
```

On Windows, use forward slashes, for example `C:/Projects/AUTONOMOUS-PARKING/autonomous_parking/parking_area_yolo_dataset`. Keep the folder name `valid` even though the YAML key is `val`.

## Train YOLOv8

Train from the included YOLOv8 nano pretrained weights:

```bash
yolo detect train model=yolov8n.pt data=parking_area_yolo_dataset/data.yaml epochs=50 imgsz=640 batch=16 device=cpu project=parking_area_runs name=yolo_training
```

Use `device=0` with a CUDA-enabled PyTorch installation to train on the first NVIDIA GPU. Reduce `batch` (for example, to `4`) if memory is limited. See the [Ultralytics training documentation](https://docs.ultralytics.com/modes/train/) for parameter details.

The repository already includes `parking_area_runs/yolo_training/`. Ultralytics normally creates an incremented directory for a new run, such as `yolo_training2`; check the output path printed in the terminal. Use that run's `weights/best.pt` for evaluation and live detection. `last.pt` stores the final checkpoint, while `results.csv` and `results.png` contain training metrics and curves.

### Optional: Use the Notebook

```bash
python -m pip install jupyter ipykernel
jupyter notebook autonomous_parking_yolo.ipynb
```

Select the kernel for your virtual environment. In the imports/configuration cell, update `DATASET_DIR` and `RUNS_DIR` to your local absolute paths, then run the cells in order. The notebook creates `parking_area_runs/data_local.yaml` from these paths and includes dataset inspection, training, validation, and test-image visualization. Its training cell uses `exist_ok=True`, so change the run name or output directory to preserve previous results.

## Model Evaluation

Evaluate the included trained model on the validation split:

```bash
yolo detect val model=parking_area_runs/yolo_training/weights/best.pt data=parking_area_yolo_dataset/data.yaml split=val imgsz=640 device=cpu project=parking_area_runs name=validation plots=True
```

Evaluate on the held-out test split after choosing your model:

```bash
yolo detect val model=parking_area_runs/yolo_training/weights/best.pt data=parking_area_yolo_dataset/data.yaml split=test imgsz=640 device=cpu project=parking_area_runs name=test_evaluation plots=True
```

Replace the model path with your new run's `best.pt` when evaluating a retrained model. Both commands report **precision**, **recall**, **mAP@50**, and **mAP@50:95**, and save plots and annotated examples in the printed output directory. Higher values generally indicate better detection performance; inspect prediction images for missed areas and false detections. See the [Ultralytics validation documentation](https://docs.ultralytics.com/modes/val/) for metric details.

The notebook's test “success rate” measures whether any detection was returned. Use the labeled test-set evaluation above to measure bounding-box accuracy.

## Run Live Camera Detection

Edit the configuration section near the top of `parking.py`. Replace its existing machine-specific `MODEL_PATH` with the included trained weights or your newly trained model:

```python
MODEL_PATH = "parking_area_runs/yolo_training/weights/best.pt"
CONFIDENCE = 0.70
CAMERA_INDEX = 0
```

With the virtual environment active and your terminal in `autonomous_parking/`, run:

```bash
python parking.py
```

The desktop window displays the camera feed with detected parking areas and the highest detection confidence. If no box meets the threshold, it displays **NO PARKING AREA DETECTED**. Close the window to stop detection and release the camera.

- **Camera does not open:** allow camera access, close other apps using it, or try `CAMERA_INDEX = 1`.
- **Model file not found:** check `MODEL_PATH`; relative paths are resolved from the terminal's working directory.
- **Few detections:** check lighting and camera framing, then try a lower `CONFIDENCE`, such as `0.50`.
- **No display / Tkinter error:** run the application in a local graphical desktop session with Tkinter installed.
