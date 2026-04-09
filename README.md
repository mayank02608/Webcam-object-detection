# 🎯 Real-Time Object Detection using YOLO & OpenCV

This project implements a **real-time object detection system** using the **Ultralytics YOLO model** and **OpenCV**. It captures live video from a webcam and detects multiple objects with bounding boxes and labels.

---

## 🚀 Features

* 📷 Real-time webcam object detection
* 🧠 Powered by YOLO (You Only Look Once)
* 🎯 Detects 80+ object classes (COCO dataset)
* 📦 Bounding boxes with labels and confidence scores
* ⚡ Fast and efficient performance

---

## 🛠️ Technologies Used

* Python
* OpenCV
* Ultralytics YOLO
* NumPy

---

## 📂 Project Structure

```
├── yolo-Weights/
│   └── yolov8n.pt
├── main.py
└── README.md
```

---

## ⚙️ Installation

### 1. Clone the Repository

```
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### 2. Install Dependencies

```
pip install ultralytics opencv-python
```

---

## ▶️ Usage

Run the script:

```
python main.py
```

* The webcam will open automatically.
* Detected objects will be shown with bounding boxes.
* Press **'q'** to exit.

---

## 🧠 Model Information

* Model: `yolov8n.pt` (Nano version for fast inference)
* Dataset: COCO (Common Objects in Context)
* Supports detection of objects like:

  * Person 👤
  * Car 🚗
  * Dog 🐶
  * Chair 🪑
  * Laptop 💻
  * and many more...

---

## 📸 Output Example

* Live video feed with:

  * Green bounding boxes
  * Blue labels with confidence scores

---

## 🔧 Customization

You can modify the following:

* 🎯 Confidence threshold:

```python
if conf > 0.5:
```

* 📏 Frame size:

```python
cap.set(3, 640)
cap.set(4, 480)
```

* 🧠 Model size:

  * `yolov8n.pt` (fastest)
  * `yolov8s.pt` (better accuracy)
  * `yolov8m.pt` (balanced)

---

## ⚡ Performance Tips

* Use GPU for faster processing:

```
model.to("cuda")
```

* Lower resolution for higher FPS

---

## 📌 Future Improvements

* 🔢 Object counting system
* 📏 Distance estimation
* 🎥 Video file input support
* 🌐 Web app integration (Flask/Streamlit)

---

## 🤝 Contributing

Contributions are welcome! Feel free to fork this repo and submit a pull request.

---

## 📄 License

This project is open-source and available under the MIT License.

---

## 🙌 Acknowledgements

* Ultralytics for YOLO implementation
* OpenCV community

---

## ⭐ Support

If you like this project, give it a ⭐ on GitHub!

---
