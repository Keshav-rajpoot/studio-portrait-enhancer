# 🎥 Studio-Quality Portrait Enhancement Pipeline

## 📌 Overview

This project converts **raw human portrait images** captured in uncontrolled conditions (mobile camera, low light, cluttered background, motion blur) into **studio-quality portraits** using a **hybrid Computer Vision + Deep Learning pipeline**.

The system enhances facial clarity, adds professional background bokeh, improves contrast and sharpness, and preserves **natural skin texture and original facial identity**, while maintaining **fast CPU-based inference**.

---

## 🎯 Problem Statement

Given a raw portrait image:

* Motion blur
* Uneven lighting
* Low contrast
* Noisy background

Generate a **studio-quality portrait** with:

* Clear and sharp face
* Natural skin texture
* Background blur (bokeh effect)
* Identity preservation
* Fast inference performance

---

## 🧠 Solution Approach (Pipeline Design)

```
Input Image
   ↓
Basic Image Enhancement (Contrast + Sharpness)
   ↓
Face Detection (MediaPipe)
   ↓
Face Enhancement (GFPGAN)
   ↓
Human Segmentation
   ↓
Background Blur (Portrait/Bokeh Effect)
   ↓
Final Studio-Quality Output
```

Each stage is modular, optimized, and production-oriented.

---

## 🔧 Technologies Used

| Component        | Technology                    |
| ---------------- | ----------------------------- |
| Language         | Python 3.10                   |
| Image Processing | OpenCV                        |
| Face Detection   | MediaPipe                     |
| Face Enhancement | GFPGAN                        |
| Deep Learning    | PyTorch                       |
| Segmentation     | MediaPipe Selfie Segmentation |
| Environment      | CPU (Fast inference)          |

---

## 📂 Project Structure

```
studio-portrait-enhancer/
│
├── app.py                     # Entry point
├── requirements.txt           # Dependencies
├── README.md
│
├── src/
│   ├── __init__.py
│   ├── post_process.py        # Contrast & sharpness enhancement
│   ├── face_detect.py         # Face detection
│   ├── face_enhance.py        # GFPGAN face enhancement
│   ├── background_blur.py     # Portrait bokeh
│   └── pipeline.py            # End-to-end pipeline
│
├── models/
│   └── gfpgan/
│       └── GFPGANv1.4.pth     # Pretrained weights
│
└── samples/
    ├── input/                 # Raw images
    └── output/                # Enhanced images
```

---

## ⚙️ Setup Instructions

### 1️⃣ Create Virtual Environment

```bash
py -3.10 -m venv .venv
.venv\Scripts\activate
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

If PyTorch fails:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

---

## ▶️ Run the Pipeline

Place raw portrait images (`.jpg` / `.png`) inside:

```
samples/input/
```

Run:

```bash
python app.py
```

Enhanced images will be saved automatically in:

```
samples/output/
```

---

## ✅ Key Features & Enhancements

### ✔ Face-Aware Enhancement

* Enhancements are **applied only to the face region**
* Prevents over-processing of background
* Preserves identity and skin texture

### ✔ Natural Skin Preservation

* No plastic or over-smoothed faces
* GFPGAN used with safety guards and controlled blending

### ✔ Professional Portrait Bokeh

* Human segmentation–based background blur
* Smooth feathered edges for studio look

### ✔ Fast Inference

* CPU-only execution
* Optimized, modular pipeline
* Suitable for real-time or batch usage

---

## 🧪 Edge Case Handling

* No face detected → image returned safely
* Invalid bounding box → skipped
* GFPGAN failure → fallback to original face
* Automatic clipping of face region

---

## ⏱ Performance

* **~1–2 seconds per image (CPU)**
* No GPU required
* Lightweight and scalable

---

## 📹 Demo

🎬 **Demo Video:**
(Attach Google Drive link showing before/after results)

---

## 🔗 Submission Details

* **GitHub Repository:** *(Attach your repo link here)*
* **Demo Video:** *(Attach Google Drive link here)*

---

## 🏁 Conclusion

This project demonstrates a **production-ready portrait enhancement system** combining classical CV techniques with deep learning models.
It focuses on **visual quality, performance, robustness, and user-centric design**, aligning closely with real-world AI imaging applications.

---
