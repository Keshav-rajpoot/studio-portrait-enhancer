# 🎨 Studio-Quality Portrait Enhancement Pipeline

This project converts **raw human portrait images** (captured using mobile cameras in uncontrolled conditions) into **studio-quality portrait images** using a **fast, production-oriented computer vision + deep learning pipeline**.

The solution focuses on:

* Preserving **facial identity**
* Maintaining **natural skin texture**
* Achieving **professional portrait aesthetics**
* Keeping **fast inference performance**

---

## 🚀 Features

✔ Motion blur handling (when present)
✔ Identity-preserving face enhancement
✔ Natural skin texture preservation
✔ Background blur (portrait / bokeh effect)
✔ Improved contrast and sharpness
✔ Clean, modular, production-ready codebase

---

## 🧠 Pipeline Overview

```
Input Image
   ↓
Face Detection (MediaPipe)
   ↓
Face Enhancement (GFPGAN)
   ↓
Background Segmentation
   ↓
Background Blur (Bokeh)
   ↓
Contrast & Sharpness Enhancement
   ↓
Studio-Quality Output
```

---

## 🛠️ Tech Stack

| Component        | Tool        |
| ---------------- | ----------- |
| Language         | Python 3.10 |
| Computer Vision  | OpenCV      |
| Face Detection   | MediaPipe   |
| Face Enhancement | GFPGAN      |
| Deep Learning    | PyTorch     |
| Image Processing | NumPy       |

---

## 📂 Project Structure

```
studio-portrait-enhancer/
│
├── app.py                  # Entry point
├── requirements.txt        # Dependencies
├── README.md
│
├── src/
│   ├── face_detect.py      # Face detection
│   ├── face_enhance.py     # GFPGAN-based face enhancement
│   ├── background_blur.py # Portrait bokeh effect
│   ├── post_process.py    # Contrast & sharpness
│   └── pipeline.py        # Full pipeline orchestration
│
├── samples/
│   └── input/              # Raw input images
```

> ⚠️ **Model weights and generated outputs are intentionally excluded from the repository** to keep the repo lightweight and GitHub-compliant.

---

## 📦 Model Weights (Required)

This project uses **GFPGAN v1.4** for face enhancement.

Download the model manually and place it here:

```
models/gfpgan/GFPGANv1.4.pth
```

📥 Official source:
[https://github.com/TencentARC/GFPGAN](https://github.com/TencentARC/GFPGAN)

---

## ▶️ How to Run

### 1️⃣ Create Virtual Environment

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Add Input Images

Place `.jpg` or `.png` images inside:

```
samples/input/
```

### 4️⃣ Run the Pipeline

```bash
python app.py
```

Enhanced images will be generated with:

* Improved face clarity
* Natural skin tones
* Studio-style background blur

---

## ⚡ Performance Notes

* Designed for **CPU inference**
* Modular pipeline enables future GPU acceleration
* Optimized to avoid unnecessary processing
* Face enhancement applied **only to detected face regions**

---

## 🎯 Design Decisions

* **No training from scratch** — uses proven pretrained models
* **Selective enhancement** — avoids over-processing
* **User-first output** — natural, realistic portraits
* **Fast inference mindset** — suitable for real-world deployment


## 👨‍💻 Author

**Keshav Singh Rajpoot**
Machine Learning Engineer Candidate
FOG Pvt Ltd – Round 1 Assessment


Just tell me.
