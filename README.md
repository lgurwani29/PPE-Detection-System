# 🦺 AI-Based Personal Protective Equipment Detection System
![Python](https://img.shields.io/badge/Python-3.12-blue)

![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-success)

![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-red)
## Overview

The AI-Based Personal Protective Equipment (PPE) Detection System is a Computer Vision application developed using YOLOv8 and Streamlit.

The system automatically detects workers and Personal Protective Equipment from uploaded images, helping estimate workplace safety compliance.

---

## Features

- 👤 Human Detection
- ⛑ Helmet Detection
- 🦺 Safety Vest Detection
- 🧤 Gloves Detection
- 🥾 Safety Boots Detection
- 📊 PPE Compliance Assessment
- 📈 Safety Score Estimation
- 📥 Download Annotated Images
- 🌐 Streamlit Web Application

---

## Technologies Used

- Python
- YOLOv8 (Ultralytics)
- Streamlit
- OpenCV
- Pandas
- Pillow

---

## Dataset

- Custom PPE Detection Dataset
- Total Images: **4401**
- Model: **YOLOv8 Nano**
- Classes:
  - Human
  - Helmet
  - Safety Vest
  - Gloves
  - Boots

---

## Project Structure

```
PPE_detection/

│── app.py

│── best.pt

│── requirements.txt

│── README.md

│── outputs/

│── dataset/
```

---

## Installation

Clone the repository

```bash
git clone https://github.com/lgurwani29/PPE-Detection-System.git
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
streamlit run app.py
```

---

## Working

1. Upload an image.
2. Adjust confidence threshold.
3. Click **Analyze PPE Compliance**.
4. View detections.
5. Review PPE Compliance.
6. Download the annotated image.

---

## Applications

- Construction Sites
- Manufacturing Industries
- Warehouses
- Mining Operations
- Oil & Gas Industry
- Smart Industrial Monitoring

---

## Limitations

- Reduced accuracy in crowded scenes.
- Performance depends on image quality.
- Does not associate individual PPE items with specific workers.

---

## Future Scope

- CCTV Integration
- Live Webcam Detection
- Worker Tracking
- Real-Time Alerts
- Cloud Database
- Mobile Application

---

## Developer

**Lavanya Guruwani**

B.Tech Computer Science & Engineering

AI-Based Personal Protective Equipment Detection System

---

## License

This project is developed for academic and educational purposes.