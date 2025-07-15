# FaceTrack - Face Recognition Attendance System

FaceTrack is a Flask-based web application for automatic employee attendance using face recognition. It uses OpenCV, `face_recognition`, and a webcam to detect and recognize multiple faces in real time, logging their attendance efficiently.

---

## 🚀 Features

- ✅ Face registration with photo and phone number
- ✅ Login using only name and phone number (no face scan required)
- ✅ Real-time face recognition from webcam
- ✅ Auto "Check-In" and "Check-Out" based on repeated recognition
- ✅ Supports **multiple people** in one frame
- ✅ Attendance stored in CSV format
- ✅ Bootstrap-based responsive UI

---

## 📸 Face Recognition Workflow

1. User registers with name, phone number, and face scan.
2. At runtime, camera captures frames and recognizes known faces.
3. On first recognition ➜ **Check-In** logged.
4. On second recognition ➜ **Check-Out** logged.
5. System resets to allow repeated cycles per day.

---

## 🛠️ Tech Stack

- **Backend**: Python, Flask
- **Frontend**: HTML, CSS (Bootstrap), JavaScript
- **Face Recognition**: `face_recognition`, `OpenCV`
- **Database**: CSV-based (easily extendable to SQL)
- **Deployment-ready**: Can be containerized or deployed to cloud platforms



