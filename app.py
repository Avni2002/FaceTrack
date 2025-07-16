from flask import Flask, request, jsonify, render_template, Response
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime, date
import face_recognition
import numpy as np
import base64
import pickle
import cv2
import os
import time
from collections import defaultdict
from mtcnn.mtcnn import MTCNN

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
detector = MTCNN()
recent_faces = defaultdict(lambda: 0)
COOLDOWN_SECONDS = 120

class Employee(db.Model):
    __tablename__ = 'employees'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    phone = db.Column(db.String(20), unique=True, nullable=False)
    department = db.Column(db.String(50))
    face_encoding = db.Column(db.LargeBinary, nullable=False)

class Attendance(db.Model):
    __tablename__ = 'attendance'
    id = db.Column(db.Integer, primary_key=True)
    employee_id = db.Column(db.Integer, db.ForeignKey('employees.id'), nullable=False)
    date = db.Column(db.String(10), nullable=False)  
    check_in_time = db.Column(db.String(8), nullable=False)  
    check_out_time = db.Column(db.String(8))
    employee = db.relationship('Employee', backref='attendances')

def decode_base64_image(data_uri):
    header, encoded = data_uri.split(",", 1)
    img_bytes = base64.b64decode(encoded)
    img_array = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    return img

def detect_faces_mtcnn(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = detector.detect_faces(rgb)
    boxes = []
    h, w, _ = frame.shape
    for r in results:
        x, y, width, height = r['box']
        x = max(0, x)
        y = max(0, y)
        x2 = min(w, x + width)
        y2 = min(h, y + height)

        # tighten box by small margin (optional tweak)
        margin = 10
        top = max(y + margin, 0)
        bottom = min(y2 - margin, h)
        left = max(x + margin, 0)
        right = min(x2 - margin, w)

        # Ensure valid box after margin adjustment
        if bottom > top and right > left:
            boxes.append((top, right, bottom, left))  # face_recognition format
    return boxes


def get_today_attendance_count(date_str):
    return Attendance.query.filter_by(date=date_str).count()

@app.route('/')
def home():
    today_str = date.today().strftime('%Y-%m-%d')
    today_count = get_today_attendance_count(today_str)
    return render_template('home.html', today_count=today_count)

@app.route('/register', methods=['POST'])
def register():
    data = request.json
    name = data.get('name')
    phone = data.get('phone')
    department = data.get('department')
    face_img_data = data.get('face_image')

    if not all([name, phone, face_img_data]):
        return jsonify({"error": "Missing required fields"}), 400

    img = decode_base64_image(face_img_data) 
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  

    os.makedirs("debug_images", exist_ok=True)
    cv2.imwrite("debug_images/latest_register.jpg", img)  

    face_locations = detect_faces_mtcnn(rgb_img)
    if len(face_locations) == 0:
        return jsonify({"error": "No face detected"}), 400

    encodings = face_recognition.face_encodings(rgb_img, face_locations)
    if len(encodings) == 0:
        return jsonify({"error": "Could not encode face"}), 400

    face_encoding = encodings[0]

    pickled_encoding = pickle.dumps(face_encoding)

    existing = Employee.query.filter_by(phone=phone).first()
    if existing:
        return jsonify({"error": "Phone number already registered"}), 400

    new_employee = Employee(name=name, phone=phone, department=department, face_encoding=pickled_encoding)
    db.session.add(new_employee)
    db.session.commit()

    return jsonify({"message": "Registration successful"})

def load_employees_encodings():
    employees = Employee.query.all()
    encodings = []
    names = []
    for emp in employees:
        enc = pickle.loads(emp.face_encoding)
        encodings.append(enc)
        names.append(emp.name)
    return encodings, names

@app.route('/checkin', methods=['POST'])
def check_in():
    data = request.json
    face_img_data = data.get('face_image')

    if not face_img_data:
        return jsonify({"error": "No image provided"}), 400

    img = decode_base64_image(face_img_data)
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    os.makedirs("debug_images", exist_ok=True)
    cv2.imwrite("debug_images/latest_checkin.jpg", img)

    face_locations = detect_faces_mtcnn(rgb_img)
    if len(face_locations) == 0:
        return jsonify({"error": "No face detected"}), 400

    face_encodings = face_recognition.face_encodings(rgb_img, face_locations)
    if len(face_encodings) == 0:
        return jsonify({"error": "Face detected but could not encode"}), 400

    known_encodings, known_names = load_employees_encodings()
    if not known_encodings:
        return jsonify({"error": "No registered employees to compare"}), 400

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_encodings, face_encoding)
        face_distances = face_recognition.face_distance(known_encodings, face_encoding)

        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            employee_name = known_names[best_match_index]
            employee = Employee.query.filter_by(name=employee_name).first()

            today_str = datetime.now().strftime("%Y-%m-%d")
            open_checkin = Attendance.query.filter_by(employee_id=employee.id, date=today_str, check_out_time=None).first()

            if open_checkin:
                return jsonify({"error": "Already checked in, please check out first"}), 400

            now_time = datetime.now().strftime("%H:%M:%S")
            new_attendance = Attendance(employee_id=employee.id, date=today_str, check_in_time=now_time)
            db.session.add(new_attendance)
            db.session.commit()

            return jsonify({"message": f"Check-in successful for {employee_name} at {now_time}."})

    return jsonify({"error": "Face not recognized"}), 400

@app.route('/checkout', methods=['POST'])
def check_out():
    data = request.json
    face_img_data = data.get('face_image')
    if not face_img_data:
        return jsonify({"error": "No image provided"}), 400

    img = decode_base64_image(face_img_data)
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    os.makedirs("debug_images", exist_ok=True)
    cv2.imwrite("debug_images/latest_checkout.jpg", img)

    face_locations = detect_faces_mtcnn(rgb_img)
    if len(face_locations) == 0:
        return jsonify({"error": "No face detected"}), 400

    face_encodings = face_recognition.face_encodings(rgb_img, face_locations)
    if len(face_encodings) == 0:
        return jsonify({"error": "Face detected but could not encode"}), 400

    known_encodings, known_names = load_employees_encodings()
    if not known_encodings:
        return jsonify({"error": "No registered employees to compare"}), 400

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_encodings, face_encoding)
        face_distances = face_recognition.face_distance(known_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)

        if matches[best_match_index]:
            employee_name = known_names[best_match_index]
            employee = Employee.query.filter_by(name=employee_name).first()

            today_str = datetime.now().strftime("%Y-%m-%d")
            open_checkin = Attendance.query.filter_by(employee_id=employee.id, date=today_str, check_out_time=None).first()

            if not open_checkin:
                return jsonify({"error": "No check-in found today. Please check in first."}), 400

            now_time = datetime.now().strftime("%H:%M:%S")
            open_checkin.check_out_time = now_time
            db.session.commit()

            return jsonify({"message": f"Check-out successful for {employee_name} at {now_time}."})

    return jsonify({"error": "Face not recognized"}), 400

@app.route('/attendance')
def attendance_log():
    selected_date = request.args.get('date')

    query = db.session.query(Attendance, Employee).join(Employee, Attendance.employee_id == Employee.id)

    if selected_date:
        query = query.filter(Attendance.date == selected_date)

    records = query.order_by(Attendance.date.desc(), Attendance.check_in_time.desc()).all()

    return render_template('attendance.html', records=records, selected_date=selected_date)

@app.route('/attendance')
def attendance_page():
    return render_template('attendance.html')

@app.route('/register')
def register_page():
    return render_template('register.html')

@app.route('/checkin')
def checkin_page():
    return render_template('checkin.html')

@app.route('/checkout')
def checkout_page():
    return render_template('checkout.html')

@app.route('/export')
def export_attendance_csv():
    today_str = date.today().strftime('%Y-%m-%d')
    filename = f"attendance_log_{today_str}.csv"

    records = (
        db.session.query(Attendance, Employee)
        .join(Employee, Attendance.employee_id == Employee.id)
        .order_by(Attendance.date.desc(), Attendance.check_in_time.desc())
        .all()
    )

    def generate():
        header = ['Employee Name', 'Phone', 'Department', 'Date', 'Check-in Time', 'Check-out Time']
        yield ','.join(header) + '\n'
        for attendance, employee in records:
            row = [
                employee.name,
                employee.phone,
                employee.department or '',
                attendance.date,
                attendance.check_in_time,
                attendance.check_out_time or ''
            ]
            yield ','.join(row) + '\n'

    return Response(generate(), mimetype='text/csv',
                    headers={'Content-Disposition': f'attachment; filename={filename}'})

@app.route('/mark-attendance', methods=['POST'])
def mark_attendance():
    data = request.json
    face_img_data = data.get('face_image')
    if not face_img_data:
        return jsonify({"error": "No image provided"}), 400

    try:
        img = decode_base64_image(face_img_data)
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except Exception as e:
        return jsonify({"error": f"Failed to process image: {e}"}), 400

    try:
        face_locations = detect_faces_mtcnn(rgb_img)
    except Exception as e:
        face_locations = []
        print(f"Face detection failed: {e}")

    face_encodings = face_recognition.face_encodings(rgb_img, face_locations)
    if len(face_encodings) == 0:
        return jsonify({"error": "No recognizable face found"}), 400

    known_encodings, known_names = load_employees_encodings()
    if not known_encodings:
        return jsonify({"error": "No registered employees to compare"}), 400

    today_str = datetime.now().strftime("%Y-%m-%d")
    now_time = datetime.now().strftime("%H:%M:%S")
    now_epoch = time.time()

    results = []
    annotated_img = img.copy()
    
    bounding_boxes = []

    for i, face_encoding in enumerate(face_encodings):
        top, right, bottom, left = face_locations[i]
        matches = face_recognition.compare_faces(known_encodings, face_encoding)
        face_distances = face_recognition.face_distance(known_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)

        if matches[best_match_index]:
            employee_name = known_names[best_match_index]
            employee = Employee.query.filter_by(name=employee_name).first()
            last_seen = recent_faces[employee_name]

            if now_epoch - last_seen < COOLDOWN_SECONDS:
                
                cv2.rectangle(annotated_img, (left, top), (right, bottom), (0, 255, 255), 2)
                cv2.putText(annotated_img, "Cooldown", (left, top - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                pass
            else:
                
                today_str = datetime.now().strftime("%Y-%m-%d")
                now_time = datetime.now().strftime("%H:%M:%S")
                open_entry = Attendance.query.filter_by(
                    employee_id=employee.id,
                    date=today_str,
                    check_out_time=None
                ).first()

                if open_entry:
                    open_entry.check_out_time = now_time
                    action = f"Checked out {employee_name} at {now_time}"
                else:
                    new_attendance = Attendance(
                        employee_id=employee.id,
                        date=today_str,
                        check_in_time=now_time
                    )
                    db.session.add(new_attendance)
                    action = f"Checked in {employee_name} at {now_time}"

                recent_faces[employee_name] = now_epoch
                results.append(action)

                # ✅ Show green box
                cv2.rectangle(annotated_img, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(annotated_img, employee_name, (left, top - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        else:
            # ❌ Unknown face — show red box
            cv2.rectangle(annotated_img, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.putText(annotated_img, "Unknown", (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            results.append("Unknown face detected")

    db.session.commit()

    os.makedirs("debug_images", exist_ok=True)
    cv2.imwrite("debug_images/marked_frame.jpg", annotated_img)

    return jsonify({"message": results})


if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)
