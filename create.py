import cv2
from mtcnn.mtcnn import MTCNN
import time

# Initialize webcam and MTCNN detector
cap = cv2.VideoCapture(0)
detector = MTCNN()

print("[INFO] Starting webcam...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    start_time = time.time()

    # Convert BGR to RGB (MTCNN needs RGB)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = detector.detect_faces(rgb_frame)

    # Draw bounding boxes
    for r in results:
        x, y, w, h = r['box']
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        if 'confidence' in r:
            cv2.putText(frame, f"{r['confidence']:.2f}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Calculate and show FPS
    end_time = time.time()
    fps = 1 / (end_time - start_time)
    cv2.putText(frame, f"FPS: {fps:.2f}", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    cv2.imshow("MTCNN Face Detection Test", frame)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
