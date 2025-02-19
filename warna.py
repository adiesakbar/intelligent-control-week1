import cv2
import numpy as np

def detect_color(frame, lower, upper, color_name, color_bgr):
    mask = cv2.inRange(hsv, lower, upper)
    result = cv2.bitwise_and(frame, frame, mask=mask)
    
    # Menemukan kontur
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 500:  # Filter area kecil
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color_bgr, 2)
            cv2.putText(frame, color_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_bgr, 2)

# Inisialisasi kamera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Kamera tidak dapat dibuka!")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Gagal mendapatkan frame!")
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Rentang warna dalam HSV
    colors = [
        (np.array([0, 120, 70]), np.array([10, 255, 255]), "Red", (0, 0, 255)),
        (np.array([36, 50, 70]), np.array([89, 255, 255]), "Green", (0, 255, 0)),
        (np.array([90, 50, 70]), np.array([130, 255, 255]), "Blue", (255, 0, 0))
    ]

    for lower, upper, name, bgr in colors:
        detect_color(frame, lower, upper, name, bgr)

    cv2.imshow("Color Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()