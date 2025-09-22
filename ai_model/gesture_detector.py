import cv2
import numpy as np
import requests
import time

cap = cv2.VideoCapture(0)

sos_last_triggered = 0 
COOLDOWN_SECONDS = 5

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    roi = frame[100:400, 100:400]
    cv2.rectangle(frame, (100, 100), (400, 400), (0, 255, 0), 2)

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (35, 35), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        max_contour = max(contours, key=cv2.contourArea)

        if cv2.contourArea(max_contour) > 1000:
            hull = cv2.convexHull(max_contour, returnPoints=False)
            if hull is not None and len(hull) > 3:
                defects = cv2.convexityDefects(max_contour, hull)
                if defects is not None:
                    finger_count = 0

                    for i in range(defects.shape[0]):
                        s, e, f, d = defects[i, 0]
                        start = tuple(max_contour[s][0])
                        end = tuple(max_contour[e][0])
                        far = tuple(max_contour[f][0])

                        a = np.linalg.norm(np.array(end) - np.array(start))
                        b = np.linalg.norm(np.array(far) - np.array(start))
                        c = np.linalg.norm(np.array(far) - np.array(end))
                        angle = np.arccos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c + 1e-5))

                        if angle <= np.pi / 2:
                            finger_count += 1
                            cv2.circle(roi, far, 5, (0, 0, 255), -1)

                    if finger_count >= 4:  # Open palm
                        current_time = time.time()
                        if current_time - sos_last_triggered > COOLDOWN_SECONDS:
                            sos_last_triggered = current_time
                            print("[ALERT] Open Palm Detected - SOS Triggered!")

                            try:
                                response = requests.post(
                                    "http://127.0.0.1:5000/ai/sos",  # 👈 Updated endpoint
                                    json={
                                        "source": "gesture",
                                        "location": "Camera 1",
                                        "message": "SOS triggered by open palm"
                                    }
                                )
                                if response.status_code == 200:
                                    print("[INFO] Alert sent to backend successfully.")
                                else:
                                    print("[ERROR] Backend responded with status:", response.status_code)
                            except Exception as e:
                                print("[ERROR] Could not send alert:", e)

                            cv2.putText(frame, "SOS GESTURE DETECTED", (50, 80),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

            cv2.drawContours(roi, [max_contour], -1, (255, 0, 0), 2)

    cv2.imshow("Gesture Detection", frame)
    cv2.imshow("Threshold", thresh)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()

