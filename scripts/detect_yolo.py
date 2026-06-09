from ultralytics import YOLO
import cv2
import time
import winsound

model = YOLO('runs/detect/runs/drowsiness/yolov8_model/weights/best.pt')

EYES_CLOSED_THRESHOLD = 2.0  
last_open_time = time.time() 
is_drowsy = False

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    results = model(frame, conf=0.5, verbose=False)
    

    boxes = results[0].boxes
    current_labels = [model.names[int(c)] for c in boxes.cls]


    if 'neutral' in current_labels:
        last_open_time = time.time()
        is_drowsy = False
    elif 'microsleep' in current_labels:
        closed_duration = time.time() - last_open_time
        if closed_duration >= EYES_CLOSED_THRESHOLD:
            is_drowsy = True
            
    if 'yawning' in current_labels:
        cv2.putText(frame, "DANG NGAP!", (50, 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 3)

    annotated_frame = results[0].plot()

    if is_drowsy:
        cv2.rectangle(annotated_frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), 10)
        cv2.putText(annotated_frame, "!!! CANH BAO NGU GAT !!!", (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
        winsound.Beep(2500, 500) 


    cv2.imshow("Drowsiness Detection System", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()