import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load trained model
model = load_model("hand_gesture_mobilenet.h5")

# Your class labels (edit according to actual dataset meaning)
labels = [
    "Class 0", "Class 1", "Class 2", "Class 3", "Class 4", "Class 5", "Class 6",
    "Class 7", "Class 8", "Class 9", "Class 10", "Class 11", "Class 12", 
    "Class 13", "Class 14", "Class 15", "Class 16", "Class 17", "Class 18",
    "Class 19", "Class 20", "Class 21", "Class 22", "Class 23", "Class 24", 
    "Class 25", "Class 26"
]

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess for MobileNet
    img = cv2.resize(frame, (224, 224))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    # Predict
    pred = model.predict(img)
    class_id = np.argmax(pred)
    confidence = np.max(pred)

    # Text display
    text = f"{labels[class_id]} ({confidence*100:.1f}%)"
    cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2)

    cv2.imshow("Hand Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
