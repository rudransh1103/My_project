
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# --- 2. Load your trained model ---
model = load_model("asl_cnn_model.h5")  # ✅ This should already exist from earlier

# --- 3. Set the image size and class labels ---
IMG_SIZE = 64  # or whatever your model was trained with
class_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
    'U', 'V', 'W', 'X', 'Y', 'Z',
    'DELETE', 'NOTHING', 'SPACE']  # all 29 classes

# --- 4. (Optional) test with static image ---
# Code to test a single image, if you want to try before webcam

# --- 5. ADD THIS BLOCK AT THE END — Real-Time Webcam Detection ---
cap = cv2.VideoCapture(0)
print("Starting real-time ASL detection... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize the frame for prediction
    roi = cv2.resize(frame, (64, 64))
    roi = roi.astype("float32") / 255.0
    roi = np.expand_dims(roi, axis=0)

    # Predict the sign
    prediction = model.predict(roi)
    predicted_class = np.argmax(prediction)
    print("Predicted index:", predicted_class)
    print("Total labels:", len(class_names))
    label = class_names[predicted_class]

    # Show result on video
    cv2.putText(frame, f'Predicted: {label}', (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
    cv2.imshow("ASL Real-Time Detection", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
