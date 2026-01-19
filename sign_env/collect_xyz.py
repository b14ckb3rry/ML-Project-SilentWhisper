import cv2
import mediapipe as mp
import pickle

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)

data = []
labels = []

# 🔑 KEY → LABEL MAP (EDIT THIS)
label_map = {
    ord('1'): "Hello",
    ord('2'): "Good",
    ord('3'): "Yes",
    ord('4'): "No",
    ord('5'): "Thanks"
}

current_label = None

print("Press number keys to select label")
print("1=Hello | 2=Good | 3=Yes | 4=No | 5=Thanks")
print("Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    key = cv2.waitKey(1) & 0xFF

    # Change label
    if key in label_map:
        current_label = label_map[key]
        print(f"Current label: {current_label}")

    if result.multi_hand_landmarks and current_label:
        hand_landmarks = result.multi_hand_landmarks[0]
        features = []

        for lm in hand_landmarks.landmark:
            features.extend([lm.x, lm.y, lm.z])

        if len(features) == 63:
            data.append(features)
            labels.append(current_label)

        mp_draw.draw_landmarks(
            frame,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS
        )

        cv2.putText(
            frame,
            f"Label: {current_label}",
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

    cv2.imshow("Collect XYZ Multi-Label", frame)

    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

with open("data_xyz.pickle", "wb") as f:
    pickle.dump({"data": data, "labels": labels}, f)

print("Saved samples:", len(data))
