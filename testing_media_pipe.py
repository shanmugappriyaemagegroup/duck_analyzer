VIDEO_PATH = r"C:\Users\EmageVision\Documents\duck_classification\DuckVidNew\only_ducks1_more_scattered.mp4"

import cv2, mediapipe as mp
hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=2,
                                 min_detection_confidence=0.1,   # very low to see everything
                                 min_tracking_confidence=0.1)
cap = cv2.VideoCapture(VIDEO_PATH)
total = 0
while True:
    ok, frame = cap.read()
    if not ok: break
    total += 1
    res = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if res.multi_handedness:
        for h in res.multi_handedness:
            score = h.classification[0].score
            print(f"frame {total}: hand score={score:.3f}")
cap.release()