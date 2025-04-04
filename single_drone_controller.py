import cv2
import numpy as np
import mediapipe as mp
import pandas as pd


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf

gestures = ["UP", "DOWN", "LEFT", "RIGHT", "FORWARD", "BACKWARD"]



def main():

    model = tf.keras.models.load_model("/home/akansh_26/AIMS-DTU/Hand_Gesture_Drone_Controller/model/model_single_drone(no speed control).h5")

    cam = cv2.VideoCapture(0)

    mpHands = mp.solutions.hands
    hand_model = mpHands.Hands(max_num_hands=1)
    mpDraw = mp.solutions.drawing_utils

    while True:
        ret, frame = cam.read()
        frame = cv2.flip(frame, 1)
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        predictions = hand_model.process(img)
        landmark_list = []

        if predictions.multi_hand_landmarks:
            for handLms in predictions.multi_hand_landmarks:
                mpDraw.draw_landmarks(frame, handLms, mpHands.HAND_CONNECTIONS)

                for i in range(21):
                    landmark_list += [handLms.landmark[i].x, handLms.landmark[i].y] 

                prediction = np.argmax(model.predict(pd.DataFrame(landmark_list).transpose()))
                cv2.putText(frame, gestures[prediction], (10, 30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 1)

        cv2.imshow("Camera", frame)
        if cv2.waitKey(50) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            cam.release()
            break



if __name__ == "__main__":
    main()