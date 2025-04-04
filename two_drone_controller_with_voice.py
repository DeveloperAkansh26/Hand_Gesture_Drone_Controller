import cv2
import numpy as np
import mediapipe as mp
import pandas as pd
import speech_recognition
import multiprocessing
import threading
import google.generativeai as genai

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["GEMINI_API_KEY"] = ""

genai.configure(api_key=os.environ["GEMINI_API_KEY"])

import tensorflow as tf

gestures = ["UP", "DOWN", "LEFT", "RIGHT", "FORWARD", "BACKWARD", "CHANGE1", "CHANGE2"]

# state_list = ["Constant-Velocity", "Speed-Controlled"]

state_l = "Constant-Velocity"
state_r = "Constant-Velocity"

frame = np.zeros((640, 480))

model = genai.GenerativeModel("gemini-2.0-flash-exp")



def main():

    p1 = threading.Thread(target=gesture_recognizer)
    p2 = threading.Thread(target=speech_recognizer)

    p1.start()
    p2.start()

    p1.join()
    p2.join()



def gesture_recognizer():

    global frame
    gestures = ["UP", "DOWN", "LEFT", "RIGHT", "FORWARD", "BACKWARD", "CHANGE1", "CHANGE2"]

    model = tf.keras.models.load_model("/home/akansh_26/AIMS-DTU/Hand_Gesture_Drone_Controller/model/model_two_drone.h5")

    cam = cv2.VideoCapture(0)

    mpHands = mp.solutions.hands
    hand_model = mpHands.Hands(max_num_hands=2)
    mpDraw = mp.solutions.drawing_utils

    frame_n = 1
    prediction_l = 0
    prediction_r = 0

    while True:

        global state_r
        global state_l

        ret, frame = cam.read()
        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (1080, 720))
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        x_dim = frame.shape[1]
        y_dim = frame.shape[0]

        cv2.line(frame, (int(x_dim / 2), 0), (int(x_dim / 2), y_dim), (0, 255, 0), 1)

        predictions = hand_model.process(img)

        if predictions.multi_hand_landmarks:
            for handLms in predictions.multi_hand_landmarks:

                landmark_list = []
                mpDraw.draw_landmarks(frame, handLms, mpHands.HAND_CONNECTIONS)

                for i in range(21):
                    landmark_list += [handLms.landmark[i].x, handLms.landmark[i].y] 
                
                x_cords = np.array([handLms.landmark[i].x for i in range(21)])
                mx = x_cords.min()
                mn = x_cords.max()

                fing_pos = finger_pos(handLms)

                if frame_n >= 10:
                    frame_n = 0

                    # RIGHT SIDE
                    if mn > 0.5 and mx > 0.5:

                        if state_r == "Constant-Velocity":

                            cv2.putText(frame, "Constant-Velocity", (int(x_dim / 2) + 100, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                            prediction_r = np.argmax(model.predict(pd.DataFrame(landmark_list).transpose()))

                            if prediction_r == 6:
                                state_r = "Speed-Controlled"                         
                            else:
                                cv2.putText(frame, gestures[prediction_r], (int(x_dim / 2) + 20, 80), cv2.FONT_HERSHEY_COMPLEX, 0.75, (0, 255, 0), 1)
                        
                        else:
                            cv2.putText(frame, "Speed-Controlled", (int(x_dim / 2) + 100, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

                            # Two-Finger Control
                            if fing_pos[0] == 1 and fing_pos[1] == 1 and fing_pos[2] == 0 and fing_pos[3] == 0:

                                x = int(np.mean([float(handLms.landmark[8].x), float(handLms.landmark[12].x)]) * x_dim)
                                y = int(np.mean([float(handLms.landmark[8].y), float(handLms.landmark[12].y)]) * y_dim)

                                cv2.circle(frame, (x, y), 10, (0, 255, 0), -1)
                                cv2.line(frame, (int(x_dim / 2) + 50, int(y_dim / 3) + 10), (int(x_dim) - 100, int(y_dim / 3) + 10), (255, 0, 0), 1)
                                cv2.line(frame, (int((int(x_dim) - 100 + int(x_dim / 2) + 50) / 2), 20), (int((int(x_dim) - 100 + int(x_dim / 2) + 50) / 2), int(y_dim * (2 / 3))), (255, 0, 0), 1)

                                x_c = int((int(x_dim) - 100 + int(x_dim / 2) + 50) / 2)
                                y_c = int(y_dim / 3) + 10

                                ux = np.abs(x - x_c) / (int(x_dim) - 100 - x_c)
                                uy = np.abs(y - y_c) / (int(y_dim * (2 / 3)) - y_c)

                                print("RIGHT-DRONE")
                                print(f"U_x = {ux}")
                                print(f"U_y = {uy}")


                            else:
                                prediction_r = np.argmax(model.predict(pd.DataFrame(landmark_list).transpose()))
                                
                                if prediction_r == 4:
                                    print(f"Forward: {handLms.landmark[0].z}")
                                elif prediction_r == 5:
                                    print(f"Backward: {-handLms.landmark[0].z}")
                                elif prediction_r == 7:
                                    state_r = "Constant-Velocity"
                    

                    # LEFT SIDE
                    elif mn < 0.5 and mx < 0.5:

                        if state_l == "Constant-Velocity":

                            cv2.putText(frame, "Constant-Velocity", (100, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                            prediction_l = np.argmax(model.predict(pd.DataFrame(landmark_list).transpose()))

                            if prediction_l == 6:
                                state_l = "Speed-Controlled"
                            else:
                                cv2.putText(frame, gestures[prediction_l], (20, 80), cv2.FONT_HERSHEY_COMPLEX, 0.75, (0, 255, 0), 1)
                                
                        
                        else:
                            cv2.putText(frame, "Speed-Controlled", (100, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

                            # Two-Finger Control
                            if fing_pos[0] == 1 and fing_pos[1] == 1 and fing_pos[2] == 0 and fing_pos[3] == 0:

                                x = int(np.mean([float(handLms.landmark[8].x), float(handLms.landmark[12].x)]) * x_dim)
                                y = int(np.mean([float(handLms.landmark[8].y), float(handLms.landmark[12].y)]) * y_dim)

                                cv2.circle(frame, (x, y), 10, (0, 255, 0), -1)
                                cv2.line(frame, (int(x_dim / 2) - 50, int(y_dim / 3) + 10), (100, int(y_dim / 3) + 10), (255, 0, 0), 1)
                                cv2.line(frame, (int((100 + int(x_dim / 2) - 50) / 2), 20), (int((100 + int(x_dim / 2) - 50) / 2), int(y_dim * (2 / 3))), (255, 0, 0), 1)

                                x_c = int((100 + int(x_dim / 2) - 50) / 2)
                                y_c = int(y_dim / 3) + 10

                                ux = np.abs(x - x_c) / (int(x_dim / 2) - 50 - x_c)
                                uy = np.abs(y - y_c) / (int(y_dim * (2 / 3)) - y_c)

                                print("LEFT-DRONE")
                                print(f"U_x = {ux}")
                                print(f"U_y = {uy}")


                            else:
                                prediction_l = np.argmax(model.predict(pd.DataFrame(landmark_list).transpose()))
                                
                                if prediction_l == 4:
                                    print(f"Forward: {handLms.landmark[0].z}")
                                elif prediction_l == 5:
                                    print(f"Backward: {-handLms.landmark[0].z}")
                                elif prediction_l == 7:
                                    state_l = "Constant-Velocity"
                    

                    else:
                        cv2.putText(frame, state_r, (int(x_dim / 2) + 100, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                        cv2.putText(frame, state_l, (100, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                
                else:

                    if mn > 0.5 and mx > 0.5:
                        if state_r == "Constant-Velocity":
                            cv2.putText(frame, gestures[prediction_r], (int(x_dim / 2) + 20, 80), cv2.FONT_HERSHEY_COMPLEX, 0.75, (0, 255, 0), 1)
                        else:
                            cv2.putText(frame, "Speed-Controlled", (int(x_dim / 2) + 100, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

                            # Two-Finger Control
                            if fing_pos[0] == 1 and fing_pos[1] == 1 and fing_pos[2] == 0 and fing_pos[3] == 0:

                                x = int(np.mean([float(handLms.landmark[8].x), float(handLms.landmark[12].x)]) * x_dim)
                                y = int(np.mean([float(handLms.landmark[8].y), float(handLms.landmark[12].y)]) * y_dim)

                                cv2.circle(frame, (x, y), 10, (0, 255, 0), -1)
                                cv2.line(frame, (int(x_dim / 2) + 50, int(y_dim / 3) + 10), (int(x_dim) - 100, int(y_dim / 3) + 10), (255, 0, 0), 1)
                                cv2.line(frame, (int((int(x_dim) - 100 + int(x_dim / 2) + 50) / 2), 20), (int((int(x_dim) - 100 + int(x_dim / 2) + 50) / 2), int(y_dim * (2 / 3))), (255, 0, 0), 1)

                                x_c = int((int(x_dim) - 100 + int(x_dim / 2) + 50) / 2)
                                y_c = int(y_dim / 3) + 10

                                ux = np.abs(x - x_c) / (int(x_dim) - 100 - x_c)
                                uy = np.abs(y - y_c) / (int(y_dim * (2 / 3)) - y_c)

                                print("RIGHT-DRONE")
                                print(f"U_x = {ux}")
                                print(f"U_y = {uy}")

                    elif mn < 0.5 and mx < 0.5:
                        if state_l == "Constant-Velocity":
                            cv2.putText(frame, gestures[prediction_l], (20, 80), cv2.FONT_HERSHEY_COMPLEX, 0.75, (0, 255, 0), 1)
                        else:
                            cv2.putText(frame, "Speed-Controlled", (100, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

                            # Two-Finger Control
                            if fing_pos[0] == 1 and fing_pos[1] == 1 and fing_pos[2] == 0 and fing_pos[3] == 0:

                                x = int(np.mean([float(handLms.landmark[8].x), float(handLms.landmark[12].x)]) * x_dim)
                                y = int(np.mean([float(handLms.landmark[8].y), float(handLms.landmark[12].y)]) * y_dim)

                                cv2.circle(frame, (x, y), 10, (0, 255, 0), -1)
                                cv2.line(frame, (int(x_dim / 2) - 50, int(y_dim / 3) + 10), (100, int(y_dim / 3) + 10), (255, 0, 0), 1)
                                cv2.line(frame, (int((100 + int(x_dim / 2) - 50) / 2), 20), (int((100 + int(x_dim / 2) - 50) / 2), int(y_dim * (2 / 3))), (255, 0, 0), 1)

                                x_c = int((100 + int(x_dim / 2) - 50) / 2)
                                y_c = int(y_dim / 3) + 10

                                ux = np.abs(x - x_c) / (int(x_dim / 2) - 50 - x_c)
                                uy = np.abs(y - y_c) / (int(y_dim * (2 / 3)) - y_c)

                                print("LEFT-DRONE")
                                print(f"U_x = {ux}")
                                print(f"U_y = {uy}")

                frame_n += 1


        cv2.putText(frame, state_r, (int(x_dim / 2) + 100, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, state_l, (100, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Camera", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            cam.release()
            break



def speech_recognizer():
    
    global frame
    recognizer = speech_recognition.Recognizer()

    while True:
        try:
            with speech_recognition.Microphone() as mic:

                print("Listening.....")

                recognizer.adjust_for_ambient_noise(mic)
                audio = recognizer.listen(mic)
                text = recognizer.recognize_google(audio)

                print(text)

                if "flip" in text.lower():
                    print("Command Executed: FLIP")

                elif "rotate" in text.lower():
                    print("Command Executed: ROTATE")

                else:
                    response = model.generate_content(f"{text}")
                    print(response.text)
                
        except:
            recognizer = speech_recognition.Recognizer()
            continue


def finger_pos(handLms):
    pos = np.array([1, 1, 1, 1])

    if handLms.landmark[8].y > handLms.landmark[6].y:
        pos[0] = 0
    
    if handLms.landmark[12].y > handLms.landmark[10].y:
        pos[1] = 0

    if handLms.landmark[16].y > handLms.landmark[14].y:
        pos[2] = 0

    if handLms.landmark[20].y > handLms.landmark[18].y:
        pos[3] = 0
    
    return pos


if __name__ == "__main__":
    main()
