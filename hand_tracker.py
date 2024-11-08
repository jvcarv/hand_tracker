import cv2
import mediapipe as mp
import numpy as np
import time
from typing import Union

# Tipagem
confidence = float
webcam_image = np.ndarray
rgb_tuple = tuple[int, int, int]


class Detector():
    def __init__(self,
                 mode: bool = False,
                 number_hands: int = 2,
                 model_complexity: int = 1,
                 min_detec_confidence: confidence = 0.5,
                 min_tracking_confidence: confidence = 0.5) -> None:
        # Parametros necessários para inicializar
        self.mode = mode
        self.max_num_hands = number_hands
        self.complexity = model_complexity
        self.detection_con = min_detec_confidence
        self.tracking_con = min_tracking_confidence

        # Inicializar o Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(self.mode,
                                         self.max_num_hands,
                                         self.complexity,
                                         self.detection_con,
                                         self.tracking_con,
                                         )
        self.tip_ids = [4, 8, 12, 16, 20]
        self.mp_draw = mp.solutions.drawing_utils

    def find_hands(self, 
                   img: webcam_image,
                   draw_hands: bool = True):
        # Convertendo a imagem para RGB
        img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Coletar resultados para processo e análise das mãos
        self.results = self.hands.process(img_RGB)
        if self.results.multi_hand_landmarks and draw_hands:
            for hand in self.results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(img, hand, self.mp_hands.HAND_CONNECTIONS)
        return img

    def find_position(self, 
                      img: webcam_image,
                      hand_number: int = 0):
        self.required_landmark_list = []

        my_hand = None

        if self.results.multi_hand_landmarks:
            height, width, _ = img.shape
            my_hand = self.results.multi_hand_landmarks[hand_number]
            for id, lm in enumerate(my_hand.landmark):
                center_x, center_y = int(lm.x * width), int(lm.y*height)

                self.required_landmark_list.append(id, center_x, center_y)

        return my_hand


# Teste de Classe
if __name__ == "__main__":
    # Declarando a classe
    Detec = Detector()

    # Captura de imagem
    capture = cv2.VideoCapture(0)

    while True:
        # Captura do frame
        _, frame = capture.read()

        # Manipulação de frame
        img = Detec.find_hands(frame)  # draw_hands = false
        landmark_list = Detec.find_position(img)
        if landmark_list:
            print(landmark_list)
            
        # Mostrando o frame
        cv2.imshow('Hand Tracking', frame)

        # Quitando
        if cv2.waitKey(20) & 0xFF==ord('d'):
            break
