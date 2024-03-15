import mediapipe as mp
import cv2
import numpy as np

class facemesh:
    def __init__(self):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_face_mesh = mp.solutions.face_mesh
    # def facedetect2(self, image):
    #     with self.mp_face_mesh.FaceMesh(
    #         min_detection_confidence=0.5,
    #         min_tracking_confidence=0.5) as face_mesh:
    #         image.flags.writeable = False
    #         results = face_mesh.process(image)
    #         # Draw the face mesh annotations on the image.
    #         image.flags.writeable = True
    #         if results.multi_face_landmarks:
    #             for face_landmarks in results.multi_face_landmarks:
    #                 shape = image.shape
    #                 image = image[0:shape[0]-50, 0:shape[1]-10]
    #                 M = cv2.getRotationMatrix2D((shape[1]/2, shape[0]/2),
    #                                                 10,
    #                                                 1)
    #                 image = cv2.warpAffine(image, M, (shape[1], shape[0]))

    def facedetect(self, image):
        with self.mp_face_mesh.FaceMesh(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as face_mesh:
            results = face_mesh.process(image)
            shape = image.shape
            for face_landmarks in results.multi_face_landmarks:
                for idx, cen in enumerate(face_landmarks.landmark):
                    if idx == 6:
                        x = int(cen.x * shape[1])
                        y = int(cen.y * shape[0])
                        image = image[y - 100: y + 100, x - 70: x + 70]
        return image