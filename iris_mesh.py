import mediapipe as mp
import cv2


class irisDetect:
    def __init__(self):
        self.mp_iris = mp.solutions.iris
        self.mp_draw = mp.solutions.drawing_utils

    def iris_detect(self, image):
        with self.mp_iris.Iris() as iris:
            results = iris.process(image)
            for id, cen in enumerate(results.face_landmarks_with_iris.landmark):
                shape = image.shape
                if id == 473:
                    xx = shape[1]
                    yy = shape[0]
                    image = cv2.circle(image, (xx, yy), 1, (0,0,255), 1)
                if id == 468:
                    xx = shape[1]
                    yy = shape[0]
                    image = cv2.circle(image, (xx, yy), 1, (0,0,255), 1)
            if results.face_landmarks_with_iris:
                self.mp_draw.draw_iris_landmarks(image, results.face_landmarks_with_iris)
        return image