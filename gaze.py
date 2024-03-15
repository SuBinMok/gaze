
import threading
from connecting_unity import WebSocketClient
import cv2
import pyautogui
import numpy as np
import time
import datetime
import mediapipe as mp

COUNT = 0
screen_width, screen_height = pyautogui.size()

def draw2(img, a, b):
    img = cv2.circle(img, (a, b), 50, (0, 0, 0), -1)
    return img
def cali():
    # time.sleep(5) #eye_movement()에서 카메라가 켜진 후 시작하기 위해 5초간 timesleep
    count = 0
    img = np.zeros((screen_height, screen_width, 3), np.uint8)
    a =0 ; b= 0
    for i in range(5):
        cv2.namedWindow("img", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("img", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        img = cv2.rectangle(img, (0, 0), (screen_width, screen_height), (255, 255, 255), -1)
        if i == 0:
            a = 50
            b = 50
        elif i == 1:
            a = screen_width - 50
            b = 50
        elif i == 2:
            a = 50
            b = screen_height - 50
        elif i == 3:
            a = screen_width - 50
            b = screen_height - 50
        elif i == 4:
            a = int(screen_width / 2)
            b = int(screen_height / 2)

        img = draw2(img, a, b)
        cv2.imshow('img', img)
        key = cv2.waitKey(1) & 0xFF
        if (key == 27):
            break
        time.sleep(5) #한 점에 대해 calibration하는 시간
        count +=1
    cv2.destroyWindow('img')

def full():
    img = cv2.imread('./ref.png')

    cv2.namedWindow("CalibrationImage", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("CalibrationImage", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    cv2.imshow('CalibrationImage', img)
    cv2.waitKey(1000*60)
    cv2.destroyWindow('CalibrationImage')

def rate(W, H, x, y):
    W0=0
    W1=W
    H0= 0
    H1= H

    a = np.sqrt((W0 - x)**2 + (H0 - y)**2)
    b = np.sqrt((W1 - x)**2 + (H0 - y)**2)
    g = np.sqrt((W0 - x)**2 + (H1 - y)**2)
    d = np.sqrt((W1 - x)**2 + (H1 - y)**2)
    return a, b, g, d

def correct(alpha, beta, gamma, delta):
    a_g = alpha / gamma
    b_d = beta / delta
    a_d = alpha / delta
    b_g = beta / gamma

    br1 = float("{:.2f}".format(a_d))
    br2 = float("{:.2f}".format(b_g))

    return br1, br2

def calculate(eye, avatar_3d, monitor):
    eye_inv = np.linalg.inv(eye)
    alpha = np.dot(avatar_3d, eye_inv)
    beta = np.dot(monitor, eye_inv)
    return alpha, beta

def fill_matrix(cali_matrix_x, cali_matrix_y, eye_matrix, count): #calibraion 계산 metrix 채움
    length = len(cali_matrix_x)
    x = sum(cali_matrix_x) / length
    y = sum(cali_matrix_y) / length
    eye_matrix[0][count] = x
    eye_matrix[1][count] = y
    eye_matrix[2][count] = x * x
    eye_matrix[3][count] = y * y
    eye_matrix[4][count] = 1
    return eye_matrix

def dotdot(x, y, alpha): #
    E = [x,
         y,
         x * x,
         y * y,
         1]
    M = np.dot(alpha, E)
    return M
def minmax(M):
    if M[0] > 25:
        M[0] = 25
    elif M[0] < -25:
        M[0] = -25
    if M[1] > 195:
        M[1] = 195
    elif M[1] < 165:
        M[1] = 165
    return M


mp_iris = mp.solutions.iris
mp_draw = mp.solutions.drawing_utils
def eye_movement_mediapipe():
    num = 4
    # 아래 bound 들은 건들지 말것.
    ag_bound = 0.0022
    bd_bound = 0.0025
    ad_bound = 0.0037
    bg_bound = 0.0021
    W = WebSocketClient()
    W.connection()
    M = np.zeros((2,1))
    E = np.zeros((5,1))
    M_l = np.zeros((5,1))
    M_r = np.zeros((5,1))
    frame_cnt = 0
    Seconds = 6
    cali_left_x = []; cali_left_y = []; cali_right_x = []; cali_right_y = []  # x, y 좌표 저장
    alpha_l = np.zeros((2, 5));alpha_r = np.zeros((2, 5))  # calibration result

    br1 = 0.0; br2 = 0.0; br3 = 0.0; br4 = 0.0
    lx = 0; ly=0; rx=0 ; ry = 0
    le1 = 0; le2=0; le3=0; le4=0
    monitor = [[50, 1870, 50, 1870, 960],
               [50, 50, 1030, 1030, 540]]
    avatar3d = [[-25, 25, -25, 25, 0],
                [195, 195, 165, 165, 180]]

    now_lx =0;now_ly=0;now_rx=0;now_ry =0
    eye_l = np.ones((5, 5)); eye_r = np.ones((5, 5))  # calibration eye
    count = 0;
    cnt = 0
    re = ''
    dist = 0
    cccc = 0
    head = ['test15', 'test17', 'test17']
    for i in range(len(head)):
        cap = cv2.VideoCapture(0)
        fps_pre = time.time()
        current_time = datetime.datetime.now()
        current_time = current_time.replace(microsecond=0)
        fix_prev_time = 0
        FPS = 300
        with mp_iris.Iris() as iris:
            while cap.isOpened():
                fps_cur = time.time()
                fps = 1 / (fps_cur - fps_pre)
                fps_pre = fps_cur
                success, image = cap.read()
                fix_current_time = time.time() - fix_prev_time
                if not success:
                    print("Ignoring empty camera frame.")
                    break
                elif (success is True) and (fix_current_time > 1./FPS):
                    fix_prev_time = time.time()
                    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
                    image = cv2.resize(image, dsize=(0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
                    image.flags.writeable = False
                    results = iris.process(image)
                    image.flags.writeable = True
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    if cv2.waitKey(1) == ord('s'):
                        threading.Timer(6, cali).start()
                        # time.sleep(3)
                        cccc = 1
                    if cccc == 1:
                        for id, cen in enumerate(results.face_landmarks_with_iris.landmark):
                            shape = image.shape
                            left_w = int(shape[1] / 2)
                            frame_cnt = frame_cnt + 1
                            new_time = current_time + datetime.timedelta(seconds=Seconds)

                            if count > 4 and count < 10:#calibration 계산
                                print("10 > count > 4 ")
                                alpha_l, beta_l = calculate(eye_l, avatar_3d=avatar3d, monitor=monitor)
                                alpha_r, beta_r = calculate(eye_r, avatar_3d=avatar3d, monitor=monitor)
                                count = 999
                                cnt = 4
                                print(alpha_l)
                            elif count < 5: #calibration을 위해 홍채 x,y metrix로 모음
                                now_time = datetime.datetime.now()
                                now_time = now_time.replace(microsecond=0)
                                if id == 468:
                                    lx = int(cen.x * shape[1])
                                    ly = int(cen.y * shape[0])
                                    cali_left_x.append(lx)
                                    cali_left_y.append(ly)
                                if id == 473:
                                    rx = int(cen.x * shape[1])
                                    ry = int(cen.y * shape[0])
                                    cali_right_x.append(rx)
                                    cali_right_y.append(ry)

                                if now_time == new_time: #calibration 한 점을 바라보는 시간동안 모아진 x, y 평균 내서 metrix에 넣는 곳  # frame_cnt == 90:#
                                    print("frmae_cnt")
                                    current_time = datetime.datetime.now()
                                    current_time = current_time.replace(microsecond=0)

                                    eye_l = fill_matrix(cali_left_x, cali_left_y, eye_l, count)
                                    eye_r = fill_matrix(cali_right_x, cali_right_y, eye_r, count)
                                    cali_left_x.clear()
                                    cali_left_y.clear()
                                    cali_right_x.clear()
                                    cali_right_y.clear()
                                    frame_cnt = 0
                                    count += 1
                            if id == 473: # 왼쪽 오른쪽 값을 한번에 보내려고 해놓은 것. 건드리면 코드 안돌아감.. 없애려고 했는데 ㅋ...
                                if cnt == 4:#calibration 이후 실시간으로 들어온 값을 계산하여 아바타에 맞게 교정함.
                                    rx = int(cen.x * shape[1])
                                    ry = int(cen.y * shape[0])
                                    alpha, beta, gamma, delta = rate(left_w, shape[0], rx, ry)
                                    le3, le4 = correct(alpha, beta, gamma, delta)
                                    #실시간으로 계산되어 교정된 좌표들의 흔들림 교정
                                    if (br3 - ag_bound * 4 <= le3) and (le3 <= br3 + ag_bound * 4):
                                        now_rx = now_rx
                                        br3 = br3
                                    else:
                                        now_rx = rx
                                        br3 = le3
                                    if (br4 - bd_bound * 4 <= le4) and (le4 <= br4 + bd_bound * 4):
                                        now_ry = now_ry
                                        br4 = br4
                                    else:
                                        now_ry = ry
                                        br4 = le4

                                M_l = dotdot(now_lx, now_ly, alpha_l)
                                M_l = minmax(M_l)
                                M_r = dotdot(now_rx, now_ry, alpha_r)
                                M_r = minmax(M_r)

                                #현재는 아바타에 오른쪽 눈 값만 넘어가서 눈이 각각 따로 움직이지 않음.
                                re = str(int(M_r[0])) + ',' + str(int(M_r[1])) + ',' \
                                     + str(int(M_r[0])) + ',' + str(int(M_r[1]))


                                try:
                                    W.send_data(re)
                                except:
                                    W.connection()

                    str_ = "FPS : %0.1f" % fps
                    cv2.putText(image, str_, (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0))
                    cv2.imshow('frame', image)
                    if cv2.waitKey(1) == ord('e'):
                        break
def main():
    eye_movement_mediapipe()

if __name__ == '__main__':
    main()