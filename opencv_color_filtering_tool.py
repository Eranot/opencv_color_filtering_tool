import cv2
import numpy as np
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("-f", "--file", dest="file",
                    help="File to open", metavar="FILE")
parser.add_argument("-v", "--video",
                    help="Use this flag if your file is a video", action='store_true')
parser.add_argument("-r", "--resize",
                    help="Resizes the image by the float passed", default=1.0, type=float)

args = parser.parse_args()

# BGR
bgr_b_min = 0
bgr_b_max = 255
bgr_g_min = 0
bgr_g_max = 255
bgr_r_min = 0
bgr_r_max = 255

# HLS
hls_h_min = 0
hls_h_max = 255
hls_l_min = 0
hls_l_max = 255
hls_s_min = 0
hls_s_max = 255

# HSV
hsv_h_min = 0
hsv_h_max = 255
hsv_s_min = 0
hsv_s_max = 255
hsv_v_min = 0
hsv_v_max = 255

def test_bgr():
    global image
    converted = image
    lower = np.uint8([bgr_b_min, bgr_g_min, bgr_r_min])
    upper = np.uint8([bgr_b_max, bgr_g_max, bgr_r_max])
    mask = cv2.inRange(converted, lower, upper)
    new_image = cv2.bitwise_and(converted, converted, mask=mask)
    cv2.imshow('bgr', new_image)

def test_hls():
    global image
    converted = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    lower = np.uint8([hls_h_min, hls_l_min, hls_s_min])
    upper = np.uint8([hls_h_max, hls_l_max, hls_s_max])
    mask = cv2.inRange(converted, lower, upper)
    new_image = cv2.bitwise_and(converted, converted, mask=mask)
    cv2.imshow('hls', new_image)

def test_hsv():
    global image
    converted = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    lower = np.uint8([hsv_h_min, hsv_s_min, hsv_v_min])
    upper = np.uint8([hsv_h_max, hsv_s_max, hsv_v_max])
    mask = cv2.inRange(converted, lower, upper)
    new_image = cv2.bitwise_and(converted, converted, mask=mask)
    cv2.imshow('hsv', new_image)

def on_slider_bgr_b_min(value):
    global bgr_b_min
    bgr_b_min = value
    test_bgr()

def on_slider_bgr_b_max(value):
    global bgr_b_max
    bgr_b_max = value
    test_bgr()

def on_slider_bgr_g_min(value):
    global bgr_g_min
    bgr_g_min = value
    test_bgr()

def on_slider_bgr_g_max(value):
    global bgr_g_max
    bgr_g_max = value
    test_bgr()

def on_slider_bgr_r_min(value):
    global bgr_r_min
    bgr_r_min = value
    test_bgr()

def on_slider_bgr_r_max(value):
    global bgr_r_max
    bgr_r_max = value
    test_bgr()

def on_slider_hls_h_min(value):
    global hls_h_min
    hls_h_min = value
    test_hls()

def on_slider_hls_h_max(value):
    global hls_h_max
    hls_h_max = value
    test_hls()

def on_slider_hls_l_min(value):
    global hls_l_min
    hls_l_min = value
    test_hls()

def on_slider_hls_l_max(value):
    global hls_l_max
    hls_l_max = value
    test_hls()

def on_slider_hls_s_min(value):
    global hls_s_min
    hls_s_min = value
    test_hls()

def on_slider_hls_s_max(value):
    global hls_s_max
    hls_s_max = value
    test_hls()

def on_slider_hsv_h_min(value):
    global hsv_h_min
    hsv_h_min = value
    test_hsv()

def on_slider_hsv_h_max(value):
    global hsv_h_max
    hsv_h_max = value
    test_hsv()

def on_slider_hsv_s_min(value):
    global hsv_s_min
    hsv_s_min = value
    test_hsv()

def on_slider_hsv_s_max(value):
    global hsv_s_max
    hsv_s_max = value
    test_hsv()

def on_slider_hsv_v_min(value):
    global hsv_v_min
    hsv_v_min = value
    test_hsv()

def on_slider_hsv_v_max(value):
    global hsv_v_max
    hsv_v_max = value
    test_hsv()

def create_trackbars():
    cv2.createTrackbar('h_min', 'hls', 0, 255, on_slider_hls_h_min)
    cv2.createTrackbar('h_max', 'hls', 255, 255, on_slider_hls_h_max)
    cv2.createTrackbar('l_min', 'hls', 0, 255, on_slider_hls_l_min)
    cv2.createTrackbar('l_max', 'hls', 255, 255, on_slider_hls_l_max)
    cv2.createTrackbar('s_min', 'hls', 0, 255, on_slider_hls_s_min)
    cv2.createTrackbar('s_max', 'hls', 255, 255, on_slider_hls_s_max)

    cv2.createTrackbar('b_min', 'bgr', 0, 255, on_slider_bgr_b_min)
    cv2.createTrackbar('b_max', 'bgr', 255, 255, on_slider_bgr_b_max)
    cv2.createTrackbar('g_min', 'bgr', 0, 255, on_slider_bgr_g_min)
    cv2.createTrackbar('g_max', 'bgr', 255, 255, on_slider_bgr_g_max)
    cv2.createTrackbar('r_min', 'bgr', 0, 255, on_slider_bgr_r_min)
    cv2.createTrackbar('r_max', 'bgr', 255, 255, on_slider_bgr_r_max)

    cv2.createTrackbar('h_min', 'hsv', 0, 255, on_slider_hsv_h_min)
    cv2.createTrackbar('h_max', 'hsv', 255, 255, on_slider_hsv_h_max)
    cv2.createTrackbar('s_min', 'hsv', 0, 255, on_slider_hsv_s_min)
    cv2.createTrackbar('s_max', 'hsv', 255, 255, on_slider_hsv_s_max)
    cv2.createTrackbar('v_min', 'hsv', 0, 255, on_slider_hsv_v_min)
    cv2.createTrackbar('v_max', 'hsv', 255, 255, on_slider_hsv_v_max)

    img_shortcuts = np.zeros((100, 455))
    cv2.putText(img_shortcuts, 'Quit = q', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img_shortcuts, 'Play/Pause video = Enter/Spacebar', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow('shortcuts', img_shortcuts)

def resize_image(frame):
    if args.resize != 1.0:
        frame = cv2.resize(frame, (int(frame.shape[1] * args.resize), int(frame.shape[0] * args.resize)))
    return frame

def wait_for_enter_or_q():
    while True:
        key = cv2.waitKey(1)
        if key == 13 or key == 32:
            return 'enter'
        elif key == ord('q'):
            return 'q'

if not args.video:
    while True:
        image = cv2.imread(args.file)
        image = resize_image(image)
        test_bgr()
        test_hls()
        test_hsv()
        create_trackbars()
        if(wait_for_enter_or_q() == 'q'):
            break
else:
    video_capture = cv2.VideoCapture(args.file)
    ret, frame = video_capture.read()
    if ret:
        image = frame
        image = resize_image(image)
        test_bgr()
        test_hls()
        test_hsv()
    create_trackbars()
    should_close = False

    if(wait_for_enter_or_q() == 'enter'):
        while not should_close:
            video_capture = cv2.VideoCapture(args.file)

            while (video_capture.isOpened()):
                ret, frame = video_capture.read()
                if ret:
                    image = frame
                    image = resize_image(image)
                    test_bgr()
                    test_hls()
                    test_hsv()
                    key = cv2.waitKey(1)
                    if key == ord('q'):
                        should_close = True
                        break
                    elif key == 13 or key == 32:
                        wait_for_enter_or_q()
                else:
                    if(wait_for_enter_or_q() == 'q'):
                        should_close = True
                    break

    video_capture.release()
cv2.destroyAllWindows()
