# ------------Step 1: Use VideoCapture in OpenCV------------
import cv2
import dlib
from math import sqrt, inf
from moviepy.editor import VideoFileClip

# Face detection with dlib
detector = dlib.get_frontal_face_detector()

# Landmark prediction
landmarks_file = "/Users/lizziehernandez/Desktop/Science Fair 2021/shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(landmarks_file)

# eye landmarks
left_eye_landmarks = [36, 37, 38, 39, 40, 41]
right_eye_landmarks = [42, 43, 44, 45, 46, 47]
nose_landmarks = [31, 35]
BLINK_RATIO_THRESHOLD = 5.7

# Face and eye classifiers from OpenCv
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eyes_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

count = 0
blinking = False

eye_m_north = +inf
eye_m_south = -inf
eye_m_west = +inf
eye_m_east = -inf
nose_width_max = -inf
nose_width_min = inf

def midpoint(point1 ,point2):
    return int((point1.x + point2.x)/2), int((point1.y + point2.y)/2)


def euclidean_distance(point1 , point2):
    return sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)


def detect_and_display(frame):
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frame_gray = cv.equalizeHist(frame_gray)

    # Detect faces
    faces = face_cascade.detectMultiScale(frame_gray)
    faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)

    
    for (x, y, w, h) in faces[:1]:
        center = (x + w//2, y + h//2)
        frame = cv.ellipse(frame, center, (w//2, h//2), 0, 0, 360, (255, 0, 255), 4)
        faceROI = frame_gray[y:y+h,x:x+w]

        # In each face, detect eyes
        eyes = eyes_cascade.detectMultiScale(faceROI, 1.3, 10)
        for (x2, y2, w2, h2) in eyes:
            eye_center = (x + x2 + w2//2, y + y2 + h2//2)
            radius = int(round((w2 + h2) * 0.25))
            frame = cv.circle(frame, eye_center, radius, (255, 0, 0), 4)


cap = cv.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Display the resulting frame
    detect_and_display(frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv.destroyAllWindows()
