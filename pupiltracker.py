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


def update_eyebbbox(eye_center):
    global eye_m_west, eye_m_north, eye_m_south, eye_m_east

    if eye_center[0] < eye_m_west:
        eye_m_west = eye_center[0]
    if eye_center[1] < eye_m_north:
        eye_m_north = eye_center[1]
    if eye_center[0] > eye_m_east:
        eye_m_east = eye_center[0]
    if eye_center[1] > eye_m_south:
        eye_m_south = eye_center[1]


def eye_area():
    return (eye_m_south - eye_m_north) * (eye_m_east - eye_m_west)


def get_blink_ratio(eye_points, facial_landmarks):
    # loading all the required points
    corner_left = (facial_landmarks.part(eye_points[0]).x,
                   facial_landmarks.part(eye_points[0]).y)
    corner_right = (facial_landmarks.part(eye_points[3]).x,
                    facial_landmarks.part(eye_points[3]).y)

    center_top = midpoint(facial_landmarks.part(eye_points[1]),
                          facial_landmarks.part(eye_points[2]))
    center_bottom = midpoint(facial_landmarks.part(eye_points[5]),
                             facial_landmarks.part(eye_points[4]))

    # calculating distance
    horizontal_length = euclidean_distance(corner_left, corner_right)
    vertical_length = euclidean_distance(center_top, center_bottom)

    ratio = horizontal_length / vertical_length

    return ratio


def get_nose_width(facial_landmarks):
    global nose_width_min, nose_width_max
    corner_left = (facial_landmarks.part(nose_landmarks[0]).x,
                   facial_landmarks.part(nose_landmarks[0]).y)

    corner_right = (facial_landmarks.part(nose_landmarks[1]).x,
                   facial_landmarks.part(nose_landmarks[1]).y)

    distance = euclidean_distance(corner_left, corner_right)

    if distance < nose_width_min:
        nose_width_min = distance
    if distance > nose_width_max:
        nose_width_max = distance


def detect_blinking(frame):
    global count, blinking
    # detecting faces in the frame
    faces, _, _ = detector.run(image=frame, upsample_num_times=0,
                               adjust_threshold=0.0)

    # -----Step 4: Detecting Eyes using landmarks in dlib-----
    for face in faces:

        landmarks = predictor(frame, face)

        # -----Step 5: Calculating blink ratio for one eye-----
        left_eye_ratio = get_blink_ratio(left_eye_landmarks, landmarks)
        right_eye_ratio = get_blink_ratio(right_eye_landmarks, landmarks)
        blink_ratio = (left_eye_ratio + right_eye_ratio) / 2

        if blink_ratio > BLINK_RATIO_THRESHOLD:
            # Blink detected! Do Something!
            cv2.putText(frame, "BLINKING", (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        2, (255, 255, 255), 2, cv2.LINE_AA)

            if not blinking:
                count += 1
                blinking = True
        else:
            blinking = False


def detect_and_display(frame, display=True):
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.equalizeHist(frame_gray)

    detect_blinking(frame_gray)

    # Detect faces
    faces = face_cascade.detectMultiScale(frame_gray)
    faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)

    for (x, y, w, h) in faces[:1]:
        center = (x + w//2, y + h//2)
        if display:
            frame = cv2.ellipse(frame, center, (w//2, h//2), 0, 0, 360, (255, 0, 255), 4)
        faceROI = frame_gray[y:y+h,x:x+w]

        # In each face, detect eyes
        eyes = eyes_cascade.detectMultiScale(faceROI, 1.3, 10)
        if len(eyes) < 2:
            return

        eyes = sorted(eyes, key=lambda e: e[0])
        for (x2, y2, w2, h2) in eyes[:1]:
            eye_center = (x + x2 + w2//2, y + y2 + h2//2)
            if display:
                frame = cv2.circle(frame, eye_center, 10, (255, 0, 0), 4)
            dist_from_face_center = (abs(eye_center[0] - center[0]), abs(eye_center[1] - center[1]))
            update_eyebbbox(dist_from_face_center)

    if display:
        cv2.imshow('Capture - Face detection', frame)


# filename should be 0 for webcam
def get_video_data(filename, display=True):
    cap = cv2.VideoCapture(filename)

    while True:
        # capturing frame
        retval, frame = cap.read()

        # exit the application if frame not found
        if not retval:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        # detect_blinking(frame)
        detect_and_display(frame, display=display)

        if display:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # releasing the VideoCapture object
    cap.release()
    cv2.destroyAllWindows()

    # return data : blinks per minute, eye movement measure
    if not display:
        duration = VideoFileClip(filename).duration / 60
        print('Blinks per minute', count / duration)
    else:
        duration = 60  # placeholder

    print('Eye area movement', eye_area())
    return count / duration, eye_area()


def save_data_to_file(original_fname, save_to):
    bpm, eyemov = get_video_data(original_fname, display=False)

    with open(save_to, "a") as text_file:
        text_file.write('BPM ' + str(bpm) + '\n')
        text_file.write('EYEMOV ' + str(eyemov) + '\n')


if __name__ == '__main__':
    fname = '/'
    save = '/'

    save_data_to_file(fname, save)
