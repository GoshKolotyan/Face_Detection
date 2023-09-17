import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture("Videos/Face_Video_1.mp4")
width = 800
height = 600
pTime = 0


mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
faceDetecion = mpFaceDetection.FaceDetection(min_detection_confidence=0.75)


while True:
    success, img = cap.read()

    if not success:
        break


    img = cv2.resize(img, (width, height))


    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceDetecion.process(imgRGB)
    print(results)

    if results.detections:
        for id, detection in enumerate(results.detections):
            # mpDraw.draw_detection(img, detection)
            # print(id, detection)
            # print(f"Score of Detection {detection.score}")
            print(detection.location_data.relative_bounding_box.xmin)
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, ic = img.shape
            bbox = int((bboxC.xmin * iw)), int(bboxC.ymin * ih), \
                   int(bboxC.width * iw), int(bboxC.height * ih)

            cv2.rectangle(img, bbox, (255, 0, 255), 2)

            cv2.putText(img, f"{int(detection.score[0]*100)}%",
                    (bbox[0], bbox[1]-20), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0,255), 2)


    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 4, (0, 255, 0), 5)

    cv2.imshow('Image', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
       break