from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *
import matplotlib.pyplot as plt

cap = cv2.VideoCapture("../videos/Video Project.mp4")
model = YOLO("../yolo weights/yolov8n.pt")
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]
mask = cv2.imread("full mask.png")
tracker = Sort(max_age=20,min_hits=3,iou_threshold=0.3)
limits = [400,297,673,297]
totalcount_left = []
totalcount_right = []
track_history = {}

line_limit_up = [164, 265, 533, 295]
line_limit_down = [502, 369, 47, 322]


lane1 = np.array([[440, 287], [375, 355], [480, 368], [519, 293]], dtype=np.int32)
lane2 = np.array([[438, 287], [356, 280], [261, 344], [372, 355]], dtype=np.int32)
lane3 = np.array([[353, 280], [273, 272], [163, 332], [262, 344]], dtype=np.int32)
lane4 = np.array([[275, 273], [191, 269], [56, 323], [163, 333]], dtype=np.int32)


lanes_left_info = [
    {'poly': lane1, 'name': 'Lane 1', 'color': (114, 63, 116)}, # بنفسجي
    {'poly': lane2, 'name': 'Lane 2', 'color': (117, 113, 52)}, # كحلي/أخضر
    {'poly': lane3, 'name': 'Lane 3', 'color': (56, 117, 118)}, # زيتوني
    {'poly': lane4, 'name': 'Lane 4', 'color': (85, 69, 220)}   # أحمر
]
# --- الخطوط الحمراء (الحدود العلوية والسفلية) ---
line_limit_up_right = [698, 303, 1146, 291]
line_limit_down_right = [723, 383, 1262, 347]

# --- مضلعات المسارات الأربعة (Lanes) ---
lane1_r = np.array([[731, 303], [772, 379], [876, 372], [806, 299]], dtype=np.int32)
lane2_r = np.array([[810, 301], [877, 373], [991, 365], [893, 296]], dtype=np.int32)
lane3_r = np.array([[895, 297], [992, 365], [1093, 357], [975, 294]], dtype=np.int32)
lane4_r = np.array([[1112, 290], [1228, 346], [1094, 357], [973, 294]], dtype=np.int32)

# قائمة المسارات المحدثة مع ألوانها (BGR)
lanes_right_info = [
    {'poly': lane1_r, 'color': (114, 63, 116)},
    {'poly': lane2_r, 'color': (117, 113, 52)},
    {'poly': lane3_r, 'color': (56, 117, 118)},
    {'poly': lane4_r, 'color': (85, 69, 220)}
]
while True:
    ret, frame = cap.read()
    if not ret: break

    imgRegion = cv2.bitwise_and(frame, mask)

    results = model(imgRegion, stream=True)
    all_detections = np.empty((0, 5))
    overlay = frame.copy()


    all_lanes = lanes_right_info + lanes_left_info

    for lane in all_lanes:
        cv2.fillPoly(overlay, [lane['poly']], color=lane['color'])


    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)


    red = (0, 0, 255)

    cv2.line(frame, (line_limit_up_right[0], line_limit_up_right[1]), (line_limit_up_right[2], line_limit_up_right[3]),
             red, 3)
    cv2.line(frame, (line_limit_down_right[0], line_limit_down_right[1]),
             (line_limit_down_right[2], line_limit_down_right[3]), red, 3)

    cv2.line(frame, (line_limit_up[0], line_limit_up[1]), (line_limit_up[2], line_limit_up[3]), red, 3)
    cv2.line(frame, (line_limit_down[0], line_limit_down[1]), (line_limit_down[2], line_limit_down[3]), red, 3)


    for lane in all_lanes:
        cv2.polylines(frame, [lane['poly']], isClosed=True, color=(255, 255, 255), thickness=1)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            currentClass = classNames[cls]

            if currentClass in ["car", "truck", "bus", "motorbike"] and conf > 0.3:
                all_detections = np.vstack([all_detections, [x1, y1, x2, y2, conf]])


    resultTracker = tracker.update(all_detections)

    for res in resultTracker:
        x1, y1, x2, y2, id = map(int, res)


        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)
        cvzone.putTextRect(frame, f' {classNames[cls]}', (max(0, x1), max(35, y1)), scale=1, thickness=1)


        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.circle(frame, (cx, cy), 5, (255, 0, 00), cv2.FILLED)
        if id not in track_history:
            track_history[id] = []

        track_history[id].append((cx, cy))


        if len(track_history[id]) > 30:
            track_history[id].pop(0)


        points = np.array(track_history[id], np.int32)
        points = points.reshape((-1, 1, 2))
        cv2.polylines(frame, [points], isClosed=False, color=(255, 0, 0), thickness=1)



        green = (0, 255, 0)

        if line_limit_up[0] < cx < line_limit_up[2] and line_limit_up[1] <= cy <= line_limit_down[1] :
            if id not in totalcount_left:
                totalcount_left.append(id)
                cv2.line(frame, (line_limit_up[0], line_limit_up[1]), (line_limit_up[2], line_limit_up[3]), green, 3)
                cv2.line(frame, (line_limit_down[0], line_limit_down[1]), (line_limit_down[2], line_limit_down[3]), green,
                         3)
        if line_limit_up_right[0] < cx < line_limit_up_right[2] and line_limit_up_right[1]  <= cy <= line_limit_down_right[1]:
            if id not in totalcount_right:
                totalcount_right.append(id)
                cv2.line(frame, (line_limit_up_right[0], line_limit_up_right[1]),
                         (line_limit_up_right[2], line_limit_up_right[3]),
                         green, 3)
                cv2.line(frame, (line_limit_down_right[0], line_limit_down_right[1]),
                         (line_limit_down_right[2], line_limit_down_right[3]), green, 3)

    cvzone.putTextRect(frame,f"count left : {len(totalcount_left)}",(50,50))
    cvzone.putTextRect(frame, f"count right : {len(totalcount_right)}", (830,50))
    cv2.imshow("YOLOv8 Detection", frame)

    cv2.waitKey(1)

