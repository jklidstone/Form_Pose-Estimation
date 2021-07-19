import cv2 as cv
import numpy as np
import argparse
import settings
import csv
from math import atan2, degrees
from training import train_classifier_km

# import imutils
import time

# Setup / terminal input parse (dataset, image, etc.)
args = settings.parser.parse_args()

if args.dataset == "COCO":
    BODY_PARTS = settings.COCO_BODY_PARTS
    PART_PAIRS = settings.COCO_PART_PAIRS
elif args.dataset == "MPI":
    BODY_PARTS = settings.MPI_BODY_PARTS
    PART_PAIRS = settings.MPI_PART_PAIRS
else:
    BODY_PARTS = settings.BODY25_BODY_PARTS
    PART_PAIRS = settings.BODY25_PART_PAIRS

inWidth = args.width
inHeight = args.height

# ...

net = cv.dnn.readNetFromCaffe(args.proto, args.model)

frame = cv.imread(args.input)
frameWidth = frame.shape[1]
frameHeight = frame.shape[0]

inp = cv.dnn.blobFromImage(
    frame, 1.0 / 255, (inWidth, inHeight), (0, 0, 0), swapRB=False, crop=False
)
net.setInput(inp)
start_t = time.time()
out = net.forward()

print("time is ", time.time() - start_t)
windowName = "Form Guidance - POC"
cv.namedWindow(windowName, cv.WINDOW_AUTOSIZE)


def GetAngleOfLineBetweenTwoPoints(p1, p2):
    xDiff = p2[0] - p1[0]
    yDiff = p2[1] - p1[1]
    return degrees(atan2(yDiff, xDiff))


# features TODO: more features!
lknee = []
rknee = []
lwrist = []
rwrist = []
chest = []
points = []
for i in range(len(BODY_PARTS)):
    # Slice heatmap of corresponging body's part.
    heatMap = out[0, i, :, :]

    # Originally, we try to find all the local maximums. To simplify a sample
    # we just find a global one. However only a single pose at the same time
    # could be detected this way.
    _, conf, _, point = cv.minMaxLoc(heatMap)
    x = (frameWidth * point[0]) / out.shape[3]
    y = (frameHeight * point[1]) / out.shape[2]

    # get angles between center torso (14) and LWrist (7) and RWrist (4)

    # Add a point if it's confidence is higher than threshold.
    points.append((int(x), int(y)) if conf > args.thr else None)
    if conf > args.thr:
        if args.dataset == "MPI":
            if i == 14:
                chest = [x, y]
            if i == 12:
                lknee = [x, y]
            if i == 9:
                rknee = [x, y]
            if i == 7:
                lwrist = [x, y]
            if i == 4:
                rwrist = [x, y]
        # TODO: conditional for mpi, body25

print(lwrist)
print(rwrist)
print(chest)
print(GetAngleOfLineBetweenTwoPoints(lwrist, chest) - 180)
print(GetAngleOfLineBetweenTwoPoints(rwrist, chest))
print(GetAngleOfLineBetweenTwoPoints(lknee, chest) - 180)
print(GetAngleOfLineBetweenTwoPoints(rknee, chest))

# image processing is done, store data if we are collecting to train classifier, continue on with UI (point plotting on image, etc.)

if args.trainpath != "":
    # store features
    # classification of motion, pulled from image title -- requires captioning of pictures (!)
    image_title_split = args.input.split("_")
    rowToInsert = [
        lwrist[0],
        lwrist[1],
        rwrist[0],
        rwrist[1],
        chest[0],
        chest[1],
        lknee[0],
        lknee[1],
        rknee[0],
        rknee[1],
        (GetAngleOfLineBetweenTwoPoints(lwrist, chest) - 180),
        (GetAngleOfLineBetweenTwoPoints(rwrist, chest)),
        (GetAngleOfLineBetweenTwoPoints(lknee, chest) - 180),
        (GetAngleOfLineBetweenTwoPoints(rknee, chest)),
        image_title_split[0],
    ]
    with open(args.trainpath, "a") as f:
        writer = csv.writer(f)
        writer.writerow(rowToInsert)

# TODO: If not adding to training data, use current frame (or image) in model returned by train_classifier_km(..)
#                                       to capture movement (squat, ohp, deadlift, etc.) then compare angles to
#                                       in various body_part_from -> body_part_two tuples against acceptable range
train_classifier_km()

# mapping on image
for pair in PART_PAIRS:
    body_part_from = pair[0]
    body_part_to = pair[1]
    # assert partFrom in BODY_PARTS
    # assert partTo in BODY_PARTS

    from_id = BODY_PARTS[body_part_from]
    to_id = BODY_PARTS[body_part_to]
    if points[from_id] and points[to_id]:
        cv.line(frame, points[from_id], points[to_id], (255, 74, 0), 3)
        cv.ellipse(
            frame, points[from_id], (4, 4), 0, 0, 360, (255, 255, 255), cv.FILLED
        )
        cv.ellipse(frame, points[to_id], (4, 4), 0, 0, 360, (255, 255, 255), cv.FILLED)
        cv.putText(
            frame,
            str(from_id),
            points[to_id],
            cv.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2,
            cv.LINE_AA,
        )
        cv.putText(
            frame,
            str(from_id),
            points[to_id],
            cv.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2,
            cv.LINE_AA,
        )

cv.imshow(windowName, frame)
cv.imwrite("result_" + args.input, frame)
