import argparse


# Parser/CLI

parser = argparse.ArgumentParser(
    description="""Pose Estimative via OpenPose (https://github.com/CMU-Perceptual-Computing-Lab/openpose) via OpenCV.
                 Utilize Pose Estimation to determine points of interest on human body, record angles between them, then
                then a classfier on movements and their respective body part <-> body part angles."""
)
parser.add_argument("--input", help="Path to input image.")
parser.add_argument("--proto", help="Path to .prototxt")
parser.add_argument("--model", help="Path to .caffemodel")
parser.add_argument(
    "--dataset",
    help="Specify what kind of model was trained. "
    "It could be (COCO, MPI) depends on dataset.",
)
parser.add_argument(
    "--trainpath",
    default="",
    help="Path to stored angles/points for classifier. Also considered a bool on whether or not the path is empty",
)
parser.add_argument(
    "--thr", default=0.1, type=float, help="Threshold value for pose parts heat map"
)
parser.add_argument(
    "--width", default=368, type=int, help="Resize input to specific width."
)
parser.add_argument(
    "--height", default=368, type=int, help="Resize input to specific height."
)

# COCO Features

COCO_BODY_PARTS = {
    "Nose": 0,
    "Neck": 1,
    "RShoulder": 2,
    "RElbow": 3,
    "RWrist": 4,
    "LShoulder": 5,
    "LElbow": 6,
    "LWrist": 7,
    "RHip": 8,
    "RKnee": 9,
    "RAnkle": 10,
    "LHip": 11,
    "LKnee": 12,
    "LAnkle": 13,
    "REye": 14,
    "LEye": 15,
    "REar": 16,
    "LEar": 17,
    "Background": 18,
}

COCO_PART_PAIRS = [
    ["Neck", "RShoulder"],
    ["Neck", "LShoulder"],
    ["RShoulder", "RElbow"],
    ["RElbow", "RWrist"],
    ["LShoulder", "LElbow"],
    ["LElbow", "LWrist"],
    ["Neck", "RHip"],
    ["RHip", "RKnee"],
    ["RKnee", "RAnkle"],
    ["Neck", "LHip"],
    ["LHip", "LKnee"],
    ["LKnee", "LAnkle"],
    ["Neck", "Nose"],
    ["Nose", "REye"],
    ["REye", "REar"],
    ["Nose", "LEye"],
    ["LEye", "LEar"],
]

# MPI Features

MPI_BODY_PARTS = {
    "Head": 0,
    "Neck": 1,
    "RShoulder": 2,
    "RElbow": 3,
    "RWrist": 4,
    "LShoulder": 5,
    "LElbow": 6,
    "LWrist": 7,
    "RHip": 8,
    "RKnee": 9,
    "RAnkle": 10,
    "LHip": 11,
    "LKnee": 12,
    "LAnkle": 13,
    "Chest": 14,
    "Background": 15,
}

MPI_PART_PAIRS = [
    ["Head", "Neck"],
    ["Neck", "RShoulder"],
    ["RShoulder", "RElbow"],
    ["RElbow", "RWrist"],
    ["Neck", "LShoulder"],
    ["LShoulder", "LElbow"],
    ["LElbow", "LWrist"],
    ["Neck", "Chest"],
    ["Chest", "RHip"],
    ["RHip", "RKnee"],
    ["RKnee", "RAnkle"],
    ["Chest", "LHip"],
    ["LHip", "LKnee"],
    ["LKnee", "LAnkle"],
]

# BODY_25 Features

BODY25_BODY_PARTS = {
    "Nose": 0,
    "Neck": 1,
    "RShoulder": 2,
    "RElbow": 3,
    "RWrist": 4,
    "LShoulder": 5,
    "LElbow": 6,
    "LWrist": 7,
    "MidHip": 8,
    "RHip": 9,
    "RKnee": 10,
    "RAnkle": 11,
    "LHip": 12,
    "LKnee": 13,
    "LAnkle": 14,
    "REye": 15,
    "LEye": 16,
    "REar": 17,
    "LEar": 18,
    "LBigToe": 19,
    "LSmallToe": 20,
    "LHeel": 21,
    "RBigToe": 22,
    "RSmallToe": 23,
    "RHeel": 24,
    "Background": 25,
}

BODY25_PART_PAIRS = [
    ["Neck", "MidHip"],
    ["Neck", "RShoulder"],
    ["Neck", "LShoulder"],
    ["RShoulder", "RElbow"],
    ["RElbow", "RWrist"],
    ["LShoulder", "LElbow"],
    ["LElbow", "LWrist"],
    ["MidHip", "RHip"],
    ["RHip", "RKnee"],
    ["RKnee", "RAnkle"],
    ["MidHip", "LHip"],
    ["LHip", "LKnee"],
    ["LKnee", "LAnkle"],
    ["Neck", "Nose"],
    ["Nose", "REye"],
    ["REye", "REar"],
    ["Nose", "LEye"],
    ["LEye", "LEar"],
    ["RShoulder", "REar"],
    ["LShoulder", "LEar"],
    ["LAnkle", "LBigToe"],
    ["LBigToe", "LSmallToe"],
    ["LAnkle", "LHeel"],
    ["RAnkle", "RBigToe"],
    ["RBigToe", "RSmallToe"],
    ["RAnkle", "RHeel"],
]
