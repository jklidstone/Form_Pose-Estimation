# ------------------------- POSE MODELS -------------------------
# Downloading body pose (COCO and MPI).
OPENPOSE_URL="http://posefs1.perception.cs.cmu.edu/OpenPose/models/"
MODELS_FOLDER="models/"

# ------------------------- POSE MODELS -------------------------
# Body (BODY_25)
BODY_25_FOLDER=${MODELS_FOLDER}"body_25/"
BODY_25_MODEL=${BODY_25_FOLDER}"pose_iter_584000.caffemodel"
wget -c ${OPENPOSE_URL}${BODY_25_MODEL} -P ${BODY_25_FOLDER}

# Body (COCO)
COCO_FOLDER=${MODELS_FOLDER}"coco/"
COCO_MODEL=${COCO_FOLDER}"pose_iter_440000.caffemodel"
wget -c ${OPENPOSE_URL}${COCO_MODEL} -P ${COCO_FOLDER}
# Alternative: it will not check whether file was fully downloaded
# if [ ! -f $COCO_MODEL ]; then
#     wget ${OPENPOSE_URL}$COCO_MODEL -P $COCO_FOLDER
# fi

# Body (MPI)
MPI_FOLDER=${MODELS_FOLDER}"mpi/"
MPI_MODEL=${MPI_FOLDER}"pose_iter_160000.caffemodel"
wget -c ${OPENPOSE_URL}${MPI_MODEL} -P ${MPI_FOLDER}

