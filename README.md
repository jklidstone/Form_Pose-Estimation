# Form_Pose-Estimation
Work-in-progress/POC weightlifting exercise classifier and form suggestion application. Heavily leverages CMU's OpenPose.

The core idea is that a computer can recognize a weightlifting exercise (for now, one of: Squat, Overhead-Press, Deadlift) by the relationship of
a user's knees, wrists, and chest (mainly the angles between these structures). 

## Getting started:
- Get MPI, COCO, and Body_25 models via getModels.sh
- install opencv-python, pandas, numpy, and sklearn

Trial with (Windows -- OSX only needs to replace py with python):
- py run_pose.py --input ohp_7rest.png --proto models/mpi/pose_deploy_linevec_faster_4_stages.prototxt  --model models/mpi/pose_iter_160000.caffemodel --dataset MPI

You should get something like this:
![](https://github.com/jklidstone/Form_Pose-Estimation/blob/main/Form_Estimation/ohp_7rest.png?raw=true)
![](https://github.com/jklidstone/Form_Pose-Estimation/blob/main/Form_Estimation/result_ohp_7rest.png?raw=true)

If you wish to save the image/frame extracted information to a .csv, you can do so with:
- py run_pose.py --input sample.jpg --proto models/mpi/pose_deploy_linevec_faster_4_stages.prototxt  --model models/mpi/pose_iter_160000.caffemodel --dataset MPI --trainpath YOUR_PATH_HERE

there is already some data (although not enough!) in training.csv if you wish to use that.

## Some Pipeline guide/explanation:

As of now, the application exists through two primary files: run_pose.py and training.py
### run_pose.py
run_pose.py utilizes CLI to gather the image to be processed from the user along with the model they wish to use (of MPI, COCO, Body_25 - Body_25 will probably be discarded in the future).
The script extracts all points of interest via global maximums (can do local for multiple people in frame, but - we only want to track one user so this will probably remain).
'Key' features (as defined by whoever works on this!) are extracted (wrists, knees, chest) and the angles between said structures are as well. This is then shuttled over
to training.py for some basic ML via sklearn (if the --training arg is set to a specific path containing a .csv). The remaining work then plots body parts and lines between pairings.
These points and lines will be utilized when the application is upgraded to live classification via video to determine 'good'/'bad' form - the lines between a user's body part pairs
will be color coded depending on where the angle exists on a range (i.e. for Squat - can determine knees bent too far inward/outward).

### training.py
Simple script utilizing sklearn to take in a .csv containing extracted features from run_pose.py on exercise images. Data is manipulated a bit using pandas then interacted
with via sklearn's k-nearest-neighbors. Current classification isn't ~too~ good right now -- because (apparently!) it is extremely hard to find large quantities of
weightlifting images.

## Upcoming TODOs:

- (Maybe) Dockerize the application to make setup and interaction a little more streamlined. Not so big of a deal now since I'm the only person touching this, but could be
useful and cool.

- More Data! Need a lot more images to train classifiers.

- Pip freeze into requirements.txt (once versions are decided)

- Video input/output


### Disclaimer:
May axe this in favor of a new build utilizing https://google.github.io/mediapipe/solutions/pose


