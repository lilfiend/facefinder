# Command to run. Use cnn or hog model to encode faces. cnn is more accurate but its slow, hog is less accurate but its fast.
# Note that facial detection is hardcoded to use cnn. Since this isn't realtime it shouldnt be a big deal, if its too slow for you change line 
# python encode.py --dataset dataset --output cnn.pickle --model cnn


# import all the things
from imutils import paths
import face_recognition
import argparse
import pickle
import cv2
import os

# That feel when newly born code handles arguments better than you
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--dataset", required=True, help="path to the directory that holds the directories of each person you have datasets for")
ap.add_argument("-e", "--output", required=True, help="path/name of encoded faces, eg cnn.spkl")
ap.add_argument("-d", "--model", type=str, default="cnn", help="face encoding method to use: either `hog` or `cnn`")
args = vars(ap.parse_args())

# Git gud images
print("[INFO] quantifying faces...")
imagePaths = list(paths.list_images(args["dataset"]))

# Arrays are fun
knownEncodings = []
knownNames = []

# loop over the image paths
for (i, imagePath) in enumerate(imagePaths):
	
	print("[INFO] processing image {}/{}".format(i + 1, len(imagePaths)))

	# extract the person name from the image path
	name = imagePath.split(os.path.sep)[-2]

	# Image loads all BGR but we want RGB, thankfully corsair--err cv2 has us covered.
	image = cv2.imread(imagePath)
	rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

	#This is the face detection part, using CNN/HOG for the face detection is a bit intensive... but results are great
	boxes = face_recognition.face_locations(rgb, model=args["model"])

	# Encode faces, more jitters is more accurate but takes longer, eg 10 is 10x longer
	encodings = face_recognition.face_encodings(rgb, boxes, num_jitters=100)
	# loop over the encodings
	for encoding in encodings:
		# add encoding and name to arrays for saving
		knownEncodings.append(encoding)
		knownNames.append(name)

# dump the facial encodings + names to disk
print("[INFO] serializing encodings...")
data = {"encodings": knownEncodings, "names": knownNames}
f = open(args["output"], "wb")
f.write(pickle.dumps(data))
f.close()


