# Command to run. Use cnn or hog model to encode faces. cnn is more accurate but its slow, hog is less accurate but its fast.
# Note that facial detection is hardcoded to use cnn. Since this isn't realtime it shouldnt be a big deal, if its too slow for you change line 
# python batchencode.py --dataset dataset --output cnn.pickle --model cnn


# import all the things
from imutils import paths
import face_recognition
import argparse
import pickle
import cv2
import os
import concurrent.futures

# That feel when newly born code handles arguments better than you
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--dataset", required=True, help="path to the directory that holds the directories of each person you have datasets for")
ap.add_argument("-e", "--output", required=True, help="path/name of encoded faces, eg cnn.spkl")
ap.add_argument("-s", "--size", type=int, default=25, help="How many images to process at once, default is 25")
ap.add_argument("-d", "--model", type=str, default="cnn", help="face encoding method to use: either `hog` or `cnn`")
args = vars(ap.parse_args())

# Git gud images
print("[INFO] quantifying faces...")
imagePaths = list(paths.list_images(args["dataset"]))

# Encode faces, more jitters is more accurate but takes longer, eg 10 is 10x longer
def encoder(rgb, box):
    encodings = face_recognition.face_encodings(rgb, box, num_jitters=100, model="large")
    for encoding in encodings:
        # add encoding to arrays for saving
        knownEncodings.append(encoding)
# Arrays are fun



data = pickle.loads(open(args["output"], "rb").read())

knownEncodings = data["encodings"]
knownNames = data["names"]
rgbar = []
names = []
frames = 0
processes = []

# Loop over the image paths
for (i, imagePath) in enumerate(imagePaths):
    #if i < len(knownNames) -1:
    #    continue
    # Extract the person name from the image path
    name = imagePath.split(os.path.sep)[-2]
    knownNames.append(name)
    # Image loads all BGR but we want RGB
    image = cv2.imread(imagePath)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Add rgb to an array for batching later
    rgbar.append(rgb)
    # Start of batch processing
    if len(rgbar) == args["size"]:
        #This is the batch face detection part, no recognition here. Upsampling can require large amounts of vram
        # If you are getting out of memory errors its probably because the batch size or your individual dataset images are too large
        boxes = face_recognition.batch_face_locations(rgbar, number_of_times_to_upsample=0, batch_size=args["size"])
        # Define max threads equal to batch size. Multithreading is needed for batching as face_encodings doesn't have a batch version in the face_recognition library
        with concurrent.futures.ThreadPoolExecutor(max_workers=args["size"]) as executor:
            for rgb, box in zip(rgbar, boxes):
                # Add threads to processes for printout later and start the threads
                processes.append(executor.submit(encoder, rgb, box))
            # Just for printout
            for task in concurrent.futures.as_completed(processes):
                print("[INFO] processing image {}/{}".format(frames + 1, len(imagePaths)))
                frames += 1
            executor.shutdown()
        # Clearing arrays next batch
        rgbar = []
        processes = []
    os.remove(imagePath)
# dump the facial encodings + names to disk
print("[INFO] serializing encodings...")
data = {"encodings": knownEncodings, "names": knownNames}
f = open(args["output"], "wb")
f.write(pickle.dumps(data))
f.close()


