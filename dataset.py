# Command to run. Use cnn model for realtime detection with gpu acceleration or hog for cpu detection
# Note you could just take pictures of faces and dump them into dataset folder, the only reason we do face detection is so you can see what it is going to train against.
# python dataset.py --video 0 --model cnn -- output dataset/yourname

# import the necessary packages
from imutils.video import VideoStream
import face_recognition
import argparse
import imutils
import time
import cv2
import os

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True, help = "CNN or HOG for facial detection")
ap.add_argument("-o", "--output", required=True, help="path to output directory")
ap.add_argument("-v", "--video", required=True, help = "video stream url, ie http://192.168.1.2:8081/video.cgi or 0 for usb webcam")
args = vars(ap.parse_args())


# Start video stream
print("[INFO] starting video stream...")

if args["video"] == "0":
	vs = VideoStream(src=0).start()
	#this fixes a very specifc bug on Microsoft's lifecam cinema webcam. Two guesses who owns one ;)
	time.sleep(2.0)
else:
	vs = VideoStream(src=args["video"]).start()

#Start that tally of how many images you've saved of your face
total = 0

# While video stream does its thing, do your thing
while True:

	# Video stream? Nah can't use that, how about an image? Yeah... that'll do just fine...
	frame = vs.read()

	# Backups are important.... But really you want to save the full size image and not the small one....
	orig = frame.copy()

	#Comment out this line if you want to preview the full resolution of your camera
	#Don't comment out this line unless you have all the vram and a ballin' GPU
	frame = imutils.resize(frame, width=400)

	# Convert input frame from BGR to RGB (for face recognition)
	rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

	#This is the face detection part, using CNN/HOG for the face detection is a bit intensive... but results are great
	boxes = face_recognition.face_locations(rgb, model=args["model"])

	# Draw a box around that face
	for (top, right, bottom, left) in boxes:
		cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

	# It's like my daddy always said "To see the fruits of this scripts labor one must cv2.imshow"
	# but seriously if you don't have a gui this is why shit isn't working
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(50) & 0xFF
 
	# Save non-resized image to disk, now with **COUNTING**
	if key == ord("s"):
		p = os.path.sep.join([args["output"], "{}.png".format(
		str(total).zfill(5))])
		cv2.imwrite(p, orig)
		total += 1

	# Q is for quitters who don't live CTRL + C life
	elif key == ord("q"):
		break

# Cleanup, cleanup, everybody cleanup
print("[INFO] {} pictures of your face were saved".format(total))
cv2.destroyAllWindows()
vs.stop()
