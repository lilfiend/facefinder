# Command to run. Use cnn model for realtime detection with gpu acceleration or hog for cpu detection
# python facefinder.py --video 0 --model cnn  --trained cnn.pickle

# import everything from everywhere.
from imutils.video import VideoStream
import face_recognition
import argparse
import imutils
import pickle
import time
import cv2

# That feel when newly born code handles arguments better than you do.
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True, help = "CNN or HOG for facial detection")
ap.add_argument("-e", "--trained", required=True, help= "path to pickled trained facial encodings")
ap.add_argument("-v", "--video", required=True, help = "video stream url, ie http://192.168.1.2:8081/video.cgi or 0 for usb webcam")
args = vars(ap.parse_args())

# Load trained faces
print("[INFO] loading trained faces...")
data = pickle.loads(open(args["trained"], "rb").read())

# Start video stream
print("[INFO] starting video stream...")

if args["video"] == "0":
	vs = VideoStream(src=0).start()
	#this fixes a very specifc bug on Microsoft's lifecam cinema webcam. Two guesses who owns one ;)
	time.sleep(2.0)
else:
	vs = VideoStream(src=args["video"]).start()

# While video stream does its thing, do your thing
while True:

	# Video stream? Nah can't use that, how about an image? Yeah... that'll do just fine...
	frame = vs.read()

	#Comment out this line if you want to use the full resolution of your camera
	#Don't comment out this line unless you have all the vram and a ballin' GPU
	frame = imutils.resize(frame, width=400)

	# Convert input frame from BGR to RGB (for face recognition)
	rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

	#This is the face detection part, using CNN/HOG for the face detection is a bit intensive... but results are great
	boxes = face_recognition.face_locations(rgb, model=args["model"])
	# compute the facial embeddings for each face bounding box
	encodings = face_recognition.face_encodings(rgb, boxes)
	names = []

	# loop over the trained facial embeddings
	for encoding in encodings:

		# This is where the face comparison happens, the important thing that happens here is you can change the comparison tolerence (lower is stricter).
		# Undefined it's 0.6, which will provide a lot of false positives. I've found that 0.35 - 0.4 is about as strict as you want to get.
		matches = face_recognition.compare_faces(data["encodings"],
			encoding,0.4)
		name = "STRANGER" #DANGER

		# Check to see if we have found a match
		if True in matches:
			# Find the indexes of matched faces then count how many times it matched. #dictionarypride
			matchedIdxs = [i for (i, b) in enumerate(matches) if b]
			counts = {}

			# Get loopy with matched indexes for each recognized face. Keep count
			for i in matchedIdxs:
				name = data["names"][i]
				counts[name] = counts.get(name, 0) + 1

			# Fuck electoral colleges, every vote matters here, winner is the recognized face. Tie breaker defaults to who was first, no cutting.
			name = max(counts, key=counts.get)

		# Name worked hard, it gets to move to the fancy new array development
		names.append(name)

	# Remember how we detected faces? And we recognized faces? Well how about we draw that shit out for you.
	for ((top, right, bottom, left), name) in zip(boxes, names):
		cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
		y = top - 15 if top - 15 > 15 else top + 15
		cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
			0.75, (0, 255, 0), 2)

	# It's like my daddy always said "To see the fruits of this scripts labor one must cv2.imshow"
	# but seriously if you don't have a gui this is why shit isn't working
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# Q is for quitters who don't live CTRL + C life
	if key == ord("q"):
		break

	#Small note here, if you want a live read of who is in frame checkout that names array

# Cleanup, cleanup, everybody cleanup
cv2.destroyAllWindows()
vs.stop()
