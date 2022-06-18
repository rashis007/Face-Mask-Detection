# This is the program to detect a face from a live Video Stream and predict the presence of a face mask. 
# First, we should import the necessary packages.
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2
import os

def detect_and_predict_mask(frame, faceNet, maskNet):
	# Get the dimensions of the image frame and create a blob using it.
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
		(104.0, 177.0, 123.0))

	# Pass the blob through the neural network and obtain the face detections.
	faceNet.setInput(blob)
	detections = faceNet.forward()
	print(detections.shape)

	# Initialize the list of faces, their corresponding locations and the list of predictions 
	# from the mask detection neural network.
	faces = []
	locs = []
	preds = []

	# Loop over the detections.
	for i in range(0, detections.shape[2]):

		confidence = detections[0, 0, i, 2]

		# Filter out the weak detections by ensuring that the confidence of the detection has a greater 
		# value than the minimum confidence, then calculate the coordinates of the bounding box.
		if confidence > 0.5:
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# Also, ensure the bounding boxes fall within the dimensions of
			# the frame.
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			# Extract the facial Regions of Interest from the frame and convert it from 
			# BGR channel to RGB channel ordering, resize it to 224x224 pixels and 
			# preprocess the input.
			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)

			# Append the faces and bounding boxes to their respective
			# lists.
			faces.append(face)
			locs.append((startX, startY, endX, endY))

	# Ensure that a prediction is made if the value of faces is greater than zero.
	# Also, for faster results, all predictions are made on batches instead of 
	# individual faces. 
	if len(faces) > 0:
		faces = np.array(faces, dtype="float32")
		preds = maskNet.predict(faces, batch_size=32)

	# Return a 2-tuple of the face locations and their corresponding
	# predictions in the neural network.
	return (locs, preds)

# Load the serialized face detector model from the hard disk.
prototxtPath = r"face_detector\deploy.prototxt"
weightsPath = r"face_detector\res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# Load the trained face mask detector model from the hard disk.
maskNet = load_model("mask_detector.model")

# Start the video stream.
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()

while True:
	# Get the frame from the video stream and set it a maximum width of 1050 pixels.
	frame = vs.read()
	frame = imutils.resize(frame, width=1050)

	# Detect faces in the video stream frame and determine if they are wearing a
	# face mask or not using the model.
	(locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

	# Loop over the detected face locations and their corresponding
	# predictions in the neural network and unpack the bounding
	# box and label.
	for (box, pred) in zip(locs, preds):
		(startX, startY, endX, endY) = box
		(mask, withoutMask) = pred

		# Set a color for each classification label and bounding box rectangle.
		label = "Mask" if mask > withoutMask else "No Mask"
		color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

		# Also, include the detection probability next to the classification label.
		label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

		# Put the label and bounding box rectangle on the output
		# frame in the video stream.
		cv2.putText(frame, label, (startX, startY - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
		cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

	# Display the frame in the output,
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# On pressing the key 'q', exit from the loop.
	if key == ord("q"):
		break

# Close all the program windows and stop the video stream.
cv2.destroyAllWindows()
vs.stop()
