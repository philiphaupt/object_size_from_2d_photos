# Aim set up python openCV script to measure object size from image
#reticulate::repl_python()
# Clear pyton vars
#import sys
#-------------------
# List all variables in the global namespace
# all_vars = list(globals().keys())

# # Exclude modules and variables starting with an underscore
# exclude = set(sys.modules.keys()) | set(globals().keys()) | {'__builtins__', '__doc__', '__name__', '__loader__', '__spec__', '__package__'}
# vars_to_clear = [var for var in all_vars if not var.startswith('_') and var not in exclude]
# 
# # Clear all variables in the global namespace
# for var in vars_to_clear:
#     del globals()[var]
#-----------------

# Click here to download the source code to this post
# Last updated on July 8, 2021.
# 
# size_of_objects_example_02
# Measuring the size of an object (or objects) in an image has been a heavily requested tutorial on the PyImageSearch blog for some time now — and it feels great to get this post online and share it with you.
# 
# Today’s post is the second in a three part series on measuring the size of objects in an image and computing the distances between them.
# 
# Last week, we learned an important technique: how reliably order a set of rotated bounding box coordinates in a top-left, top-right, bottom-right, and bottom-left arrangement.
# 
# Today we are going to utilize this technique to aid us in computing the size of objects in an image. Be sure to read the entire post to see how it’s done!
# 
# Update July 2021: Added section on how to improve object size measurement accuracy by performing a proper camera calibration with a checkerboard.
# 
# Looking for the source code to this post?
# JUMP RIGHT TO THE DOWNLOADS SECTION 
# Measuring the size of objects in an image with OpenCV
# Measuring the size of objects in an image is similar to computing the distance from our camera to an object — in both cases, we need to define a ratio that measures the number of pixels per a given metric.
# 
# I call this the “pixels per metric” ratio, which I have more formally defined in the following section.
# 
# The “pixels per metric” ratio
# In order to determine the size of an object in an image, we first need to perform a “calibration” (not to be confused with intrinsic/extrinsic calibration) using a reference object. Our reference object should have two important properties:
# 
# Property #1: We should know the dimensions of this object (in terms of width or height) in a measurable unit (such as millimeters, inches, etc.).
# Property #2: We should be able to easily find this reference object in an image, either based on the placement of the object (such as the reference object always being placed in the top-left corner of an image) or via appearances (like being a distinctive color or shape, unique and different from all other objects in the image). In either case, our reference should should be uniquely identifiable in some manner.
# In this example, we’ll be using the United States quarter as our reference object and throughout all examples, ensure it is always the left-most object in our image:

# By guaranteeing the quarter is the left-most object, we can sort our object contours from left-to-right, grab the quarter (which will always be the first contour in the sorted list), and use it to define our pixels_per_metric, which we define as:
# 
# pixels_per_metric = object_width / know_width
# 
# A US quarter has a known_width of 0.955 inches. Now, suppose that our object_width (measured in pixels) is computed be 150 pixels wide (based on its associated bounding box).
# 
# The pixels_per_metric is therefore:
# 
# pixels_per_metric = 150px / 0.955in = 157px
# 
# Thus implying there are approximately 157 pixels per every 0.955 inches in our image. Using this ratio, we can compute the size of objects in an image.
#--------------------------
# In my example image, as follows:
# To automatically count the number of pixels between two points on an image, you can use the cv2.line() function to draw a line between the two points and then use the cv2.countNonZero() function to count the number of non-zero (i.e. white) pixels in the line.

import cv2
import numpy as np

# Load image and convert to grayscale
image = cv2.imread("./images/IMG_001.JPG") # example image
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Define two points to measure distance between
point1 = (337 , 520)
point2 = (1038, 520)

# Draw line between points
cv2.line(image, point1, point2, (0, 0, 255), 2)

# Count non-zero pixels in line
line_mask = np.zeros_like(gray)
cv2.line(line_mask, point1, point2, 255, 1)
num_pixels = cv2.countNonZero(cv2.bitwise_and(gray, line_mask))

# Display image with line and measured distance
cv2.putText(image, '16mm = {} pixels'.format(num_pixels), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
#cv2.putText(image, '16 mm'.format(mm)), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
cv2.imshow('Line Distance', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

#Display the same image but rescale it so that I can see the whole image
# Calculate the new dimensions
scale_percent = 50 # percent of original size
width = int(image.shape[1] * scale_percent / 100)
height = int(image.shape[0] * scale_percent / 100)
dim = (width, height)
# Resize the image
resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
# Display the resized image
cv2.imshow("Image with bounding boxes", resized)
cv2.waitKey(0)
# RESULT:
# The answer for this image is 626 pixels in 1 centimeter- so it is important to have the setup as consistent as possible - that way we can use teh same value throughout. I.e. the height of the tripod above the image.


# Measuring the size of objects with computer vision
# Now that we understand the “pixels per metric” ratio, we can implement the Python driver script used to measure the size of objects in an image.
#TO CHECK: # 16 mm  = 702 pixels (may need to convert to inches for the code below?) which is 1 inch - which is the w paramter below
702/16
# Open up a new file, name it object_size.py , and insert the following code:
#-------------------------
#https://pyimagesearch.com/2016/03/28/measuring-size-of-objects-in-an-image-with-opencv/

# import the necessary packages
from scipy.spatial import distance as dist
from imutils import perspective #pip install --upgrade imutils if not already installed
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2

#Now that we understand the “pixels per metric” ratio, we can implement the Python driver script used to measure the size of objects in an image.

# Lines 2-8 import our required Python packages. We’ll be making heavy use of the imutils package in this example, so if you don’t have it installed, make sure you install it before proceeding:

def midpoint(ptA, ptB):
	return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)



# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to the input image")
ap.add_argument("-w", "--width", type=float, required=True,
	help="width of the left-most object in the image (in mm)")
args = vars(ap.parse_args(["-i", "./images/IMG_001b.JPG", "-w", "43.875"])) # width given in inches from above calculation RESULT
#pixels per mm0
#example below
#args = vars(ap.parse_args(["-i", "C:/Users/Phillip Haupt/Pictures/image_object_size_example.JPG", "-w", "157"]))
# We then parse our command line arguments on Lines 14-19. We require two arguments, --image , 
# which is the path to our input image containing the objects we want to measure, 
# and --width , which is the width (in inches) of our reference object, 
# presumed to be the left-most object in our --image

# load the image, 
#image = cv2.imread(args["C:/Users/Phillip Haupt/Pictures/IMG_4913.JPG"]) #image = cv2.imread(args["C:/Users/Phillip Haupt/Pictures/work photos/20210415_123223.JPG"])#IMG-20201007-WA0000
#image = cv2.imread("C:/Users/Phillip Haupt/Pictures/work photos/20210415_123223.JPG")# can't get the args to work
image = cv2.imread(args["image"])

# convert it to grayscale, and blur it slightly
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
resized = cv2.resize(gray, dim, interpolation = cv2.INTER_AREA)
cv2.imshow("Edged image with bounding boxes", resized)
gray = cv2.GaussianBlur(gray, (7, 7), 0)
resized = cv2.resize(gray, dim, interpolation = cv2.INTER_AREA)
cv2.imshow("Edged image with bounding boxes", resized)

# perform edge detection, then perform a dilation + erosion to
# close gaps in between object edges
edged = cv2.Canny(gray, 5, 10)
edged = cv2.dilate(edged, None, iterations=1)
edged = cv2.erode(edged, None, iterations=1)

resized = cv2.resize(edged, dim, interpolation = cv2.INTER_AREA)
cv2.imshow("Edged image with bounding boxes", resized)

# find contours in the edge map
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
# sort the contours from left-to-right and initialize the
# 'pixels per metric' calibration variable
(cnts, _) = contours.sort_contours(cnts)
pixelsPerMetric = None


# loop over the contours individually
for c in cnts:
	# if the contour is not sufficiently large, ignore it
	if cv2.contourArea(c) < 60000:
		continue
	# compute the rotated bounding box of the contour
	orig = image.copy()
	box = cv2.minAreaRect(c)
	box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
	box = np.array(box, dtype="int")
	# order the points in the contour such that they appear
	# in top-left, top-right, bottom-right, and bottom-left
	# order, then draw the outline of the rotated bounding
	# box
	box = perspective.order_points(box)
	cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)
	# loop over the original points and draw them
	for (x, y) in box:
		cv2.circle(orig, (int(x), int(y)), 5, (0, 0, 255), -1)


  # unpack the ordered bounding box, then compute the midpoint
	# between the top-left and top-right coordinates, followed by
	# the midpoint between bottom-left and bottom-right coordinates
	(tl, tr, br, bl) = box
	(tltrX, tltrY) = midpoint(tl, tr)
	(blbrX, blbrY) = midpoint(bl, br)
	# compute the midpoint between the top-left and top-right points,
	# followed by the midpoint between the top-righ and bottom-right
	(tlblX, tlblY) = midpoint(tl, bl)
	(trbrX, trbrY) = midpoint(tr, br)
	# draw the midpoints on the image
	cv2.circle(orig, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
	cv2.circle(orig, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
	cv2.circle(orig, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
	cv2.circle(orig, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)
	# draw lines between the midpoints
	cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),
		(255, 0, 255), 2)
	cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),
		(255, 0, 255), 2)

	# compute the Euclidean distance between the midpoints
	dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
	dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
	# if the pixels per metric has not been initialized, then
	# compute it as the ratio of pixels to supplied metric
	# (in this case, inches)
	if pixelsPerMetric is None:
		pixelsPerMetric = dB / args["width"]


	# compute the size of the object
	dimA = dA / pixelsPerMetric
	dimB = dB / pixelsPerMetric
	# draw the object sizes on the image
	cv2.putText(orig, "{:.1f}mm".format(dimA),
		(int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,
		0.65, (255, 255, 255), 2)
	cv2.putText(orig, "{:.1f}mm".format(dimB),
		(int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,
		0.65, (255, 255, 255), 2)
	
	
	# show the output image
	#cv2.imshow("Image with bounding boxes", orig)
	#cv2.waitKey(0)

  # Calculate the new dimensions
  scale_percent = 40 # percent of original size
  width = int(image.shape[1] * scale_percent / 100)
  height = int(image.shape[0] * scale_percent / 100)
  dim = (width, height)
  # Resize the image
  resized = cv2.resize(orig, dim, interpolation = cv2.INTER_AREA)
  # Display the resized image
  cv2.imshow("Image with bounding boxes", resized)
  cv2.waitKey(0)


##-- run the code above in a terminal with the line:
$ python object_size.py --image images/image_object_size_example.JPG --width 0.955
$ python object_size.py --image images/IMG_001.JPG --width 43.875
