# Aim set up python openCV script to measure object size from image


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

# Measuring the size of objects with computer vision
# Now that we understand the “pixels per metric” ratio, we can implement the Python driver script used to measure the size of objects in an image.
# 
# Open up a new file, name it object_size.py , and insert the following code:

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

def midpoint(ptA, ptB):
	return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)


