from keras.models import load_model
import numpy as np
import cv2
from PIL import Image, ImageOps
from picamera2 import Picamera2
import gpiod
import time
from motor import *

# Open the GPIO chip
chip = gpiod.Chip('gpiochip4')

# Open the GPIO line for the button
button_line = chip.get_line(2)
button_line.request(consumer="BUTTON", type=gpiod.LINE_REQ_DIR_IN)

# Define the number of rows and columns for the grid
rows = 3
cols = 6

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the keras model
model = load_model("keras_model.h5", compile=False)

# Load the labels
class_names = open("labels.txt", "r").readlines()

# Create the array of the right shape to feed into the keras model
# The 'length' or number of images you can put into the array is
# determined by the first position in the shape tuple, in this case 1
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# function processes an image which allows for easy detection of contours 
def processImage(frame):
    '''
    
    '''
    frame_cvt = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_blur = cv2.GaussianBlur(frame_cvt, (5, 5), 0)
    frame_edge = cv2.Canny(frame_blur, 30, 80)
    kernel = np.ones((5, 5), np.uint8)
    frame_dilated = cv2.dilate(frame_edge, kernel, iterations=1)
    return frame_dilated

# crops outer border from image
def cropImage(original_image):
    # Get image dimensions
    height, width, _ = original_image.shape

    # calculate coordinates for crop
    start_y = round(height * 0.18436)
    end_y = round(height * 0.93855)
    start_x = round(width * 0.06699)
    end_x = round(width * 0.93301)

    # perform crop
    original_image = original_image[start_y:end_y, start_x:end_x]
    return original_image

# splits image into 18 sub images and saves
def splitImage(original_image):
    # Get image dimensions
    height, width, _ = original_image.shape

    # Calculate the size of each sub-image
    sub_image_height = height // rows
    sub_image_width = width // cols

    # Loop through each row and column to crop and save the sub-images
    for i in range(rows):
        for j in range(cols):
            # Calculate the starting and ending coordinates for the current sub-image
            start_y = i * sub_image_height
            end_y = (i + 1) * sub_image_height
            start_x = j * sub_image_width
            end_x = (j + 1) * sub_image_width

            # Crop the sub-image from the original image
            sub_image = original_image[start_y:end_y, start_x:end_x]

            # Save or process the sub-image as needed
            sub_image_path = f'sub_image_{i}_{j}.jpg'
            cv2.imwrite(sub_image_path, sub_image)
    
#function will execute any given command
def executeCommand(command):
    if "forward" in command:
        moveForward()
    elif "backward" in command:
        moveBackward()
    elif "rotate" in command:
        rotate90d_cw()
    
# function uses tensorflow ai for determining content of images taken
def applyTF():
    prediction_out = "prediction.txt"

    # Loop through each row and column to interpret every image
    for i in range(rows):
        for j in range(cols):
            image_path = f'sub_image_{i}_{j}.jpg'
            image = Image.open(image_path).convert("RGB")

            # resizing the image to be at least 224x224 and then cropping from the center
            size = (224, 224)
            image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

            # turn the image into a numpy array
            image_array = np.asarray(image)

            # Normalize the image
            normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

            # Load the image into the array
            data[0] = normalized_image_array

            # Predicts the model
            prediction = model.predict(data)

            index = np.argmax(prediction)
            class_name = class_names[index]
            confidence_score = prediction[0][index]

            # Print prediction and confidence score
            with open(prediction_out, 'a') as file:
                file.write(f"Class: {class_name[2:]}")
                file.write(f"Confidence Score: {confidence_score}\n\n")
                executeCommand(f"Class: {class_name[2:]}")


camera = Picamera2()

config = camera.create_preview_configuration(raw={'format': 'SRGGB10_CSI2P', 'size': (3280, 2464)})
camera.configure(config)

camera.start()

print("starting program...")
time.sleep(1)

'''
loop processes a frame to find contours and displays the largest found contour. pressing s will take a photo and process any found combinations of blocks using the Keras model.
'''
while True:

    frame = np.array(camera.capture_image("main"))
    frame_edge = processImage(frame)
    contours, h = cv2.findContours(frame_edge, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        max_contour = max(contours, key=cv2.contourArea)
        epsilon = 0.02 * cv2.arcLength(max_contour, True)
        approx = cv2.approxPolyDP(max_contour, epsilon, True)
        x, y, w, h = cv2.boundingRect(approx)
        if cv2.contourArea(approx) > 5000:
            # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
            object_only = frame[y:y+h, x:x+w]
            cv2.imshow('My Smart Scanner', object_only)
            # Read the value of the button
            value = button_line.get_value()
            if cv2.waitKey(1) == ord('s') or value == 0:
                #splitImage(cropImage(object_only))
                splitImage(object_only)
                applyTF()
