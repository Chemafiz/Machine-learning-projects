import matplotlib.pyplot as plt
import cv2
import pickle
import os
import subprocess

#open the file with paint
subprocess.Popen("%s %s" % (r"mspaint.exe", r"image.png"))
if input():
    pass
#loading model
loaded_model = pickle.load(open('model.pickle', 'rb'))

#loading image
image = cv2.imread("image.png", 0)
down_points = (28, 28)
image_resized = cv2.resize(image, down_points, interpolation=cv2.INTER_NEAREST)
image_final = abs(image_resized - [255])

#predict
image_vector = image_final.reshape(1, -1)
digit = loaded_model.predict(image_vector)
print(f"The written digit was predicted to be --> {digit[0]}")

