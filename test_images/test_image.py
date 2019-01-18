import numpy as np
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip
import glob

img  = cv2.imread('test1.jpg')
print(img.shape)
