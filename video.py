# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML
import cv2
import matplotlib.pyplot as plt

def process_image(image):
   cv2.imshow('frame',image)
 #  plt.imshow(image)
 #  plt.show()
   return image


white_output = 'test_videos_output/solidWhiteRight.mp4'


clip1 = VideoFileClip("project_video.mp4")
white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
white_clip.write_videofile(white_output, audio=False)

HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(white_output))


