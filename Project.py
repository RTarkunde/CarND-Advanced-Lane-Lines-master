###############################################################################
# Advance Lane detection program December 2018
# Udacity KPIT scholarship Self riving scholarship Term 1 Project 2
# Author: Rahul Tarkunde
###############################################################################
#Critical imports
import numpy as np
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip
import glob

###############################################################################
#Some globals for this project
HORIZONTAL_CORNERS = 9
VERTICAL_CORNERS   = 6
nx = HORIZONTAL_CORNERS
offset = 100

font         = cv2.FONT_HERSHEY_SIMPLEX
text_pos1    = (10,500)
text_pos2    = (10,550)
text_pos3    = (10,600)
font_scale   = 1
font_color   = (255,255,255)
line_type    = 2 #cv2.LINE_AA

debug_font         = cv2.FONT_HERSHEY_SIMPLEX
debug_text_pos    = (10,400)
debug_font_color   = (255,0, 0)

image_width  = 1280
image_height = 720
 # Define conversions in x and y from pixels space to meters
ym_per_pix   = 30 /720 # meters per pixel in y dimension
xm_per_pix   = 3.7/700 # meters per pixel in x dimension
vehicle_center = image_width*xm_per_pix/2

#################################
# Sanity check limits
# Curvature limits in meters
high_curvature_limit  = 15000
low_curvature_limit   =   300
lane_width_high_limit =   5.0
lane_width_low_limit  =   3.0
################################################################################
# Class for Lines
# Define a class to receive the characteristics of each line detection
# We aren't using all fields but this has all the recommended fields
class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.recent_xfitted = []
        #average x values of the fitted line over the last n iterations
        self.bestx = None
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        #radius of curvature of the line in some units
        #self.radius_of_curvature = None
        self.curvature = None
        #distance in meters of vehicle center from the line
        self.line_base_pos = None
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float')
        #x values for detected line pixels
        self.allx = None
        #y values for detected line pixels
        self.ally = None
        # Approximate lane width in unwrapped meters at 100 pixels depth
        self.linediff = 0

################################################################################
# Test image converter. COnverts the sample test images from test_images folders
# and puts then in output_images folder. Does not return anything
################################################################################
def convert_testimages(mtx, dist):
    image_files = glob.glob('test_images/*.jpg')
    for idx, fname in enumerate(image_files):
        test_image = cv2.imread(fname)
        img_size = (test_image.shape[1], test_image.shape[0])
        dst = cv2.undistort(test_image, mtx, dist, None, mtx)
        out_file = 'output_images/' + fname
        #print(out_file)
        cv2.imwrite(out_file, dst)
        return

################################################################################
# Read and caliberate the camera with images
################################################################################
def caliberate_camera():
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.
    img_size  = []
    objp = np.zeros((VERTICAL_CORNERS*HORIZONTAL_CORNERS, 3), np.float32)
    objp[:,:2] = np.mgrid[0:HORIZONTAL_CORNERS, 0:VERTICAL_CORNERS].T.reshape(-1, 2)

    # Make a list of calibration images
    calib_images = glob.glob('camera_cal/calibration*.jpg')
    # Step through the list and search for chessboard corners
    for idx, fname in enumerate(calib_images):
        img  = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # assume that the sizes are same and get overwritten
        img_size = (img.shape[1], img.shape[0])
        # Find the chessboard corners
        ret, chessbrd_corners = cv2.findChessboardCorners(gray, (HORIZONTAL_CORNERS, VERTICAL_CORNERS), None)
        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(chessbrd_corners)
            out_file = 'output_images/' + fname
             # Draw and display the corners
            cv2.drawChessboardCorners(img, (HORIZONTAL_CORNERS,VERTICAL_CORNERS), chessbrd_corners, ret)
            cv2.imwrite(out_file, img)
        else:
            print('Cannot find chessboard corners for image '+fname)

    # Do camera calibration given object points and image points
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)

    # Convert the images
    convert_testimages(mtx, dist)
    return mtx, dist, chessbrd_corners

################################################################################
# Undistort the image
################################################################################
def undistort_image(image, mtx, dist):
    return cv2.undistort(image, mtx, dist, None, mtx)

################################################################################
# Color transorm, gradients and create a thersholded binary image
################################################################################
def create_thresholded_image(img, s_thresh=(170, 255), sx_thresh=(20, 100)):
    img = np.copy(img)
    ############################################################################
    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]
    sobelx = cv2.Sobel(s_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
    ############################################################################
    # Absolute x derivative to accentuate lines away from horizontal
    abs_sobelx = np.absolute(sobelx)
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1
    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    # Stack each channel
    color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary)) * 255
    return color_binary

################################################################################
# Apply perspective. It moves the camera to top for top view
################################################################################
def apply_perspective(img, corners):
    img_size = (img.shape[1], img.shape[0])
    y_offset = 40
    x_correct = 50
    # cv2.imwrite('./output_images/un_wrapped_perspective.jpg', img)
    # src = np.float32([corners[0], corners[nx-1], corners[-1], corners[-nx]])
    # src = np.float32([[291, 720-y_offset], [1103, 720-y_offset],[737,467],[588,467]])
    # dst = np.float32([[291, 720-y_offset], [1103, 720-y_offset],[1103,0], [291+x_correct,0]])
    src = np.float32([[588,467], [737,467], [1103, 720-y_offset], [291, 720-y_offset]])
    dst = np.float32([[291+x_correct,0], [1103,0], [1103, 720-y_offset], [291, 720-y_offset]])
    M   = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, img_size)
    # cv2.imwrite('./output_images/perspective.jpg', warped)
    return warped, M

################################################################################
# Find lane pixels by taking a histogram
################################################################################
def find_lane_pixels_histogram(binary_warped, left_line, right_line):
    #print('Entry:In find_lane_pixels')
    # Choose the number of sliding windows We take 10 instead of common 9
    nwindows =  10
    # Set the width of the windows +/- margin
    margin   = 100
    # Set minimum number of pixels found to recenter window
    minpix   =  50
    left_lane_inds  = []
    right_lane_inds = []

    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint    = np.int(histogram.shape[0]//2)
    leftx_base  = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0]//nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero  = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated later for each window in nwindows
    leftx_current  = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low  = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] -  window*window_height
        win_xleft_low   = leftx_current  - margin
        win_xleft_high  = leftx_current  + margin
        win_xright_low  = rightx_current - margin
        win_xright_high = rightx_current + margin

        # This is not required for final sub. Draw the windows
        # on the visualization image
        #cv2.rectangle(out_img,(win_xleft_low,win_y_low),
        #(win_xleft_high,win_y_high), (255,0,0), 2)
        #cv2.rectangle(out_img,(win_xright_low,win_y_low),
        #(win_xright_high,win_y_high),(0,255,0), 2)

        good_left_inds  = ((nonzeroy >= win_y_low)      &
                           (nonzeroy <  win_y_high)     &
                           (nonzerox >= win_xleft_low)  &
                           (nonzerox <  win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low)      &
                           (nonzeroy <  win_y_high)     &
                           (nonzerox >= win_xright_low) &
                           (nonzerox <  win_xright_high)).nonzero()[0]

        # Append these indices to the lists
        left_lane_inds.append (good_left_inds)
        right_lane_inds.append(good_right_inds)
        #recenter
        if len(good_left_inds)  > minpix:
            leftx_current  = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds  = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        print('Lane processing threw exception')
        pass

    # Extract left and right line pixel positions
    leftx  = nonzerox[left_lane_inds]
    lefty  = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    #print('Exit: find_lane_pixels')
    return leftx, lefty, rightx, righty


################################################################################
# Find lane pixels by narrow search
################################################################################
def find_lane_pixels_search_around_poly(left_line, right_line, binary_warped):
    # Choose the width of the margin around the previous polynomial to search
    margin = 200

    # Grab activated pixels
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Search based on activated x-values within the +/- margin of polynomial function
    # Though the fit says current its actually last frame's.
    left_lane_inds = ((nonzerox > (left_line.current_fit[0]*(nonzeroy**2) + left_line.current_fit[1]*nonzeroy +
                    left_line.current_fit[2] - margin)) & (nonzerox < (left_line.current_fit[0]*(nonzeroy**2) +
                    left_line.current_fit[1]*nonzeroy + left_line.current_fit[2] + margin)))
    right_lane_inds = ((nonzerox > (right_line.current_fit[0]*(nonzeroy**2) + right_line.current_fit[1]*nonzeroy +
                    right_line.current_fit[2] - margin)) & (nonzerox < (right_line.current_fit[0]*(nonzeroy**2) +
                    right_line.current_fit[1]*nonzeroy + right_line.current_fit[2] + margin)))

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    return leftx, lefty, rightx, righty

################################################################################
# Find lane pixels
################################################################################
def lane_line_pixels(left_line, right_line, img):

    binary_warped = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
# Find our lane pixels first
    if left_line.detected == False or right_line.detected == False:  # First frame
        leftx, lefty, rightx, righty = find_lane_pixels_histogram(binary_warped, left_line, right_line)
        left_line.detected = True
        right_line.detected = True
    else: # Subsequent frames
        leftx, lefty, rightx, righty = find_lane_pixels_search_around_poly(left_line, right_line, binary_warped)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    #  print(ploty)
    left_fitx  = []
    right_fitx = []
    left_fit   = np.polyfit(lefty,   leftx, 2)
    right_fit  = np.polyfit(righty, rightx, 2)

    try:
        left_fitx_temp  = left_fit[0]*ploty**2  + left_fit[1]*ploty  + left_fit[2]
        right_fitx_temp = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    except TypeError:
        print('The function failed to fit a line!')

    if left_line.detected == True:
        old_left_curvature  = left_line.curvature
        old_right_curvature = right_line.curvature

    measure_curvature_real(left_line, right_line, left_fitx_temp, right_fitx_temp, ploty)
    lane_width = (right_fitx_temp[100] - left_fitx_temp[100])*xm_per_pix
    # Sanity checks: Check for width of the lane approx 3.0 and absurd curvatures
    if lane_width > lane_width_low_limit and lane_width < lane_width_high_limit and \
         left_line.curvature  < high_curvature_limit and \
         left_line.curvature  > low_curvature_limit  and \
         right_line.curvature < high_curvature_limit and \
         right_line.curvature > low_curvature_limit:

        left_fitx  = left_fitx_temp
        right_fitx = right_fitx_temp
        left_line.linediff = lane_width# ((right_fitx[100] - left_fitx[100])*xm_per_pix)
         # Update the values for next search for search around poly
        left_line.current_fit    = left_fit
        right_line.current_fit   = right_fit
        left_line.line_base_pos  = left_fitx[0]*xm_per_pix
        right_line.line_base_pos = right_fitx[0]*xm_per_pix
    else: # take the old values. Some issue with the curve
        left_fit  = left_line.current_fit
        right_fit = right_line.current_fit
        # Restore curvature of previous image
        left_line.curvature  = old_left_curvature
        right_line.curvature = old_right_curvature
        # Discard the temp values and take the old ones.
        left_fitx  = left_line.current_fit [0]*ploty**2 + left_line.current_fit [1]*ploty + left_line.current_fit [2]
        right_fitx = right_line.current_fit[0]*ploty**2 + right_line.current_fit[1]*ploty + right_line.current_fit[2]

    plt.plot(right_fitx, ploty, color='yellow')
    plt.plot(left_fitx,  ploty, color='yellow')

    return left_fitx, right_fitx, ploty, left_fit, right_fit

################################################################################
# unWrapPerspective: after the polu curve processing undo the view to normal
################################################################################
def unWrapPerspective(binary_warped, image, Minv, left_fitx, right_fitx, ploty):
    # Image contains original unwarped image
    # binary_warped contains top perspective binary image
    warp_zero   = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp  = np.copy(warp_zero)
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left  = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts       = np.hstack((pts_left, pts_right))
    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    # Warp the blank back to original image space using inverse perspective matrix (Mbinary_warpedinv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (color_warp.shape[1], color_warp.shape[0]))
    # Combine the result with the original image
    result  = cv2.addWeighted(image, 1, newwarp, 0.3, 0)
    return result

################################################################################
#polyfit_data:Helper function for measure_curvature_real.Deals in meters !pixels
################################################################################
def polyfit_data(leftx, rightx, ploty):

    leftx  = leftx [::-1]  # Reverse to match top-to-bottom in y
    rightx = rightx[::-1]  # Reverse to match top-to-bottom in y

    left_fit_cr  = np.polyfit(ploty*ym_per_pix, leftx*xm_per_pix,  2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, rightx*xm_per_pix, 2)

    return ploty*ym_per_pix, left_fit_cr, right_fit_cr

################################################################################
# Calculates the curvature of polynomial functions.
################################################################################
def measure_curvature_real(left_line, right_line, leftx, rightx, ploty):

    ploty, left_fit_cr, right_fit_cr = polyfit_data(leftx, rightx, ploty)
    y_eval = np.max(ploty)

    #Implement the calculation of R_curve (radius of curvature)
    left_curverad  = ((1 + (left_fit_cr[0]*y_eval  + left_fit_cr[1])**2)**(3/2))/np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (right_fit_cr[0]*y_eval + right_fit_cr[1])**2)**(3/2))/np.absolute(2*right_fit_cr[0])
    left_line.curvature  = left_curverad
    right_line.curvature = right_curverad
    #print ('ROC = ' + str(int(left_curverad)) + ' ' +str(int(right_curverad)))

################################################################################
# A function to print information on images
################################################################################
def text_on_image(image, left_line, right_line):
    left_string  =  'Left curvature:'   + str(int(left_line.curvature))  + ' (m)'
    right_string =  'Right curvature:'  + str(int(right_line.curvature)) + ' (m)'
    lane_string  =   str()
    lane_center = (left_line.line_base_pos + right_line.line_base_pos) /2
    if   vehicle_center - lane_center < 0.0:
        lane_string = 'Car to left:' +  str("{0:.2f}".format(lane_center - vehicle_center)) + '(m)'
    elif vehicle_center - lane_center > 0.0:
        lane_string = 'Car to right:' + str("{0:.2f}".format(vehicle_center - lane_center)) + '(m)'
    else:
        lane_string = "Car in center"
    cv2.putText(image, left_string,  text_pos1, font, font_scale, font_color, line_type)
    cv2.putText(image, right_string, text_pos2, font, font_scale, font_color, line_type)
    cv2.putText(image, lane_string,  text_pos3, font, font_scale, font_color, line_type)

################################################################################
# A function to show images
################################################################################
def show_image(image):
    plt.imshow(image)
    plt.show()

################################################################################
# A function to show images with calling function for debugging
################################################################################
def debug_show_image(func_name, image):
    cv2.putText(image, func_name, debug_text_pos, font, font_scale, debug_font_color, line_type)
    return image
################################################################################
# Top level function for processing image pipeline
################################################################################
image_count =  0
def process_image(img):
    global left_fit, right_fit, image_count
    #print (image_count)
    undistorted_image = undistort_image(img, mtx, dist)
    thersholded_image = create_thresholded_image(undistorted_image)
    #show_image(thersholded_image)
    persp_image, M    = apply_perspective(thersholded_image, corners)
    #show_image(per_image)
    left_fitx, right_fitx, ploty, left_fit, right_fit = lane_line_pixels(left_line, right_line, persp_image)
    ret, Minv  = cv2.invert(M)
    final_processed_image = unWrapPerspective(persp_image, undistorted_image, Minv, left_fitx, right_fitx, ploty)
    #show_image(final_processed_image)
    text_on_image(final_processed_image, left_line, right_line)
    # image_count += 1
    # if image_count == 20:
    # cv2.imwrite('./output_images/final_image_20.jpg', final_processed_image)
    return final_processed_image
    #show_image(final_processed_image)

################################################################################
# Main program
################################################################################
# Global class instances
left_line  = Line ()
right_line = Line ()
input_video_file  = 'project_video.mp4'
processed_video   = 'project_video_out.mp4'
#input_video_file  = 'challenge_video.mp4'
#processed_video   = 'challenge_video_out.mp4'
mtx, dist,corners =  caliberate_camera()
input_video       =  VideoFileClip(input_video_file)
output_video      =  input_video.fl_image(process_image)
output_video.write_videofile(processed_video, audio=False)
"""
images = glob.glob('test_images/test*.jpg')
mtx, dist,corners = caliberate_camera()
for idx, fname in enumerate(images):
    img = cv2.imread(fname)
    outi = process_image(img)
    show_image(outi)
"""
