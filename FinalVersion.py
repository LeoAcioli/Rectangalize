import cv2
import numpy as np #math

'''#
In this first block I just let the user pick which picture they want to use for the program.
I also warn them that the rectangles drawn must not be so small, otherwise the rectangle detection
might just not work as it is intended to.
#'''
nameImage=input(f'\nPlease type the name of the image (must be in jpg format): ')
print(f'Your new image will be saved as {nameImage}Detected.jpg and detected dimensions will be displayed on screen!')
print('''
Make sure your drawn rectangles are not too close to the borders of the image or are too small!\n''')

img = cv2.pyrDown(cv2.imread(f"C:/Users/leona/Desktop/{nameImage}.jpg", cv2.IMREAD_UNCHANGED))
# threshold image, first parameter is image now in gray for easier detection. The next is the threshold,
# which is set to 127, the next is the color white. The last parameter is the type of threshold.
ret, threshed_img = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 127, 255, cv2.THRESH_BINARY)
# find contours and get the external one. Arguments going from left to right:
# Source image, contour retrieval mode, contour approximation method
contours, hier = cv2.findContours(threshed_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# with each contour, draw bounding rectangle in green and a min Area Rectangle in red 
# to improve the approximation of detection
for c in contours:
    x, y, w, h = cv2.boundingRect(c) #gets bounding rectangle
    #arguments for cv2.rectangle(the picture, (x detected,y detected), (the opposite corner of the rectangle),
    # color, and thickness)
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 4)
    if h>100 and w>100 and x!=0 and y!=0: 
        print('Wow! A new rectangle was found!') 
        print('Height components: y coordinate:',y,',h coordinate:',h)
        print('Width components: x coordinate:',x,',w coordinate:',w) 
        print('in the format (x,y,width,height) : (',x,',',y,',',w,',',h,')')
        print('\n')
        '''#
        Due to imperfections on the detection, I assumed no rectangles would be drawn too small 
        I didn't let x or y coordinates be 0 because it is detecting the contour of the image itself as a rectangle.
        So the rest of this for loop will only work if the rectangles found meeet this criteria
        #'''
    # get the min area rect
    rect = cv2.minAreaRect(c)
    box = cv2.boxPoints(rect)
    # Convert all coordinates that are floats to int
    box = np.int0(box)
    # Draw a red rectangle, (0,0,255) being red
    cv2.drawContours(img, [box], 0, (0, 0, 255))

#Get number of elements in the list
print(len(contours))
cv2.drawContours(img, contours, -1, (255, 255, 0), 1)

'''#
In this last section I output the image to the user and save it in the format I told them 
the program said it would.
#'''
cv2.imshow("contours.jpg", img)
cv2.imwrite(f"{nameImage}Detected.jpg", img)