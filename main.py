import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

png = cv.imread('ball.png', )
hsv = cv.cvtColor(png, cv.COLOR_BGR2HSV)


# lower boundary RED color range values; Hue (0 - 10)
lower1 = np.array([0, 100, 20])
upper1 = np.array([10, 255, 255])

# upper boundary RED color range values; Hue (160 - 180)
lower2 = np.array([160,100,20])
upper2 = np.array([179,255,255])

# Threshold the HSV image to get only red colors
mask1 = cv.inRange(hsv, lower1, upper1)
mask2 = cv.inRange(hsv, lower2, upper2)
mask = mask2+mask1


# drawing the mask
plt.imshow(mask), plt.show()

kernel = np.ones((5, 5), np.uint8)
mask_without_noise = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
mask_closed = cv.dilate(mask_without_noise,kernel,iterations = 6)

plt.imshow(mask_without_noise), plt.show()

#center of mass
ret, thresh = cv.threshold(mask_closed, 127, 255, 0)
contours, hierarchy = cv.findContours(thresh, 1, 2)
cnt = contours[0]
M = cv.moments(cnt)
cx = int(M['m10'] / M['m00'])
cy = int(M['m01'] / M['m00'])
cv.CHAIN_APPROX_SIMPLE
mask_closed = cv.drawContours(mask_closed,cnt,-1, (100, 100, 20), 2)
mask_closed = cv.circle(mask_closed, (cx, cy), 7, (100, 100, 20), -1)
mask_closed = cv.putText(mask_closed,"srodek", (cx - 20, cy - 20), cv.FONT_HERSHEY_SIMPLEX, 3, (100, 100, 20), 2)

plt.imshow(mask_closed),plt.show()
print(cx,cy)