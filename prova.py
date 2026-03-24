import cv2
import numpy
import time

theta = numpy.deg2rad(16.4)
T = numpy.array([
    [numpy.cos(theta), -numpy.sin(theta), 0],
    [numpy.sin(theta),  numpy.cos(theta), 0],
    [0,0,1]
], dtype=float)

print(T[:2,:])

im = cv2.imread('Esercitazione1/test_image.jpg')

r,c = im.shape[0],im.shape[1]

M = cv2.getRotationMatrix2D( (c/2 , r/2), angle=-16.4 , scale=.3)
#print(M)
out = cv2.warpAffine(im, M, (c,r), flags=cv2.INTER_LINEAR)

cv2.imshow("provacopia.jpg", out)
cv2.waitKey(0)
cv2.destroyAllWindows()
