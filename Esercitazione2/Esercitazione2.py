import cv2 as cv
import numpy as np


refence = cv.imread('Esercitazione2/reference.jpg')
moving = cv.imread('Esercitazione2/moving.jpg')

sift = cv.SIFT_create()

kp1 , desc1 = sift.detectAndCompute(cv.cvtColor(refence,cv.COLOR_BGR2GRAY), None)
kp2 , desc2 = sift.detectAndCompute(cv.cvtColor(moving,cv.COLOR_BGR2GRAY), None)


reference_sift = cv.drawKeypoints(refence, kp1, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
moving_sift = cv.drawKeypoints(moving, kp2, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

bfm = cv.BFMatcher(normType=cv.NORM_L2)

matchs_knn = bfm.knnMatch(desc1, desc2, k=2)

good = []
for m,n in matchs_knn:
    if m.distance < 0.75*n.distance:
        good.append([m])

im3 = cv.drawMatchesKnn(refence,kp1,moving,kp2,good,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

cv.imshow('KNN-match',im3)
cv.waitKey(0)
cv.destroyAllWindows()

from_points = np.float32([kp1[m[0].queryIdx].pt for m in good])
to_points = np.float32([kp2[m[0].trainIdx].pt for m in good])

M , pippo= cv.estimateAffine2D(from_points,to_points,method=cv.RANSAC)
registered = cv.warpAffine(refence, M, (refence.shape[1], refence.shape[0]))

cv.imshow('Registered Image', registered)
cv.waitKey(0)
cv.destroyAllWindows()