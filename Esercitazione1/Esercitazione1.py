import cv2 as cv
import numpy as np

im = cv.imread("Esercitazione1/test_image.jpg")
#print(im)
print("data type: ", type(im))
print("image size: ", im.shape)

#RIDURRE DIMENSIONE IMMAGINE
im = cv.resize(im, None, fx=1/3, fy=1/3)
print("image size: ", im.shape)


#VISUALIZZA TUTTI I CANALI
for k in range(3):
    cv.imshow("test_image.jpg", im[:,:,k])
    #cv.imshow("prova", np.zeros((1080,1920)))
    cv.waitKey(0)
    cv.destroyAllWindows()

#PRIMO QUARTO TUTTO NERO
im_copy = im.copy()
height, width, channel = im.shape
im_copy[:height//4 , :width//4 , :] = 0
cv.imshow("provacopia.jpg", im_copy)
cv.waitKey(0)
cv.destroyAllWindows()

#CANALE ROSSO TUTTO A 255
im_copy = im.copy()
im_copy[:,:,2] = 255
cv.imshow("provacopia.jpg", im_copy)
cv.waitKey(0)
cv.destroyAllWindows()

#CANALE BLU x2.3
im_copy = im.copy()
#im_copy_blue_channel = im_copy[:,:,0]
im_copy[:,:,0] = np.clip(im_copy[:,:,0] * 2.3, 0, 255).astype(np.uint8)
cv.imshow("provacopia.jpg", im_copy)
cv.waitKey(0)
cv.destroyAllWindows()


#NUOVA IMMAGINE Z E VIDEO
Z = np.full(im.shape, 128).astype(np.uint8)
cv.imshow("nuova_im.jpg", Z)
cv.waitKey(0)
cv.destroyAllWindows()

video = []
for t in np.arange(0, 1, 1/200):
    frame = (1-t)*Z + t*im
    video.append(frame.astype(np.uint8))

fourcc = cv.VideoWriter_fourcc(*'avc1')
h,w = frame.shape[1], frame.shape[0]
out = cv.VideoWriter('Esercitazione1/out60.mov', fourcc, 60, (h,w))

for frame in video:
    out.write(frame)

out.release()
