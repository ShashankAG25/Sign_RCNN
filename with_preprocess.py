# running the prediction on the pre-processed images

import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from keras import Model
from tensorflow import keras

model = keras.models.load_model("ieeercnn_vgg16_1.h5")

# path = "Test_sets/1.png"
path = "Test_sets1/"
out_path = "test_out_preprocess"
# np_path = "sspreprocess"
ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()


def preprocess(img):
    image = cv2.imread(img)

    result = image.copy()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.array([90, 38, 0])
    upper = np.array([145, 255, 255])
    mask = cv2.inRange(image, lower, upper)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    close = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=2)
    img_erosion = cv2.erode(close, kernel, iterations=2)
    img_dilation = cv2.dilate(img_erosion, kernel, iterations=3)
    result[img_dilation == 0] = (255, 255, 255)
    return result


img = preprocess("Test_sets1/test49.png")
# cv2.namedWindow('result', cv2.WINDOW_NORMAL)
# cv2.imshow("result", img)
# cv2.waitKey(0)
#
# # print("preprocessing starts")
# # ss.setBaseImage(img)
# # ss.switchToSelectiveSearchFast()
# # rects = ss.process()
# # imOut = img.copy()
# # for i, rect in (enumerate(rects)):
# #     x, y, w, h = rect
# # #     print(x,y,w,h)
# # #     imOut = imOut[x:x+w,y:y+h]
# #     cv2.rectangle(imOut, (x, y), (x+w, y+h), (0, 255, 0), 1, cv2.LINE_AA)
# # # plt.figure()
# # cv2.namedWindow('ssp', cv2.WINDOW_NORMAL)
# # cv2.imshow("ssp", imOut)
# # cv2.waitKey(0)

ss.setBaseImage(img)
ss.switchToSelectiveSearchFast()
ssresults = ss.process()
res = np.array(ssresults)
# np.save(os.path.join(np_path, filename), res)
imout = img.copy()
for e, result in enumerate(ssresults):
    if e < 2000:
        x, y, w, h = result
        timage = imout[y:y + h, x:x + w]
        resized = cv2.resize(timage, (224, 224), interpolation=cv2.INTER_AREA)
        img = np.expand_dims(resized, axis=0)
        out = model.predict(img)
        # print(out)
        if out[0][0] > 0.9:
             cv2.rectangle(imout, (x, y), (x + w, y + h), (0, 255, 0), 1, cv2.LINE_AA)
# plt.figure()
# cv2.imwrite(os.path.join(out_path, "result.png"), imout)
cv2.namedWindow('result', cv2.WINDOW_NORMAL)
cv2.imshow("result", imout)
cv2.waitKey(0)

# # for e, i in enumerate(os.listdir(path)):
# #     filename = i.split(".")[0] + ".png"
# #     print(filename)
# #     img = cv2.imread(os.path.join(path, filename))
# #     cv2.namedWindow('result', cv2.WINDOW_NORMAL)
# #     cv2.imshow("result", img)
# #     cv2.imwrite(os.path.join(out_path, filename), img)
# #     cv2.waitKey(0)
