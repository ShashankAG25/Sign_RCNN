import os, cv2, keras
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

path = "sign/img/"
annot = "sign/label_csv/"

# for e,i in enumerate(os.listdir(annot)):
#     if e < 10:
#         filename = i.split(".")[0]+".png"
#         print(filename)
#         img = cv2.imread(os.path.join(path,filename))
#         df = pd.read_csv(os.path.join(annot,i))
#         plt.imshow(img)
#         for row in df.iterrows():
#             x1 = int(row[1][0].split(" ")[0])
#             y1 = int(row[1][0].split(" ")[1])
#             x2 = int(row[1][0].split(" ")[2])
#             y2 = int(row[1][0].split(" ")[3])
#             cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,0), 2)
#         # plt.figure()
#         # plt.imshow(img)
#         cv2.namedWindow('Output', cv2.WINDOW_NORMAL)
#         cv2.imshow("Output", img)
#         cv2.waitKey(0)
#         break

cv2.setUseOptimized(True)
ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
#
# im = cv2.imread("sign/img/1.png")
# df = pd.read_csv("sign/label_csv/1.csv")
#
#
# gtvalues = []
# for row in df.iterrows():
#     x1 = int(row[1][0].split(" ")[0])
#     y1 = int(row[1][0].split(" ")[1])
#     x2 = int(row[1][0].split(" ")[2])
#     y2 = int(row[1][0].split(" ")[3])
#     gtvalues.append({"x1": x1, "x2": x2, "y1": y1, "y2": y2})
#
#
# ss.setBaseImage(im)
# ss.switchToSelectiveSearchFast()
# rects = ss.process()
# imOut = im.copy()
#
#
# for i, rect in (enumerate(rects)):
#     x, y, w, h = rect
#     for gtval in gtvalues:
#         if (rect == gtval).all():
#             cv2.rectangle(imOut, (x, y), (x + w, y + h), (0, 255, 0), 1, cv2.LINE_AA)
#             cv2.namedWindow('Output', cv2.WINDOW_NORMAL)
#             cv2.imshow("Output", imOut)
#             cv2.waitKey(0)
# # plt.figure()
# # plt.imshow(imOut)


train_images = []
train_labels = []


def get_iou(bb1, bb2):
    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']

    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou


image = cv2.imread("sign/img/1.png")
# cv2.namedWindow('Output', cv2.WINDOW_NORMAL)
# cv2.imshow("Output", image)
# cv2.waitKey(0)
df = pd.read_csv("sign/label_csv/1.csv")
# print(df)
gtvalues = []
for row in df.iterrows():
    x1 = int(row[1][0].split(" ")[0])
    y1 = int(row[1][0].split(" ")[1])
    x2 = int(row[1][0].split(" ")[2])
    y2 = int(row[1][0].split(" ")[3])
    gtvalues.append({"x1": x1, "x2": x2, "y1": y1, "y2": y2})
# print(gtvalues)
ss.setBaseImage(image)
ss.switchToSelectiveSearchFast()
ssresults = ss.process()

res = np.array(ssresults)
np.save("ssresult", res)

ssresults = np.load("ssresult.npy")
imout = image.copy()


counter = 0
falsecounter = 0
flag = 0
fflag = 0
bflag = 0
for e, result in enumerate(ssresults):
    if e < 2000 and flag == 0:
        for gtval in gtvalues:
            x, y, w, h = result
            iou = get_iou(gtval, {"x1": x, "x2": x + w, "y1": y, "y2": y + h})
            if counter < 30:
                if iou > 0.6:
                    print(iou)
                    print(x, y, w, h)
                    timage = imout[y:y + h, x:x + w]
                    resized = cv2.resize(timage, (224, 224), interpolation=cv2.INTER_AREA)
                    train_images.append(resized)
                    train_labels.append(1)
                    counter += 1
                    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 1, cv2.LINE_AA)
                    cv2.namedWindow('Output', cv2.WINDOW_NORMAL)
                    cv2.imshow("Output", image)
                    cv2.waitKey(0)
            else:
                fflag = 1
            if falsecounter < 30:
                if iou < 0.6:
                    timage = imout[y:y + h, x:x + w]
                    resized = cv2.resize(timage, (224, 224), interpolation=cv2.INTER_AREA)
                    train_images.append(resized)
                    train_labels.append(0)
                    falsecounter += 1

            else:
                bflag = 1


        if fflag == 1 and bflag == 1:
            print("inside")
print(train_labels)
X_new = np.array(train_images)
y_new = np.array(train_labels)

np.save("x_new_exp", X_new)
np.save("y_new_exp", y_new)
#
