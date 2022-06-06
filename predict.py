import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from keras import Model
from tensorflow import keras

model = keras.models.load_model("ieeercnn_vgg16_1.h5")


# path = "Test_sets/1.png"
path = "Test_sets1/"
out_path = "Test_results"
np_path = "sspreprocess"


# img = cv2.imread(path)
# cv2.imwrite(str(i)+"predict.png",img)
#
# X_new = np.load("x_new.npy")
# y_new = np.load("y_new.npy")

# plt.plot(hist.history['loss'])
# plt.plot(hist.history['val_loss'])
# plt.title("model loss")
# plt.ylabel("Loss")
# plt.xlabel("Epoch")
# plt.legend(["Loss","Validation Loss"])
# plt.show()
# plt.savefig('chart loss.png')


# im = X_new[160]
# plt.imshow(im)
# img = np.expand_dims(im, axis=0)
# out= model.predict(img)
# if out[0][0] > out[0][1]:
#     print("signature")
# else:
#     print("not sign")
# z=0
#
ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
#
#
#
#
for e, i in enumerate(os.listdir(path)):
    filename = i.split(".")[0] + ".png"
    print(filename)
    img = cv2.imread(os.path.join(path, filename))
    ss.setBaseImage(img)
    ss.switchToSelectiveSearchFast()
    ssresults = ss.process()
    res = np.array(ssresults)
    np.save(os.path.join(np_path,filename),res)
    imout = img.copy()

    for e, result in enumerate(ssresults):
        if e < 2000:
            x, y, w, h = result
            timage = imout[y:y + h, x:x + w]
            resized = cv2.resize(timage, (224, 224), interpolation=cv2.INTER_AREA)
            img = np.expand_dims(resized, axis=0)
            out = model.predict(img)
            if out[0][0] > 0.9:
                cv2.rectangle(imout, (x, y), (x + w, y + h), (0, 255, 0), 1, cv2.LINE_AA)
    # plt.figure()
    cv2.imwrite(os.path.join(out_path, filename), imout)
    # cv2.namedWindow('result', cv2.WINDOW_NORMAL)
    # cv2.imshow("result", imout)
    # cv2.waitKey(0)

# for e, i in enumerate(os.listdir(path)):
#     filename = i.split(".")[0] + ".png"
#     print(filename)
#     img = cv2.imread(os.path.join(path, filename))
#     cv2.namedWindow('result', cv2.WINDOW_NORMAL)
#     cv2.imshow("result", img)
#     cv2.imwrite(os.path.join(out_path, filename), img)
#     cv2.waitKey(0)
