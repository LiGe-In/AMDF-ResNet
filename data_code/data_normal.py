import numpy as np
from skimage import io

path="../dataset/image2/"
data=io.imread(path+"ms4.tif")
# print data.shape

data=data.transpose(2,0,1)
c, h, w = data.shape
for k in range(c):
    data[k, :, :] = (data[k, :, :] - np.min(data[k, :, :])) * 255.0 / (
        np.max(data[k, :, :]) - np.min(data[k, :, :]))
data=data.astype(np.float32)
for k in range(c):
    data[k, :, :] = (data[k, :, :] - np.mean(data[k, :, :]))*1.0/np.std(data[k, :, :])

np.save(path+"data_normal.npy",data)