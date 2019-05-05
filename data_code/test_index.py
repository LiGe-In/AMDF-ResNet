import numpy as np
from skimage import io

path="../Data/image2/"
gt=io.imread(path+'gt.tif')
h,w=gt.shape
xyc_test_all=np.zeros((h*w,3))
for i in range(h):
    for j in range(w):
        # print(i * w + j)
        xyc_test_all[i * w + j, 0] = i
        xyc_test_all[i * w + j, 1] = j
        xyc_test_all[i * w + j, 2] = gt[i,j]
np.save(path+"xy_test_all.npy",xyc_test_all)

inds_label=np.where(gt!=0)
xy_test_all_label=np.zeros((len(inds_label[0]),2))
for i in range(len(inds_label[0])):
    xy_test_all_label[i,0]=inds_label[0][i]
    xy_test_all_label[i, 1] = inds_label[1][i]
np.save(path+"xy_test_all_label.npy",xy_test_all_label)

