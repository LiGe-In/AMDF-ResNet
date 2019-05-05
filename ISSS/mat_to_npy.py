import numpy as np
import scipy.io as sio
from skimage import io

path="../dataset/image2/"
select_xy_mat=sio.loadmat("./sp_xyc_train2000.mat")
select_xy=select_xy_mat['select_xyc_all']

np.save(path+"sp_xyc_train2000.npy",select_xy)

gt=io.imread(path+'gt.tif')

for i in range(select_xy.shape[0]):
    gt[int(select_xy[i,0]),int(select_xy[i,1])]=0
inds_test=np.where(gt!=0)
xys_test=np.zeros((inds_test[0].shape[0],3))
for i in range(inds_test[0].shape[0]):
    xys_test[i,0]=inds_test[0][i]
    xys_test[i, 1] = inds_test[1][i]
    xys_test[i, 2]=gt[int(xys_test[i,0]),int(xys_test[i,1])]
np.save(path+"sp_xyc_test_2000.npy",xys_test)