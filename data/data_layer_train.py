#coding:utf-8

import caffe
import numpy as np


class DataLayer(caffe.Layer):
    """data layer used for training."""
    def _shuffle_xyc_inds(self):
        """Randomly permute the training data."""
        self._perm = np.random.permutation(np.arange(self._xyc_inds.shape[0]))
        self._cur = 0

    def _get_next_minibatch_inds(self):
        """Return the indices for the next minibatch."""
        if self._cur + self.TRAIN_BATCH >= self._xyc_inds.shape[0]:
            self._shuffle_xyc_inds()

        db_inds = self._perm[self._cur:self._cur + self.TRAIN_BATCH]
        self._cur += self.TRAIN_BATCH
        return db_inds

    def _pad(self,hsi_48,xyc_arrays):
        d = self.TRAIN_IMAGE_SIZE//2
        c, h, w = hsi_48.shape
        hsi_48_pad = np.zeros((c, h + 2 * d, w + 2 * d))
        hsi_48_pad[:, d:h + d, d:w + d] = hsi_48
        for i in range(d):
            hsi_48_pad[:, i, :] = hsi_48_pad[:, 2 * d - i, :]
            hsi_48_pad[:, h + i, :] = hsi_48_pad[:, 2 * (h + d - 1) - (h + i), :]
            hsi_48_pad[:, :, i] = hsi_48_pad[:, :, 2 * d - i]
            hsi_48_pad[:, :, w + i] = hsi_48_pad[:, :, 2 * (w + d - 1) - (w + i)]
        for i in range(xyc_arrays.shape[0]):
            xyc_arrays[i, 0] += d
            xyc_arrays[i, 1] += d
            xyc_arrays[i, 2] -=1 #将[1,20]变为[0,19]
        return hsi_48_pad,xyc_arrays

    def setup(self, bottom, top):
        """Setup the DataLayer."""
        self._cur = 0
        self.TRAIN_BATCH = 64
        self.TRAIN_IMAGE_SIZE = 32
        self.CHANNELS = 4
        self._name_to_top_map = {
            'data': 0,
            'labels': 1}
        top[0].reshape(self.TRAIN_BATCH, self.CHANNELS, self.TRAIN_IMAGE_SIZE, self.TRAIN_IMAGE_SIZE)
        top[1].reshape(self.TRAIN_BATCH)

        xyc_arrays = np.load("../dataset/image2/sp_xyc_train_2000.npy")
        hsi_48 = np.load("../dataset/image2/data_normal.npy")

        print hsi_48.shape

        hsi_48,xyc_arrays = self._pad(hsi_48,xyc_arrays)

        self._hsi = hsi_48
        self._xyc_inds = xyc_arrays
        self._shuffle_xyc_inds()
        self._minibatch_data_blob=np.zeros((self.TRAIN_BATCH,self.CHANNELS, self.TRAIN_IMAGE_SIZE,self.TRAIN_IMAGE_SIZE),
                    dtype=np.float32)
        self._minibatch_labels_blob = np.zeros((self.TRAIN_BATCH), dtype=np.float32)

    def _get_blob(self,x,y):
        d = self.TRAIN_IMAGE_SIZE // 2
        patch=self._hsi[:,int(x-d):int(x+d),int(y-d):int(y+d)]
        return patch
    def forward(self, bottom, top):
        """Get blobs and copy them into this layer's top blob vector."""
        db_inds = self._get_next_minibatch_inds()

        for i in range(len(db_inds)):
            ind=db_inds[i]
            xyc=self._xyc_inds[ind]
            self._minibatch_labels_blob[i] = xyc[2]
            each_blob=self._get_blob(xyc[0],xyc[1])
            self._minibatch_data_blob[i,:,:,:]=each_blob
        # Reshape net's input blobs
        # Copy data into net's input blobs
        # data
        top[0].reshape(*(self._minibatch_data_blob.shape))
        top[0].data[...] = self._minibatch_data_blob.astype(np.float32, copy=False)
        # Reshape net's input blobs
        # Copy data into net's input blobs
        # label
        top[1].reshape(*(self._minibatch_labels_blob.shape))
        top[1].data[...] = self._minibatch_labels_blob.astype(np.float32, copy=False)


    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

