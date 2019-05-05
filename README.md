Adaptive Multiscale Deep Fusion Residual Network for Remote Sensing Image Classification

The important samples selection strategy (ISSS) is implemented in matlab. The adaptive multiscale deep fusion residual network (AMDF-ResNet)is implemented using the deep learning framework of Caffe. 

The folder dataset is datasets, where image1 refers to Vancouver Level 1B Image, and image2 refers to Vancouver Level 1C Image. The folder models are models for training and testing.
In the folder caffe, based on the original caffe module, the relevant code for implementing the adaptive fusion module is added. The folder data_code contains the code for data processing, where data_normal.py normalizes the data, and test_index.py save the coordinates of the test pixels in order to reduce the memory usage. When using different datasets, the corresponding data path in the code needs to be replaced.

1.Build Caffe and pycaffe
cd ./caffe
make all && make pycaffe

2.Test
The folder output contains the weights trained using our method. Using these weights to test, the corresponding classification map and the quantitative result are obtained. The quantitative result is saved in res.txt.
cd ./Code
python test_net.py


3.Train
First run the code of the ISSS folder to get the samples of training and testing.
Run vl_demo_slic.m to get the superpixel segmentation result.
Run gradient_ROEWA.m to get the gradient map.
Run select_sample.m to get the pixels for training.
Run mat_to_npy.py to get the pixels for testing.
Then train the model.
cd ./Code
python test_net.py

