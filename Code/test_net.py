import sys
sys.path.append('../caffe/python')
sys.path.append('../caffe/python/caffe')
import caffe
import argparse
from pylab import *
from skimage import io
import time
from sklearn import metrics
from tqdm import tqdm

if __name__ == '__main__':
    data_folder="image1"
    output_folder = "image1_2000"
    save_path = "../output_sp/"+output_folder+"/"
    path = "../Data/"+data_folder+"/"

    parser = argparse.ArgumentParser()
    parser.add_argument('--def', dest='prototxt', help='prototxt file defining the network',
                        default="../models/test_net.prototxt", type=str)
    parser.add_argument('--net', dest='caffemodel', help='model to test',
                        default=save_path+'amdf_iter_10000.caffemodel', type=str)
    args =  parser.parse_args()

    caffe.set_mode_gpu()
    caffe.set_device(0)

    net = caffe.Net(args.prototxt, args.caffemodel, caffe.TEST)
    cls=[]
    cls_prob=[]
    print 'testing...'
    xyc_arrays = np.load(path+"xy_test_all_label.npy")
    start = time.clock()
    batch_size=300
    for cur in tqdm(range(0,xyc_arrays.shape[0],batch_size)):
        net.forward()
        feature=net.blobs['score'].data
        if cur + batch_size >= xyc_arrays.shape[0]:
            cls.extend(net.blobs['score'].data.argmax(1).tolist()[-(xyc_arrays.shape[0]-cur):])
        else:
            cls.extend(net.blobs['score'].data.argmax(1).tolist())

    print len(cls)

    gt = io.imread(path+"gt.tif")
    h, w = gt.shape

    gt_pred = np.zeros((h ,w))
    gt_prob = np.zeros((h, w))
    for k in range(len(cls)):
        gt_pred[int(xyc_arrays[k][0]),int(xyc_arrays[k][1])]= int(cls[k])+1#  class_list[int(cls[k])]
    GT = np.array(gt_pred, dtype=np.uint8)
    io.imsave(save_path+"GT.tif", GT)
    print(save_path+"GT.tif saved")
    elapsed = time.clock() - start
    print("all time: %.1f s" % elapsed)
    def colorshow_a(gt, colors):
        x = gt.shape[0]
        y = gt.shape[1]
        c = np.ones((x, y, 3),dtype=np.uint8) * 255
        for i in range(x):
            for j in range(y):
                c[i, j, :] = colors[gt[i, j]]
        c = np.array(c, dtype=np.uint8)
        return c


    def colorshow(label, colors, gt):
        # assert label.shape == gt.shape
        x = label.shape[0]
        y = label.shape[1]
        c = np.ones((x, y, 3), dtype=np.uint8) * 255
        for i in range(x):
            for j in range(y):
                if gt[i, j] == 0:
                    c[i, j, :] = colors[0]
                else:
                    c[i, j, :] = colors[label[i, j]]
        c = np.array(c, dtype=np.uint8)
        return c


    colors = {"image1": [[0, 0, 0],
                         [255, 0, 0],
                         [0, 255, 0],
                         [0, 0, 255],
                         [241, 132, 14],
                         [255, 255, 0],
                         [210, 180, 140],
                         [255, 192, 203],
                         [170, 210, 140],
                         [150, 6, 205],
                         [0, 255, 255],
                         [255, 0, 255]],
              "image2": [[0, 0, 0],
                         [255, 0, 0],
                         [0, 255, 0],
                         [0, 0, 255],
                         [241, 132, 14],
                         [255, 255, 0],
                         [0, 136, 78],
                         [255, 192, 203],
                         [0, 255, 255]],
              "image3": [[0, 0, 0],
                         [0, 255, 0],
                         [255, 255, 0],
                         [255, 0, 0],
                         [241, 132, 14],
                         [0, 255, 255],
                         [255, 192, 203],
                         [170, 210, 140],
                         [150, 6, 205]],
              "image4": [[0, 0, 0],
                         [255, 153, 255],
                         [255, 153, 51],
                         [0, 255, 255],
                         [255, 0, 0],
                         [153, 102, 51],
                         [0, 255, 0],
                         [0, 0, 255]]}

    color1 = colorshow_a(GT, colors[data_folder])
    color2 = colorshow(GT, colors[data_folder], gt)
    io.imsave(save_path+"color1.tif", color1)
    io.imsave(save_path + "color2.tif", color2)


    def eval(y_true, y_pred):
        confusion_matrix = metrics.confusion_matrix(y_true, y_pred)
        overall_accuracy = metrics.accuracy_score(y_true, y_pred)
        acc_for_each_class = metrics.recall_score(y_true, y_pred, average=None)
        average_accuracy = np.mean(acc_for_each_class)
        kappa = metrics.cohen_kappa_score(y_true, y_pred, labels=None)
        return overall_accuracy, average_accuracy, kappa, acc_for_each_class, confusion_matrix

    # path = "../Data/image2/"
    GT = io.imread(path + "gt.tif")
    h, w = GT.shape

    pre_gt = io.imread(save_path+"GT.tif")
    xyc_arrays = np.load(path + "sp_xyc_test_2000.npy")

    a = GT.max()
    b = pre_gt.max()
    a1 = GT.min()
    b1 = pre_gt.min()
    inds = np.where(GT != 0)
    y_true = GT[inds]
    y_pred = pre_gt[inds]

    test_y_true1 = np.zeros((xyc_arrays.shape[0]))
    test_y_true2 = np.zeros((xyc_arrays.shape[0]))
    test_y_pred = np.zeros((xyc_arrays.shape[0]))
    for i in range(xyc_arrays.shape[0]):
        test_y_pred[i] = pre_gt[int(xyc_arrays[i, 0]), int(xyc_arrays[i, 1])]
        test_y_true1[i] = xyc_arrays[i, 2]
        test_y_true2[i] = GT[int(xyc_arrays[i, 0]), int(xyc_arrays[i, 1])]
    overall_accuracy,average_accuracy,kappa,acc_for_each_class,confusion_matrix=eval(test_y_true2,test_y_pred)
    print("test")
    print('OA: {0:.4f}'.format(overall_accuracy)),
    print('AA: {0:.4f}'.format(average_accuracy)),
    print('kappa: {0:.4f}'.format(kappa))
    np.set_printoptions(precision=4)
    print('acc_for_each_class:')
    print(np.array(acc_for_each_class))
    # print('confusion_matrix:')
    # print(confusion_matrix)
