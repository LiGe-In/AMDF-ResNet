import sys
sys.path.append('../caffe/python')
sys.path.append('../caffe/python/caffe')
import caffe
import argparse
from pylab import *
import time
# import matplotlib.pyplot as plt
if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--solver_folder', default='../Solver/', help='path to the solver FOLDER. [DEFAULT=../Solver/]')
    parser.add_argument('--solver_file', default='solver.prototxt', help='path to the solver NAME. [DEFAULT=solver.prototxt]')
    args =  parser.parse_args()

    caffe.set_mode_gpu()
    caffe.set_device(0)
    
    SOLVER_FULL_PATH = args.solver_folder + args.solver_file
    solver = None
    solver = caffe.get_solver(SOLVER_FULL_PATH)
 
    solver.net.forward()  # train net

    niter = 10000
    # losses will also be stored in the log
    train_loss = zeros(niter)

    start=time.clock()
    start_time=time.time()
    train_start_time = time.time()
    train_start_clock = time.clock()
    # the main solver loop
    for it in range(niter):
        solver.step(1)  # SGD by Caffe
        train_loss[it] = solver.net.blobs['loss'].data
        if it % (100) == 0:
            train_elapsed_clock = time.clock() - train_start_clock
            train_start_clock = time.clock()
            print("100 iters time clock: %.1f s" % (train_elapsed_clock))

    elapsed = time.clock() - start
    print("all time clock: %.1f s" % elapsed)

    elapsed_time = time.time() - start_time
    print("all time: %.1f s" % elapsed)
     # plot the train loss and test accuracy
    _, ax1 = subplots()
    ax1.plot(arange(niter), train_loss)
    ax1.set_xlabel('iteration')
    ax1.set_ylabel('train loss')
    show()
