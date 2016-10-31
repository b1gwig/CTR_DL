# pylint: skip-file

# This is the MXNet version of CDL used in the KDD 2016 hands-on tutorial for MXNet. Note that the code is a simplified version of CDL and is used for demonstration only (you can also find the corresponding IPython notebook). We do not use pretrain and sigmoid activation (as used in the matlab/C++ code for CDL) in this version of code, which may harm the performance.

# To run the code, please type in (after installing MXNet):
# python cdl.py
# in the command line.

# Hao WANG, 2016.8.24

from mxnet.misc import FactorScheduler
import mxnet as mx
import numpy as np
import logging
import data
from math import sqrt
from autoencoder import AutoEncoderModel
import os

if __name__ == '__main__':
    lambda_u = 1 # lambda_u in CDL
    lambda_v = 10 # lambda_v in CDL
    K = 50
    num_iter = 34000
    batch_size = 256

    np.random.seed(1234) # set seed
    lv = 1e-2 # lambda_v/lambda_n in CDL

    l_rate = 0.1

    # setup output dir
    dir_save = 'cdl_out'
    if not os.path.isdir(dir_save):
        os.system('mkdir %s' % dir_save)
    fp = open(dir_save+'/cdl.log','w')
    print 'lambda_v/lambda_u/ratio/K: %f/%f/%f/%d' % (lambda_v,lambda_u,lv,K)
    fp.write('lambda_v/lambda_u/ratio/K: %f/%f/%f/%d\n' % \
            (lambda_v,lambda_u,lv,K))
    fp.close()

    # setup logger
    logging.basicConfig(level=logging.DEBUG)

    X = data.get_mult()
    R = data.read_user()

    ae_model = AutoEncoderModel(xpu=mx.cpu(2),
                                dims=[X.shape[1],100,K],
                                pt_dropout=0.2)

    train_X = X

    V = np.random.rand(train_X.shape[0],K)/10
    lambda_v_rt = np.ones((train_X.shape[0],K))*sqrt(lv)
    learnign_rate_factor = FactorScheduler(step=20000, factor=l_rate)

    U, V, theta, BCD_loss = ae_model.finetune(train_X, R, V,
                                              lambda_v_rt, lambda_u, lambda_v,
                                              dir_save, batch_size, num_iter, 'sgd', l_rate=l_rate, decay=0.0,
                                              lr_scheduler=learnign_rate_factor)
    #ae_model.save('cdl_pt.arg')
    np.savetxt(dir_save+'/final-U.dat',U,fmt='%.5f',comments='')
    np.savetxt(dir_save+'/final-V.dat',V,fmt='%.5f',comments='')
    np.savetxt(dir_save+'/final-theta.dat',theta,fmt='%.5f',comments='')

    #ae_model.load('cdl_pt.arg')
    Recon_loss = lambda_v/lv*ae_model.eval(train_X,V,lambda_v_rt)
    print "Training error: %.3f" % (BCD_loss+Recon_loss)
    fp = open(dir_save+'/cdl.log','a')
    fp.write("Training error: %.3f\n" % (BCD_loss+Recon_loss))
    fp.close()
    #print "Validation error:", ae_model.eval(val_X)
