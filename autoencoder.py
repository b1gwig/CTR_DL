# pylint: skip-file
import mxnet as mx
import numpy as np
import model
import logging
from solver import Solver, Monitor
try:
   import cPickle as pickle
except:
   import pickle


class AutoEncoderModel(object):
    def __init__(self, xpu, dims, pt_dropout):
        self.xpu = xpu
        self.loss = None
        self.args = {}
        self.args_grad = {}
        self.args_mult = {}
        self.auxs = {}

        self.N = len(dims) - 1
        self.dims = dims
        self.stacks = []
        self.pt_dropout = pt_dropout

        self.data = mx.symbol.Variable('data')
        self.V = mx.symbol.Variable('V')
        self.lambda_v_rt = mx.symbol.Variable('lambda_v_rt')

        # create stack
        for i in range(self.N):
            if i == 0:
                idropout = None
            else:
                idropout = pt_dropout
            if i == self.N-1:
                odropout = None
            else:
                odropout = pt_dropout

            istack, iargs, iargs_grad, iargs_mult, iauxs = self.make_stack(i, self.data, dims[i], dims[i+1],
                                                idropout, odropout)
            self.stacks.append(istack)
            self.args.update(iargs)
            self.args_grad.update(iargs_grad)
            self.args_mult.update(iargs_mult)
            self.auxs.update(iauxs)

        # create encoder
        self.encoder, self.internals = self.make_encoder(self.data, dims)

        # create decoder
        self.decoder = self.make_decoder(self.encoder, dims)

        fe_loss = mx.symbol.LinearRegressionOutput(data=self.lambda_v_rt*self.encoder,
            label=self.lambda_v_rt*self.V)
        fr_loss = mx.symbol.LinearRegressionOutput(data=self.decoder, label=self.data)
        self.loss = mx.symbol.Group([fe_loss, fr_loss])

    def make_stack(self, istack, data, num_input, num_hidden, idropout=None,
                   odropout=None):
        x = data
        if idropout:
            x = mx.symbol.Dropout(data=x, p=idropout)

        x = mx.symbol.FullyConnected(name='encoder_%d'%istack, data=x, num_hidden=num_hidden)
        x = mx.symbol.Activation(data=x, act_type='relu')

        if odropout:
            x = mx.symbol.Dropout(data=x, p=odropout)
        x = mx.symbol.FullyConnected(name='decoder_%d'%istack, data=x, num_hidden=num_input)

        x = mx.symbol.Activation(data=x, act_type='relu')
        x = mx.symbol.LinearRegressionOutput(data=x, label=data)

        args = {'encoder_%d_weight'%istack: mx.nd.empty((num_hidden, num_input), self.xpu),
                'encoder_%d_bias'%istack: mx.nd.empty((num_hidden,), self.xpu),
                'decoder_%d_weight'%istack: mx.nd.empty((num_input, num_hidden), self.xpu),
                'decoder_%d_bias'%istack: mx.nd.empty((num_input,), self.xpu),}
        args_grad = {'encoder_%d_weight'%istack: mx.nd.empty((num_hidden, num_input), self.xpu),
                     'encoder_%d_bias'%istack: mx.nd.empty((num_hidden,), self.xpu),
                     'decoder_%d_weight'%istack: mx.nd.empty((num_input, num_hidden), self.xpu),
                     'decoder_%d_bias'%istack: mx.nd.empty((num_input,), self.xpu),}
        args_mult = {'encoder_%d_weight'%istack: 1.0,
                     'encoder_%d_bias'%istack: 2.0,
                     'decoder_%d_weight'%istack: 1.0,
                     'decoder_%d_bias'%istack: 2.0,}
        auxs = {}

        init = mx.initializer.Uniform(0.07)
        for k,v in args.items():
            init(k,v)

        return x, args, args_grad, args_mult, auxs

    def make_encoder(self, data, dims):
        x = data
        internals = []
        N = len(dims) - 1
        for i in range(N):
            x = mx.symbol.FullyConnected(name='encoder_%d'%i, data=x, num_hidden=dims[i+1])

            # if first input layer, then activation
            if i < N-1:
                x = mx.symbol.Activation(data=x, act_type='relu')

            internals.append(x)
        return x, internals

    def make_decoder(self, feature, dims):
        x = feature
        N = len(dims) - 1
        for i in reversed(range(N)):
            x = mx.symbol.FullyConnected(name='decoder_%d'%i, data=x, num_hidden=dims[i])
            if i > 0:
                x = mx.symbol.Activation(data=x, act_type='relu')
        return x

    def layerwise_pretrain(self, X, batch_size, n_iter, optimizer, l_rate, decay, lr_scheduler=None):
        def l2_norm(label, pred):
            return np.mean(np.square(label-pred))/2.0
        solver = Solver(optimizer, momentum=0.9, wd=decay, learning_rate=l_rate, lr_scheduler=lr_scheduler)
        solver.set_metric(mx.metric.CustomMetric(l2_norm))
        solver.set_monitor(Monitor(1000))
        data_iter = mx.io.NDArrayIter({'data': X}, batch_size=batch_size, shuffle=True,
                                      last_batch_handle='roll_over')
        for i in range(self.N):
            if i == 0:
                data_iter_i = data_iter
            else:
                X_i = model.extract_feature(self.internals[i-1], self.args, self.auxs,
                                            data_iter, X.shape[0], self.xpu).values()[0]
                data_iter_i = mx.io.NDArrayIter({'data': X_i}, batch_size=batch_size,
                                                last_batch_handle='roll_over')
            logging.info('Pre-training layer %d...'%i)
            solver.solve(self.xpu, self.stacks[i], self.args, self.args_grad, self.auxs, data_iter_i,
                         0, n_iter, {}, False)

    def finetune(self, X, R, V, lambda_v_rt, lambda_u, lambda_v, dir_save, batch_size, n_iter, optimizer, l_rate, decay, lr_scheduler=None):

        def l2_norm(label, pred):
           return np.mean(np.square(label-pred))/2.0

        solver = Solver(optimizer, momentum=0.9, wd=decay, learning_rate=l_rate, lr_scheduler=lr_scheduler)

        solver.set_metric(mx.metric.CustomMetric(l2_norm))
        solver.set_monitor(Monitor(1000))

        data_iter = mx.io.NDArrayIter({'data': X, 'V': V, 'lambda_v_rt':
            lambda_v_rt},
                batch_size=batch_size, shuffle=False,
                last_batch_handle='pad')
        logging.info('Fine tuning...')
        # self.loss is the net
        U, V, theta, BCD_loss = solver.solve(X, R, V,
                                             lambda_v_rt, lambda_u, lambda_v,
                                             dir_save, batch_size, self.xpu, self.loss,
                                             self.args, self.args_grad, self.auxs,
                                             data_iter, 0, n_iter)
        return U, V, theta, BCD_loss

    # modified by hog
    def eval(self, X, V, lambda_v_rt):
        batch_size = 100
        data_iter = mx.io.NDArrayIter({'data': X, 'V': V, 'lambda_v_rt':
            lambda_v_rt},
            batch_size=batch_size, shuffle=False,
            last_batch_handle='pad')
        # modified by hog
        Y = model.extract_feature(self.loss[1], self.args, self.auxs, data_iter,
                                 X.shape[0], self.xpu).values()[0]
        return np.sum(np.square(Y-X))/2.0

    def save(self, fname):
        args_save = {key: v.asnumpy() for key, v in self.args.items()}
        with open(fname, 'w') as fout:
            pickle.dump(args_save, fout)

    def load(self, fname):
        with open(fname) as fin:
            args_save = pickle.load(fin)
            for key, v in args_save.items():
                if key in self.args:
                    self.args[key][:] = v