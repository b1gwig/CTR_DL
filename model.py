# pylint: skip-file
import mxnet as mx
import numpy as np
import logging
from solver import Solver, Monitor
try:
   import cPickle as pickle
except:
   import pickle


def extract_feature(sym, args, auxs, data_iter, N, xpu=mx.cpu()):
    input_buffs = [mx.nd.empty(shape, ctx=xpu) for k, shape in data_iter.provide_data]
    input_names = [k for k, shape in data_iter.provide_data]
    args = dict(args, **dict(zip(input_names, input_buffs)))
    exe = sym.bind(xpu, args=args, aux_states=auxs)
    outputs = [[] for i in exe.outputs]
    output_buffs = None

    data_iter.hard_reset()
    for batch in data_iter:
        for data, buff in zip(batch.data, input_buffs):
            data.copyto(buff)
        exe.forward(is_train=False)
        if output_buffs is None:
            output_buffs = [mx.nd.empty(i.shape, ctx=mx.cpu()) for i in exe.outputs]
        else:
            for out, buff in zip(outputs, output_buffs):
                out.append(buff.asnumpy())
        for out, buff in zip(exe.outputs, output_buffs):
            out.copyto(buff)
    for out, buff in zip(outputs, output_buffs):
        out.append(buff.asnumpy())
    outputs = [np.concatenate(i, axis=0)[:N] for i in outputs]
    return dict(zip(sym.list_outputs(), outputs))