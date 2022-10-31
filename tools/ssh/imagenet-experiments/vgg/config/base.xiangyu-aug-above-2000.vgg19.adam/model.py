#!/usr/bin/env python2
# -*- coding: utf-8 -*-
# $File: model.py
# $Date: Tue Mar 21 17:31:03 2017 +0800
# $Author: Xinyu Zhou <zxytim[at]gmail[dot]com>


from megskull.network import MLPNetworkBuilder, RawNetworkBuilder
from megskull.graph import iter_dep_opr
from megskull.opr.netsrc import ParamProvider
from neupeak import model as O

from common import config

def proc(x):
    return O.relu(x)

def extract(x):
    nl = O.nonlinearity.Identity()

    def ch(n):
        return n

    # BGR
    #image_mean = O.shared('image_mean', value=np.array([104.00699, 116.66877, 122.67892], 'float32'))
    #x = x - image_mean.dimshuffle('x', 0, 'x', 'x')
    x = x / 256

    chs = list(map(ch, [64, 128, 256, 512, 512]))

    for group in range(1, 6):
        for i in range(1, 5 if group > 2 else 3):
            x = O.conv2d_bn_cw('conv{}_{}'.format(group, i), x, kernel_shape=3, padding=1, output_nr_channel=chs[group - 1], nonlinearity=nl)
            x = proc(x)

        x = O.pool2d('pool{}'.format(group), x, 2, mode='max')

    x = O.fully_connected_bn_cw('fc1', x, output_dim=ch(4096), nonlinearity=O.nonlinearity.ReLU())
    x = O.fully_connected_bn_cw('fc2', x, output_dim=ch(4096), nonlinearity=O.nonlinearity.ReLU())
    x = O.fully_connected_bn_cw('fct', x, output_dim=config.nr_class, nonlinearity=nl)

    # please follow the convention of nameing the final output layer as 'pred'
    x = O.softmax('pred', x)
    return x

def get():
    input_image = O.input('data', shape=config.input_shape)
    label = O.input('label', shape=(config.minibatch_size, ))

    x = input_image

    x = extract(x)

    pred = x


    losses = dict()
    loss_xent = O.cross_entropy(pred, label, name='loss_xent')
    losses['loss_xent'] = loss_xent

    loss_wowd = sum(losses.values())
    O.utils.hint_loss_subgraph(list(losses.values()), loss_wowd)
    loss_wwd = O.weight_decay(loss_wowd, {'*:W': 1e-5})
    loss_weight_decay = loss_wwd - loss_wowd
    loss_weight_decay.vflags.data_parallel_reduce_method = 'sum'
    # loss = sum(losses.values())  # loss is the sum of losses

    loss = loss_wwd
    losses['loss_weight_decay'] = loss_weight_decay

    network = RawNetworkBuilder(inputs=[input_image], outputs=[pred], loss=loss)
    extra_outputs = losses.copy()
    extra_outputs.update(
        misclassify=O.misclassify(pred, label),
        accuracy=O.accuracy(pred, label),
        accuracy10=O.accuracy_top_n(pred, label, 10),
    )

    network.extra['extra_outputs'] = extra_outputs
    network.extra['extra_config'] = {
        'monitor_vars': sorted(extra_outputs.keys()),
    }

    return network


# vim: foldmethod=marker
