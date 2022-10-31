#!/usr/bin/env python2
# -*- coding: utf-8 -*-
# $File: common.py
# $Date: Tue Mar 21 18:03:18 2017 +0800
# $Author: Xinyu Zhou <zxytim[at]gmail[dot]com>

from neupeak.utils import get_caller_base_dir, get_default_log_dir
import os
import hashlib
import getpass


class Config:
    base_dir = get_caller_base_dir()

    log_dir = get_default_log_dir(base_dir)
    '''where to write all the logging information during training
    (includes saved models)'''

    log_model_dir = os.path.join(log_dir, 'models')
    '''where to write model snapshots to'''

    log_file = os.path.join(log_dir, 'log.txt')

    exp_name = os.path.basename(log_dir)
    '''name of this experiment'''

    minibatch_size = 128
    nr_channel = 3  # input channels
    image_shape = (224, 224)
    nr_class = 1000

    @property
    def input_shape(self):
        return (self.minibatch_size,
                self.nr_channel) + self.image_shape

    def abs_path(self, path):
        '''convert relative path to base_dir to absolute path
        :return: absolute path '''
        return os.path.join(self.base_dir, path)

    rela_path = abs_path

    def make_servable_name(self, dataset_name, dep_files):
        '''make a unique servable name.

        .. note::
            The resulting servable name is composed by the content of
            dependency files and the original dataset_name given.

        :param dataset_name: an dataset identifier, usually the argument
            passed to dataset.py:get
        :type dataset_name: str

        :param dep_files: files that the constrution of the dataset depends on.
        :type dep_files: list of str

        '''
        parts = []
        for path in dep_files:
            with open(path, 'rb') as f:
                parts.append(self._md5(f.read()))
        return ('neupeak:' + getpass.getuser() + ':' + '.'.join(parts) + '.' +
                dataset_name)

    def _md5(self, s):
        m = hashlib.md5()
        m.update(s)
        return m.hexdigest()


config = Config()


# vim: foldmethod=marker
