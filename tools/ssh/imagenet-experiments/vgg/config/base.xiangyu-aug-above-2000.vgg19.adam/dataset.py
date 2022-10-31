#!/usr/bin/env python2
# -*- coding: utf-8 -*-
# $File: dataset.py
# $Date: Wed Jan 11 01:53:15 2017 +0800
# $Author: Xinyu Zhou <zxytim[at]gmail[dot]com>



from neupeak.dataset.server import create_servable_dataset
from neupeak.dataset.meta import GeneratorDataset, StackMinibatchDataset
from neupeak.dataset.lmdb import LMDBJsonDataset
from neupeak.utils.meta import function_pipe
from neupeak.utils.misc import stable_rng
from neupeak.utils import imgproc

from meghair.train.base import DatasetMinibatch
from meghair.utils import logconf
from meghair.utils.imgproc import imdecode
from meghair.utils.misc import ic01_to_i01c, i01c_to_ic01, list2nparray
from meghair.dpflow import control, InputPipe
import getpass
import random

from common import config

# import augmentor
# from augmentor import augmentor_name

import cv2
import numpy as np
from tqdm import tqdm

logger = logconf.get_logger(__name__)


import imagenet_crop

def augment(img, rng, do_training):
    if not do_training:
        # img_shape on 256x crop
        h, w = img.shape[:2]

        # 1. scale the image such that the minimum edge length is 256 using
        #    bilinear interpolation
        scale = 256 / min(h, w)
        th, tw = map(int, (h * scale, w * scale))
        img = cv2.resize(img, (tw, th), interpolation=cv2.INTER_LINEAR)
        th, tw = img.shape[:2]

        # 2. center crop 224x244 patch from the image
        sx, sy = (tw-config.image_shape[1])//2, (th-config.image_shape[0])//2
        img = img[sy:sy+config.image_shape[0], sx:sx+config.image_shape[1]]

    else:
        # data augmentation from fb.resnet.torch
        # https://github.com/facebook/fb.resnet.torch/blob/master/datasets/imagenet.lua

        def scale(img, size):
            s = size / min(img.shape[0], img.shape[1])
            h, w = int(round(img.shape[0]*s)), int(round(img.shape[1]*s))
            return cv2.resize(img, (w, h))


        def center_crop(img, shape):
            h, w = img.shape[:2]
            sx, sy = (w-shape[1])//2, (h-shape[0])//2
            img = img[sy:sy+shape[0], sx:sx+shape[1]]
            return img


        def random_sized_crop(img):
            NR_REPEAT = 10

            h, w = img.shape[:2]
            area = h * w
            ar = [3./4, 4./3]
            for i in range(NR_REPEAT):
                target_area = rng.uniform(0.08, 1.0) * area
                target_ar = rng.choice(ar)
                nw = int(round( (target_area * target_ar) ** 0.5 ))
                nh = int(round( (target_area / target_ar) ** 0.5 ))

                if rng.rand() < 0.5:
                    nh, nw = nw, nh

                if nh <= h and nw <= w:
                    sx, sy = rng.randint(w - nw + 1), rng.randint(h - nh + 1)
                    img = img[sy:sy+nh, sx:sx+nw]

                    return cv2.resize(img, config.image_shape[::-1])

            size = min(config.image_shape[0], config.image_shape[1])
            return center_crop(scale(img, size), config.image_shape)


        def grayscale(img):
            w = list2nparray([0.114, 0.587, 0.299]).reshape(1, 1, 3)
            gs = np.zeros(img.shape[:2])
            gs = (img * w).sum(axis=2, keepdims=True)

            return gs


        def brightness_aug(img, val):
            alpha = 1. + val * (rng.rand() * 2 - 1)
            img = img * alpha

            return img


        def contrast_aug(img, val):
            gs = grayscale(img)
            gs[:] = gs.mean()
            alpha = 1. + val * (rng.rand() * 2 - 1)
            img = img * alpha + gs * (1 - alpha)

            return img


        def saturation_aug(img, val):
            gs = grayscale(img)
            alpha = 1. + val * (rng.rand() * 2 - 1)
            img = img * alpha + gs * (1 - alpha)

            return img


        def color_jitter(img, brightness, contrast, saturation):
            augs = [(brightness_aug, brightness),
                    (contrast_aug, contrast),
                    (saturation_aug, saturation)]
            random.shuffle(augs)

            for aug, val in augs:
                img = aug(img, val)

            return img


        def lighting(img, std):
            eigval = list2nparray([ 0.2175, 0.0188, 0.0045 ])
            eigvec = list2nparray([
                                [ -0.5836, -0.6948,  0.4203 ],
                                [ -0.5808, -0.0045, -0.8140 ],
                                [ -0.5675, 0.7192, 0.4009 ],
                                ])
            if std == 0:
                return img

            alpha = rng.randn(3) * std
            bgr = eigvec * alpha.reshape(1, 3) * eigval.reshape(1, 3)
            bgr = bgr.sum(axis=1).reshape(1, 1, 3)
            img = img + bgr

            return img


        def horizontal_flip(img, prob):
            if rng.rand() < prob:
                return img[:, ::-1]
            return img


        # img = random_sized_crop(img)
        x0, y0, x1, y1 = imagenet_crop.imagenet_standard_crop(
            width=img.shape[1],
            height=img.shape[0],
            complexity=10000,
            phase='TRAIN', standard='latest')

        # random interpolation
        assert config.image_shape[0] == config.image_shape[1], (
            config.image_shape)
        size = config.image_shape[0]
        img = imgproc.resize_rand_interp(
            rng, img[y0:y1,x0:x1], (size, size))

        img = color_jitter(img, brightness=0.4, contrast=0.4, saturation=0.4)
        img = lighting(img, 0.1)
        img = horizontal_flip(img, 0.5)

        img = np.minimum(255, np.maximum(0, img))

    return np.rollaxis(img, 2).astype('uint8')


def get(dataset_name):

    source_name = dict(
        train='brain.example.imagenet.source.train',
        validation='brain.example.imagenet.source.val')[dataset_name]

    servable_name = config.make_servable_name(
        dataset_name, list(map(config.rela_path,
            ['common.py', 'dataset.py'])))

    do_training = (dataset_name == 'train')

    nr_images_in_epoch = dict(
        train=5000 * 256,
        validation=196 * 256)[dataset_name]

    rng = stable_rng(stable_rng)

    def generator():
        source_pipe = InputPipe(source_name, buffer_size=1000)
        # Pipes in the same group will never receive the same data
        source_pipe.set_policy('GROUP_ID', servable_name)

        with control(io=[source_pipe]):
            for img_jpeg, label in source_pipe:
                img = imdecode(img_jpeg)[:,:,:3]
                ic01 = augment(img, rng, do_training)
                yield DatasetMinibatch(data=ic01, label=np.array(label),
                                       check_minibatch_size=False)

    dataset = GeneratorDataset(
        generator, nr_minibatch_in_epoch=nr_images_in_epoch)
    dataset = StackMinibatchDataset(dataset, config.minibatch_size)

    dataset = create_servable_dataset(
        dataset, servable_name, dataset.nr_minibatch_in_epoch,
        serve_type='combiner')

    return dataset


# vim: foldmethod=marker
