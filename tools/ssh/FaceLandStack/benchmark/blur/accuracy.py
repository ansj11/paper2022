import time
from collections import OrderedDict

from benchmark.dataset import BlurrinessImage
from benchmark.tester import TestUnitBase
from landstack.utils.visualization import draw, quick_draw, montage
from landstack.utils import misc
from landstack.train import env
from IPython import embed
import tqdm
import numpy as np
import os
import cv2
from itertools import chain

def get_inverse_num(scores, pair_list):
    scores = scores.flatten()
    num_list = []
    for pair in pair_list:
        num = 0
        for l1, l2 in pair:
            if scores[l1] >= scores[l2]:
                num += 1
        num_list.append(num)
    return num_list

def softmax(x):
    x = np.float32(x)
    x = x - x.max(axis=1, keepdims=True)
    x = np.exp(x)
    return x / x.sum(axis=1, keepdims=True)

def get_loss(scores, pair_list):
    scores = scores.flatten()[list(chain.from_iterable(pair_list))]
    score_p_top1 = softmax(scores.reshape(-1, 2))[:, -1]
    loss = -(np.log(np.clip(score_p_top1, 1e-4, 1-1e-4)).mean())
    return loss


class TestUnit(TestUnitBase):
    def __init__(self, caches):
        name = "blur.accuracy Tester ({}-{})".format(time.strftime("%x"), time.strftime("%X"))
        super(TestUnit, self).__init__(name, caches)

    def gen_kv_dict_all(self, image_shape, val_keys, is_color, legacy):
        if self._kv_dict_all is None:
            val_ds = BlurrinessImage(keys=val_keys)
            self._kv_dict_all = val_ds.load(
                is_color=is_color,
                legacy=legacy,
                image_shape=image_shape,
            )
            all_in_kv_dict = OrderedDict()
            for k, v in self._kv_dict_all.items():
                for kk, vv in v.items():
                    all_in_kv_dict.setdefault(kk, []).append(vv)
            for k, v in all_in_kv_dict.items():
                self._kv_dict_all.setdefault('all-in', OrderedDict())[k] = np.concatenate(v)
            self._caches.setdefault('kv_dict_all', self._kv_dict_all)
        else:
            print("Using cached data")
        return self._kv_dict_all

    def gen_inf_outputs_data(self, net, devices, nr_loop=10, noise=20, seed=11):
        assert self._kv_dict_all, 'Please generate kv_dict_all first'
        inf_func = env.compile_inf_func(net.outputs, devices=devices)
        self._inf_outputs_data = OrderedDict()
        rng = misc.get_rng(seed)
        for k, v in self._kv_dict_all.items():
            self._inf_outputs_data.setdefault(k, []).append(env.forward(inf_func, {'img':v['img']}, batch_size=2028))
            print("\tLooping {} times to compute stability".format(nr_loop))
            for _ in tqdm.tqdm(range(nr_loop)):
                img_feed = np.uint8(rng.rand(*v['img'].shape) * noise + v['img']).clip(0, 255)
                self._inf_outputs_data.setdefault(k, []).append(env.forward(inf_func, {'img':img_feed}, batch_size=2048))
            assert len(self._inf_outputs_data[k]) == nr_loop + 1
        return self._inf_outputs_data

    def compute_metric(self, lm_name, badcase, out_root='/tmp/blur.accuracy'):
        assert self._kv_dict_all and self._inf_outputs_data, 'Please gen kv_dict_all and inf_outputs_data first'
        res_dict = OrderedDict()
        if badcase:
            misc.ensure_dir(out_root, erase=True)

        for key in self._kv_dict_all.keys():
            imgs = self._kv_dict_all[key]['img'].transpose(0, 2, 3, 1)
            pair_list = self._kv_dict_all[key]['pairs']
            tuple5_list = self._kv_dict_all[key]['tuple5']
            scores = self._inf_outputs_data[key][0][lm_name].flatten()
            num_list = get_inverse_num(scores, pair_list)
            loss = get_loss(scores, pair_list)
            res_dict0 = OrderedDict([
                ('#inverse', np.mean(num_list)),
                ('loss', loss),
            ])
            if len(self._inf_outputs_data[key]) > 1:
                score_trace = np.concatenate([v[lm_name] for v in self._inf_outputs_data[key][1:]], axis=1)
                std = np.mean(np.std(score_trace, axis=1))
                res_dict0['stability(gaussian)'] = std
            res_dict[key] = res_dict0

            if badcase:
                num2img_dict = OrderedDict()
                for i in range(len(num_list)):
                    if num_list[i] > 0:
                        scores0 = scores[tuple5_list[i]]
                        imgs0 = imgs[tuple5_list[i]]
                        imgs1 = imgs0[np.argsort(scores0)]
                        imgs0 = np.concatenate([imgs0, np.zeros_like(imgs0)], axis=0)[:5] # in case of less than 5
                        imgs1 = np.concatenate([imgs1, np.zeros_like(imgs1)], axis=0)[:5]
                        num2img_dict.setdefault(num_list[i], []).append(imgs0)
                        num2img_dict.setdefault(num_list[i], []).append(imgs1)

                for num, v in num2img_dict.items():
                    v0 = np.vstack(v)
                    save_root = os.path.join(out_root, key, str(num))
                    misc.ensure_dir(save_root)
                    for i in range(int(np.ceil(len(v) / 50.))):
                        img_copy = montage(v0[i*50:(i+1)*50], nrow=10, ncol=5, canvas_height_width=(1280, 640))
                        save_file = os.path.join(save_root, '{}.jpg'.format(i))
                        cv2.imwrite(save_file, img_copy)

        return res_dict

def main(model, devices, args_input, caches):
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-k', '--keys', type=str, default=None)
    parser.add_argument('--lm_name', default='clearness')
    parser.add_argument('--badcase', action='store_true', help='whether to log badcase')
    parser.add_argument('-s', '--image_shape', default='112,112')
    parser.add_argument('--is_color', action='store_true')
    parser.add_argument('--legacy', action='store_true')
    parser.add_argument('--nr_loop', type=int, default=0, help='#loop to compute std')
    parser.add_argument('--noise', type=float, default=20, help='noisy intensity')
    parser.add_argument('--seed', type=int, default=11, help='random seed')
    parser.add_argument('--lowbit', action='store_true')
    parser.add_argument('--on-mobile', action='store_true', help='test acc on mobile')
    parser.add_argument('--profile', default=None)
    parser.add_argument('--debug', action='store_true', help='enable -c dbg')
    parser.add_argument('--network-outputs', default='/tmp/network_outputs.pkl')
    parser.add_argument('--sfa', default=7, type=int)
    parser.add_argument('--bit_midout', default=16, type=int)

    args, unknownargs = parser.parse_known_args(args_input)

    # params
    val_keys = args.keys and args.keys.split(',')
    args.image_shape = tuple(map(int, args.image_shape.split(',')))
    assert len(args.image_shape) == 2, args.image_shape

    # load model
    assert len(args.lm_name.split(',')) == 1, args.lm_name
    net = misc.load_network_and_extract(model, args.lm_name)

    # init TestUnit
    t = TestUnit(caches=caches)
    t.gen_kv_dict_all(
        val_keys=val_keys,
        image_shape=args.image_shape,
        is_color=args.is_color,
        legacy=args.legacy,
    )

    if args.lowbit or args.on_mobile:
        command = "lsk-test gen.speed {} --lm_name {} --keys {} --no_info --network-outputs {} --sfa {} --bit_midout {} --dataset eyeoc.accuracy".format(
            model, args.lm_name, args.keys, args.network_outputs, args.sfa, args.bit_midout)
        command = command + ' --lowbit' if args.lowbit else command
        command = command + ' --debug' if args.debug else command
        command = command + ' --profile {}'.format(args.profile) if args.profile is not None else command
        print("Invoking sys command {}".format(command))
        res = os.system(command)
        if res:
            print("Error exit")
        else:
            inf_outputs_data = misc.load_pickle(args.network_outputs)
            t._inf_outputs_data = inf_outputs_data
    else:
        t.gen_inf_outputs_data(
            net=net,
            devices=devices,
            nr_loop=args.nr_loop,
            noise=args.noise
        )
    return t.compute_metric(
        lm_name=args.lm_name,
        badcase=args.badcase,
    )



