import time
from collections import OrderedDict

from benchmark.dataset import GazeLockImage
from benchmark.tester import TestUnitBase
from landstack.utils.visualization import draw, quick_draw, montage
from landstack.utils import misc
from landstack.train import env
from IPython import embed
import numpy as np
import os
import cv2


class TestUnit(TestUnitBase):
    def __init__(self, caches):
        name = "gaze.lock Tester ({}-{})".format(time.strftime("%x"), time.strftime("%X"))
        super(TestUnit, self).__init__(name, caches)

    def gen_kv_dict_all(self, width_height, val_keys, is_color):
        if self._kv_dict_all is None:
            val_ds = GazeLockImage(keys=val_keys)
            self._kv_dict_all = val_ds.load(
                width_height=width_height,
                is_color=is_color,
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

    def gen_inf_outputs_data(self, net, devices):
        assert self._kv_dict_all, 'Please generate kv_dict_all first'
        inf_func = env.compile_inf_func(list(net.outputs), devices=devices)
        self._inf_outputs_data = OrderedDict()
        for k, v in self._kv_dict_all.items():
            self._inf_outputs_data.setdefault(k, env.forward(inf_func, v, batch_size=1024))
        return self._inf_outputs_data

    def compute_metric(self, lm_name):
        assert self._kv_dict_all and self._inf_outputs_data, 'Please gen kv_dict_all and inf_outputs_data first'
        res_dict = OrderedDict()
        loss_all = []
        for key in self._kv_dict_all.keys():
            gt = self._kv_dict_all[key]['lock'].flatten()
            probs =  self._inf_outputs_data[key][lm_name]
            assert probs.shape[1] == 1, probs.shape
            probs = probs.flatten()

            fpr, tpr, thresholds, auc = misc.compute_roc(probs, gt)
            strict_1 = misc.find_tpr(fpr, tpr, thresholds, 0.99)
            no_strict_1 = misc.find_fpr(fpr, tpr, thresholds, 0.01)
            ps = [
                ['lock(fpr/tpr/thr)', *strict_1],
                ['no_lock(fpr/tpr/thr)', *no_strict_1],
            ]
            res_dict0 = OrderedDict([
                (v[0], '{:.3f}/{:.3f}/{:.3f}'.format(*v[1:])) for v in ps
            ])
            res_dict0['auc'] = '{:.4g}'.format(auc)
            val_loss = (-(gt* np.log(probs+ 1e-5) + (1-gt)*np.log(1-probs+1e-5))).mean()
            res_dict0['val_loss'] = val_loss
            res_dict0['#lock(1)'] = np.sum(gt)
            res_dict0['#no_lock(0)'] = len(gt) - np.sum(gt)
            res_dict0['#total'] = len(gt)
            res_dict[key] = res_dict0
            if key != 'all-in':
                loss_all.append(val_loss)

        print('avg loss: {:.3f}'.format(np.mean(loss_all)))
        return res_dict

def main(model, devices, args_input, caches):
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-k', '--keys', type=str, default=None)
    parser.add_argument('--lm_name', default='s-pred-lock-patch')
    parser.add_argument('--is_color', action='store_true')
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

    # load model
    assert len(args.lm_name.split(',')) == 1, args.lm_name
    net = misc.load_network_and_extract(model, args.lm_name)

    # init TestUnit
    t = TestUnit(caches=caches)
    t.gen_kv_dict_all(
        val_keys=val_keys,
        width_height=(48, 32),
        is_color=args.is_color,
    )

    if args.lowbit or args.on_mobile:
        command = "lsk-test gen.speed {} --lm_name {} --keys {} --no_info --network-outputs {} --sfa {} --bit_midout {} --dataset lock.accuracy".format(
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

        )
    return t.compute_metric(
        lm_name=args.lm_name,
    )



