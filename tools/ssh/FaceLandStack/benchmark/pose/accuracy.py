import time
from collections import OrderedDict
from benchmark.dataset import pose
from benchmark.tester import TestUnitBase
from landstack.utils.visualization import draw
from landstack.utils import misc
from landstack.train import env
from IPython import embed
import os
import math
import numpy as np

class TestUnit(TestUnitBase):
    def __init__(self, caches):
        name = "Accuracy Tester ({}-{})".format(time.strftime("%x"), time.strftime("%X"))
        super(TestUnit, self).__init__(name, caches)

    def gen_kv_dict_all(self, image_shape, val_keys, is_color, norm_type):
        if self._kv_dict_all is None:
            val_ds = pose.PoseImage(keys=val_keys)
            self._kv_dict_all = val_ds.load(
                align=True,
                norm_type=norm_type,
                image_shape=image_shape,
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

    def gen_inf_outputs_data(self, devices, net):
        assert self._kv_dict_all, 'Please generate kv_dict_all first'
        inf_func = env.compile_inf_func(net.outputs, devices=devices)
        self._inf_outputs_data = OrderedDict()
        for k, v in self._kv_dict_all.items():
            self._inf_outputs_data.setdefault(k, env.forward(inf_func, v))
        return self._inf_outputs_data

    def compute_metric(self):
        assert self._kv_dict_all and self._inf_outputs_data, 'Please gen kv_dict_all and inf_outputs_data first'
        res_dict = OrderedDict()
        tau = math.pi * 2
        for k in self._kv_dict_all.keys():
            pose_gt = self._kv_dict_all[k]['pose'].reshape(-1,2)
            pose_pred = self._inf_outputs_data[k]
            pose_pred = list(pose_pred.values())[0] if isinstance(pose_pred, dict) else pose_pred
            if pose_pred.shape[1] == 3:
                pose_pred = pose_pred[:, 1:]
            pose_pred = pose_pred.reshape(-1,2)

            pose_gt = pose_gt%tau
            pose_pred = pose_pred%tau
            t = pose_gt-pose_pred
            pitchl1, yawl1, pitchl2, yawl2 = [],[],[],[]
            for i in range(len(pose_gt)):
                pitchl1.append(min(abs(t[i][0]), abs(t[i][0]+tau), abs(t[i][0]-tau)))
                yawl1.append(min(abs(t[i][1]), abs(t[i][1]+tau), abs(t[i][1]-tau)))
                pitchl2.append(min((t[i][0])**2, (t[i][0]+tau)**2, (t[i][0]-tau)**2))
                yawl2.append(min((t[i][1])**2, (t[i][1]+tau)**2, (t[i][1]-tau)**2))
            pitchl1_avg = np.float32(pitchl1).mean()
            yawl1_avg = np.float32(yawl1).mean()
            pitchl2_avg = np.float32(pitchl2).mean()
            yawl2_avg = np.float32(yawl2).mean()
            res_dict0 = dict(
                pitch_l1 = pitchl1_avg,
                yaw_l1 = yawl1_avg,
                l1 = pitchl1_avg + yawl1_avg,
                pitch_l2 = pitchl2_avg,
                yaw_l2 = yawl2_avg,
                l2 = pitchl2_avg + yawl2_avg
            )
            res_dict[k] = res_dict0

        return res_dict

def main(model, devices, args_input, caches):
    """args_input is a list of rest arguments"""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-k', '--keys', type=str, default=None)
    parser.add_argument('--components', type=str, default=None)
    parser.add_argument('--lm_name', default='s-pose')
    parser.add_argument('-s', '--image_shape', default='112,112')
    parser.add_argument('--norm', choices=['geom', 'pupil', 'area'], default='geom')
    parser.add_argument('--is_color', action='store_true')
    parser.add_argument('--lowbit', action='store_true')
    parser.add_argument('--on-mobile', action='store_true', help='test acc on mobile')
    parser.add_argument('--profile', default=None)
    parser.add_argument('--debug', action='store_true', help='enable -c dbg')
    parser.add_argument('--network-outputs', default='/tmp/network_outputs.pkl')
    parser.add_argument('--sfa', default=7)
    parser.add_argument('--bit_midout', default=16)

    args, unknownargs = parser.parse_known_args(args_input)

    # params
    val_keys = args.keys and args.keys.split(',')
    args.image_shape = tuple(map(int, args.image_shape.split(',')))
    assert len(args.image_shape) == 2, args.image_shape

    # load model
    net = misc.load_network_and_extract(model, args.lm_name)

    # init TestUnit
    t = TestUnit(caches=caches)
    t.gen_kv_dict_all(
        val_keys=val_keys,
        image_shape=args.image_shape,
        norm_type=args.norm,
        is_color=args.is_color,
    )

    if args.lowbit or args.on_mobile:
        command = "lsk-test gen.speed {} --lm_name {} --keys {} --no_info --network-outputs {} --sfa {} --bit_midout {}".format(
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
    return t.compute_metric()



