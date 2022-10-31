import time
from collections import OrderedDict

from benchmark.dataset import GazeLockImage
from benchmark.tester import TestUnitBase
from landstack.utils.visualization import draw, montage
from landstack.utils import misc
from landstack.train import env
import numpy as np
import os
import cv2


class TestUnit(TestUnitBase):
    def __init__(self, caches):
        name = "lock.thr Tester ({}-{})".format(time.strftime("%x"), time.strftime("%X"))
        super(TestUnit, self).__init__(name, caches)

    def gen_kv_dict_all(self, width_height, val_keys, is_color, badcase):
        if self._kv_dict_all is None:
            val_ds = GazeLockImage(keys=val_keys)
            self._kv_dict_all = val_ds.load(
                width_height=width_height,
                is_color=is_color,
                # n_samples=2000,
            )

            all_in_kv_dict = OrderedDict()
            for k, v in self._kv_dict_all.items():
                for kk, vv in v.items():
                    all_in_kv_dict.setdefault(kk, []).append(vv)
            for k, v in all_in_kv_dict.items():
                self._kv_dict_all.setdefault('all-in', OrderedDict())[k] = np.concatenate(v)

            # read raw info
            if badcase:
                kv_dict_raw = val_ds.load(is_color=True, align=False)
                for k, v in kv_dict_raw.items():
                    self._kv_dict_all[k]['img_raw'] = v['img_patch']
                    self._kv_dict_all[k]['lm_raw'] = v['label']

            self._caches.setdefault('kv_dict_all', self._kv_dict_all)
        else:
            print("Using cached data")
        return self._kv_dict_all

    def gen_inf_outputs_data(self, net, devices, badcase):
        assert self._kv_dict_all, 'Please generate kv_dict_all first'
        inf_func = env.compile_inf_func(list(net.outputs), devices=devices)

        self._inf_outputs_data = OrderedDict()
        for k, v in self._kv_dict_all.items():
            self._inf_outputs_data.setdefault(k, env.forward(inf_func, v, batch_size=1024))
        return self._inf_outputs_data

    def compute_metric(self, lm_name, thr, badcase, out_root='/tmp/lock.thr'):
        assert self._kv_dict_all and self._inf_outputs_data, 'Please gen kv_dict_all and inf_outputs_data first'
        res_dict = OrderedDict()
        if badcase:
            misc.ensure_dir(out_root, erase=True)

        loss_all = []
        for key in self._kv_dict_all.keys():
            gt = self._kv_dict_all[key]['lock'].flatten()
            probs =  self._inf_outputs_data[key][lm_name].flatten()
            pred = probs >= thr
            acc = (pred == gt).mean()

            fn = (1-pred) * gt # predict open actually close
            fp = pred * (1-gt) # predict close actually open
            close_fpr = fn.sum() / gt.sum()
            open_fpr = fp.sum() / (1-gt).sum()
            fp = (pred * (1-gt)) # predict close actually open
            res_dict0 = OrderedDict([
                ('lock_fpr', close_fpr),
                ('no_lock_fpr', open_fpr),
                ('acc', acc),
                ('thr', thr),
                ('#lock(1)', gt.sum()),
                ('#no_lock(0)', len(gt)-gt.sum())
            ])
            res_dict[key] = res_dict0
            if key != 'all-in':
                loss_all.extend([acc])

            if badcase and key != 'all-in':
                nori_id = self._kv_dict_all[key]['nori_id']
                img_out = self._kv_dict_all[key]['img_patch']
                lm_out = self._kv_dict_all[key]['label']
                img_raw = self._kv_dict_all[key]['img_raw']
                ds_save_root = os.path.join(out_root, key)
                fp_root = os.path.join(ds_save_root, 'thr{:.3f}-predict_lock_actually_no_lock'.format(thr))
                fn_root = os.path.join(ds_save_root, 'thr{:.3f}-predict_no_lock_actually_lock'.format(thr))
                misc.ensure_dir(fp_root)
                misc.ensure_dir(fn_root)

                for j in range(len(img_raw)):
                    if fp[j]:
                        img_copy10 = draw(img_raw[j], text='0', canvas_height_width=(320, 480), wait=False, show=False)
                        img_copy11 = draw(misc.ensure_hwc(img_raw[j])[:, ::-1, :], text='0', canvas_height_width=(320, 480), wait=False, show=False)
                        img_copy2 = draw(img_out[j], lm_out[j], canvas_height_width=(320, 480), wait=False, show=False)
                        img_copy = montage([img_copy10, img_copy11, img_copy2], nrow=3, ncol=1, canvas_height_width=(640, 480))
                        save_file = os.path.join(fp_root, '{}-{}-{:.5g}.jpg'.format(j, nori_id[j], probs[j]))
                        cv2.imwrite(save_file, img_copy)
                    if fn[j]:
                        img_copy10 = draw(img_raw[j], text='1', canvas_height_width=(320, 480), wait=False, show=False)
                        img_copy11 = draw(misc.ensure_hwc(img_raw[j])[:, ::-1, :], text='1', canvas_height_width=(320, 480), wait=False, show=False)
                        img_copy2 = draw(img_out[j], lm_out[j], canvas_height_width=(320, 480), wait=False, show=False)
                        img_copy = montage([img_copy10, img_copy11, img_copy2], nrow=3, ncol=1, canvas_height_width=(640, 480))
                        save_file = os.path.join(fn_root, '{}-{}-{:.5g}.jpg'.format(j, nori_id[j], probs[j]))
                        cv2.imwrite(save_file, img_copy)
        print("avg acc: {:.4f}".format(np.mean(loss_all)))
        return res_dict

def main(model, devices, args_input, caches):
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-k', '--keys', type=str, default=None)
    parser.add_argument('--lm_name', default='s-pred-lock-patch')
    parser.add_argument('--badcase', action='store_true', help='whether to log ')
    parser.add_argument('--thr', type=float, help='strict threshold', required=True)
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
        badcase=args.badcase,
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
            badcase=args.badcase,
        )
    return t.compute_metric(
        lm_name=args.lm_name,
        thr = args.thr,
        badcase=args.badcase,
    )



