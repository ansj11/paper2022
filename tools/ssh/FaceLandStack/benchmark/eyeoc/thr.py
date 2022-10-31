import time
from collections import OrderedDict

from benchmark.dataset import EyeOCImage
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
        name = "eyeoc.thr Tester ({}-{})".format(time.strftime("%x"), time.strftime("%X"))
        super(TestUnit, self).__init__(name, caches)
        self.n_points = 81

    def gen_kv_dict_all(self, image_shape, val_keys, is_color, norm_type, badcase):
        if self._kv_dict_all is None:
            val_ds = EyeOCImage(keys=val_keys)
            self._kv_dict_all = val_ds.load(
                align=True,
                norm_type=norm_type,
                image_shape=image_shape,
                is_color=is_color,
            )

            # read raw info
            if badcase:
                kv_dict_raw = val_ds.load()
                for k, v in kv_dict_raw.items():
                    self._kv_dict_all[k]['img_raw'] = v['img']
                    self._kv_dict_all[k]['lm_raw'] = v['label']

            self._caches.setdefault('kv_dict_all', self._kv_dict_all)
        else:
            print("Using cached data")
        return self._kv_dict_all

    def gen_inf_outputs_data(self, net, devices, badcase):
        assert self._kv_dict_all, 'Please generate kv_dict_all first'
        if badcase:
            warp_imgs = misc.find_opr_by_name(net.outputs, ['eyeoc:left_eye_patch', 'eyeoc:right_eye_patch'])
            inf_func = env.compile_inf_func(list(net.outputs)+warp_imgs, devices=devices)
        else:
            inf_func = env.compile_inf_func(list(net.outputs), devices=devices)

        self._inf_outputs_data = OrderedDict()
        for k, v in self._kv_dict_all.items():
            self._inf_outputs_data.setdefault(k, env.forward(inf_func, v))
        return self._inf_outputs_data

    def compute_metric(self, lm_name, thr, badcase, out_root='/tmp/eyeoc.thr'):
        assert self._kv_dict_all and self._inf_outputs_data, 'Please gen kv_dict_all and inf_outputs_data first'
        res_dict = OrderedDict()
        if badcase:
            misc.ensure_dir(out_root, erase=True)

        for key in self._kv_dict_all.keys():
            gt = self._kv_dict_all[key]['eyeoc'].flatten()
            sig_key, pred_key = lm_name.split(',')
            lm_pred = self._inf_outputs_data[key][pred_key]
            probs =  self._inf_outputs_data[key][sig_key]
            left_eye = probs[:, 0]
            right_eye = probs[:, 1]
            probs = probs.reshape(len(gt), -1).mean(axis=1).flatten() # compatible for nx2 and nx1 tensor
            pred = probs >= thr
            left_pred = left_eye.flatten() >= thr
            right_pred = right_eye.flatten() >= thr
            left_eye_acc = (left_pred == gt).mean()
            right_eye_acc = (right_pred == gt).mean()

            fn = (1-pred) * gt # predict open actually close
            fp = pred * (1-gt) # predict close actually open
            close_fpr = fn.sum() / gt.sum()
            open_fpr = fp.sum() / (1-gt).sum()
            fp = (pred * (1-gt)) # predict close actually open
            res_dict0 = OrderedDict([
                ('close_fpr', close_fpr),
                ('open_fpr', open_fpr),
                ('left_acc', left_eye_acc),
                ('right_acc', right_eye_acc),
                ('thr', thr),
                ('#close(1)', gt.sum()),
                ('#open(0)', len(gt)-gt.sum())
            ])
            res_dict[key] = res_dict0

            if badcase:
                left_eye_patch = (self._inf_outputs_data[key]['eyeoc:left_eye_patch'] * 128 + 128).astype(np.uint8).transpose(0, 2, 3,1)
                right_eye_patch = (self._inf_outputs_data[key]['eyeoc:right_eye_patch'] * 128 + 128).astype(np.uint8).transpose(0, 2, 3, 1)
                nori_id = self._kv_dict_all[key]['nori_id']
                img_out = self._kv_dict_all[key]['img']
                lm_out = self._kv_dict_all[key]['label']
                img_raw = self._kv_dict_all[key]['img_raw']
                ds_save_root = os.path.join(out_root, key)
                fp_root = os.path.join(ds_save_root, 'thr{:.3f}-predict_close_actually_open'.format(thr))
                fn_root = os.path.join(ds_save_root, 'thr{:.3f}-predict_open_actually_close'.format(thr))
                misc.ensure_dir(fp_root)
                misc.ensure_dir(fn_root)

                for j in range(len(img_raw)):
                    if fp[j]:
                        img_copy1 = draw(img_raw[j], text='0', canvas_height_width=(640, 480), wait=False, show=False)
                        img_copy2 = draw(img_out[j], lm_out[j], lm_pred[j], canvas_height_width=(640, 480), wait=False, show=False)
                        left_eye_copy = draw(left_eye_patch[j], canvas_height_width=(320, 480), text='left_eye_{:.3f}'.format(left_eye[j]), font_scale=2, wait=False, show=False)
                        right_eye_copy = draw(right_eye_patch[j], canvas_height_width=(320, 480), text='right_eye_{:.3f}'.format(right_eye[j]), font_scale=2, wait=False, show=False)
                        merge0 = montage([left_eye_copy, right_eye_copy], nrow=2, ncol=1, canvas_height_width=(640, 480))
                        img_copy = montage([img_copy1, img_copy2, merge0], nrow=1, ncol=3, canvas_height_width=(640, 1440))
                        save_file = os.path.join(fp_root, '{}-{}-{:.5g}.jpg'.format(j, nori_id[j], probs[j]))
                        cv2.imwrite(save_file, img_copy)
                    if fn[j]:
                        img_copy1 = draw(img_raw[j], text='1', canvas_height_width=(640, 480), wait=False, show=False)
                        img_copy2 = draw(img_out[j], lm_out[j], lm_pred[j], canvas_height_width=(640, 480), wait=False, show=False)
                        left_eye_copy = draw(left_eye_patch[j], canvas_height_width=(320, 480), text='left_eye_{:.3f}'.format(left_eye[j]), font_scale=2, wait=False, show=False)
                        right_eye_copy = draw(right_eye_patch[j], canvas_height_width=(320, 480), text='right_eye_{:.3f}'.format(right_eye[j]), font_scale=2, wait=False, show=False)
                        merge0 = montage([left_eye_copy, right_eye_copy], nrow=2, ncol=1, canvas_height_width=(640, 480))
                        img_copy = montage([img_copy1, img_copy2, merge0], nrow=1, ncol=3, canvas_height_width=(640, 1440))
                        save_file = os.path.join(fn_root, '{}-{}-{:.5g}.jpg'.format(j, nori_id[j], probs[j]))
                        cv2.imwrite(save_file, img_copy)
        return res_dict

def main(model, devices, args_input, caches):
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-k', '--keys', type=str, default=None)
    parser.add_argument('--lm_name', default='s-pred-eyeoc,s-pred')
    parser.add_argument('--badcase', action='store_true', help='whether to log ')
    parser.add_argument('--thr', type=float, help='strict threshold', required=True)
    parser.add_argument('--norm', choices=['geom', 'pupil', 'area'], default='geom')
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
    assert len(args.lm_name.split(',')) == 2, args.lm_name
    net = misc.load_network_and_extract(model, args.lm_name)

    # init TestUnit
    t = TestUnit(caches=caches)
    t.gen_kv_dict_all(
        val_keys=val_keys,
        image_shape=(112, 112),
        norm_type=args.norm,
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



