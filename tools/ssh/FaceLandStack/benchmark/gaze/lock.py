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
        self.n_points = 81

    def gen_kv_dict_all(self, image_shape, val_keys, is_color, norm_type, badcase):
        if self._kv_dict_all is None:
            val_ds = GazeLockImage(keys=val_keys)
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

    def gen_inf_outputs_data(self, net, devices, patch_name, badcase):
        assert self._kv_dict_all, 'Please generate kv_dict_all first'
        if badcase:
            warp_imgs = misc.find_opr_by_name(net.outputs, patch_name.split(','))
            inf_func = env.compile_inf_func(list(net.outputs)+warp_imgs, devices=devices)
        else:
            inf_func = env.compile_inf_func(list(net.outputs), devices=devices)
        self._inf_outputs_data = OrderedDict()
        for k, v in self._kv_dict_all.items():
            self._inf_outputs_data.setdefault(k, env.forward(inf_func, v))
        return self._inf_outputs_data

    def compute_metric(self, lm_name, badcase, patch_name, out_root='/tmp/lock.accuracy'):
        assert self._kv_dict_all and self._inf_outputs_data, 'Please gen kv_dict_all and inf_outputs_data first'
        res_dict = OrderedDict()
        if badcase:
            misc.ensure_dir(out_root, erase=True)

        loss_all = []
        for key in self._kv_dict_all.keys():
            gt = self._kv_dict_all[key]['lock'].flatten()
            sig_key, pred_key = lm_name.split(',')
            lm_pred = self._inf_outputs_data[key][pred_key]
            probs =  self._inf_outputs_data[key][sig_key]
            assert probs.shape[1] == 2, probs.shape
            left_eye = probs[:, 0]
            right_eye = probs[:, 1]
            probs = probs.reshape(len(gt), -1).mean(axis=1).flatten() # compatible for nx2 and nx1 tensor

            fpr, tpr, thresholds, auc = misc.compute_roc(probs, gt)
            strict_1 = misc.find_tpr(fpr, tpr, thresholds, 0.99)
            no_strict_1 = misc.find_fpr(fpr, tpr, thresholds, 0.01)
            ps = [
                ['close(fpr/tpr/thr)', *strict_1],
                ['open(fpr/tpr/thr)', *no_strict_1],
            ]
            res_dict0 = OrderedDict([
                (v[0], '{:.3f}/{:.3f}/{:.3f}'.format(*v[1:])) for v in ps
            ])
            res_dict0['auc'] = '{:.4g}'.format(auc)
            val_loss_left = (-(gt* np.log(left_eye+ 1e-5) + (1-gt)*np.log(1-left_eye+1e-5))).mean()
            val_loss_right = (-(gt* np.log(right_eye+ 1e-5) + (1-gt)*np.log(1-right_eye+1e-5))).mean()
            val_loss = (val_loss_left + val_loss_right) / 2.0
            res_dict0['val_loss'] = val_loss
            res_dict0['val_loss_left'] = val_loss_left
            res_dict0['val_loss_right'] = val_loss_right
            res_dict0['#lock(1)'] = np.sum(gt)
            res_dict0['#non-lock(0)'] = len(gt) - np.sum(gt)
            res_dict0['#total'] = len(gt)
            res_dict[key] = res_dict0
            if key != 'all-in':
                loss_all.append(val_loss)

            if badcase and 'img_raw' in self._kv_dict_all[key]:
                left_eye_patch_name, right_eye_patch_name = patch_name.split(',')
                left_eye_patch = (self._inf_outputs_data[key][left_eye_patch_name] * 128 + 128).astype(np.uint8).transpose(0, 2, 3,1)
                right_eye_patch = (self._inf_outputs_data[key][right_eye_patch_name] * 128 + 128).astype(np.uint8).transpose(0, 2, 3, 1)
                nori_id = self._kv_dict_all[key]['nori_id']
                img_out = self._kv_dict_all[key]['img']
                lm_out = self._kv_dict_all[key]['label']
                img_raw = self._kv_dict_all[key]['img_raw']
                ds_save_root = os.path.join(out_root, key)
                misc.ensure_dir(ds_save_root)

                # draw pr curve
                title='auc:{:.4g}'.format(auc)
                fig_save_file = os.path.join(ds_save_root, 'curve.png')
                quick_draw(fpr, tpr, title, 'fpr', 'tpr', fig_save_file)

                # save details data
                save_txt_file = os.path.join(ds_save_root, 'pr.txt')
                with open(save_txt_file, 'w') as f:
                    f.write('fpr\ttpr\tthreshold\n')
                    for p0, r0, t0 in zip(tpr, fpr, thresholds):
                        f.write('{:.8f}\t{:.8f}\t{:.8f}\n'.format(p0, r0, t0))
                f.close()

                # save badcase
                for ps0 in ps:
                    print(ps0)
                    p0, fpr0, tpr0, thr0 = ps0
                    p0 = p0.split('(')[0] # strip(fpr/tpr/thr)

                    pr_save_root = os.path.join(ds_save_root, '{}-fpr{:.3g}-tpr{:.4g}-thr{:.8g}'.format(p0, fpr0, tpr0, thr0))
                    fp_root = os.path.join(pr_save_root, 'fp')
                    fn_root = os.path.join(pr_save_root, 'fn')
                    misc.ensure_dir(fn_root, fp_root)

                    pred = probs >= thr0
                    fp = pred * (1 - gt)
                    fn = (1-pred)*gt

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
        print('avg loss: {:.3f}'.format(np.mean(loss_all)))
        return res_dict

def main(model, devices, args_input, caches):
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-k', '--keys', type=str, default=None)
    parser.add_argument('--lm_name', default='s-pred-lock,pred')
    parser.add_argument('--badcase', action='store_true', help='whether to log ')
    parser.add_argument('--patch_name', default='lock:left_eye_patch,lock:right_eye_patch')
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
            patch_name=args.patch_name,
            badcase=args.badcase,

        )
    return t.compute_metric(
        lm_name=args.lm_name,
        badcase=args.badcase,
        patch_name=args.patch_name,
    )



