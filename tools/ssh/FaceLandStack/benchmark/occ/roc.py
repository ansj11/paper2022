import time
from collections import OrderedDict

from benchmark.dataset import OccImage
from benchmark.tester import TestUnitBase
from landstack.utils.visualization import draw, quick_draw, montage
from landstack.utils import misc
from landstack.train import env
from IPython import embed
import os
import numpy as np
import cv2


class TestUnit(TestUnitBase):
    def __init__(self, caches):
        name = "occ.roc Tester ({}-{})".format(time.strftime("%x"), time.strftime("%X"))
        super(TestUnit, self).__init__(name, caches)
        self.n_points = 81
        self.splits = OrderedDict([
            ('contour',list(range(19))),
            ('left_eye' ,list(range(19, 28))),
            ('right_eye' ,list(range(64, 73))),
            ('eye' , list(range(19, 28)) + list(range(64, 73))),
            ('left_eyebrow' ,list(range(28, 36))),
            ('right_eyebrow' ,list(range(73, 81))),
            ('eyebrow' ,list(range(28, 36)) + list(range(73, 81))),
            ('mouth' ,list(range(36, 54))),
            ('nose' ,list(range(54, 64))),
            ('all' ,list(range(81))),
        ])
        self.point_to_component = OrderedDict()
        for k, v in self.splits.items():
            for vv in v:
                if vv not in self.point_to_component:
                    self.point_to_component.setdefault(vv, k)

    def gen_kv_dict_all(self, image_shape, val_keys, is_color, norm_type):
        if self._kv_dict_all is None:
            val_ds = OccImage(keys=val_keys)
            self._kv_dict_all = val_ds.load(
                norm_type=norm_type,
                image_shape=image_shape,
                is_color=is_color,
            )
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

    def compute_metric(self, lm_name, out_root, pr_res_out_file):
        assert self._kv_dict_all and self._inf_outputs_data, 'Please gen kv_dict_all and inf_outputs_data first'
        res_dict = OrderedDict()
        pr_res_dict = OrderedDict()
        if out_root != 'None':
            misc.ensure_dir(out_root, erase=True)
        for key in self._kv_dict_all.keys():
            img_gt = self._kv_dict_all[key]['img']
            occ_gt = self._kv_dict_all[key]['occ']
            mask_gt = self._kv_dict_all[key]['mask']

            occ_pred_sig = self._inf_outputs_data[key][lm_name[0]]
            lm_pred = self._inf_outputs_data[key][lm_name[1]].reshape(len(img_gt), -1, 2)

            for i in range(81):
                probs = occ_pred_sig[:, i][mask_gt[:, i] == 1]
                gt =  occ_gt[:, i][mask_gt[:, i] == 1]
                fpr, tpr, thresholds, auc = misc.compute_roc(probs, gt)
                occ_100 = misc.find_tpr(fpr, tpr, thresholds, 0.99)
                occ_1000 = misc.find_tpr(fpr, tpr, thresholds, 0.999)
                ps = [
                    ['fpr100', *occ_100],
                    ['fpr1000', *occ_1000],
                ]

                pr_res_dict.setdefault(key, []).append(ps)
                comp_name = '{}-{}-{}'.format(key, self.point_to_component[i], i)
                res_dict0 = OrderedDict([
                    (v[0], '{:.3f}/{:.5f}/{:.8f}'.format(*v[1:])) for v in ps
                ])
                res_dict0['auc'] = auc
                res_dict[comp_name] = res_dict0


                if out_root != 'None':
                    p0, fpr0, tpr0, thr0 = ps[0]
                    print(ps[0])

                    save_root = os.path.join(out_root, key,
                                             '{}-{}-{}-{:.3f}-{:.4f}-{:.6g}'.format(self.point_to_component[i], i, p0, fpr0, tpr0, thr0))
                    misc.ensure_dir(save_root, erase=True)

                    pred = occ_pred_sig[:, i] >= thr0
                    fp = ((pred * (1 - occ_gt[:, i])) * mask_gt[:, i]).flatten()
                    fn = (((1 - pred) * occ_gt[:, i]) * mask_gt[:, i]).flatten()

                    fp_imgs, fn_imgs = [], []
                    for j in range(len(img_gt)):
                        mark = [''] * 81
                        mark[i] = 'X'
                        if fp[j]:
                            fp_imgs.append(draw(img_gt[j], lm_pred[j], lm_pred[j][occ_gt[j]==1], mark=mark, show=False, wait=False))
                        if fn[j]:
                            fn_imgs.append(draw(img_gt[j], lm_pred[j], lm_pred[j][occ_gt[j]==1], mark=mark, show=False, wait=False))
                    if len(fp_imgs) > 0:
                        fp_img_compy = montage(fp_imgs, 7, 7, (2100, 2100))
                        cv2.imwrite(os.path.join(save_root, 'fp.png'), fp_img_compy)

                    if len(fn_imgs) > 0:
                        fn_img_compy = montage(fn_imgs, 7, 7, (2100, 2100))
                        cv2.imwrite(os.path.join(save_root, 'fn.png'), fn_img_compy)
                    title='auc:{:.4g}'.format(auc)
                    fig_save_file = os.path.join(save_root, 'curve.png')
                    quick_draw(fpr, tpr, title, 'fpr', 'tpr', fig_save_file)


            probs = occ_pred_sig[mask_gt==1]
            gt = occ_gt[mask_gt == 1]
            fpr, tpr, thresholds, auc = misc.compute_roc(probs, gt)
            occ_100 = misc.find_tpr(fpr, tpr, thresholds, 0.99)
            occ_1000 = misc.find_tpr(fpr, tpr, thresholds, 0.999)
            ps = [
                ['fpr100', *occ_100],
                ['fpr1000', *occ_1000],
            ]
            res_dict0 = OrderedDict([
                (v[0], '{:.3f}/{:.5f}/{:.8f}'.format(*v[1:])) for v in ps
            ])
            res_dict0['auc'] = auc
            res_dict[key] = res_dict0

            if out_root != 'None':
                save_root = os.path.join(out_root, key)
                title='auc:{:.4g}, {}:{:.3f}/{:.5f}/{:.8f}, {}:{:.3f}/{:.5f}/{:.8f}'.format(auc, *ps[0], *ps[1])
                fig_save_file = os.path.join(save_root, 'curve-all.png')
                quick_draw(fpr, tpr, title, 'fpr', 'tpr', fig_save_file)
        if pr_res_out_file is not None:
            misc.dump_pickle(pr_res_dict, pr_res_out_file)
        return res_dict

def main(model, devices, args_input, caches):
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-k', '--keys', type=str, default='validation-valid')
    parser.add_argument('--lm_name', default='s-pred-occ-sig,s-pred')
    parser.add_argument('--occ-out-root', default='/tmp/occ.roc')
    parser.add_argument('-o', '--out', default=None)
    parser.add_argument('--norm', choices=['geom', 'pupil', 'area'], default='geom')
    parser.add_argument('--is-color', action='store_true')
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
    net = misc.load_network_and_extract(model, args.lm_name.split(','))

    # init TestUnit
    t = TestUnit(caches=caches)
    t.gen_kv_dict_all(
        val_keys=val_keys,
        image_shape=(112, 112),
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
    return t.compute_metric(
        out_root=args.occ_out_root,
        lm_name=args.lm_name.split(','),
        pr_res_out_file=args.out,
    )



