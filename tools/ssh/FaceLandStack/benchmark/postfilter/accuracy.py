import time
from collections import OrderedDict

from benchmark.dataset import postfilter
from benchmark.tester import TestUnitBase
from landstack.utils.visualization import draw, quick_draw
from landstack.utils import misc
from landstack.train import env
import itertools
import os
import cv2
import numpy as np

class TestUnit(TestUnitBase):
    def __init__(self, caches):
        name = "Accuracy Tester ({}-{})".format(time.strftime("%x"), time.strftime("%X"))
        super(TestUnit, self).__init__(name, caches)

    def gen_kv_dict_all(self, val_keys):
        if self._kv_dict_all is None:
            val_ds = postfilter.PostImage(keys=val_keys)
            self._kv_dict_all = val_ds.load()
            self._caches.setdefault('kv_dict_all', self._kv_dict_all)
        else:
            print("Using cached data")

        all_in_kv_dict = OrderedDict()
        for k, v in self._kv_dict_all.items():
            for kk, vv in v.items():
                all_in_kv_dict.setdefault(kk, []).append(vv)
        for k, v in all_in_kv_dict.items():
            self._kv_dict_all.setdefault('all-in', OrderedDict())[k] = np.concatenate(v)

        return self._kv_dict_all

    def gen_inf_outputs_data(self, devices, net):
        assert self._kv_dict_all, 'Please generate kv_dict_all first'
        inf_func = env.compile_inf_func(net.outputs, devices=devices)
        self._inf_outputs_data = OrderedDict()
        for k, v in self._kv_dict_all.items():
            self._inf_outputs_data.setdefault(k, env.forward(inf_func, v))
        return self._inf_outputs_data

    def compute_metric(self, badcase, pr, out_root='/tmp/postfilter.accuracy'):
        assert self._kv_dict_all and self._inf_outputs_data, 'Please gen kv_dict_all and inf_outputs_data first'
        res_dict = OrderedDict()
        if out_root != 'None':
            misc.ensure_dir(out_root, erase=True)

        for key in self._kv_dict_all.keys():
            gt = self._kv_dict_all[key]['postfilter']
            probs = list(self._inf_outputs_data[key].values())[0].flatten()

            fpr, tpr, thresholds, auc = misc.compute_roc(probs, gt)
            if (pr is None):
                pr_list = ['0.01']
            else:
                pr_list = list(pr.split(','))

            for PR_raw in pr_list:
                PR = float(PR_raw)
                strict_1 = misc.find_tpr(fpr, tpr, thresholds, 1-PR)
                no_strict_1 = misc.find_fpr(fpr, tpr, thresholds, PR)
                ps = [
                    ['strict1(fpr/tpr/thr)', *strict_1],
                    ['no_strict1(fpr/tpr/thr)', *no_strict_1],
                ]
                res_dict0 = OrderedDict([
                    (v[0], '{:.3f}/{:.3f}/{:.3f}'.format(*v[1:])) for v in ps
                ])
                res_dict0['auc'] = '{:.4g}'.format(auc)
                eps=1e-7
                val_loss = -(gt * np.log(np.maximum(probs,eps)) + (1-gt) * np.log(np.maximum(1-probs,eps))).mean()
                res_dict0['val_loss'] = '{:.6f}'.format(val_loss)
                res_dict0['#pos'] = np.sum(gt)
                res_dict0['#neg'] = len(gt) - np.sum(gt)
                res_dict[key+str(PR)] = res_dict0

                if badcase:
                    nori_id = self._kv_dict_all[key]['nori_id']
                    img_out = self._kv_dict_all[key]['img']
                    ds_save_root = os.path.join(out_root, key+str(PR))
                    misc.ensure_dir(ds_save_root)

                    # draw pr curve
                    title='auc:{:.4g}'.format(auc)
                    fig_save_file = os.path.join(ds_save_root, 'curve.png')
                    quick_draw(fpr, tpr, title, 'fpr', 'tpr', fig_save_file)

                    # save details data
                    save_txt_file = os.path.join(ds_save_root, 'pr.txt')
                    with open(save_txt_file, 'w') as f:
                        f.write('tpr\tfpr\tthreshold\n')
                        for p0, r0, t0 in zip(tpr, fpr, thresholds):
                            f.write('{:.8f}\t{:.8f}\t{:.8f}\n'.format(p0, r0, t0))
                    f.close()

                    # save badcase
                    for ps0 in ps:
                        print(ps0)
                        p0, fpr0, tpr0, thr0 = ps0
                        p0 = p0.split('1')[0] # strip(fpr/tpr/thr)

                        pr_save_root = os.path.join(ds_save_root, '{}-fpr{:.3g}-tpr{:.4g}-thr{:.8g}'.format(p0, fpr0, tpr0, thr0))
                        fp_root = os.path.join(pr_save_root, 'fp')
                        fn_root = os.path.join(pr_save_root, 'fn')
                        misc.ensure_dir(fn_root, fp_root)

                        pred = probs >= thr0
                        fp = pred * (1 - gt)
                        fn = (1-pred)*gt

                        for j in range(len(img_out)):
                            if fp[j]:
                                img_copy2 = draw(img_out[j], text='0-prob_{:.3f}'.format(probs[j]), font_scale=2, canvas_height_width=(480, 480), wait=False, show=False)
                                save_file = os.path.join(fp_root, '{}-{}-{:.5g}.jpg'.format(j, nori_id[j], probs[j]))
                                cv2.imwrite(save_file, img_copy2)
                            if fn[j]:
                                img_copy2 = draw(img_out[j], text='1-prob_{:.3f}'.format(probs[j]), font_scale=2, canvas_height_width=(480, 480),wait=False, show=False)
                                save_file = os.path.join(fn_root, '{}-{}-{:.5g}.jpg'.format(j, nori_id[j], probs[j]))
                                cv2.imwrite(save_file, img_copy2)

        return res_dict

def main(model, devices, args_input, caches):
    """args_input is a list of rest arguments"""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-k', '--keys', type=str, default=None)
    parser.add_argument('--components', type=str, default=None)
    parser.add_argument('--lm_name', default='prob')
    parser.add_argument('--badcase', action='store_true')
    parser.add_argument('--norm', choices=['geom', 'pupil', 'area'], default='geom')
    parser.add_argument('--is_color', action='store_true')
    parser.add_argument('--on-mobile', action='store_true', help='test acc on mobile')
    parser.add_argument('--profile', default=None)
    parser.add_argument('--fpr', default=None)
    parser.add_argument('--debug', action='store_true', help='enable -c dbg')
    parser.add_argument('--network-outputs', default='/tmp/network_outputs.pkl')

    args, unknownargs = parser.parse_known_args(args_input)

    # params
    val_keys = args.keys and args.keys.split(',')

    # load model
    net = misc.load_network_and_extract(model, args.lm_name)

    # init TestUnit
    t = TestUnit(caches=caches)
    t.gen_kv_dict_all(
        val_keys=val_keys,
    )

    t.gen_inf_outputs_data(
        net=net,
        devices=devices,
    )
    return t.compute_metric(
        badcase=args.badcase,
        pr=args.fpr
    )



