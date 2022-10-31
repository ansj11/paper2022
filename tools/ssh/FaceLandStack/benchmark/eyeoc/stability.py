import time
from collections import OrderedDict

from benchmark.dataset import EyeOCImage
from benchmark.tester import TestUnitBase
from landstack.utils import misc, transforms, augment
from landstack.train import env
import numpy as np
import tqdm
from landstack.utils.visualization import draw, quick_draw, montage
from IPython import embed


class TestUnit(TestUnitBase):
    def __init__(self, caches):
        name = "eyeoc.stability Tester ({}-{})".format(time.strftime("%x"), time.strftime("%X"))
        super(TestUnit, self).__init__(name, caches)
        self.n_points = 81

    def gen_kv_dict_all(self, image_shape, val_keys, is_color, norm_type, badcase):
        if self._kv_dict_all is None:
            val_ds = EyeOCImage(keys=val_keys)
            self._kv_dict_all = val_ds.load(
                align=False,
                is_color=is_color,
            )
            self._caches.setdefault('kv_dict_all', self._kv_dict_all)
        else:
            print("Using cached data")
        return self._kv_dict_all

    def compute_metric(self, net, lm_name):
        res_dict = OrderedDict()
        mean_face_file = '/unsullied/sharefs/_research_facelm/Isilon-datashare/baseimg/extra/mean_face_81.pkl'
        mean_face = misc.load_pickle(mean_face_file)
        inf_func = env.compile_inf_func(net.outputs)


        loss_all_x, loss_all_y = [], []
        for key in self._kv_dict_all.keys():

            # compute trans_x
            trace_kv_dict = OrderedDict()
            for i in tqdm.tqdm(range(-15, 15)):
                aug = augment.AugmentBase(
                    mean_face=mean_face, image_shape=(112, 112),
                    transforms_pre=transforms.Compose([
                        transforms.Jitter(
                            scale=1.0,
                            rotate=0.0,
                            trans_x=0.01*i,
                            trans_y=0,
                            rng=None,
                        )
                    ]),
                    transforms_post=transforms.Compose([
                        transforms.ConvertToGray(),
                        transforms.HWC2CHW(),
                        transforms.NormalizeLandmark(image_shape=(112, 112))
                    ])
                )

                img = self._kv_dict_all[key]['img']
                eyeoc = self._kv_dict_all[key]['eyeoc']
                label = self._kv_dict_all[key]['label']
                aligned_dict = OrderedDict()
                for img0, eyeoc0, label0 in zip(img, eyeoc, label):
                    img_out, lm_out, _ = aug(img0, label0, label0.copy())
                    aligned_dict.setdefault('img', []).append(img_out)
                    aligned_dict.setdefault('label', []).append(lm_out)
                    aligned_dict.setdefault('eyeoc', []).append(eyeoc0)
                aligned_dict['img'] = np.uint8(aligned_dict['img'])
                aligned_dict['label'] = np.float32(aligned_dict['label'])
                aligned_dict['eyeoc'] = np.float32(aligned_dict['eyeoc'])

                outputs_data = env.forward(inf_func, aligned_dict)
                eyeoc_data = outputs_data[lm_name]

                trace_kv_dict['trans_x_{}'.format(i)] = dict(
                    img=aligned_dict['img'],
                    lm = np.float32(aligned_dict['label'])*112,
                    eyeoc=eyeoc_data
                )

            eyeoc_batch = []
            for k, v in trace_kv_dict.items():
                eyeoc_batch.append(v['eyeoc'])
            eyeoc_batch = np.float32(eyeoc_batch).transpose(1, 0, 2)

            stds = []
            for eyeoc0 in eyeoc_batch:
                eyeoc_ref = eyeoc0[15]
                eyeoc_diff = eyeoc0 - eyeoc_ref
                eyeoc_diff_avg = eyeoc_diff.mean(axis=1, keepdims=True)
                eyeoc_diff = np.concatenate([eyeoc_diff, eyeoc_diff_avg], axis=1)
                std = eyeoc_diff.std(axis=0)
                stds.append(std)
            std0 = np.mean(stds, axis=0)
            res_dict0 = OrderedDict()
            res_dict0['trans_x_left'] = '{:.3f}'.format(std0[0])
            res_dict0['trans_x_right'] = '{:.3f}'.format(std0[1])
            res_dict0['trans_x_avg'] = '{:.3f}'.format(std0[2])
            if key != 'all-in':
                loss_all_x.append(std0[2])

            # compute trans_y
            trace_kv_dict = OrderedDict()
            for i in tqdm.tqdm(range(-15, 15)):
                aug = augment.AugmentBase(
                    mean_face=mean_face, image_shape=(112, 112),
                    transforms_pre=transforms.Compose([
                        transforms.Jitter(
                            scale=1.0,
                            rotate=0.0,
                            trans_x=0.0,
                            trans_y=0.01*i,
                            rng=None,
                        )
                    ]),
                    transforms_post=transforms.Compose([
                        transforms.ConvertToGray(),
                        transforms.HWC2CHW(),
                        transforms.NormalizeLandmark(image_shape=(112, 112))
                    ])
                )

                img = self._kv_dict_all[key]['img']
                eyeoc = self._kv_dict_all[key]['eyeoc']
                label = self._kv_dict_all[key]['label']
                aligned_dict = OrderedDict()
                for img0, eyeoc0, label0 in zip(img, eyeoc, label):
                    img_out, lm_out, _ = aug(img0, label0, label0.copy())
                    aligned_dict.setdefault('img', []).append(img_out)
                    aligned_dict.setdefault('label', []).append(lm_out)
                    aligned_dict.setdefault('eyeoc', []).append(eyeoc0)
                aligned_dict['img'] = np.uint8(aligned_dict['img'])
                aligned_dict['label'] = np.float32(aligned_dict['label'])
                aligned_dict['eyeoc'] = np.float32(aligned_dict['eyeoc'])

                outputs_data = env.forward(inf_func, aligned_dict)
                eyeoc_data = outputs_data[lm_name]

                trace_kv_dict['trans_x_{}'.format(i)] = dict(
                    img=aligned_dict['img'],
                    lm = np.float32(aligned_dict['label'])*112,
                    eyeoc=eyeoc_data
                )

            eyeoc_batch = []
            for k, v in trace_kv_dict.items():
                eyeoc_batch.append(v['eyeoc'])
            eyeoc_batch = np.float32(eyeoc_batch).transpose(1, 0, 2)

            stds = []
            for eyeoc0 in eyeoc_batch:
                eyeoc_ref = eyeoc0[15]
                eyeoc_diff = eyeoc0 - eyeoc_ref
                eyeoc_diff_avg = eyeoc_diff.mean(axis=1, keepdims=True)
                eyeoc_diff = np.concatenate([eyeoc_diff, eyeoc_diff_avg], axis=1)
                std = eyeoc_diff.std(axis=0)
                stds.append(std)
            std0 = np.mean(stds, axis=0)
            res_dict0['trans_y_left'] = '{:.3f}'.format(std0[0])
            res_dict0['trans_y_right'] = '{:.3f}'.format(std0[1])
            res_dict0['trans_y_avg'] = '{:.3f}'.format(std0[2])

            res_dict[key] = res_dict0
            if key != 'all-in':
                loss_all_y.append(std0[2])

        print('avg loss trans_x: {:.3f},  trans_y: {:.3f}'.format(np.mean(loss_all_x), np.mean(loss_all_y)))
        return res_dict

def main(model, devices, args_input, caches):
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-k', '--keys', type=str, default=None)
    parser.add_argument('--lm_name', default='s-pred-eyeoc')
    parser.add_argument('--badcase', action='store_true', help='whether to log ')
    parser.add_argument('--patch_name', default='left_eye_patch,right_eye_patch')
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
    assert len(args.lm_name.split(',')) == 1, args.lm_name
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

    return t.compute_metric(
        net = net,
        lm_name=args.lm_name,
    )



