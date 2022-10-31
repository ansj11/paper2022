from collections import OrderedDict
import numpy as np
import cv2

import pickle as pkl
import nori2 as nori
from landstack.utils import misc, transforms, augment

class DenseImage(object):
    def __init__(self, nori_file, keys, name='dld'):
        self.nori_file = nori_file
        self.name = name
        self.keys = [keys] if not isinstance(keys, (tuple, list)) else keys

    def load(self, align=False, **kwargs):
        print("Loading data from {}\nkeys: {}".format(self.nori_file, ','.join(self.keys)))
        fn = nori.Fetcher()
        nori_dict = misc.load_pickle(self.nori_file)

        # load img and label
        is_color = kwargs.get('is_color', False)
        kv_dict_all = OrderedDict()
        for key in self.keys:
            assert key in nori_dict, key
        for key in self.keys:
            selected_nori_id_list = nori_dict[key]
            n_samples = min(kwargs.get('n_samples', len(selected_nori_id_list)), len(selected_nori_id_list))
            print("\tFound {}/{} samples to load from {}".format(n_samples, len(selected_nori_id_list), key))

            img_raw, lm_raw = [], []
            for nori_id in selected_nori_id_list[:n_samples]:
                r = pkl.loads(fn.get(nori_id))
                img = misc.str2img(r['img'])
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)[:, :, np.newaxis] if not is_color else img
                lm = r[self.name]
                img_raw.append(img)
                lm_raw.append(lm)
            lm_raw = np.array(lm_raw).astype('float32')
            kv_dict = dict(
                img = img_raw,
                label = lm_raw,
            )
            kv_dict_all.setdefault(key, kv_dict)

        # align data and return
        if align:
            mean_face = misc.load_pickle(kwargs['mean_face_file'])
            mean_face = mean_face['ld124_81']
            image_shape = kwargs.get('image_shape', (112, 112))
            mat = kwargs.get('trans_mat', False)
            norm_type = kwargs.get('norm_type', None)
            aug = augment.AugmentDenseWithNorm(
                mean_face=mean_face, image_shape=image_shape,
                transforms_post=transforms.Compose([
                    transforms.HWC2CHW(),
                    transforms.NormalizeLandmark(image_shape=image_shape),
                ])
            )
            kv_dict_align_all = OrderedDict()
            for key, kv_dict in kv_dict_all.items():
                img_raw = kv_dict['img']
                lm_raw = kv_dict['label']

                img_new, lm_new, norm_new, mat_new = [], [], [], []
                for img, lm in zip(img_raw, lm_raw):
                    img_out, lm_out, norm_out, mat_out = aug(img, lm[:124,:], lm, norm_type=norm_type, mat=True)
                    img_new.append(img_out)
                    lm_new.append(lm_out)
                    norm_new.append(norm_out)
                    mat_new.append(mat_out)
                img_new = np.array(img_new).astype('float32')
                lm_new = np.array(lm_new).astype('float32')
                norm_new = np.array(norm_new).astype('float32').reshape(-1, 1)
                mat_new = np.array(mat_new).astype('float32')
                kv_dict_align = dict(
                    img = img_new,
                    label = lm_new,
                    norm = norm_new
                )

                if mat:
                    kv_dict_align['mat'] = mat_new
                kv_dict_align_all.setdefault(key, kv_dict_align)
            return kv_dict_align_all
        # return raw data
        else:
            return kv_dict_all

