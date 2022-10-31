from collections import OrderedDict
import numpy as np
import tqdm

from landstack.utils import misc, transforms, augment
from landstack.utils import geom

class OccImage(object):
    def __init__(self, nori_file=None, keys=None):
        self.nori_file = nori_file or '/unsullied/sharefs/_research_facelm/Isilon-datashare/proj/occ/landmark_occ_val_170927.info'
        self.keys = keys

    def load(self, mean_face_file=None, flip_idx_file=None, image_shape=(112, 112),
             is_color=False, n_samples=None, norm_type='geom', align_func=geom.align_to):
        ds_kv_dict = misc.load_pickle(self.nori_file)
        self.keys  = self.keys or list(ds_kv_dict.keys())
        for key in self.keys:
            assert key in ds_kv_dict, key

        print("Loading data from {}\nkeys: {}".format(self.nori_file, ','.join(self.keys)))
        kv_dict_all = OrderedDict()
        flip_idx_file = flip_idx_file or '/unsullied/sharefs/_research_facelm/Isilon-datashare/baseimg/extra/flip_idx_81.pkl'
        mean_face_file = mean_face_file or '/unsullied/sharefs/_research_facelm/Isilon-datashare/baseimg/extra/mean_face_81.pkl'
        aug = augment.AugmentBase(
            mean_face=mean_face_file, image_shape=image_shape,
            transforms_post=transforms.Compose([
                transforms.HWC2CHW(),
                transforms.NormalizeLandmark(image_shape=image_shape),
                transforms.AddNorm(norm_type=norm_type),
            ])
        )
        aug_flip = augment.AugmentBase(
            mean_face=mean_face_file, image_shape=image_shape,
            transforms_post=transforms.Compose([
                transforms.RandomHorizontalFlip(flip_idx=flip_idx_file, rng=None),
                transforms.HWC2CHW(),
                transforms.NormalizeLandmark(image_shape=image_shape),
                transforms.AddNorm(norm_type=norm_type),
            ])
        )

        for key in self.keys:
            nori_kv_dict = ds_kv_dict[key]
            n_samples0 = min(n_samples or len(nori_kv_dict), len(nori_kv_dict))
            print("\tFound {}/{} samples to load from {}".format(n_samples0, len(nori_kv_dict), key))
            img_out_batch, lm_out_batch, occ_batch, mask_batch, nori_id_batch, norm_out_batch =  [], [], [], [], [], []
            for nori_id, info in tqdm.tqdm(list(nori_kv_dict.items())[:n_samples0]):
                img_raw, lm_raw, _ = misc.unpack_lm(nori_id, is_color=is_color)
                img_out, lm_out, param_dict_out = aug(img_raw, lm_raw, lm_raw.copy(), align_func=align_func)
                no_flip_occ = info['no_flip']['occ']
                no_flip_mask = info['no_flip']['mask']
                norm_out = param_dict_out['norm']
                img_out_batch.append(img_out)
                lm_out_batch.append(lm_out)
                occ_batch.append(no_flip_occ)
                mask_batch.append(no_flip_mask)
                nori_id_batch.append(nori_id)
                norm_out_batch.append(norm_out)

                if 'flip' in info:
                    img_out, lm_out, param_dict_out = aug_flip(img_raw, lm_raw, lm_raw.copy(),
                                                               occ=info['flip']['occ'], align_func=align_func)
                    flip_occ = param_dict_out['occ']
                    flip_mask = info['flip']['mask']
                    norm_out = param_dict_out['norm']

                    img_out_batch.append(img_out)
                    lm_out_batch.append(lm_out)
                    occ_batch.append(flip_occ)
                    mask_batch.append(flip_mask)
                    nori_id_batch.append(nori_id)
                    norm_out_batch.append(norm_out)

            kv_dict = dict(
                img = np.uint8(img_out_batch),
                label = np.float32(lm_out_batch),
                occ = np.float32(occ_batch),
                mask = np.float32(mask_batch),
                nori_id = nori_id_batch,
                norm=np.float32(norm_out_batch).flatten(),
            )
            kv_dict_all.setdefault(key, kv_dict)
        return kv_dict_all


if __name__ == '__main__':
    from landstack.utils.visualization import draw, montage
    from IPython import embed
    import cv2
    import os
    ds = OccImage()
    val_kv_dict = ds.load()['validation-valid']
    img = val_kv_dict['img']
    label = val_kv_dict['label']
    occ = val_kv_dict['occ']
    mask = val_kv_dict['mask']
    img_all= []
    print((occ*mask).sum(axis=0), ((1-occ)*mask).sum(axis=0))
    for img0, label0, occ0 in zip(img, label, occ):
        img_all.append(draw(img0, label0, label0[occ0==1], show=True, wait=True))
    save_root = '/tmp/dataset/occ'
    misc.ensure_dir(save_root, erase=True)
    for i in range(10):
        img_copy = montage(img_all[i*100:(i+1)*100], 10, 10, (3000, 3000))
        save_file = os.path.join(save_root, '{}.png'.format(i))
        cv2.imwrite(save_file, img_copy)




