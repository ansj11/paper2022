from collections import OrderedDict
import numpy as np

from landstack.utils import misc, transforms, augment
import tqdm

def encode(mouthocc):
    assert mouthocc in ['no_occ', 'mask', 'respirator', 'others']
    if mouthocc == 'no_occ':
        return [1.0, 0, 0, 0]
    elif mouthocc == 'mask':
        return [0, 1.0, 0, 0]
    elif mouthocc == 'respirator':
        return [0, 0, 1.0, 0]
    else:
        return [0, 0, 0, 1.0]

class MouthOccImage(object):
    """mouth occlusion"""
    def __init__(self, nori_file=None, keys=None):
        self.nori_file = nori_file or '/unsullied/sharefs/_research_facelm/Isilon-datashare/proj/mouthocc/landmark_mouthocc_val_171127.info'
        self.keys = keys

    def load(self, mean_face_file=None, image_shape=(112, 112),
             align=True, is_color=False, n_samples=None, norm_type='geom'):
        ds_kv_dict = misc.load_pickle(self.nori_file)
        self.keys  = self.keys or list(ds_kv_dict.keys())
        assert isinstance(self.keys, (tuple, list))
        for key in self.keys:
            assert key in ds_kv_dict, key

        print("Loading data from {}\nkeys: {}".format(self.nori_file, ','.join(self.keys)))
        kv_dict_all = OrderedDict()
        for key in self.keys:
            nori_kv_dict = ds_kv_dict[key]
            n_samples0 = min(n_samples or len(nori_kv_dict), len(nori_kv_dict))
            print("\tFound {}/{} samples to load from {}".format(n_samples0, len(nori_kv_dict), key))
            img_raw_batch, lm_raw_batch, mouthocc_batch, nori_id_batch = [], [], [], []
            for nori_id, value in tqdm.tqdm(list(nori_kv_dict.items())[:n_samples0]):
                img_raw, lm_raw, _ = misc.unpack_lm(nori_id, is_color=True)
                assert img_raw.shape == (256, 256, 3)
                img_raw_batch.append(img_raw)
                lm_raw_batch.append(lm_raw)
                mouthocc_batch.append(encode(value['mouthocc']))
                nori_id_batch.append(nori_id)
            kv_dict = dict(
                img = img_raw_batch,
                label = lm_raw_batch,
                mouthocc = mouthocc_batch,
                nori_id = nori_id_batch,
            )
            kv_dict_all.setdefault(key, kv_dict)

        if align:
            mean_face_file = mean_face_file or '/unsullied/sharefs/_research_facelm/Isilon-datashare/baseimg/extra/mean_face_81.pkl'
            if is_color:
                aug = augment.AugmentBase(
                    mean_face=mean_face_file, image_shape=image_shape,
                    transforms_post=transforms.Compose([
                        transforms.HWC2CHW(),
                        transforms.NormalizeLandmark(image_shape=image_shape),
                        transforms.AddNorm(norm_type=norm_type, n_points=81)
                    ])
                )
            else:
                aug = augment.AugmentBase(
                    mean_face=mean_face_file, image_shape=image_shape,
                    transforms_post=transforms.Compose([
                        transforms.ConvertToGray(),
                        transforms.HWC2CHW(),
                        transforms.NormalizeLandmark(image_shape=image_shape),
                        transforms.AddNorm(norm_type=norm_type, n_points=81)
                    ])
                )

            kv_dict_align_all = OrderedDict()
            for key, kv_dict in kv_dict_all.items():
                img_raw_batch = kv_dict['img']
                lm_raw_batch = kv_dict['label']
                mouthocc_batch = kv_dict['mouthocc']
                nori_id_batch = kv_dict['nori_id']

                img_out_batch, lm_out_batch, norm_out_batch, mat_out_batch = [], [], [], []
                for img_raw, lm_raw in zip(img_raw_batch, lm_raw_batch):
                    img_out, lm_out, param_dict = aug(img_raw, lm_raw, lm_raw.copy())
                    mat_out = param_dict['mat']
                    norm_out = param_dict['norm']
                    img_out_batch.append(img_out)
                    lm_out_batch.append(lm_out)
                    norm_out_batch.append(norm_out)
                    mat_out_batch.append(mat_out)
                kv_dict_align = dict(
                    img = np.uint8(img_out_batch),
                    label = np.float32(lm_out_batch),
                    norm = np.float32(norm_out_batch).flatten(),
                    mouthocc = np.float32(mouthocc_batch),
                    mat = np.float32(mat_out_batch),
                    nori_id = nori_id_batch,
                )
                kv_dict_align_all.setdefault(key, kv_dict_align)
            return kv_dict_align_all
        else:
            return kv_dict_all

if __name__ == '__main__':
    from IPython import embed
    from landstack.utils.visualization import draw
    ds = MouthOccImage()
    kv_dict_all = ds.load(align=True)
    for ds_name, kv_dict in kv_dict_all.items():
        img = kv_dict['img']
        label = kv_dict['label']
        mouthocc = kv_dict['mouthocc']
        for img0, label0, mouthocc0 in zip(img, label, mouthocc):
            print(img0.shape, mouthocc0)
            draw(img0, font_scale=2)
        embed()

