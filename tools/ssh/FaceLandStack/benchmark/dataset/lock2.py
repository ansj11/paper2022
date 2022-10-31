from collections import OrderedDict
import numpy as np
from landstack.utils import misc, transforms
from landstack.utils.augment import AugmentGaze, AugmentBaseGeneral
import tqdm

def get_post_transform(width_height=(48, 32), is_color=False, flip_idx=None, rng=None):
    aug_list = []
    if flip_idx is not None:
        aug_list.append(transforms.RandomHorizontalFlip(flip_idx=flip_idx, rng=rng))
    if not is_color:
        aug_list.append(transforms.ConvertToGray())
    aug_list.append(transforms.HWC2CHW())
    aug_list.append(transforms.NormalizeLandmark(width_height=width_height))
    aug = transforms.Compose(aug_list)
    return aug

class GazeLockImage(object):
    def __init__(self, nori_file=None, keys=None):
        self.nori_file = nori_file or '/unsullied/sharefs/_research_facelm/Isilon-datashare/proj/lock/landmark_lock_val_171215.info'
        self.keys = keys

    def load(self, width_height=(48, 32), align=True, is_color=False, n_samples=None, scale=0.8):
        ds_kv_dict = misc.load_pickle(self.nori_file)
        self.keys  = self.keys or list(ds_kv_dict.keys())
        assert isinstance(self.keys, (tuple, list))
        for key in self.keys:
            assert key in ds_kv_dict, key

        print("Loading data from {}\nkeys: {}".format(self.nori_file, ','.join(self.keys)))
        _kv_dict_all = OrderedDict()
        for key in self.keys:
            nori_kv_dict = ds_kv_dict[key]
            n_samples0 = min(n_samples or len(nori_kv_dict), len(nori_kv_dict))
            print("\tFound {}/{} samples to load from {}".format(n_samples0, len(nori_kv_dict), key))
            img_raw_batch, lm_raw_batch, lock_batch, nori_id_batch, lr_batch = [], [], [], [], []
            for nori_id, value in tqdm.tqdm(list(nori_kv_dict.items())[:n_samples0]):
                img_raw, _, _ = misc.unpack_lm(nori_id, is_color=True)
                lm_raw = value['lm']
                img_raw_batch.append(img_raw)
                lm_raw_batch.append(lm_raw)
                lock_batch.append(value['lock'] == 'lock')
                nori_id_batch.append(nori_id)
                lr_batch.append(value['which'])
            _kv_dict = dict(
                img_patch = img_raw_batch,
                label = lm_raw_batch,
                lock = lock_batch,
                lr = lr_batch,
                nori_id = nori_id_batch,
            )
            _kv_dict_all.setdefault(key, _kv_dict)

        if align:
            mean_face_file = '/unsullied/sharefs/_research_facelm/Isilon-datashare/baseimg/extra/mean_face_81.pkl'
            mean_face = misc.load_pickle(mean_face_file)
            left_eye_idx = np.arange(19, 28)
            right_eye_idx = np.arange(64, 73)
            left_mean_eye = AugmentGaze.gen_mean_eye(mean_face[left_eye_idx], scale, width_height=width_height)
            right_mean_eye = AugmentGaze.gen_mean_eye(mean_face[right_eye_idx], scale, width_height=width_height)
            flip_idx = [0, 5, 3, 2, 4, 1, 6, 8, 7]

            aug_left = AugmentBaseGeneral(
                mean_face=left_mean_eye, width_height=width_height,
                transforms_post=get_post_transform(width_height, is_color, flip_idx, None) # post process flip
            )
            aug_right = AugmentBaseGeneral(
                mean_face=right_mean_eye, width_height=width_height,
                transforms_post=get_post_transform(width_height, is_color, None, None) # post process without flip
            )

            kv_dict_align_all = OrderedDict()
            for key, _kv_dict in _kv_dict_all.items():
                img_raw_batch = _kv_dict['img_patch']
                lm_raw_batch = _kv_dict['label']
                lock_batch = _kv_dict['lock']
                nori_id_batch = _kv_dict['nori_id']
                lr_batch = _kv_dict['lr']

                img_out_batch, lm_out_batch, mat_out_batch = [], [], []
                for img_raw, lm_raw, lr_raw in zip(img_raw_batch, lm_raw_batch, lr_batch):
                    assert lr_raw in ['left', 'right']
                    aug = aug_left if lr_raw == 'left' else aug_right
                    img_out, lm_out, param_dict = aug(img_raw, lm_raw, lm_raw.copy())
                    mat_out = param_dict['mat']
                    img_out_batch.append(img_out)
                    lm_out_batch.append(lm_out)
                    mat_out_batch.append(mat_out)
                kv_dict_align = dict(
                    img_patch = np.uint8(img_out_batch),
                    label = np.float32(lm_out_batch),
                    lock = np.float32(lock_batch),
                    mat = np.float32(mat_out_batch),
                    nori_id = nori_id_batch,
                )
                kv_dict_align_all.setdefault(key, kv_dict_align)
            return kv_dict_align_all
        else:
            return _kv_dict_all

if __name__ == '__main__':
    from landstack.utils.visualization import draw
    from IPython import embed
    ds = GazeLockImage()
    kv_dict_all = ds.load(align=True, n_samples=1000)
    for ds_name, kv_dict in kv_dict_all.items():
        img = kv_dict['img_patch']
        label = kv_dict['label']
        lock = kv_dict['lock']
        for img0, label0, lock0 in zip(img, label, lock):
            if lock0 == 0.0:
                continue
            print(img0.shape)
            draw(img0, text=str(int(lock0)), canvas_height_width=(256, 256))



