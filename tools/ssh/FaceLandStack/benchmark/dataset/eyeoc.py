from collections import OrderedDict
import numpy as np
from landstack.utils import misc, transforms, augment
import tqdm

class EyeOCImage(object):
    """eye open closed"""
    def __init__(self, nori_file=None, keys=None):
        self.nori_file = nori_file or '/unsullied/sharefs/_research_facelm/Isilon-datashare/proj/eyeoc/landmark_eyeoc_val_171217.info'
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
            img_raw_batch, lm_raw_batch, eyeoc_batch, nori_id_batch = [], [], [], []
            for nori_id, value in tqdm.tqdm(list(nori_kv_dict.items())[:n_samples0]):
                assert value['eyeoc'] in ['close', 'open'], value
                img_raw, lm_raw, _ = misc.unpack_lm(nori_id, is_color=True)
                img_raw_batch.append(img_raw)
                lm_raw_batch.append(lm_raw)
                eyeoc_batch.append(value['eyeoc'] == 'close')
                nori_id_batch.append(nori_id)
            kv_dict = dict(
                img = img_raw_batch,
                label = lm_raw_batch,
                eyeoc = eyeoc_batch,
                nori_id = nori_id_batch,
            )
            kv_dict_all.setdefault(key, kv_dict)

        if align:
            mean_face_file = mean_face_file or '/unsullied/sharefs/_research_facelm/Isilon-datashare/landmark/extra/mean_face_81.pkl'
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
                eyeoc_batch = kv_dict['eyeoc']
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
                    eyeoc=np.float32(eyeoc_batch).flatten(),
                    mat =np.float32(mat_out_batch),
                    nori_id=nori_id_batch,
                )
                kv_dict_align_all.setdefault(key, kv_dict_align)
            return kv_dict_align_all
        else:
            return kv_dict_all

if __name__ == '__main__':
    from IPython import embed
    from landstack.utils.visualization import draw
    ds = EyeOCImage(keys=['attention'])
    kv_dict_all = ds.load(align=True, n_samples=100)
    for ds_name, kv_dict in kv_dict_all.items():
        img = kv_dict['img']
        label = kv_dict['label']
        eyeoc = kv_dict['eyeoc']
        for img0, label0, eyeoc0 in zip(img, label, eyeoc):
            draw(img0, text=str(int(eyeoc0)))
        embed()

