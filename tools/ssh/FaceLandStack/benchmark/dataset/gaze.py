from collections import OrderedDict
import numpy as np
from landstack.utils import misc, transforms
from landstack.utils import augment
import tqdm


def encode(dict_data):
    return np.concatenate([
        np.array(dict_data['iris']).reshape(-1, 4, 2), # 4 point
        np.array(dict_data['eye_center']).reshape(-1, 1, 2), # 1 point
        np.array(dict_data['gaze_point']).reshape(-1, 1, 2), # 1 point
        np.array(dict_data['interior_margin']).reshape(-1, 8, 2) # 8 point
    ], axis=1).squeeze()


def decode(array_data):
    array_data = np.float32(array_data).reshape(-1, 14, 2)
    res = dict(
        iris = array_data[:, :4, :].squeeze(),
        eye_center = array_data[:, 4, :].squeeze(),
        gaze_point = array_data[:, 5, :].squeeze(),
        interior_margin = array_data[:, 6:, :].squeeze(),
    )
    return res


class GazeImage(object):
    def __init__(self, nori_file=None, keys=None):
        self.nori_file = nori_file or '/unsullied/sharefs/_research_facelm/Isilon-datashare/proj/gaze/landmark_gaze_val_171216.info'
        self.keys = keys

    def load(self, mean_face_file=None, width_height=(48, 32),
             align=True, is_color=False, n_samples=None, scale=1.0, ratio0=16):
        nori_dict = misc.load_pickle(self.nori_file)
        self.keys = self.keys or list(nori_dict.keys())
        kv_dict_all = OrderedDict()
        print("Loading data from {}\nkeys: {}".format(self.nori_file, ','.join(self.keys)))
        for key in self.keys:
            selected_nori_id_list = nori_dict[key]
            n_samples0 = min(n_samples or len(selected_nori_id_list), len(selected_nori_id_list))
            print("\tFound {}/{} samples to load from {}".format(n_samples0, len(selected_nori_id_list), key))

            img_raw_batch, lm_raw_batch, nori_id_batch, pose_batch = [], [], [], []
            for nori_id, value in tqdm.tqdm(list(selected_nori_id_list.items())[:n_samples0]):
                img_raw, _, _ = misc.unpack_lm(nori_id,is_color=True)
                lm_raw = encode(value) # first 6 points are gaze, the others are eyes for align
                img_raw_batch.append(img_raw)
                lm_raw_batch.append(lm_raw)
                nori_id_batch.append(nori_id)
                pose_batch.append(np.float32([value.get('pitch', 0.0), value['yaw']]))
            kv_dict = dict(
                img_gaze = img_raw_batch,
                label = np.float32(lm_raw_batch),
                pose_gaze = np.float32(pose_batch),
                nori_id = nori_id_batch,
            )
            kv_dict_all.setdefault(key, kv_dict)


        if align:
            mean_face = misc.load_pickle(mean_face_file or '/unsullied/sharefs/_research_facelm/Isilon-datashare/baseimg/extra/mean_face_81.pkl')
            mean_eye = np.vstack([mean_face[64:68], mean_face[69:73]])
            # m = augment.AugmentGaze.gen_mean_face(mean_eye, scale=1.0, width_height=(256, 256))
            # mean_face = mean_face[64:73] # right-eye without pupils
            # draw(np.ones((3, 300, 300))*255, m, canvas_height_width=(300, 300), mark=True)
            if is_color:
                aug = augment.AugmentGaze(
                    mean_face=mean_eye, width_height=width_height,
                    scale=scale,
                    transforms_post=transforms.Compose([
                        transforms.HWC2CHW(),
                        transforms.NormalizeLandmark(width_height=width_height),
                    ])
                )
            else:
                aug = augment.AugmentGaze(
                    mean_face=mean_eye, width_height=width_height,
                    scale=scale,
                    transforms_post=transforms.Compose([
                        transforms.ConvertToGray(),
                        transforms.HWC2CHW(),
                        transforms.NormalizeLandmark(width_height=width_height),
                    ])
                )
            kv_dict_align_all = OrderedDict()
            for key, kv_dict in kv_dict_all.items():
                img_raw_batch = kv_dict['img_gaze']
                lm_raw_batch = kv_dict['label']
                nori_id_batch = kv_dict['nori_id']
                pose_batch = kv_dict['pose_gaze']

                img_out_batch, lm_out_batch = [], []
                mat_batch, mat_gp_batch = [], []
                for img_raw, lm_raw in zip(img_raw_batch, lm_raw_batch):
                    img_out, lm_out, param_out = aug(img_raw, lm_raw, lm_raw.copy(), ratio0=ratio0)
                    img_out_batch.append(img_out)
                    lm_out_batch.append(lm_out)
                    mat_batch.append(param_out['mat'])
                    mat_gp_batch.append((param_out['mat_gp']))

                kv_dict_align = dict(
                    img_gaze = np.uint8(img_out_batch),
                    label=np.float32(lm_out_batch),
                    pose_gaze = np.float32(pose_batch),
                    nori_id = nori_id_batch,
                    mat=np.float32(mat_batch),
                    mat_gp=np.float32(mat_gp_batch),
                )
                kv_dict_align_all.setdefault(key, kv_dict_align)
            return kv_dict_align_all
        else:
            return kv_dict_all


if __name__ == '__main__':
    from landstack.utils.visualization import draw
    from landstack.utils import geom
    from IPython import embed
    ds = GazeImage(keys=['lock'])
    kv_dict_all0 = ds.load(align=False, n_samples=100)
    kv_dict_all = ds.load(align=True, n_samples=100)
    width_height = np.float32([48, 32])
    for ds_name, kv_dict in kv_dict_all.items():
        img_raw  = kv_dict_all0[ds_name]['img_gaze']
        img = kv_dict['img_gaze']
        r = decode(kv_dict['label'])
        lm = r['interior_margin']
        iris = r['iris']
        eye_center = r['eye_center']
        gaze_point = r['gaze_point']
        pose = kv_dict['pose_gaze']
        mat = kv_dict['mat']
        mat_gp = kv_dict['mat_gp']
        for img_raw0, img0, iris0, eye_center0, gaze_point0, lm0, pose0, mat0, mat_gp0 in \
                zip(img_raw, img, iris, eye_center, gaze_point, lm, pose, mat, mat_gp):
            eye_center0  = eye_center0 *width_height
            iris0 = iris0 * width_height
            lm0 = lm0 *width_height
            gaze_point0 = gaze_point0 * width_height
            eye_center1 = geom.lm_affine_general(eye_center0, np.linalg.inv(mat0))
            iris1 = geom.lm_affine_general(iris0, np.linalg.inv(mat0))
            lm1 = geom.lm_affine_general(lm0, np.linalg.inv(mat0))
            gaze_point1 = geom.lm_affine_general(gaze_point0, np.linalg.inv(mat_gp0))
            draw(img0, eye_center0, gaze_point0, lm0, iris0, text='{:.2f}, {:.2f}'.format(pose0[0], pose0[1]), line=True, canvas_height_width=(320, 480), wait=False, name='align')
            draw(img_raw0, eye_center1, gaze_point1, lm1, iris1, text='{:.2f}, {:.2f}'.format(pose0[0], pose0[1]), line=True, canvas_height_width=(256, 256), name='raw')

