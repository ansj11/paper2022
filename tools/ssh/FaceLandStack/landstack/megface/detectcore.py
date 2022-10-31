import mgfpy
from mgfpy import MegFace, LogLevel, DeviceType, Image, Orient
from mgfpy import DetectorRectContext, LandmarkContext, DetectorContext, TrackerContext
from landstack.utils import misc
import numpy as np
import os
import re

class MegFaceAPI:
    MEGFACE_LIB_PATH = '/unsullied/sharefs/_rsde/megface/v2.5.0/lib/libmegface.so'
    FACE_MODEL_ROOT = '/unsullied/sharefs/_research_facelm/Isilon-modelshare/megface/FaceModel'
    DET_RECT_CONFIG = 'detector_rect.densebox.small.v1.3.conf' # get raw detect rectangle
    DET_81_CONFIG = 'lmk.postfilter.small.v1.2.conf' # get detector naive lm
    LM_81_CONFIG = 'detector.mobile.v3.accurate.conf'  # get final lm
    TRACK_81_CONFIG = 'tracker.mobile.v3.fast.conf' # get track lm

    def __init__(self, megface_lib_path=None, face_model_root=None, version='2.5', device='gpu0'):
        self.megface_lib_path = os.path.realpath(megface_lib_path or MegFaceAPI.MEGFACE_LIB_PATH)
        self.face_model_root = os.path.realpath(face_model_root or MegFaceAPI.FACE_MODEL_ROOT)
        self.face_model_root = os.path.join(self.face_model_root, 'config', 'out', 'megface', version)

        # parse devices
        assert 'gpu' in device or 'cpu' in device, device
        self.device_type = DeviceType.MGF_GPU if 'gpu' in device else DeviceType.MGF_CPU
        self.device_id = int(re.findall('\d+', device)[0])
        self.device_opt = {'dev_type': self.device_type, 'dev_id': self.device_id, 'stream_id': 0}
        self.tracker_opt = {'type': mgfpy.TrackerType.MGF_SYNC_DETECTOR_TRACKER,
            'sync': {
                'tracker_device': { 'dev_type': self.device_type, 'dev_id': self.device_id, 'stream_id': 0 },
                'detector_options': { 'roi': {'left':0, 'top':0, 'right':0, 'bottom':0}, 'min_face': 10, 'orient': mgfpy.Orient.MGF_UP, 'work_load': 1}, 
                'missing_tolerance': 3, 'grid_num_row': 1, 'grid_num_column': 1, 'max_num_faces':100
             }
        }

        # init megface instance
        MegFace.init(self.megface_lib_path)
        MegFace.set_log_level(LogLevel.MGF_LOG_ERROR)

        # instance
        self._det_rect_handler = None
        self._det_81_handler = None
        self._lm_81_handler = None
        self._track_81_handler = None
        self.register_det_rect_config(MegFaceAPI.DET_RECT_CONFIG)
        self.register_det_81_config(MegFaceAPI.DET_81_CONFIG)
        self.register_lm_81_config(MegFaceAPI.LM_81_CONFIG)
        self.register_track_81_config(MegFaceAPI.TRACK_81_CONFIG)

    def register_det_rect_config(self, config):
        conf_path = os.path.join(self.face_model_root, config)
        if os.path.exists(conf_path):
            self._det_rect_handler = DetectorRectContext(
                config_path=conf_path,
                settings={'device': self.device_opt}
            )

    def register_det_81_config(self, config):
        conf_path = os.path.join(self.face_model_root, config)
        if os.path.exists(conf_path):
            self._det_81_handler = LandmarkContext(
                config_path=conf_path,
                settings={'device': self.device_opt}
            )

    def register_lm_81_config(self, config):
        conf_path = os.path.join(self.face_model_root, config)
        if os.path.exists(conf_path):
            self._lm_81_handler = DetectorContext(
                config_path=conf_path,
                settings={'device': self.device_opt}
            )

    def register_track_81_config(self, config):
        conf_path = os.path.join(self.face_model_root, config)
        if os.path.exists(conf_path):
            self._track_81_handler = TrackerContext(
                config_path=conf_path,
                settings=self.tracker_opt
            )

def getmaxface(faces, thres):
    maxarea = 0
    maxface = None
    for face in faces:
        if face['conf'] < thres:
            continue
        pts = face['pts'].reshape(-1,2)
        x0, x1, y0, y1 = pts[:,0].min(), pts[:,0].max(), pts[:,1].min(), pts[:,1].max()
        ss = (x1-x0) * (y1-y0)
        if ss > maxarea:
            maxarea = ss
            maxface = face
    if maxface is None:
        return maxface
    else:
        return [maxface]

def getoverlapface(faces, gt, thres=0):
    gt = gt.reshape(-1,2)
    x0,x1,y0,y1 = gt[:,0].min(),gt[:,0].max(),gt[:,1].min(),gt[:,1].max()
    ssd = (x1-x0)*(y1-y0)
    maxscore, maxface = -1,[]
    for face in faces:
        res = face['pts']
        x2,x3,y2,y3 = res[:,0].min(),res[:,0].max(),res[:,1].min(),res[:,1].max()
        sst = (x3-x2)*(y3-y2)
        xx, yy = min(x1,x3)-max(x0,x2), min(y1,y3)-max(y0,y2)
        if xx>0 and yy>0:
            olp = xx * yy / min(sst, ssd)
            if olp > 0.5 and face['conf']>maxscore:
                maxscore = face['conf']
                maxface = face
    if maxscore < thres:
        return []
    else:
        return [maxface]

def detect81(mgf, img, faces):
    assert mgf._det_81_handler is not None
    outfaces = []
    for face in faces:
        lm = mgf._det_81_handler.predict([mgfimg], [face])[0]
        x0, x1, y0, y1 = face['rect']['left'], face['rect']['right'], face['rect']['top'], face['rect']['bottom']
        rect = np.array([x0, y0, x1, y1]).reshape(-1,2)
        res = misc.megface2landmark(lm['points'])
        outfaces.append({'pts':res, 'conf':lm['score'], 'rect':rect})
    outfaces = sorted(outfaces, key=lambda x:np.prod(np.max(x['rect'], axis=1)- np.min(x['rect'], axis=1)), reverse=True)
    return outfaces

def detect(mgf, img, stage='lm_81', thres=0.2, maxface=False, gt=None, options=None):
    assert stage in ['det_rect', 'det_81', 'lm_81'], stage
    mgfimg = mgfpy.Image.from_cv2_image(img)
    faces = []
    detector_options = {'roi': { 'left': 0, 'top': 0, 'right': 0, 'bottom': 0 }, 'min_face': 70, 'orient': mgfpy.Orient.MGF_UP}
    if isinstance(options, dict):
        detector_options.update(options)

    if stage == 'det_rect':
        assert mgf._det_rect_handler is not None
        result = mgf._det_rect_handler.detect(mgfimg, detector_options)
        for face in result['items']:
            x0, x1, y0, y1 = face['rect']['left'], face['rect']['right'], face['rect']['top'], face['rect']['bottom']
            res = np.array([x0, y0, x1, y1]).reshape(-1,2)
            faces.append({'pts':res, 'conf':face['confidence']})
    elif stage == 'det_81':
        assert mgf._det_rect_handler is not None and mgf._det_81_handler is not None
        result = mgf._det_rect_handler.detect(mgfimg, detector_options)['items']
        for face in result:
            lm = mgf._det_81_handler.predict([mgfimg], [face])[0]
            x0, x1, y0, y1 = face['rect']['left'], face['rect']['right'], face['rect']['top'], face['rect']['bottom']
            rect = np.array([x0, y0, x1, y1]).reshape(-1,2)
            res = misc.megface2landmark(lm['points'])
            faces.append({'pts':res, 'conf':lm['score'], 'rect':rect})
        faces = sorted(faces, key=lambda x:np.prod(np.max(x['rect'], axis=1)- np.min(x['rect'], axis=1)), reverse=True)
    else:
        assert mgf._lm_81_handler is not None
        result = mgf._lm_81_handler.detect(mgfimg, detector_options)
        for face in result['items']:
            res = misc.megface2landmark(face['landmark']['points'])
            faces.append({'pts':res, 'conf':face['confidence']})
        faces = sorted(faces, key=lambda x:x['conf'], reverse=True)

    if len(faces) == 0:
        return []

    # get the same face wit given gt
    if gt is not None:
        return getoverlapface(faces, gt)
    # get maxface with confidence more than given thres
    elif maxface:
        return getmaxface(faces, thres)    
    else:
        return faces

def track(mgf, img):
    mgfimg = mgfpy.Image.from_cv2_image(img)
    ret = mgf._track_81_handler.track_frame(mgfimg)
    return ret

def loadyuv(imgpath):
    names = imgpath.split('_')
    h, w = int(names[-1].split('.')[0]), int(names[-2])
    with open(imgpath, 'rb') as f:
        img = f.read()
        if img is None:
            return None
    img = misc.yuv2rgb(np.fromstring(img, np.uint8),h,w)
    print(imgpath, img.shape)
    return img

if __name__ == '__main__':
    from IPython import embed
    from landstack.utils.visualization import draw
    img_root = '/tmp/badcase_pack'
    for root, dirs, files in os.walk(img_root):
        for fname in files:
            full_path = os.path.join(root, fname)
            img = misc.read_yuv(full_path, height=480, width=640, rgb=True)
            det = MegFaceAPI(
                megface_lib_path='/home/chenxi/projects/megface-v2/lib/libmegface.so',
            )
            det.register_det_rect_config('detector_rect.densebox.small.v1.conf')
            det.register_det_81_config('lmk.postfilter.small.v1.2.conf')
            res = detect(det, img, stage='det_81')
            if len(res) > 0:
                draw(img, res[0]['pts'])
                embed()


