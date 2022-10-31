import os
import pickle as pkl
import numpy as np
import hashlib
from multiprocessing import Process, Pipe
from IPython import embed
import cv2
import nori2 as nori

def ensure_dir(*paths, erase=False):
    import shutil
    for path in paths:
        if os.path.exists(path) and erase:
            print("Removing old folder {}".format(path))
            try:
                shutil.rmtree(path)
            except Exception as e:
                print("Try to use sudo")
                import traceback
                traceback.print_exc()
                os.system('sudo rm -rf {}'.format(path))
        if not os.path.exists(path):
            print("Creating folder {}".format(path))
            try:
                os.makedirs(path)
            except Exception as e:
                print("Try to use sudo")
                import traceback
                traceback.print_exc()
                os.system('sudo mkdir -p {}'.format(path))

def ensure_writable(path):
    path = os.path.realpath(path)
    if not os.path.exists(path):
        print("Could not find {}, do nothing".format(path))
    else:
        os.system('sudo chmod +w {}'.format(path))


def ensure_hwc(img):
    if img.ndim == 2:
        img = img[:, :, np.newaxis]
    if img.shape[0] == 1 or img.shape[0] == 3:
        img = np.transpose(img, (1, 2, 0))
    if img.shape[-1] == 1:
        img = np.tile(img, (1, 1, 3))
    assert img.shape[-1] == 3, img.shape
    return img

def load_pickle(path):
    path = expand_path(path)
    print("Loading pickle object from {} => ".format(path), end='', flush=True)
    with open(path, 'rb') as f:
        v = pkl.load(f)
    print('Done')
    return v

def dump_pickle(obj, path):
    path = expand_path(path)
    with open(path, 'wb') as f:
        print("Dumping pickle object to {} => ".format(path), end='', flush=True)
        pkl.dump(obj, f)
    print('Done')


def write_txt(txt, path):
    txt = "\n".join(txt) if isinstance(txt, (tuple, list)) else txt
    assert isinstance(txt, str), type(txt)
    with open(path, 'w') as f:
        print("Writing to {}".format(path), end='')
        f.write(txt)
    print(' => Done')

def expand_path(path):
    return os.path.abspath(os.path.expanduser(path))

def str2img(str_b):
    return cv2.imdecode(np.fromstring(str_b, np.uint8), cv2.IMREAD_COLOR)

def img2str(img):
    return cv2.imencode('.jpg', img)[1].tostring()

def md5(fname):
    m = hashlib.md5()
    if os.path.exists(fname):
        fh = open(fname, 'rb')
        while True:
            data = fh.read(8192)
            if not data:
                break
            m.update(data.encode("UTF-8"))
        fh.close()
    else:
        m.update(fname.encode("UTF-8"))
    return m.hexdigest()

class AsyncTask(object):
    def __init__(self, func, name, *args, **kwargs):
        # Pipe() return (conn1, conn2), conn1 can be used for receive and conn2 used for send
        self.parent_conn, self.child_conn = Pipe()
        self.name = name
        self.res = None

        def wrapper(conn):
            v = func(*args, **kwargs)
            conn.send(v)

        self.task = Process(target=wrapper, args=(self.child_conn, ))

    def start(self):
        self.task.start()

    def fetch(self):
        if self.res is None:
            print("Waiting {} to finish".format(self.name))
            self.res = self.parent_conn.recv()
            self.task.join()
        return self.res

def block_sys_call(command):
    import subprocess
    command = " ".join(command) if isinstance(command, (tuple, list)) else command
    print("Executing {}".format(command))
    subprocess.call(command, bufsize=4096, shell=True)


def format_time(elapse):
    elapse = int(elapse)
    hour = elapse // 3600
    minute = elapse % 3600 // 60
    seconds = elapse % 60
    return "{:02d}:{:02d}:{:02d}".format(hour, minute, seconds)

def find_opr_by_name(dest_oprs, names):
    from megskull.graph import iter_dep_opr
    import collections
    """dest_oprs should be var nodes, names is str"""
    assert names is not None, names
    if isinstance(names, str):
        names = names.split(',')
    assert np.all([isinstance(v, str) for v in names])
    if isinstance(dest_oprs, dict):
        dest_oprs = list(dest_oprs.values())
    found = collections.OrderedDict()
    for opr in iter_dep_opr(dest_oprs):
        if opr.name in names:
            found.setdefault(opr.name, opr)
    not_found = set(names) - set(found.keys())
    assert len(not_found) == 0, 'could not found {}'.format(','.join(list(not_found)))
    res = [found[k] for k in names]
    return res

def load_network_and_extract(model_file, opr_names=None):
    from neupeak.utils.cli import load_network
    from megskull.network import RawNetworkBuilder
    print("Loading network from {}".format(model_file))
    if isinstance(model_file, str):
        net = load_network(model_file)
    else:
        net = model_file
    outputs = list(net.outputs) if isinstance(net.outputs, (tuple, list)) else [net.outputs]
    if net.loss_var is not None:
        outputs += [net.loss_var]
    if opr_names is not None:
        if isinstance(opr_names, str):
            opr_names = opr_names.split(',')
        oprs = find_opr_by_name(outputs, opr_names)
        oprs = sorted(oprs, key=lambda x:opr_names.index(x.name)) # keep order
        net = RawNetworkBuilder(inputs=[], outputs=oprs)
    return net

def load_videos(video_file, multiplier=1, meta=True):
    import cv2
    cap = cv2.VideoCapture(video_file)
    fps = cap.get(cv2.CAP_PROP_FPS) // multiplier
    frames = []
    print("Loading from {} => ".format(video_file), end='')
    counter = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if counter % multiplier == 0:
            frames.append(frame)
        counter += 1
    if len(frames) == 0:
        print("WRN: read 0 frame from {}".format(video_file))
    print(len(frames))
    if meta:
        return frames, fps
    else:
        return frames

def serialize_dict(kv_dict):
    assert isinstance(kv_dict, dict), type(kv_dict)
    r = []
    for k, v in kv_dict.items():
        if isinstance(v, (tuple, list, np.ndarray)) and len(v) == 1:
            r.append('{}:{:.4f}'.format(k, v[0]))
        elif isinstance(v, float):
            r.append('{}:{:.4f}'.format(k, v))
        else:
            r.append('{}:{}'.format(k, v.__repr__()))

    return r

def compute_precision_recall(probs, gt):
    from sklearn import metrics
    if isinstance(gt, np.ndarray):
        gt = gt.flatten()
    if isinstance(probs, np.ndarray):
        probs = probs.flatten()
    precision, recall, thresholds = metrics.precision_recall_curve(gt, probs)
    auc = metrics.auc(recall, precision)
    precision = precision[:-1][::-1]
    recall = recall[:-1][::-1]
    thresholds = thresholds[::-1]
    return precision, recall, thresholds, auc

def compute_roc(probs, gt):
    from sklearn import metrics
    if isinstance(gt, np.ndarray):
        gt = gt.flatten()
    if isinstance(probs, np.ndarray):
        probs = probs.flatten()
    fpr, tpr, thresholds = metrics.roc_curve(gt, probs)
    auc = metrics.auc(fpr, tpr)
    return fpr, tpr, thresholds, auc

def find_fpr(fpr, tpr, thresholds, target):
    idx = np.argsort(np.abs(fpr - target))[0]
    return fpr[idx], tpr[idx], thresholds[idx]

def find_tpr(fpr, tpr, thresholds, target):
    idx = np.argsort(np.abs(tpr - target))[0]
    return 1-tpr[idx], 1-fpr[idx], thresholds[idx]

def summary_precision_recall(precision, recall, threshold, N=5):
    default = [0.0, 0.0, 0.0]
    res = []
    for i in range(N):
        left = 1 - i * 0.01
        try:
            p = list(filter(lambda x:left<=x[0], zip(precision, recall, threshold)))[-1]
        except Exception as e:
            p = default
        p = list(p)
        p.insert(0, 'p{:03d}'.format(int(100*left)))
        res.append(p)
    return res

def merge_to_video(out_file, frames, width, height, fourcc=None, fps=40):
    fourcc = fourcc or cv2.VideoWriter_fourcc(*'H264')
    out = cv2.VideoWriter(out_file, fourcc, int(fps), (width, height))
    for frame in frames:
        out.write(frame)
    out.release()

def merge_imgs(out_root, frames, fnames=None, erase=False):
    ensure_dir(out_root, erase=erase)
    fnames = fnames or ['{}.png'.format(i) for i in range(len(frames))]
    for fname, frame in zip(fnames, frames):
        save_file = os.path.join(out_root, fname)
        cv2.imwrite(save_file, frame)

def count_files(walk_root, file_format='jpg,png'):
    c = 0
    for root, dirs, files in os.walk(walk_root):
        for fname in files:
            if np.any([fname.endswith(v) for v in file_format.split(',')]):
                c += 1
    return c

def getneib81(i):
    index = {0:[9,18],1:[1,2],2:[1,3],3:[2,4],4:[3,5],5:[4,6],6:[5,7],7:[6,8],8:[7,9],9:[8,0],10:[10,11],11:[10,12],12:[11,13],13:[12,14],14:[13,15],15:[14,16],16:[15,17],17:[16,18],18:[17,0],19:[21,22],20:[26,21],21:[20,19],22:[19,24],23:[20,24],24:[22,27],25:[26,27],26:[20,25],27:[25,24],28:[33,29],29:[28,30],30:[29,31],31:[30,32],32:[35,31],33:[28,34],34:[33,35],35:[34,32],36:[48,39],37:[40,43],38:[36,44],39:[36,40],40:[39,37],41:[44,45],42:[43,45],43:[37,42],44:[38,41],45:[51,42],46:[49,52],47:[48,53],48:[36,47],49:[36,46],50:[53,51],51:[50,45],52:[46,45],53:[47,50],54:[54,55],55:[54,61],56:[61,57],57:[56,60],58:[58,59],59:[58,62],60:[57,62],61:[55,61],62:[59,62],63:[63,57],64:[66,67],65:[71,66],66:[65,64],67:[64,69],68:[65,69],69:[72,67],70:[71,72],71:[65,70],72:[70,69],73:[78,74],74:[73,75],75:[76,77],76:[75,77],77:[80,76],78:[73,79],79:[78,80],80:[79,77]}
    return index[i]


def centrotate(src, angle, scale=1.):
    w = src.shape[1]
    h = src.shape[0]
    rangle = np.deg2rad(angle)
    nw = (abs(np.sin(rangle)*h) + abs(np.cos(rangle)*w))*scale
    nh = (abs(np.cos(rangle)*h) + abs(np.sin(rangle)*w))*scale
    rot_mat = cv2.getRotationMatrix2D((nw*0.5, nh*0.5), angle, scale)
    rot_move = np.dot(rot_mat, np.array([(nw-w)*0.5, (nh-h)*0.5,0]))
    rot_mat[0,2] += rot_move[0]
    rot_mat[1,2] += rot_move[1]
    return cv2.warpAffine(src, rot_mat, (int(nw+0.5), int(nh+0.5)), flags=cv2.INTER_LANCZOS4)

def rotatecv(imgin, fliptype):
    """
    anti-clockwise
    """
    img = imgin.copy()
    if fliptype == 'rot90':             # left 90
        img = centrotate(img, 90)
    elif fliptype == 'rot180':          # left 180
        img = centrotate(img, 180)
    elif fliptype == 'rot270':          # right 90
        img = centrotate(img, 270)
    return img

def rotateld(ldin, fliptype, h, w):
    ldin = ldin.reshape(-1,2)
    ld = ldin.copy()
    if fliptype == 'rot90':
        ld[:,1] = ldin[:,0]
        ld[:,0] = h-ldin[:,1]-1
    elif fliptype == 'rot180':
        ld[:,0] = w-ldin[:,0]-1
        ld[:,1] = h-ldin[:,1]-1
    elif fliptype == 'rot270':
        ld[:,0] = ldin[:,1]
        ld[:,1] = w-ldin[:,0]-1
    return ld

def format_table(res):
    import pandas as pd
    pd.options.display.float_format = '{:.5g}'.format
    pd.set_option('expand_frame_repr', False)
    pd.set_option('display.max_rows', 500)
    index = list(res.keys())
    columns = list(list(res.values())[0].keys())
    data = [list(v.values()) for v in res.values()]
    df = pd.DataFrame(data=data, index=index, columns=columns)
    return df.__repr__()

# yuv to rgb
def yuv2gray(yuv,h,w):
    gray = yuv[:w*h]
    gray = gray.reshape(h, w)
    return gray

def yuv2rgb(yuv,h,w):
    w2, h2 = int(w/2), int(h/2)
    y = yuv[:(w*h)].reshape(h, w)
    u = yuv[(w*h):(w*h+w2*h2)].reshape(h2, w2)
    v = yuv[(w*h+w2*h2):].reshape(h2, w2)

    u = cv2.resize(u, (0,0), fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    v = cv2.resize(v, (0,0), fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    rgb = cv2.merge([y,u,v])
    rgb = cv2.cvtColor(rgb, cv2.COLOR_YUV2BGR)
    return rgb

def read_yuv(path, height, width, rgb=True):
    if os.path.exists(path):
        with open(path, 'rb') as f:
            img = np.fromstring(f.read(), np.uint8)
            if img is not None:
                if rgb:
                    return yuv2rgb(img, height, width)
                else:
                    return yuv2gray(img, height, width)
            else:
                print("Load {} fail".format(path))
                return None
    else:
        print("Could not find {}".format(path))
        return None

# origin names to label names
ground_keys = ['contour_chin','contour_left1','contour_left2','contour_left3','contour_left4','contour_left5','contour_left6','contour_left7','contour_left8','contour_left9','contour_right1','contour_right2','contour_right3','contour_right4','contour_right5','contour_right6','contour_right7','contour_right8','contour_right9','left_eye_bottom','left_eye_left_corner','left_eye_lower_left_quarter','left_eye_lower_right_quarter','left_eye_pupil','left_eye_right_corner','left_eye_top','left_eye_upper_left_quarter','left_eye_upper_right_quarter','left_eyebrow_left_corner','left_eyebrow_lower_left_quarter','left_eyebrow_lower_middle','left_eyebrow_lower_right_quarter','left_eyebrow_right_corner','left_eyebrow_upper_left_quarter','left_eyebrow_upper_middle','left_eyebrow_upper_right_quarter','mouth_left_corner','mouth_lower_lip_bottom','mouth_lower_lip_left_contour1','mouth_lower_lip_left_contour2','mouth_lower_lip_left_contour3','mouth_lower_lip_right_contour1','mouth_lower_lip_right_contour2','mouth_lower_lip_right_contour3','mouth_lower_lip_top','mouth_right_corner','mouth_upper_lip_bottom','mouth_upper_lip_left_contour1','mouth_upper_lip_left_contour2','mouth_upper_lip_left_contour3','mouth_upper_lip_right_contour1','mouth_upper_lip_right_contour2','mouth_upper_lip_right_contour3','mouth_upper_lip_top','nose_contour_left1','nose_contour_left2','nose_contour_left3','nose_contour_lower_middle','nose_contour_right1','nose_contour_right2','nose_contour_right3','nose_left','nose_right','nose_tip','right_eye_bottom','right_eye_left_corner','right_eye_lower_left_quarter','right_eye_lower_right_quarter','right_eye_pupil','right_eye_right_corner','right_eye_top','right_eye_upper_left_quarter','right_eye_upper_right_quarter','right_eyebrow_left_corner','right_eyebrow_lower_left_quarter','right_eyebrow_lower_middle','right_eyebrow_lower_right_quarter','right_eyebrow_right_corner','right_eyebrow_upper_left_quarter','right_eyebrow_upper_middle','right_eyebrow_upper_right_quarter']

label_keys = ['left_eye_pupil','left_eye_left_corner','left_eye_right_corner','left_eye_top','left_eye_bottom','left_eye_upper_left_quarter','left_eye_lower_left_quarter','left_eye_upper_right_quarter','left_eye_lower_right_quarter','right_eye_pupil','right_eye_left_corner','right_eye_right_corner','right_eye_top','right_eye_bottom','right_eye_upper_left_quarter','right_eye_lower_left_quarter','right_eye_upper_right_quarter','right_eye_lower_right_quarter','left_eyebrow_left_corner','left_eyebrow_right_corner','left_eyebrow_upper_middle','left_eyebrow_lower_middle','left_eyebrow_upper_left_quarter','left_eyebrow_lower_left_quarter','left_eyebrow_upper_right_quarter','left_eyebrow_lower_right_quarter','right_eyebrow_left_corner','right_eyebrow_right_corner','right_eyebrow_upper_middle','right_eyebrow_lower_middle','right_eyebrow_upper_left_quarter','right_eyebrow_lower_left_quarter','right_eyebrow_upper_right_quarter','right_eyebrow_lower_right_quarter','nose_tip','nose_contour_lower_middle','nose_contour_left1','nose_contour_right1','nose_contour_left2','nose_contour_right2','nose_left','nose_right','nose_contour_left3','nose_contour_right3','mouth_left_corner','mouth_right_corner','mouth_upper_lip_top','mouth_upper_lip_bottom','mouth_upper_lip_left_contour1','mouth_upper_lip_right_contour1','mouth_upper_lip_left_contour2','mouth_upper_lip_right_contour2','mouth_upper_lip_left_contour3','mouth_upper_lip_right_contour3','mouth_lower_lip_top','mouth_lower_lip_bottom','mouth_lower_lip_left_contour1','mouth_lower_lip_right_contour1','mouth_lower_lip_left_contour2','mouth_lower_lip_left_contour3','mouth_lower_lip_right_contour3','mouth_lower_lip_right_contour2','contour_left1','contour_right1','contour_chin','contour_left2','contour_left3','contour_left4','contour_left5','contour_left6','contour_left7','contour_left8','contour_left9','contour_right2','contour_right3','contour_right4','contour_right5','contour_right6','contour_right7','contour_right8','contour_right9']

# {'x': x, 'y': y}
def landmark2label(gt, width, height):
    label_names = {name: i for i, name in enumerate(label_keys)}
    ground_names = {name: i for i, name in enumerate(ground_keys)}
    ground2label = [ground_names[i] for i in label_keys]
    label2ground = [label_names[i] for i in ground_keys]

    gt = gt.reshape(-1,2)
    out = []
    for x,y in gt[ground2label]:
        x = x / width
        y = y / height
        x = min(x,1.8)
        x = max(x,-0.8)
        y = min(y,1.8)
        y = max(y,-0.8)
        out.append({'x': x, 'y': y})
    return out

def label2landmark(gt, width, height):
    label_names = {name: i for i, name in enumerate(label_keys)}
    ground_names = {name: i for i, name in enumerate(ground_keys)}
    ground2label = [ground_names[i] for i in label_keys]
    label2ground = [label_names[i] for i in ground_keys]

    gt = np.array(gt).reshape(-1,2)
    out = []
    for x,y in gt[label2ground]:
        out.append([x*width, y*height])
    out = np.array(out).reshape(-1,2)
    return out

# origin names to standard megface names (dim 84)
megface_labels={0:'left_eyebrow_left_corner',1:'left_eyebrow_right_corner',2:'left_eyebrow_upper_left_quarter',3:'left_eyebrow_upper_middle',4:'left_eyebrow_upper_right_quarter',5:'left_eyebrow_lower_left_quarter',6:'left_eyebrow_lower_middle',7:'left_eyebrow_lower_right_quarter',16:'right_eyebrow_left_corner',17:'right_eyebrow_right_corner',18:'right_eyebrow_upper_left_quarter',19:'right_eyebrow_upper_middle',20:'right_eyebrow_upper_right_quarter',21:'right_eyebrow_lower_left_quarter',22:'right_eyebrow_lower_middle',23:'right_eyebrow_lower_right_quarter',32:'left_eye_left_corner',33:'left_eye_right_corner',34:'left_eye_top',35:'left_eye_bottom',36:'left_eye_center',37:'left_eye_pupil',38:'left_eye_upper_left_quarter',39:'left_eye_upper_right_quarter',40:'left_eye_lower_left_quarter',41:'left_eye_lower_right_quarter',48:'right_eye_left_corner',49:'right_eye_right_corner',50:'right_eye_top',51:'right_eye_bottom',52:'right_eye_center',53:'right_eye_pupil',54:'right_eye_upper_left_quarter',55:'right_eye_upper_right_quarter',56:'right_eye_lower_left_quarter',57:'right_eye_lower_right_quarter',64:'nose_left',65:'nose_right',66:'nose_tip',67:'nose_contour_lower_middle',68:'nose_contour_left1',69:'nose_contour_left2',70:'nose_contour_left3',71:'nose_contour_right1',72:'nose_contour_right2',73:'nose_contour_right3',80:'mouth_left_corner',81:'mouth_right_corner',82:'mouth_upper_lip_top',83:'mouth_upper_lip_bottom',84:'mouth_lower_lip_top',85:'mouth_lower_lip_bottom',86:'mouth_upper_lip_left_contour1',87:'mouth_upper_lip_left_contour2',88:'mouth_upper_lip_left_contour3',89:'mouth_upper_lip_right_contour1',90:'mouth_upper_lip_right_contour2',91:'mouth_upper_lip_right_contour3',92:'mouth_lower_lip_left_contour1',93:'mouth_lower_lip_left_contour2',94:'mouth_lower_lip_left_contour3',95:'mouth_lower_lip_right_contour1',96:'mouth_lower_lip_right_contour2',97:'mouth_lower_lip_right_contour3',98:'mouth_middle',112:'contour_chin',113:'contour_left1',114:'contour_left2',115:'contour_left3',116:'contour_left4',117:'contour_left5',118:'contour_left6',119:'contour_left7',120:'contour_left8',121:'contour_left9',145:'contour_right1',146:'contour_right2',147:'contour_right3',148:'contour_right4',149:'contour_right5',150:'contour_right6',151:'contour_right7',152:'contour_right8',153:'contour_right9'}
megface_idxes=[0,1,2,3,4,5,6,7,16,17,18,19,20,21,22,23,32,33,34,35,36,37,38,39,40,41,48,49,50,51,52,53,54,55,56,57,64,65,66,67,68,69,70,71,72,73,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,112,113,114,115,116,117,118,119,120,121,145,146,147,148,149,150,151,152,153]

# face['landmark']['points']
def landmark2megface(ld):
    ld = ld.reshape(-1,2)
    ground_idxes = {name:i for i, name in enumerate(ground_keys)}
    res = {}
    for i in megface_idxes:
        name = megface_labels[i]
        if name in ground_idxes:
            idx = ground_idxes[name]
            res[i] = {'x':ld[idx,0], 'y':ld[idx,1]}
    res[36] = res[37].copy()
    res[52] = res[53].copy()
    mouth = ld[36:54].mean(axis=0)
    res[98] = {'x': mouth[0], 'y':mouth[1]}
    resout = []
    for i in megface_idxes:
        resout.append(res[i])
    return resout


def megface2landmark(face):
    megface_keys = {megface_labels[i]:idx for idx, i in enumerate(megface_idxes)}
    #megface_keys = {megface_labels[i]:i for i in megface_labels}
    ground_idxes = [megface_keys[i] for i in ground_keys]
    res = []
    for i in range(len(ground_idxes)):
        pt = face[ground_idxes[i]]
        res.append([pt['x'], pt['y']])
    res = np.array(res).reshape(-1,2)
    return res

# origin names to megface feature names (dim 83)
megfacefeat_names = ['CONTOUR_CHIN','CONTOUR_LEFT1','CONTOUR_LEFT2','CONTOUR_LEFT3','CONTOUR_LEFT4','CONTOUR_LEFT5','CONTOUR_LEFT6','CONTOUR_LEFT7','CONTOUR_LEFT8','CONTOUR_LEFT9','CONTOUR_RIGHT1','CONTOUR_RIGHT2','CONTOUR_RIGHT3','CONTOUR_RIGHT4','CONTOUR_RIGHT5','CONTOUR_RIGHT6','CONTOUR_RIGHT7','CONTOUR_RIGHT8','CONTOUR_RIGHT9','LEFTEYE_BOTTOM','LEFTEYE_LEFTCORNER','LEFTEYE_LOWERLEFTQUARTER','LEFTEYE_LOWERRIGHTQUARTER','LEFTEYE_PUPIL','LEFTEYE_RIGHTCORNER','LEFTEYE_TOP','LEFTEYE_UPPERLEFTQUARTER','LEFTEYE_UPPERRIGHTQUARTER','LEFTEYEBROW_LEFTCORNER','LEFTEYEBROW_LOWERLEFTQUARTER','LEFTEYEBROW_LOWERMIDDLE','LEFTEYEBROW_LOWERRIGHTQUARTER','LEFTEYEBROW_RIGHTCORNER','LEFTEYEBROW_UPPERLEFTQUARTER','LEFTEYEBROW_UPPERMIDDLE','LEFTEYEBROW_UPPERRIGHTQUARTER','MOUTH_LEFTCORNER','MOUTH_LOWERLIPBOTTOM','MOUTH_LOWERLIPLEFTCONTOUR1','MOUTH_LOWERLIPLEFTCONTOUR2','MOUTH_LOWERLIPLEFTCONTOUR3','MOUTH_LOWERLIPRIGHTCONTOUR1','MOUTH_LOWERLIPRIGHTCONTOUR2','MOUTH_LOWERLIPRIGHTCONTOUR3','MOUTH_LOWERLIPTOP','MOUTH_RIGHTCORNER','MOUTH_UPPERLIPBOTTOM','MOUTH_UPPERLIPLEFTCONTOUR1','MOUTH_UPPERLIPLEFTCONTOUR2','MOUTH_UPPERLIPLEFTCONTOUR3','MOUTH_UPPERLIPRIGHTCONTOUR1','MOUTH_UPPERLIPRIGHTCONTOUR2','MOUTH_UPPERLIPRIGHTCONTOUR3','MOUTH_UPPERLIPTOP','NOSE_CONTOURLEFT1','NOSE_CONTOURLEFT2','NOSE_CONTOURLEFT3','NOSE_CONTOURLOWERMIDDLE','NOSE_CONTOURRIGHT1','NOSE_CONTOURRIGHT2','NOSE_CONTOURRIGHT3','NOSE_LEFT','NOSE_RIGHT','NOSE_TIP','RIGHTEYE_BOTTOM','RIGHTEYE_LEFTCORNER','RIGHTEYE_LOWERLEFTQUARTER','RIGHTEYE_LOWERRIGHTQUARTER','RIGHTEYE_PUPIL','RIGHTEYE_RIGHTCORNER','RIGHTEYE_TOP','RIGHTEYE_UPPERLEFTQUARTER','RIGHTEYE_UPPERRIGHTQUARTER','RIGHTEYEBROW_LEFTCORNER','RIGHTEYEBROW_LOWERLEFTQUARTER','RIGHTEYEBROW_LOWERMIDDLE','RIGHTEYEBROW_LOWERRIGHTQUARTER','RIGHTEYEBROW_RIGHTCORNER','RIGHTEYEBROW_UPPERLEFTQUARTER','RIGHTEYEBROW_UPPERMIDDLE','RIGHTEYEBROW_UPPERRIGHTQUARTER']

'''
def landmark2megfacefeat(ld):
    if len(ld_dict) == 81:
        ld_dict[LEFTEYE_CENTER.name] = ld_dict[LEFTEYE_PUPIL.name]
        ld_dict[RIGHTEYE_CENTER.name] = ld_dict[RIGHTEYE_PUPIL.name]
    ld_dict = {}
    for pt, tag in zip(face['points'], face['tags']):
        tag = LandmarkTag(tag)
        ld_dict[tag.name] = (float(pt['x']), float(pt['y']))
        ld_dict = ld_81p_to_83p(ld_dict)
    return ld_dict
'''

def megfacefeat2landmark(kv_dict):
    assert isinstance(kv_dict, dict), type(kv_dict)
    #assert len(kv_dict) == len(megfacefeat_names), '{} vs {}'.format(len(kv_dict), len(megfacefeat_names))
    lm = np.zeros((len(megfacefeat_names), 2))
    for k, v in kv_dict.items():
        if k not in megfacefeat_names:
            continue
        lm[megfacefeat_names.index(k)] = np.float32(v)
    return lm.astype(np.float32)

fn = nori.Fetcher()
def unpack_lm(nori_id, **kwargs):
    r = pkl.loads(fn.get(nori_id))
    img = str2img(r.pop('img'))
    if 'ld' in r:
        lm = r.pop('ld')
    elif 'lm' in r:
        lm = r.pop('lm')
    else:
        lm = None
    if isinstance(lm, dict):
        lm = megfacefeat2landmark(lm)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)[:, :, np.newaxis] if not kwargs.get('is_color', False) else img
    return img, lm, r

def unpack_img(nori_id, **kwargs):
    img_str = fn.get(nori_id)
    if img_str is None:
        return None
    img = str2img(img_str)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)[:, :, np.newaxis] if not kwargs.get('is_color', False) else img
    return img

def get_rng(seed=None):
    if seed is None:
        import time
        seed = int(time.time()) * 1000 % (2**32-1)
    rng = np.random.RandomState(seed)
    return rng

def choose2d(data, num=1, seed=11, rng=None):
    rng = rng or np.random.RandomState(seed)
    select_values = []
    for i in range(num):
        key1 = rng.choice(len(data), 1)[0]
        key2 = rng.choice(len(data[key1]), 1)[0]
        select_values.append(data[key1][key2])
    return select_values


# Copied from:
# https://git-core.megvii-inc.com/brain-user/MegHair/blob/master/meghair/utils/imgproc/__init__.py
def imdecode(data, *, require_chl3=True, require_alpha=False):
    """decode images in common formats (jpg, png, etc.)
    :param data: encoded image data
    :type data: :class:`bytes`
    :param require_chl3: whether to convert gray image to 3-channel BGR image
    :param require_alpha: whether to add alpha channel to BGR image
    :rtype: :class:`numpy.ndarray`
    """
    img = cv2.imdecode(np.fromstring(data, np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None and len(data) >= 3 and data[:3] == b'GIF':
        # cv2 doesn't support GIF, try PIL
        img = _gif_decode(data)
    assert img is not None, 'failed to decode'
    if img.ndim == 2 and require_chl3:
        img = img.reshape(img.shape + (1,))
    if img.shape[2] == 1 and require_chl3:
        img = np.tile(img, (1, 1, 3))
    if img.ndim == 3 and img.shape[2] == 3 and require_alpha:
        assert img.dtype == np.uint8
        img = np.concatenate([img, np.ones_like(img[:, :, :1]) * 255], axis=2)
    return img

def _gif_decode(data):
    try:
        import io
        from PIL import Image
        im = Image.open(io.BytesIO(data))
        im = im.convert('RGB')
        return cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
    except Exception:
        return

def jpeg_encode(img, quality=90):
    '''
    :param img: uint8 color image array
    :type img: :class:`numpy.ndarray`
    :param int quality: quality for JPEG compression
    :return: encoded image data
    '''
    return cv2.imencode('.jpg', img,
                        [int(cv2.IMWRITE_JPEG_QUALITY), quality])[1].tostring()

def png_encode(input, compress_level=3):
    '''
    :param numpy.ndarray input: uint8 color image array
    :param int compress_level: compress level for PNG compression
    :return: encoded image data
    '''
    # assert len(input.shape) == 3 and input.shape[2] in [1, 3, 4]
    # assert input.dtype == np.uint8
    assert isinstance(compress_level, int) and 0 <= compress_level <= 9
    enc = cv2.imencode('.png', input,
                       [int(cv2.IMWRITE_PNG_COMPRESSION), compress_level])
    return enc[1].tostring()

def imread(path):
    with open(path, 'rb') as file:
        return imdecode(file.read())

def imwrite(path, img):
    cv2.imwrite(path, img)
