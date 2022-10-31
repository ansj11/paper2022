import os
import pickle as pkl
import numpy as np
import hashlib
from multiprocessing import Process, Pipe
from IPython import embed
import cv2
import hashlib

import mgfpy
from landstack.megface.detectcore import loadyuv, detect, MegFaceAPI
from landstack.megface.lmkcore import loadmodel, getattr, getlm

lmklargepath = '/unsullied/sharefs/_research_facelm/Isilon-modelshare/model_zoo/release/lmk.detect.v4.xlarge.171020/model.pred.prob'
lmksmallpath = '/unsullied/sharefs/_research_facelm/Isilon-modelshare/model_zoo/temporary/model.fast/lmk_fast170323base5ms'
posepath = '/unsullied/sharefs/_research_facelm/Isilon-modelshare/model_zoo/temporary/model.extra/lmk_extra170620pose_concate'

mean_shape = pkl.load(open('/unsullied/sharefs/_research_facelm/Isilon-datashare/baseimg/extra/mean_face_81.pkl', 'rb'))
mean_shape = np.array(mean_shape).reshape(-1,2)

def initModels(device):
    mgf = MegFaceAPI(
        megface_lib_path='/unsullied/sharefs/_research_facelm/Isilon-modelshare/megface/megface26/lib/libmegface.so',
        face_model_root='/unsullied/sharefs/_research_facelm/Isilon-modelshare/model_zoo/release_conf/Model/',
        version='2.6',
        device=device,
    )
    mgf.register_det_rect_config('detector_rect.densebox.xlarge.v1.3.conf')
    mgf.register_det_81_config('lmk.postfilter.xlarge.v1.2.conf')

    lmklarge = loadmodel(lmklargepath, 'pred')
    lmksmall = loadmodel(lmksmallpath, 'pred')
    posemodel = loadmodel(posepath, 'pose')
    return mgf, lmklarge, lmksmall, posemodel

def preprocess(img, mgfdet, lmkmodel, lmksmlmodel=None, posemodel=None, maxface=False, thres=0.2):
    faces = detect(mgfdet, img, stage='det_81', thres=thres, maxface=maxface, gt=None, options={'orient':mgfpy.Orient.MGF_UP})
    if faces is None:
        return []
    outpacks = []
    for face in faces:
        if face['conf']<thres:
            continue
        ld = getlm(img, face['pts'], lmkmodel)
        ld = ld.reshape(-1,2)
        ldsml = None
        if lmksmlmodel is not None:
            ldsml = getlm(img, face['pts'], lmksmlmodel)
            ldsml = ldsml.reshape(-1,2)
        pose = None
        if posemodel is not None:
            pose = getattr(img, ld, posemodel)
        #print(ld.shape, ldsml.shape, pose)
        outpacks.append({'ld':ld, 'ldsml':ldsml, 'pose':pose})
    return outpacks

def cropface(img, gt, gt1=None):
    gt = gt.reshape(-1,2)
    x0,y0,x1,y1 = gt[:,0].min(),gt[:,1].min(),gt[:,0].max(),gt[:,1].max()
    w = (x1-x0)*1.0
    h = (y1-y0)*1.0
    x0 = int(max(0, x0-w))
    y0 = int(max(0, y0-h))
    x1 = int(min(img.shape[1], x1+w))
    y1 = int(min(img.shape[0], y1+h))
    img = img[y0:y1, x0:x1, :]
    gt[:,0] = (gt[:,0] - x0) / img.shape[1]
    gt[:,1] = (gt[:,1] - y0) / img.shape[0]
    if gt1 is not None:
        gt1[:,0] = (gt1[:,0] - x0) / img.shape[1]
        gt1[:,1] = (gt1[:,1] - y0) / img.shape[0]

    scale = min(800/img.shape[0], 800/img.shape[1])
    if scale < 1:
        outwidth = int(img.shape[1] * scale)
        outheight = int(img.shape[0] * scale)
        img = cv2.resize(img,(outwidth,outheight),interpolation=cv2.INTER_CUBIC)
    gt[:,0] *= img.shape[1]
    gt[:,1] *= img.shape[0]
    if gt1 is not None:
        gt1[:,0] *= img.shape[1]
        gt1[:,1] *= img.shape[0]

    if gt1 is not None:
        return img, gt, gt1
    else:
        return img, gt

# pack: ld, ldsml, pose,
def chooseface(img, pack):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    pose = pack['pose'].reshape(-1)
    outflag = getoutflag(gray, pack['ld'])
    blur = getblur2(gray, pack['ld'])
    err = geterrcomp3(pack['ld'], pack['ldsml'])
    light = getlight2(gray, pack['ld'])

    '''
    pack['blur'] = blur
    pack['err'] = err
    pack['light'] = light
    '''

    if outflag>10:
        pack['type'] = 'out'
    elif blur<150 and light not in ['dark']:
        pack['type'] = 'blur'
    else:
        pack['type'] = '%s_%s'%(light, getpose(pose))
    return pack

def imencode(img):
    res, img = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    img = np.array(img, np.uint8).tostring()
    return img

def _md5(s):
    m = hashlib.md5()
    m.update(s)
    return m.hexdigest()

def getmd5name(path):
    newname = _md5(path.encode("utf8"))
    return newname

##########################################################################
# basic image
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

########################################################################
def rot_scale_align(src, dst):
    src_x, src_y = src[:, 0], src[:, 1]
    dst_x, dst_y = dst[:, 0], dst[:, 1]
    d = (src**2).sum()
    a = sum(src_x*dst_x + src_y*dst_y) / d
    b = sum(src_x*dst_y - src_y*dst_x) / d
    mat = np.array([[a, -b], [b, a]])
    return mat

def alignto(lm, mean_face, output_size):
    lm = np.array(lm).reshape(-1, 2)
    mean = lm.mean(axis=0)
    mat1 = rot_scale_align(lm - mean, mean_face).T
    mat1 *= output_size
    mat2 = np.float64([[mat1[0][0], mat1[1][0], -mean[0] * mat1[0][0] - mean[1] * mat1[1][0] + output_size // 2],
                       [mat1[0][1], mat1[1][1], -mean[0] * mat1[0][1] - mean[1] * mat1[1][1] + output_size // 2]])
    return mat2

def align_to_mean_face(ld, mean_shape, inpsize):
    ld = np.array(ld).reshape(-1, 2)
    return alignto(ld[:81,:], mean_shape, inpsize)

def lm_affine(lm, mat):
    lm = np.array(lm).reshape(-1, 2)
    return np.dot(np.concatenate([lm, np.ones((lm.shape[0], 1))], axis=1), mat.T)

def inv_mat(mat):
    return np.linalg.inv(mat.tolist() + [[0, 0, 1]])[:2]

def saveimg(img, ld, ldsml, err, nid, outdir):
    pimg = cv2.imdecode(np.fromstring(img, np.uint8), cv2.IMREAD_COLOR)
    for i in range(len(ld)):
        cv2.circle(pimg, (int(ld[i,0]), int(ld[i,1])), 1, (255,0,0), 1)
        cv2.circle(pimg, (int(ldsml[i,0]), int(ldsml[i,1])), 2, (0,0,255), 2)
    outpath = '%s/%.2f_%s.jpg'%(outdir,err,nid)
    cv2.imwrite(outpath, pimg)

def getalign(img, ld, ldsml):
    INPSIZE = 200
    pimg = cv2.imdecode(np.fromstring(img, np.uint8), cv2.IMREAD_COLOR)
    mat = alignto(ld, mean_shape, INPSIZE)
    pimg = cv2.warpAffine(pimg, mat, (INPSIZE, INPSIZE), borderMode=cv2.BORDER_REPLICATE)
    pld = lm_affine(ld, mat)
    pldsml = lm_affine(ldsml, mat)
    for i in range(len(pld)):
        cv2.circle(pimg, (int(pld[i,0]), int(pld[i,1])), 1, (0,0,255), 1)
        cv2.circle(pimg, (int(pldsml[i,0]), int(pldsml[i,1])), 2, (255,0,0), 2)
    return pimg

def split(label):
    comps = dict()
    comps['contour'] = label[:19, :]
    #comps['lefteye'] = label[19:28, :]
    #comps['righteye'] = label[64:73, :]
    #comps['lefteyebrow'] = label[28:36, :]
    #comps['righteyebrow'] = label[73:, :]
    comps['eye'] = np.concatenate([label[19:28, :], label[64:73, :]], axis=1)
    comps['eyebrow'] = np.concatenate([label[28:36, :], label[73:, :]], axis=1)
    comps['mouth'] = label[36:54, :]
    comps['nose'] = label[54:64, :]
    return comps

def geterr(ld, ldsml, badlist, confs):
    norm = np.sum((ld[23]-ld[68])**2)**0.5
    comps_ld = split(ld)
    comps_ldsml = split(ldsml)
    errs = []
    for i in comps_ld:
        if i in badlist:
            continue
        err = np.mean(np.sum(((comps_ld[i]-comps_ldsml[i])**2), axis=1)**0.5/norm) * confs[i]
        errs.append(err)
    maxerr = np.array(errs).max()
    return maxerr

#get allaver:
def geterrcomp3(ld, ldsml):
    norm = np.sum((ld[23]-ld[68])**2)**0.5
    err = np.mean(np.sum(((ld-ldsml)**2), axis=1)**0.5/norm)
    errs = {'all':err}
    return errs

# ger aver
def geterrcomp2(ld, ldsml):
    norm = np.sum((ld[23]-ld[68])**2)**0.5
    comps_ld = split(ld)
    comps_ldsml = split(ldsml)
    errs = {}
    for i in comps_ld:
        err = np.mean(np.sum(((comps_ld[i]-comps_ldsml[i])**2), axis=1)**0.5/norm)
        errs[i] = err
    return errs

# get 5 maxerr points
def geterrcomp(ld, ldsml):
    norm = np.sum((ld[23]-ld[68])**2)**0.5
    comps_ld = split(ld)
    comps_ldsml = split(ldsml)
    errs = {}
    for i in comps_ld:
        err = np.sum(((comps_ld[i]-comps_ldsml[i])**2), axis=1)**0.5
        err = sorted(err, reverse=True)
        err = np.mean(err[:5])/norm
        errs[i] = err
    return errs

def getlight(img, pt81):
    mat = align_to_mean_face(pt81, mean_shape, 200)
    img = cv2.warpAffine(img, mat, (200, 200), borderMode=cv2.BORDER_REPLICATE)

    pimg = np.array(img[40:160, 40:160])
    meanv = pimg.mean()
    img1 = pimg[:60,:60]
    img2 = pimg[:60,60:]
    img3 = pimg[60:,:60]
    img4 = pimg[60:,60:]

    mean1 = img1.mean()
    mean2 = img2.mean()
    mean3 = img3.mean()
    mean4 = img4.mean()
    maxm = np.array([mean1,mean2,mean3,mean4]).max()
    minm = np.array([mean1,mean2,mean3,mean4]).min()
    if maxm < 50:
        return 'dark'
    elif maxm-minm > 40:
        return 'bright'
    else:
        return 'normal'
    #print(mean1, mean2, mean3, mean4)

def getlight2(img, pt81):
    mat = align_to_mean_face(pt81, mean_shape, 200)
    img = cv2.warpAffine(img, mat, (200, 200), borderMode=cv2.BORDER_REPLICATE)

    pimg = np.array(img[30:170, 30:170])
    meanv = pimg.mean()
    img1 = pimg[:70,:70]
    img2 = pimg[:70,70:]
    img3 = pimg[70:,:70]
    img4 = pimg[70:,70:]
    img5 = pimg[60:140,60:140]

    mean1 = img1.mean()
    mean2 = img2.mean()
    mean3 = img3.mean()
    mean4 = img4.mean()
    mean5 = img5.mean()
    maxm = np.array([mean1,mean2,mean3,mean4,mean5]).max()
    minm = np.array([mean1,mean2,mean3,mean4,mean5]).min()
    if maxm < 50:
        return 'dark'
    elif maxm-minm > 40:
        return 'bright'
    else:
        return 'normal'
    #print(mean1, mean2, mean3, mean4)

def getpose(pose):
    x = pose[0] / 3.1415926 * 20
    y = pose[1] / 3.1415926 * 20

    if np.fabs(y)<=2:
        if x<=2.5 and x>=-2.5:
            pose = 'front'
        elif x>2.5:
            pose = 'down'
        else:
            pose = 'up'
    elif np.fabs(y)<=5:
        pose = 'halfprofile'
    else:
        pose = 'profile'
    return pose

def getpose2(pose):
    x = pose[0] / 3.1415926 * 20
    y = pose[1] / 3.1415926 * 20

    if np.fabs(y)>5:
        pose = 'profile'
    elif np.fabs(x)>4:
        pose = 'updown'
    elif np.fabs(x)>2 or np.fabs(y)>2:
        pose = 'profile_updown'
    else:
        pose = 'front'

    return pose


def getoutflag(pimg, pt):
    height, width = pimg.shape
    outnum = 0
    for i in range(len(pt)):
        if pt[i,0]<0 or pt[i,1]<0 or pt[i,0]>width or pt[i,1]>height:
            outnum += 1
    #print(outnum)
    return outnum

def getblur(pimg, pt81):
    mat = align_to_mean_face(pt81, mean_shape, 80)
    pimg = cv2.warpAffine(pimg, mat, (80, 80), borderMode=cv2.BORDER_REPLICATE)

    height, width = pimg.shape
    sumVarVer, sumVarHor, sumDiffVer, sumDiffHor = 0,0,0,0
    k = 16
    scale = 1.0/(k*2-1)
    for i in range(k, height-k, 1):
        for j in range(k, width-k, 1):
            diffVer_ = np.abs(pimg[i, j] - pimg[i-1, j])
            diffHor_ = np.abs(pimg[i, j] - pimg[i, j-1])
            diffBlurVer_ = np.abs(pimg[i+k-1, j] - pimg[i-k, j]) * scale
            diffBlurHor_ = np.abs(pimg[i, j+k-1] - pimg[i, j-k]) * scale
            varVer_ = max(0, diffVer_ - diffBlurVer_)
            varHor_ = max(0, diffHor_ - diffBlurHor_)
            sumDiffVer += diffVer_
            sumDiffHor += diffHor_
            sumVarVer += varVer_
            sumVarHor += varHor_
    if sumDiffVer==0 or sumDiffHor==0:
        return 0
    blur = max((sumDiffVer - sumVarVer) / sumDiffVer, (sumDiffHor - sumVarHor) / sumDiffHor)
    '''
    print(blur)
    cv2.namedWindow("Image")
    cv2.imshow("Image", pimg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    '''
    return blur

def getblur2(pimg, pt81):
    mean_shapet = mean_shape * 1.2
    mat = align_to_mean_face(pt81, mean_shapet, 64)
    pimg = cv2.warpAffine(pimg, mat, (64, 64), borderMode=cv2.BORDER_REPLICATE)
    blur = cv2.Laplacian(pimg, cv2.CV_64F).var()
    return blur


