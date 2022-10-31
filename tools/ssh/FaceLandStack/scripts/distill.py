import os
import argparse
import json
from collections import OrderedDict
import cv2
import pickle as pkl
from landstack.utils import misc
import numpy as np
from landstack.utils import augment, transforms, geom
from IPython import embed
import rrun
import threading
import concurrent
import nori2 as nori
from nori2.multi import  MultiSourceReader
from multiprocessing import Queue, Manager, Process
from landstack.train import env
from landstack.megface.detectcore import MegFaceAPI, detect
from functools import lru_cache
import tqdm

def get_executor(name='cleaning task', nr_jobs=1, cpu=2, gpu=1, memory=10240, threads=16, is_kill=False):
    spec = rrun.RunnerSpec()
    spec.name = name
    spec.log_dir = os.path.expanduser('~/eval_logs')
    spec.scheduling_hint.group = 'users'
    spec.resources.cpu = cpu
    spec.resources.gpu = gpu
    spec.resources.memory_in_mb = memory
    spec.max_wait_time = 3600 * int(1e9)
    spec.preemptible = is_kill is True
    return rrun.RRunExecutor(spec, num_runners=nr_jobs, num_threads=threads)


def get_todo_list(source):
    todo_list = []
    if source.endswith('nori'):
        nori_file = os.path.join(source, 'meta.bin')
        with open(nori_file) as f:
            for line in list(f.readlines())[1:]:
                dataid = json.loads(line)['DataID']
                todo_list.append(dataid)
    elif os.path.isdir(source):
        for root, dirs, files in os.walk(source):
            for fname in files:
                if fname.lower().endswith('png') or fname.lower().endswith('jpg'):
                    full_path = os.path.join(root, fname)
                    todo_list.append(full_path)
    else:
        data = misc.load_pickle(source)
        if isinstance(data, (list, tuple)):
            todo_list = data
        else:
            assert isinstance(data, dict), type(data)
            todo_list = data.keys()
    return todo_list


@lru_cache(maxsize=1)
def worker_init(flag=True):
    if flag:
        print("123")

    pose_model_file = '/unsullied/sharefs/_research_facelm/Isilon-modelshare/model_zoo/temporary/model.extra/pose.crop2.base_base.concate/model.bin'
    pose_net = misc.load_network_and_extract(pose_model_file)
    inf_func_pose = env.compile_inf_func(pose_net.outputs, devices='gpu0')

    root = '/unsullied/sharefs/_research_facelm/Isilon-datashare'
    mean_face_file = os.path.join(root, 'baseimg', 'extra', 'mean_face_81.pkl')
    aug = augment.AugmentBase(
        mean_face=mean_face_file, image_shape=(112, 112),
        transforms_post=transforms.Compose([
            transforms.HWC2CHW(),
        ])
    )

    mgf = MegFaceAPI(
        megface_lib_path='/unsullied/sharefs/chenxi/isilon-share/shares/public/megface-v2/lib/libmegface.so',
        device='gpu0',
        version='2.5',
    )
    mgf.register_lm_81_config('detector.xlarge.v3.conf')

    return mgf, inf_func_pose, aug

def worker(t):
    if os.path.exists(t):
        img = cv2.imread(t, 0)
        if img is None:
            return None
        img = img[:, :, np.newaxis]
    else:
        try:
            img = misc.unpack_lm(t, is_color=False)[0]
        except:
            img = misc.unpack_img(t, is_color=False)

    mgf, inf_func_pose, aug = worker_init(True)
    res = detect(mgf, img, thres=0.9)
    if res is None:
        return None
    d_list = []
    for res0 in res:
        lm = res0['pts']
        img_out, lm_out, _ = aug(img, lm, lm.copy())
        pitch, yaw = env.forward(inf_func_pose, {'img':np.uint8(img_out).reshape(1, 1, 112, 112)})['pose'].flatten()
        d = dict(
            lm =lm,
            pitch=pitch,
            yaw=yaw
        )
        d_list.append(d)
    return d_list



def main():
    parser = argparse.ArgumentParser(description='run test script on benchmark')
    parser.add_argument(dest='source', help='input path, file or folder', default=None)
    parser.add_argument(dest='target', help='save path', default=None)
    parser.add_argument('--gpu', default=8, type=int)
    parser.add_argument('--kill', action='store_true')
    parser.add_argument('--reader_threads', default=16, type=int)
    parser.add_argument('-o', '--output', help='output path', default=None)
    args, unknownargs  = parser.parse_known_args()

    args.source = os.path.realpath(args.source)
    args.target = os.path.realpath(args.target)
    assert args.target.startswith('/unsullied'), args.target

    # load tasks
    todo_list = get_todo_list(args.source)
    todo_list = todo_list

    executors = []
    for i in range(args.gpu):
        executor = get_executor(name='gpu', cpu=2, gpu=1, memory=10240, threads=1, is_kill=args.kill)
        executors.append(executor)

    futures_dict = OrderedDict()
    for i, t in enumerate(todo_list):
        embed()
        executor = executors[i % len(executors)]
        fu = executor.submit(worker, t)
        futures_dict[fu] = t

    res_dict = OrderedDict()
    for future in tqdm.tqdm(concurrent.futures.as_completed(futures_dict.keys()), total=len(todo_list)):
        d_list = future.result()
        t = futures_dict[future]
        res_dict[t] = d_list

    misc.ensure_dir(os.path.dirname(args.target))
    misc.dump_pickle(res_dict, args.target)
    for executor in executors:
        executor.shutdown()

if __name__ == "__main__":
    main()