import os
import argparse
import cv2
import tqdm
import numpy as np
from IPython import embed
from landstack.utils import misc

def main():
    parser = argparse.ArgumentParser(description='run test script on benchmark')
    parser.add_argument(dest='input', help='input path, file or folder', default=None)
    parser.add_argument('-o', '--output', help='output path', default=None, required=True)
    parser.add_argument('--height', default=480, type=int)
    parser.add_argument('--width', default=640, type=int)
    parser.add_argument('--angle', choices=['rot90', 'rot180', 'rot270'], help='rotate image anti-clockwise')
    args, unknownargs  = parser.parse_known_args()

    if not os.path.exists(args.input):
        print("Could not found")
        exit(0)

    todo_list = []
    args.input = os.path.realpath(args.input)
    args.output = os.path.realpath(args.output)
    if os.path.isdir(args.input):
        in_root = args.input
        for root, _, files in os.walk(args.input, followlinks=True):
            for fname in files:
                full_path = os.path.join(root, fname)
                todo_list.append(full_path)
    else:
        in_root = os.path.dirname(args.input)
        todo_list = [args.input]
    misc.ensure_dir(args.output, erase=True)

    for fname in tqdm.tqdm(todo_list):
        img = cv2.imread(fname)
        if img is None:
            try:
                img = misc.read_yuv(fname, height=args.height, width=args.width, rgb=False)
                img = np.tile(img[:, :, np.newaxis], [1, 1, 3])
            except Exception as e:
                print("No recognize file {}, skip it".format(fname))
                continue
        assert img.ndim == 3  and img.shape[2] == 3

        if args.angle:
            img0 = misc.rotatecv(img, args.angle)
        else:
            img0 = img
        if not fname.endswith('png') and not fname.endswith('jpg') and not fname.endswith('bmp'):
            save_file = fname + '.jpg'
        else:
            save_file = fname
        save_file = save_file.replace(in_root, args.output)
        misc.ensure_dir(os.path.dirname(save_file))
        cv2.imwrite(save_file, img0)

if __name__ == "__main__":
    main()
