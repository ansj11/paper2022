import os
import argparse
from meghair.utils import io
from landstack.utils import detect, misc
from IPython import embed


def main():
    parser = argparse.ArgumentParser(description='run test script on benchmark')
    parser.add_argument(dest='input', help='input path, file or folder', default=None)
    parser.add_argument('--video-suffix', default='mp4,avi')
    parser.add_argument('--det-suffix', default='det')
    parser.add_argument('-o', '--output', help='output path', default=None)
    args, unknownargs  = parser.parse_known_args()
    det = detect.mdetect('gpu0', 'm')

    video_suffix = ['.{}'.format(v) for v in args.video_suffix.split(',')] # add dot
    assert len(args.det_suffix.split(',')) == 1, args.det_suffix

    video_files = []
    if any([args.input.endswith(v) for v in video_suffix]):
        video_files.append(args.input)
        args.output = args.output or os.path.dirname(args.input)
    else:
        for root, dirs, files in os.walk(args.input):
            for fname in files:
                full_path = os.path.join(root, fname)
                if any([fname.endswith(v) for v in video_suffix]):
                    video_files.append(full_path)
        args.output = args.output or args.input

    for video_file in video_files:
        frames, fps = misc.load_videos(video_file)
        det_result = []
        for frame in frames:
            r = sorted(det.detect(frame), key=lambda x:x['rect'][2] * x['rect'][3], reverse=True)
            det_result.append(r)
        det_file = os.path.join(args.output, '.'.join(os.path.basename(video_file).split('.')[:-1]+[args.det_suffix]))
        misc.dump_pickle(det_result, det_file)

if __name__ == "__main__":
    main()