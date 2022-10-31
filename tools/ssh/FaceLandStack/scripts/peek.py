import argparse
from landstack.utils import misc
import os
from IPython import embed

def vis_dict(data_dict, records, depth, indent='', mute=False):
    if not isinstance(data_dict, dict):
        if depth > 0 and not mute:
            print('{}{}'.format(indent, data_dict))
        return 1

    # prune
    if isinstance(data_dict, dict) and depth < 2:
        for k, v in data_dict.items():
            if not isinstance(v, dict):
                return len(data_dict)

    c = 0
    for idx, (k, v) in enumerate(data_dict.items()):
        c0 = vis_dict(v, records, depth-1, indent+'\t', mute=True)
        c += c0
        if depth > 0 and idx <records and not mute:
            print('{}{} => {}'.format(indent, k, c0))
        vis_dict(v, records, depth-1, indent+'\t', idx>=records or mute)

    if len(data_dict) > records:
        if depth > 0 and not mute:
            print('{}...'.format(indent))
    return c

def main():
    parser = argparse.ArgumentParser(description='run test script on benchmark')
    parser.add_argument(dest='input', help='model to load')
    parser.add_argument('--depth', default=2, type=int, help='max depth to print')
    parser.add_argument('--records', default=10, type=int, help='max records for each level')
    args, _ = parser.parse_known_args()

    if not os.path.exists(args.input):
        print("Could not found")
        exit(0)

    todo_list = []
    if os.path.isdir(args.input):
        for root, _, files in os.walk(args.input):
            for fname in files:
                full_path = os.path.join(root, fname)
                todo_list.append(full_path)
    else:
        todo_list = [args.input]


    for fname in todo_list:
        d = misc.load_pickle(fname)
        assert isinstance(d, dict), type(d)
        vis_dict(d, args.records, args.depth, '')







if __name__ == "__main__":
    main()

