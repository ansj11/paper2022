import argparse
from landstack.lowbit import convertor

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model', help='model to convert')
    parser.add_argument('output', help='save path')
    parser.add_argument('--sfa', default=7)
    parser.add_argument('--bit_midout', default=16)
    args, unknownargs = parser.parse_known_args()
    convertor.SFA = int(args.sfa)
    convertor.BIT_MIDOUT = int(args.bit_midout)
    convertor.run(args.model, args.output)

if __name__ == '__main__':
    main()
