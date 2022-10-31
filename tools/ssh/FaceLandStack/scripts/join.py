import argparse
from meghair.utils import io
from megskull.network import RawNetworkBuilder
from landstack.utils import misc
from IPython import embed


def main():
    parser = argparse.ArgumentParser(description='run test script on benchmark')
    parser.add_argument(dest='model', help='model to load')
    parser.add_argument('--join-point', help='name to join', default=None)
    parser.add_argument('--deps', help='depend oprs', default=None)
    parser.add_argument('-r', '--replace', help='old_name1:new_name1,old_name2:new_name2')
    parser.add_argument('-p', '--print-all-oprs', action='store_true', help='print all oprs')
    parser.add_argument('-o', '--output', help='output path', default=None)
    args, unknownargs  = parser.parse_known_args()

    # merge multiple models
    model_files = args.model.split(',')
    nets = [misc.load_network_and_extract(model_file) for model_file in model_files]
    assert len(nets) == 1 or args.join_point is not None
    merge_net = nets[0]
    for net in nets[1:]:
        exist_oprs_dict = merge_net.outputs_visitor.all_oprs_dict
        exist_join_oprs = [exist_oprs_dict[v] for v in args.join_point.split(',')]

        new_oprs_dict = net.outputs_visitor.all_oprs_dict
        new_join_oprs = [new_oprs_dict[v] for v in args.join_point.split(',')]

        net.outputs_visitor.replace_oprs(
            zip(new_join_oprs, exist_join_oprs), copy=False
        )

        outputs_dict = {v.name:v for v in merge_net.outputs}
        for v in net.outputs:
            if v.name not in outputs_dict:
                outputs_dict[v.name] = v
        merge_net = RawNetworkBuilder(inputs=[], outputs=list(outputs_dict.values()))

    if args.deps:
        outputs = misc.find_opr_by_name(merge_net.outputs, args.deps.split(','))
        net = RawNetworkBuilder(inputs=[], outputs=outputs)
    else:
        net = merge_net

    print("outputs: {}".format(', '.join(['{}:{}'.format(v.name, v.partial_shape) for v in net.outputs])))
    print("inputs: {}".format(', '.join(['{}:{}'.format(v.name, v.partial_shape) for v in net.outputs_visitor.all_data_providers])))

    if args.replace:
        pairs = args.replace.split(',')
        assert len(pairs) % 2 == 0, len(pairs)
        all_oprs_dict = net.outputs_visitor.all_oprs_dict
        for i in range(len(pairs)//2):
            o, n = pairs[2*i], pairs[2*i+1]
            all_oprs_dict[o].rename(n)
        print("----"*20)
        print("outputs: {}".format(', '.join(['{}:{}'.format(v.name, v.partial_shape) for v in net.outputs])))
        print("inputs: {}".format(', '.join(['{}:{}'.format(v.name, v.partial_shape) for v in net.outputs_visitor.all_data_providers])))

    if args.print_all_oprs:
        all_oprs_dict = net.outputs_visitor.all_oprs_dict
        for k, v in all_oprs_dict.items():
            print(k, v)

    if args.output is not None:
        print("Saving model to {}".format(args.output))
        io.dump(net, args.output)

if __name__ == "__main__":
    main()

