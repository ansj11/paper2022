import argparse
from collections import OrderedDict
import importlib
import os
import pandas as pd
pd.options.display.float_format = '{:.5g}'.format
pd.set_option('expand_frame_repr', False)
pd.set_option('display.max_rows', 500)
from landstack.utils import misc
import getpass
import re
import time
import glob

def hint_script(scripts=None, module_root='benchmark'):
    m = importlib.import_module(module_root)
    available_modules = []
    for root, dirs, files in os.walk(m.__path__[0]):
        for file in files:
            if file.endswith('.py') and file != '__init__.py':
                full_file = os.path.join(root, file[:-3])
                l = full_file.split('/')
                l = '.'.join(l[(l.index(module_root)+1):])
                available_modules.append(l)
    available_modules = list(filter(lambda x: '.' in x and 'dataset.' not in x, available_modules))

    # O(n^2) complexity
    scripts_out = []
    for script in scripts:
        pat = '\.'.join(list(map(lambda x: x+'[a-zA-Z_]*', script.split('.'))))
        pat = '^' + pat + '$' # pattern
        for m in available_modules:
            if re.match(pat, m) is not None and m not in scripts_out:
                scripts_out.append(m)
    return scripts_out

def gen_info_str(model, script, res):
    # md_info = []
    # for script, res in res_dict_all.items():
    index = list(res.keys())
    columns = list(list(res.values())[0].keys())
    data = [list(v.values()) for v in res.values()]
    df = pd.DataFrame(data=data, index=index, columns=columns)
    info = ['-'*60, "TestUnit: {} on {}".format(script, model), df.__repr__(), '-'*60]
    info_str = "\n".join(info)
    return info_str
        # print(info_str)
        # md_info.append(info_str)
        # res_dict_all.setdefault(script, res)
    # return md_info

def gen_md_info(model, res_dict_all):
    md_info = []
    for script, res in res_dict_all.items():
        info_str = gen_info_str(model, script, res)
        # index = list(res.keys())
        # columns = list(list(res.values())[0].keys())
        # data = [list(v.values()) for v in res.values()]
        # df = pd.DataFrame(data=data, index=index, columns=columns)
        # info = ['-'*60, "TestUnit: {} on {}".format(script, model), df.__repr__(), '-'*60]
        # info_str = "\n".join(info)
        # return info_str
        # print(info_str)
        md_info.append(info_str)
        # res_dict_all.setdefault(script, res)
    return md_info




def main():
    parser = argparse.ArgumentParser(description='run test script on benchmark')
    parser.add_argument(dest='scripts', help='script file with python package namespace')
    parser.add_argument(dest='model', help='model to evaluate')
    parser.add_argument('-u', '--upload', action='store_true', help='whether to upload to model server')
    parser.add_argument('-d', '--devices', type=str, default='gpu0')
    parser.add_argument('--prefix', help='model server root',
                        default='/unsullied/sharefs/_research_facelm/Isilon-modelshare/model_server')
    parser.add_argument('--loop', action='store_true')
    args, unknownargs  = parser.parse_known_args()

    if args.upload:
        assert os.path.exists(args.prefix), args.prefix

    # handle scripts
    scripts = hint_script(args.scripts.split(",")) # hint script with regular expression
    print("INFO: To perform test [{}]".format(', '.join(scripts)))

    # perform test
    caches_all = OrderedDict()
    seen_models = set()
    models = args.model.split(",")
    while True:
        todo_models = []
        for model in models:
            model = model.strip()
            if any(marker in model for marker in ['*', '?', '[', ']']):
                ms = sorted(glob.glob(model), key = lambda x:os.path.getctime(x))
            else:
                ms = [model]
            for m in ms:
                m = os.path.realpath(m)
                if os.path.isdir(m):
                    print("Found that {} is directory, skip it".format(m))
                else:
                    todo_models.append(m)

        # early stop if illegal
        upload_info = []
        for model in todo_models:
            if model in seen_models:
                continue
            if args.upload:
                user_root = os.path.join(args.prefix, getpass.getuser(), 'config')
                model_root = os.path.dirname(os.path.realpath(model))
                exp_name = os.path.basename(os.path.dirname(model_root))
                old_config_root = os.path.realpath(os.path.join(model_root, '..', '..', '..', 'config', exp_name))
                if not os.path.exists(old_config_root):
                    exp_name = input("FAILED: old config root {} is not exist, please hint with model_name: ".format(old_config_root))
                new_config_root = os.path.join(user_root, exp_name)
                r = 'overwrite'
                if os.path.exists(new_config_root):
                    r = input("FAILED: found duplicate target {}, overwrite/append/ignore? [O/A/I] ".format(new_config_root))
                    r = r.lower()
                    if r in ['i']:
                        continue
                    else:
                        assert r in ['o', 'a', 'overwrite', 'append']
                    r = 'overwrite' if r == 'o' else r
                    r = 'append' if r == 'a' else r
                upload_info.append((model, old_config_root, new_config_root, r))
            else:
                upload_info.append((model, None, None, None))
        if len(upload_info) > 0:
            print("The following models are to eval")
            for i, model in enumerate(list(zip(*upload_info))[0]):
                print("{}. {}".format(i+1, model))
        for model, old_config_root, new_config_root, action in upload_info:
            res_dict_all = OrderedDict()
            # print_info_list = []
            for script in scripts:
                test_module = '.'.join(['benchmark', script])
                obj = importlib.import_module(test_module)
                caches = caches_all.setdefault(script, OrderedDict())
                res = obj.main(model, args.devices, unknownargs, caches=caches) # res is {dataset1:{k1x:v1x}, dataset2:{k2x:v2x}}

                # index = list(res.keys())
                # columns = list(list(res.values())[0].keys())
                # data = [list(v.values()) for v in res.values()]
                # df = pd.DataFrame(data=data, index=index, columns=columns)
                # info = ['-'*60, "TestUnit: {} on {}".format(script, model), df.__repr__(), '-'*60]
                # info_str = "\n".join(info)
                info_str = gen_info_str(model, script, res)
                print(info_str)
                # print_info_list.append(info_str)
                res_dict_all.setdefault(script, res)
            # print_info_list = gen_md_info(model, res_dict_all)

            # execute uploading operation
            if args.upload:
                misc.ensure_writable(os.path.dirname(args.prefix))
                misc.ensure_writable(args.prefix)
                misc.ensure_dir(new_config_root)
                if os.path.exists(old_config_root):
                    command = 'cp -r {}/* {}'.format(old_config_root, new_config_root)
                    misc.block_sys_call(command)

                pkl_file = os.path.join(new_config_root, 'info.pkl')
                md_file = os.path.join(new_config_root, 'info.md')

                if action == 'append':
                    if os.path.exists(pkl_file):
                        old_res_dict_all = misc.load_pickle(pkl_file)
                        for k, v in res_dict_all.items():
                            # old_res_dict_all[k] = res_dict_all.get(k, old_res_dict_all[k])
                            # old_res_dict_all.setdefault(k, v)
                            old_res_dict_all[k] = v
                        res_dict_all = old_res_dict_all
                print_info_list = gen_md_info(model, res_dict_all)

                misc.dump_pickle(res_dict_all, pkl_file)
                misc.write_txt(print_info_list, md_file)

            seen_models.add(model)

        if not args.loop:
            break
        else:
            time.sleep(10)

if __name__ == "__main__":
    main()