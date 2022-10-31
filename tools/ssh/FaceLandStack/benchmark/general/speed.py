#!/usr/bin/env python3

import os
import os.path
import re
import json
import argparse
import getpass
import time
import logging

import requests
import requests.auth
import collections

from IPython import embed
from benchmark.tester import TestUnitBase
from meghair.utils import io
from landstack.utils import misc
import numpy as np

logger = logging.getLogger(__name__)

class JenkinsAPI(object):

    def __init__(self, baseurl, user=None, token=None, retry=None):
        self.baseurl = baseurl
        self.user = user
        self.token = token

        self.session = requests.Session()
        if user and token:
            self.session.auth = (user, token)
        if retry:
            self.session.mount(
                'http://', requests.adapters.HTTPAdapter(max_retries=retry))

    def copy_job(self, from_job_name, new_job_name):
        url = '{}/createItem'.format(self.baseurl)
        params = {'name': new_job_name,
                  'mode': 'copy',
                  'from': from_job_name}
        resp = self.session.post(url, params=params, data='')
        if 'already exists' not in resp.text:
            exist = False
            resp.raise_for_status()
        else:
            exist = True
        return exist

    def enable_job(self, job_name):
        logger.info('Enable job')
        # XXX: hack for jenkins. copied is disabled by default.
        url = '{}/job/{}/disable'.format(self.baseurl, job_name)
        self.session.post(url, data='')

        url = '{}/job/{}/enable'.format(self.baseurl, job_name)
        resp = self.session.post(url, data='')
        return resp

    def build_job(self, job_name, params, fpaths):
        if isinstance(params, dict):
            params = [{'name': k, 'value': v} for k, v in params.items()]
        assert isinstance(params, list)
        url = '{}/job/{}/build'.format(self.baseurl, job_name)
        files = {}
        for fname, (fpath, net) in fpaths.items():
            # hack for read and write
            random_file_name = os.path.join('/tmp', '{}.model'.format(np.random.randint(10000)))
            io.dump(net, random_file_name)
            with open(random_file_name, 'rb') as f:
                fbody = f.read()
            files[fname] = fbody
            os.remove(random_file_name)
            params.append({'name': fname, 'file': fname})

        data = {'json': json.dumps({'parameter': params})}
        resp = self.session.post(url, data=data, files=files)
        resp.raise_for_status()
        return resp

    def is_building(self, job_name):
        url = '{}/job/{}/api/json?depth=2&tree=builds[number,building]'.format(self.baseurl, job_name)
        resp = self.session.get(url)
        data = resp.json()
        return any(b['building'] for b in data['builds'])

    def next_build_number(self, job_name):
        url = '{}/job/{}/api/json?tree=nextBuildNumber'.format(self.baseurl, job_name)
        resp = self.session.get(url)
        return resp.json()['nextBuildNumber']

    def stream_log(self, job_name, build_number='lastBuild', quiet=False):
        log_url = '{}/job/{}/{}/logText/progressiveText'.format(
            self.baseurl, job_name, build_number)
        logger.info('Streaming log ...' if not quiet else 'Quiet mode. Waiting ...')
        resp = self.session.get(log_url)
        cnt = 0
        while resp.status_code == 404 and cnt < 30:
            time.sleep(3)
            cnt += 1
            resp = self.session.get(log_url)

        success_mark = 'Finished: SUCCESS'
        succeed = False
        if not quiet and resp.text.strip():
            print(resp.text)
        succeed = succeed or success_mark in resp.text
        has_more = resp.headers.get('X-More-Data') == 'true'
        start = int(resp.headers.get('X-Text-Size', 0))
        while has_more:
            time.sleep(3)
            resp = self.session.get(log_url, params={'start': start})
            if not quiet and resp.text.strip():
                print(resp.text)
            succeed = succeed or success_mark in resp.text
            has_more = resp.headers.get('X-More-Data') == 'true'
            start = int(resp.headers.get('X-Text-Size', 0))

        return succeed

    def job_console_url(self, job_name):
        return '{}/job/{}/lastBuild/console'.format(self.baseurl, job_name)

    def get_artifact(self, artifact, job_name, build_number='lastSuccessfulBuild'):
        url = '{}/job/{}/{}/artifact/{}'.format(
            self.baseurl, job_name, build_number, artifact)
        resp = self.session.get(url)
        if resp.status_code != 200:
            return
        return resp.text

    def get_profile_log(self, job_name, build_number='lastSuccessfulBuild'):
        return self.get_artifact('report/profile.json', job_name, build_number)

    def get_network_outputs(self, job_name, build_number='lastSuccessfulBuild'):
        return self.get_artifact('report/network_outputs.txt', job_name, build_number)

    def get_summary(self, job_name, build_number='lastSuccessfulBuild'):
        return self.get_artifact('report/test.log', job_name, build_number)


def gen_job_name(user, model_path, suffix='speedtest'):
    RE_MODEL_PATH = re.compile(r'.*/([^/]*)/models/([\w-]+)')
    model_real_path = os.path.realpath(model_path)
    m = RE_MODEL_PATH.match(model_real_path)
    if m:
        name, epoch = m.group(1), m.group(2)
        return '{user}-{name}.{epoch}-{suffix}'.format(
            user=user, name=name, epoch=epoch, suffix=suffix)

    name = os.path.basename(model_path)
    return '{user}-{name}-{suffix}'.format(user=user, name=name, suffix=suffix)


def save_profile_log(log_text, save_path):
    if not log_text:
        logger.error('Fetch profile log failed.')
        return
    with open(save_path, 'w') as f:
        f.write(log_text)
    logger.info('Profiling log saved to %s', save_path)

def save_network_outputs(outputs_text, save_path):
    if not outputs_text:
        logger.error('Fetch network outputs result failed.')
        return
    outputs = outputs_text.split('\n')
    outputs_dict = collections.OrderedDict()
    current_key = None
    current_var = None
    for line in outputs:
        line = line.strip()
        if len(line) == 0:
            continue
        elif 'key' in line:
            key, var = line.split(',')
            current_key = key.strip().split()[-1]
            current_var = var.strip().split()[-1]
            print(current_key, current_var)
        else:
            outputs_dict.setdefault(current_key, collections.OrderedDict()).setdefault(current_var, []).append(line.split())
    for key, v in outputs_dict.items():
        for var, vv in v.items():
            outputs_dict[key][var] = np.array(vv).astype('float32')

    misc.dump_pickle(outputs_dict, save_path)
    logger.info('Network outputs result saved to {}'.format(save_path))


class TestUnit(TestUnitBase):
    def __init__(self, caches):
        name = "Mobile Speed Tester ({}-{})".format(time.strftime("%x"), time.strftime("%X"))
        super(TestUnit, self).__init__(name, caches)
        is_brainpp = os.path.exists('/unsullied/sharefs')
        if is_brainpp:
            # self.jenkins_server = 'http://gpu0.chenxi.brc.sm.megvii-op.org:8080'
            self.jenkins_server = 'http://ws0.chenxi.brw.sm.megvii-op.org:8080'
        else:
            self.jenkins_server = 'http://10.169.0.67:8080'

        self.android_speed_test_template = 'mobile-speed-test'
        self.max_retry = 3

    def compute_metric(self, model_name, net, input_shapes, device, cpu, profile, debug,
                       optimize_model, output_vars, quiet, dataset, keys, network_outputs_file,
                       mutable, n_samples, det, warmup):
        jk = JenkinsAPI(self.jenkins_server, retry=self.max_retry)
        job_name = gen_job_name(user=getpass.getuser(), model_path=model_name)
        logger.info('Create job {}'.format(job_name))
        existed = jk.copy_job(self.android_speed_test_template, job_name)
        if not existed:
            jk.enable_job(job_name)

        if jk.is_building(job_name): # when processing
            logger.warning('Job {} is building.'.format(job_name))
            logger.warning(jk.job_console_url(job_name))

            r = input('Streaming log? [y/n]')
            if r not in ('N', 'n'):
                succeed = jk.stream_log(job_name)
                if succeed and profile:
                    log_text = jk.get_profile_log(job_name)
                    save_profile_log(log_text, profile)
            return
        params = {
            'INPUT_SHAPES': input_shapes,
            'BAZEL_CPU': cpu,
            'PROFILE': '1' if profile else '',
            'DEBUG': '1' if debug else '',
            'OPTIMIZE_MODEL': '1' if optimize_model else '',
            'DEVICE': device,
            'DATASET':dataset,
            'KEYS': keys,
            'MUTABLE': mutable,
            'NSAMPLES': n_samples,
            'DETECT_CONF': det,
            'WARMUP': warmup,
        }

        if output_vars:
            params['OUTPUT_VARS'] = output_vars
        fpaths = {'model': (model_name, net)}
        # logger.debug('Params: %s, %s', params, fpaths)
        logger.info('Uploading ...')
        next_build_number = jk.next_build_number(job_name)
        jk.build_job(job_name, params, fpaths)
        logger.info('Running ...')
        logger.info('View log: %s', jk.job_console_url(job_name))

        succeed = jk.stream_log(job_name, build_number=next_build_number, quiet=quiet)
        if succeed:
            summary = jk.get_summary(job_name, build_number=next_build_number)
            if profile:
                log_text = jk.get_profile_log(job_name, build_number=next_build_number)
                save_profile_log(log_text, profile)
            outputs_text = jk.get_network_outputs(job_name, build_number=next_build_number)
            save_network_outputs(outputs_text, network_outputs_file)
            pat = re.compile('avg-time.*\s\d+\.\d+ms')
            m = pat.search(summary)
            avg_time = float(m.group(0).split(':')[1].strip()[:-2]) # avg-time@{1,1,112,112}: 44.766ms
        else:
            logger.error('Job failed. log: %s', jk.job_console_url(job_name))
            avg_time = 0.0
        return avg_time



def parse_args(model, devices, args_input):
    DEVICE_CPU_MAPPING = {
        'xiaomi': 'android_armv7',
        'sumsung': 'android_armv7',
    }
    parser = argparse.ArgumentParser(description='Mobile Speed Test')
    parser.add_argument('--input_shapes',
                        help='Input shapes. Eg. 1,3,1920,1080:1,3,640,480', default='1,1,112,112')
    parser.add_argument('--lm_name', default=None)
    parser.add_argument('--cpu',
                        choices=('android_aarch64', 'android_armv7',
                                 'android_piii', 'android_piii_sse3',
                                 'piii'),
                        help='Bazel compile CPU architecture')
    parser.add_argument('-p', '--profile',
                        help='Request for profile log and save to this path')
    parser.add_argument('-O', '--optimize-model', action='store_false',
                        help='Need optimize model? Defalt is true')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Verbose log')
    parser.add_argument('-q', '--quiet', action='store_true',
                        help='Only show test result summary')
    parser.add_argument('--output-vars', help='Output vars split by `,`. '
                        'Number indicates outputs index. Eg. `0` means `net.outputs[0]`')
    parser.add_argument('--lowbit', action='store_true')
    parser.add_argument('--sfa', default=7)
    parser.add_argument('--bit_midout', default=16)
    parser.add_argument('--dataset', default='general.accuracy')
    parser.add_argument('--keys', default='validation-valid',
                        help='dataset to eval')
    parser.add_argument('--debug', action='store_true', help='enable -c opt')
    parser.add_argument('--network-outputs', default='/tmp/network_outputs.pkl')
    # 指定输入是否可以是任意形状
    parser.add_argument('--mutable', action="store_true")
    # 当type == random时，指定使用的测试样本数量
    parser.add_argument('--n_samples', type=int, default=1000)
    # 三种状态， 不使用该参数时args.det==None, 表示不进行检测；使用该参数但不传值时，表示使用默认conf进行检测，否则使用指定conf进行检测
    parser.add_argument('--det', nargs="*", help="detection before testing")
    # 设置warmup的次数， 加快测速
    parser.add_argument('--warmup', type=int, default=100)

    args, unknownargs = parser.parse_known_args(args_input)
    args.device = 'xiaomi' if 'gpu' in devices else devices
    if args.device and not args.cpu:
        args.cpu =DEVICE_CPU_MAPPING.get(args.device.lower())
    assert os.path.exists(model), 'model {} not found'.format(model)
    assert all(len(s.split(',')) == 4 for s in args.input_shapes.split(':')), 'N,C,H,W'

    return args

def main(model, devices, args_input, caches):
    args = parse_args(model, devices, args_input)

    # init logger
    logger.setLevel(logging.DEBUG if args.verbose else logging.INFO)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter('[%(levelname)s] %(asctime)s %(message)s'))
    logger.addHandler(stream_handler)


    net = misc.load_network_and_extract(model, args.lm_name)
    t = TestUnit(caches=caches)
    if args.lowbit:
        from landstack.lowbit import convertor
        convertor.SFA = int(args.sfa)
        convertor.BIT_MIDOUT = int(args.bit_midout)
        net =  convertor.run(net)

    if isinstance(args.det, list):
        if len(args.det) == 0:
            det = "default"
        else:
            det = args.det[0]
    else:
        det = "None"

    res = t.compute_metric(
        model_name=model,
        net = net,
        input_shapes=args.input_shapes,
        device=args.device,
        cpu=args.cpu,
        profile=args.profile,
        debug=args.debug,
        optimize_model=args.optimize_model,
        output_vars=args.output_vars,
        quiet=args.quiet,
        dataset=args.dataset,
        keys=args.keys,
        network_outputs_file = args.network_outputs,
        mutable = args.mutable,
        n_samples = args.n_samples,
        det =  det,
        warmup = args.warmup,
    )
    res_wrapper = collections.OrderedDict([('speed', {args.device: res})])
    return res_wrapper

if __name__ == '__main__':
    main()
