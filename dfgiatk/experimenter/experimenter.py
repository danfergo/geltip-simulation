import datetime
import math

import os
import shutil
import sys, subprocess

import __main__


# def experiment(description, config, event_listeners=None):
#     event_listeners = event_listeners or []
#
#     def _(fn):
#         fn.__meta__ = {
#             'description': description,
#             'config': config,
#             'event_listeners': event_listeners
#         }
#         return fn
#
#     return _


class E:

    def __init__(self):
        self.config = None
        self.event_listeners = None

    def ws(self, path, *args):
        path = self.config[path] if path in self.config else path
        return os.path.join(self.config['__ws__'] + '/' + path, *args)

    def out(self, path=None, *args):
        path = '' if path is None else (self.config[path] if path in self.config else path)
        return os.path.join(self.config['__out__'] + '/' + path, *args)

    def emit(self, event_name, args=None):
        args = args or {}
        [getattr(el, 'on_' + event_name)(args) for el in self.event_listeners if hasattr(el, 'on_' + event_name)]

    def __getitem__(self, item):
        try:
            if type(item) is tuple:
                return tuple([self.config[item_] for item_ in item])
            return self.config[item]
        except KeyError as e:
            raise KeyError('Configuration "' + e.args[0] + '" is missing. Check the experimenter configurations.')

    def __contains__(self, item):
        return item in self.config

    def __getattr__(self, item):
        return self[item]

    def push_config(self, config, event_listeners):
        def parse(key):
            self.config[key[1:-1]] = self.config[key]()
            self.config.pop(key)

        self.config = config
        {parse(key) for key in list(self.config) if key.startswith('{') and key.endswith('}')}

        self.event_listeners = event_listeners()


e = E()


def check_experiment_exists(key, description, tmp, append):
    outputs_path = ('/tmp/e' if tmp else os.getcwd()) + '/outputs/'
    e_dir = outputs_path + key
    # experiment_dirs = os.listdir(outputs_path)
    if append:
        return True
    if os.path.exists(e_dir):
        if tmp:
            shutil.rmtree(e_dir)
        else:
            print('\n\n---')
            print(description)
            print('---')
            print(e_dir)
            print('Already exists. DELETE EXPERIMENT ? [Y/n]')
            if input() == 'Y':
                shutil.rmtree(e_dir)
                print('done.')
            else:
                return False

    # for e_dir in experiment_dirs:
    #     f = open(outputs_path + e_dir + "/readme.md", "r")
    #     if description == f.read():
    #         if append:
    #             return e_dir
    #         elif tmp:
    #             shutil.rmtree(outputs_path + e_dir)
    #         else:

    return True


def run(description=None, entry=None, config=None, src=None, listeners=None, append=False, tmp=False, open_e=False):
    config = config or {}
    listeners = listeners or []
    src = src or 'src'

    description = description
    # key = description.strip().split('\n')[0][0:120].replace('#', '').replace('.', '').strip()
    key = os.path.basename(__main__.__file__).split('.')[0]
    event_listeners = listeners or (lambda: [])

    workspace_path = os.getcwd() + '/'
    outputs_path = ('/tmp/e' if tmp else os.getcwd()) + '/outputs/'
    config['__ws__'] = workspace_path

    # ensure tmp folder for experimenter exists, if tmp experiment
    # if tmp and not os.path.exists(workspace_path):
    #     os.mkdir(workspace_path)

    # ensure experiments outputs folder exists
    if not os.path.exists(outputs_path):
        os.mkdir(outputs_path)

    # if not check_experiment_exists(key, description, tmp, append):
    #     return

    experiment_key = key
    experiment_path = outputs_path + experiment_key + '/'
    if not os.path.exists(experiment_path):
        os.mkdir(experiment_path)

    runs_path = experiment_path + '/runs/'

    if not os.path.exists(runs_path):
        os.mkdir(runs_path)

    currentDT = datetime.datetime.now()
    run_key = currentDT.strftime("%Y-%m-%d %H:%M:%S")
    run_path = runs_path + run_key + '/'
    os.mkdir(run_path)
    config['__out__'] = run_path + '/out/'
    config['__src__'] = run_path + '/src/'
    os.mkdir(config['__out__'])
    os.mkdir(config['__src__'])

    #
    # calculates YYYYMMDDHHMM string
    # currentDT = datetime.datetime.strptime(dt, "%Y-%m-%d %H:%M:%S") \
    #     if append and dt is not True else datetime.datetime.now()

    # if not (append and dt is not True):
    # create current experiment folder

    f = open(run_path + "/readme.md", "w+")
    f.write(description.strip())
    f.close()

    f = open(run_path + "/~running", "w+")
    f.close()

    # copy src folder
    # shutil.copytree(workspace_path + src, experiment_path + src)
    shutil.copy(os.path.abspath(__main__.__file__), config['__src__'] + os.path.basename(__main__.__file__))

    # open folder
    if open_e:
        open_experiment(experiment_path)

    # prepare args
    e.push_config(config, event_listeners)
    entry()

    # if not (append and dt is not True):
    # remove /~running file
    os.remove(run_path + "/~running")

    latest_run_path = experiment_path + '/latest'
    if os.path.exists(latest_run_path):
        os.unlink(latest_run_path)

    os.symlink(run_path, latest_run_path)

    e.emit('e_end')


def open_experiment(experiment_path):
    opener = "open" if sys.platform == "darwin" else "xdg-open"
    subprocess.call([opener, experiment_path])


def query(tail=5):
    workspace_path = os.getcwd()
    outputs_path = workspace_path + '/outputs/'
    experiment_dirs = os.listdir(outputs_path)

    start = 0 if tail == -1 else len(experiment_dirs) - tail

    cs = [5, 3, 20, 30]
    description_max = 140
    description_idx = 3
    ct = ['IDX', 'STA', 'DATE', 'DESCRIPTION']
    alg = ['rjust', 'ljust', 'ljust', 'ljust']

    def row(cells, sep=' | ', pad=' '):
        print(sep.join([getattr(cells[c], alg[c])(cs[c], pad) for c in range(len(ct))]))

    def line(char='-'):
        row(['' for _ in ct], sep=''.ljust(3, char), pad=char)

    row(ct)
    line()
    for i in range(start, len(experiment_dirs)):
        e_dir = experiment_dirs[i]
        description = open(outputs_path + '/' + e_dir + '/readme.md', "r").read().strip()[:description_max]
        description_lines = math.ceil(len(description) / cs[description_idx])
        running_exists = os.path.exists(outputs_path + '/' + e_dir + '/~running')

        for ln in range(description_lines):
            d_start = ln * cs[description_idx]
            row([
                str(i) if ln == 0 else '',
                ('RUN' if running_exists else 'DON') if ln == 0 else '',
                e_dir if ln == 0 else '',
                description[d_start:d_start + cs[description_idx]]
            ])
