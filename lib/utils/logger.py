"""
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

Some simple logging functionality, inspired by rllab's logging.

Logs to a tab-separated-values file (path/to/output_directory/progress.txt)

"""
import os 
import os.path as osp
import yaml
import atexit
import shutil
import numpy as np
from collections import defaultdict
from collections import deque
from datetime import datetime 

import torch
from tensorboardX import SummaryWriter

from collections import defaultdict
from collections import deque

import time, os, io
import matplotlib.pyplot as plt
from matplotlib import patches



color2num = dict(
    gray=30,
    red=31,
    green=32,
    yellow=33,
    blue=34,
    magenta=35,
    cyan=36,
    white=37,
    crimson=38
)

def colorize(string, color, bold=False, highlight=False):
    """
    Colorize a string.
    This function was originally written by John Schulman.
    """
    attr = []
    num = color2num[color]
    if highlight: num += 10
    attr.append(str(num))
    if bold: attr.append('1')
    return '\x1b[%sm%s\x1b[0m' % (';'.join(attr), string)


def get_datetime():
    return str(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))


class SmoothedValue(object):
    """ Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """
    def __init__(self, window_size=20):
        self.deque = deque(maxlen=window_size)
        self.series = []
        self.total = 0.0
        self.count = 0

    def update(self, value):
        self.deque.append(value)
        self.series.append(value)
        self.count += 1
        self.total += value

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque))
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count


class Logger(object):
    """
    A general-purpose logger.

    Makes it easy to save diagnostics, hyperparameter configurations, the 
    state of a training run, and the trained model.
    """
    def __init__(self, output_dir=None, output_fname='progress.txt', exp_name=None, delimiter="\t"):
        """
        Initialize a Logger.

        Args:
            output_dir (string): A directory for saving results to. If 
                ``None``, defaults to a temp directory of the form
                ``/tmp/experiments/somerandomnumber``.

            output_fname (string): Name for the tab-separated-value file 
                containing metrics logged throughout a training run. 
                Defaults to ``progress.txt``. 

            exp_name (string): Experiment name. If you run multiple training
                runs and give them all the same ``exp_name``, the plotter
                will know to group them. (Use case: if you run the same
                hyperparameter configuration with multiple random seeds, you
                should give them all the same ``exp_name``.)
        """
        self.output_dir = "results/experiments/%s/%s" % (
            exp_name if (output_dir is not None) and (exp_name is not None) else 'tmp', "logs")
        )
        if osp.exists(self.output_dir):
            print("Warning: Log dir %s already exists! Storing info there anyway."%self.output_dir)
        else:
            os.makedirs(self.output_dir)
        self.output_file = open(osp.join(self.output_dir, output_fname), 'w')
        atexit.register(self.output_file.close)
        print(colorize("Logging data to %s"%self.output_file.name, 'green', bold=True))
       
        self.first_row=True
        self.log_headers = []
        self.log_current_row = {}
        self.exp_name = exp_name

        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

        self.log_dir = 
        self.tf_logger = SummaryWriter()

    def update(self, **kwargs):
        for k, v in kwargs.items(): 
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        return object.__getattr__(self, attr)

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {:.4f} ({:.4f})".format(name, meter.median, meter.global_avg)
            )
        return self.delimiter.join(loss_str)

    def log(self, msg, color='green'):
        """Print a colorized message to stdout."""
        print(colorize(msg, color, bold=True))

    def save_config(self, config):
        """
        Log an experiment configuration.

        Call this once at the top of your experiment, passing in all important
        config vars as a dict. This will serialize the config to JSON, while
        handling anything which can't be serialized in a graceful way (writing
        as informative a string as possible). 

        Example use:

        .. code-block:: python

            logger = EpochLogger(**logger_kwargs)
            logger.save_config(locals())
        """
        config_dict = yaml.load(config.dump())
        print(colorize('Saving config:\n', color='cyan', bold=True))
        print(config)
        with open(osp.join(self.output_dir, "config.yaml"), 'w') as out:
            yaml.dump(config_dict, out, default_flow_style=False)

    def dump_tabular(self):
        """
        Write all of the diagnostics from the current iteration.

        Writes both to stdout, and to the output file.
        """
        vals = []
        key_lens = [len(key) for key in self.log_headers]
        max_key_len = max(15,max(key_lens))
        keystr = '%'+'%d'%max_key_len
        fmt = "| " + keystr + "s | %15s |"
        n_slashes = 22 + max_key_len
        print("-"*n_slashes)
        for key in self.log_headers:
            val = self.log_current_row.get(key, "")
            valstr = "%8.3g"%val if hasattr(val, "__float__") else val
            print(fmt%(key, valstr))
            vals.append(val)
        print("-"*n_slashes)
        if self.output_file is not None:
            if self.first_row:
                self.output_file.write("\t".join(self.log_headers)+"\n")
            self.output_file.write("\t".join(map(str,vals))+"\n")
            self.output_file.flush()
        self.log_current_row.clear()
        self.first_row=False



class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        return object.__getattr__(self, attr)

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {:.4f} ({:.4f})".format(name, meter.median, meter.global_avg)
            )
        return self.delimiter.join(loss_str)
        

class TensorboardLogger(MetricLogger):
    def __init__(self, log_dir='logs', delimiter='\t'):
        
        super(TensorboardLogger, self).__init__(delimiter)
        self.writer = self._get_tensorboard_writer(log_dir)
        
    @staticmethod
    def _get_tensorboard_writer(log_dir):
        try:
            from tensorboardX import SummaryWriter
        except:
            raise ImportError('TensorboardX not installed!')
            
        if is_main_process():
            timestamp = datetime.fromtimestamp(time.time()).strftime('%Y%m%d-%H:%M')
            tb_logger = SummaryWriter(os.path.join(log_dir, timestamp))
            return tb_logger
        else:
            return None
    
    def update(self, iteration, ** kwargs):
        super(TensorboardLogger, self).update(**kwargs)
        if self.writer:
            for k, v in kwargs.items():
                if isinstance(v, torch.Tensor):
                    v = v.item()
                assert isinstance(v, (float, int))
                self.writer.add_scalar(k, v, iteration)
            
    def update_image(self, iteration, image, preds, targets):
        image = image.cpu().numpy()
        boxes = preds.bbox.cpu().numpy()
        boxes_gt = targets.bbox.cpu().numpy()
        cats = preds.get_field('labels').cpu().numpy()
        cats_gt = targets.get_field('labels').cpu().numpy()
        
        if self.writer:
            for cat in np.unique(np.append(cats, cats_gt)):
                if cat == 0:
                    continue
                fig, ax = plt.figure(), plt.gca()
                ax.imshow(image.transpose(1, 2, 0))
                
                for i in range(len(cats)):
                    if cats[i] == cat:
                        x1, y1, x2, y2 = boxes[i]
                        ax.add_patch(
                            patches.Rectangle(
                                (x1, y1),
                                x2 - x1,
                                y2 - y1,
                                edgecolor='r',
                                linewidth=1,
                                fill=False
                            )
                        )
                        
                for i in range(len(cats_gt)):
                    if cats_gt[i] == cat:
                        x1, y1, x2, y2 = boxes_gt[i]
                        ax.add_patch(
                            patches.Rectangle(
                                (x1, y1),
                                x2 - x1,
                                y2 - y1,
                                edgecolor='g',
                                linewidth=1,
                                fill=False
                            )
                        )
                        
                        
                plt.axis('scaled')
                plt.tight_layout()
                        
                self.writer.add_figure('train/image/{}'.format(cat), fig, iteration)

    
