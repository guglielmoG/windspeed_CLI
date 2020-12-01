import os
import argparse
import logging

# CLI setup
description='A utility to predit wind speed (absent, weak, strong) from images, using wind indicators such as flags. Produces a csv storing predicted wind speed for supplied image(s).'

parser = argparse.ArgumentParser(description=description)
parser.add_argument('path', help='path to image or folder')
parser.add_argument('-th', '--threshold', type=float, default=0.5, help='acceptance threshold for flag detection algorithm. Bounding boxes with lower values will be excluded. Must be in (0, 1). Default to 0.5')
parser.add_argument('-o', '--out-dir', default='out', help='output directory for csv and other script data. Defaults to ./out')
parser.add_argument('--show-steps', action='store_true', help='store intermediate steps to out-dir')
parser.add_argument('--video', action='store_true', help='interprets input path as video, output video frames are annotated with predicted wind speed.')
parser.add_argument('-v', '--verbose', type=int, choices=[0,1,2], default=1, help='increase output verbosity. 0=info, 1=warning, 2=error. Default 1.')

args = parser.parse_args()

path = args.path
th = args.threshold
out_dir = args.out_dir
v = args.verbose
steps = args.show_steps
video = args.video

# Set tensorflow verbosity. Needs to be done before importing it

if v == 0:
    logging.getLogger("tensorflow").setLevel(logging.INFO)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
elif v == 1:
    logging.getLogger("tensorflow").setLevel(logging.WARNING)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
else:
    logging.getLogger("tensorflow").setLevel(logging.ERROR)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import utils

# Setup logging for this module

logger = logging.getLogger('windspeed')
ch = logging.StreamHandler()
if v == 0:
    ch.setLevel(logging.INFO)
elif v == 1:
    ch.setLevel(logging.WARNING)
else:
    ch.setLevel(logging.ERROR)
formatter = logging.Formatter('%(levelname)s:%(name)s:%(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

def run_script(path, th, out_dir, steps):
    if not 0 < th < 1:
        logger.error('Invalid threshold value, must be between 0 and 1.')
        return
        
    if not os.path.exists(out_dir):
        base_dir = os.path.dirname(out_dir)
        if os.path.exists() or base_dir == '':
            os.mkdir(out_dir)
        else:
            logger.error('The system cannot find the path specified: "{}"'.format(out_dir))
            return

    if os.path.exists(path):
        if os.path.isdir(path):
            res = utils.bulk_pred_2step(path, th, steps, out_dir)
        elif video:
            utils.video_pred_2step(path, out_dir, th)
        else:
            res = utils.img_pred_2step(path, th, steps, out_dir)
    else:
        logger.error('The input path "{}" does not exist.'.format(path))
        return


    # print csv
    if not video:
        csv = [im + ', ' + res[im] for im in res]
        csv.append('')
        with open(out_dir + os.sep + 'wind_result.csv', 'w') as w:
            w.write('\n'.join(csv))

run_script(path, th, out_dir, steps)
