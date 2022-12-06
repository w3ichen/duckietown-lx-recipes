import sys
import os


def plain_progress_monitor(handler, interactive: bool = True):
    if interactive:
        # move back to the beginning of the line
        sys.stdout.write('\r')
    sys.stdout.write(f'{handler.progress.percentage}%')


# stolen from https://github.com/duckietown/yolov5/blob/dt-obj-det/utils/torch_utils.py, but not yolov5-restricted: this
# is useful for all torch conversions
def select_device(device='', batch_size=None):
    import torch
    # device = 'cpu' or '0' or '0,1,2,3'
    cpu = device.lower() == 'cpu'
    if cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # force torch.cuda.is_available() = False
    elif device:  # non-cpu device requested
        os.environ['CUDA_VISIBLE_DEVICES'] = device  # set environment variable
        assert torch.cuda.is_available(), f'CUDA unavailable, invalid device {device} requested'  # check availability

    cuda = not cpu and torch.cuda.is_available()

    return torch.device('cuda:0' if cuda else 'cpu')


def get_dfe(path):
    """
    :param path: Takes a path to a file
    :return: (directory of file, filename without the extension, file's extension)
    """

    dir, file = os.path.split(path)

    splitted = file.split(".")  # this handles files with more than one '.', such as `model.pt.wts`
    filename = ".".join(splitted[:-1])
    extension = splitted[-1]

    return dir, filename, extension


import subprocess


def run(input, exception_on_failure=False):
    try:
        program_output = subprocess.check_output(f"{input}", shell=True, universal_newlines=True,
                                                 stderr=subprocess.STDOUT)
    except Exception as e:
        if exception_on_failure:
            raise e
        program_output = e.output
    return program_output.strip()
