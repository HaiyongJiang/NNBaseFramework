import os
import GPUtil
import time
import argparse


#########################################################################
# setup CUDA_VISIBLE_DEVICES before importing pytorch
def setup_GPU(ngpu=1, xsize=10000):
    ## setup GPU
    ## detect and use the first available GPUs
    gpus = GPUtil.getGPUs()
    idxs = []
    mems = []
    counter = 0
    while len(idxs) == 0:
        for ii,gpu in enumerate(gpus):
            if gpu.memoryFree > xsize:
                idxs.append(ii)
                mems.append(gpu.memoryFree)
        if len(idxs) == 0:
            time.sleep(60)
            counter += 1
            if counter%(60*12) == 0:
                print("%d hours passed"%(counter/60))
    idxs = [v for _, v in sorted(zip(mems, idxs), reverse=True)]
    idxs = sorted(idxs[:ngpu])
    print("Training on Gpus: %s." % (
        str({"ID%d"%ii:"%fMB"%mems[ii] for ii in idxs})))
    GPU_IDS = ",".join([str(v) for v in idxs])
    os.environ["CUDA_VISIBLE_DEVICES"] = GPU_IDS
    return list(range(len(idxs)))


def get_arg_parser():
    # Arguments
    parser = argparse.ArgumentParser(
        description='Train a 3D reconstruction model.'
    )
    parser.add_argument('config', type=str, help='Path to config file.')
    parser.add_argument('--no-cuda', action='store_true', help='Do not use cuda.')
    parser.add_argument('--no-restore', action='store_false', help='Restore network if available.')
    parser.add_argument('--opt_layers', type=str, default=".*", help="layer regex to optimize")
    parser.add_argument('--ngpu', type=int, default=1, help='the number of gpu to use .')
    parser.add_argument('--train', action='store_false', help='train the network.')
    parser.add_argument('--test', action='store_true', help='test the network.')
    return parser


