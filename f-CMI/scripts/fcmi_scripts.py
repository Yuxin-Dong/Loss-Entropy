import argparse
import os
import subprocess
import time

import torch

from nnlib.nnlib.data_utils.base import register_parser as register_fn


local_functions = {}  # storage for registering the functions below


def multi_gpu_launcher(commands):
    """
    Launch commands on the local machine, using all GPUs in parallel.
    """
    print('WARNING: using experimental multi_gpu_launcher.')
    try:
        # Get list of GPUs from env, split by ',' and remove empty string ''
        # To handle the case when there is one extra comma: `CUDA_VISIBLE_DEVICES=0,1,2,3, python3 ...`
        available_gpus = [x for x in os.environ['CUDA_VISIBLE_DEVICES'].split(',') if x != '']
    except Exception:
        # If the env variable is not set, we use all GPUs
        available_gpus = [str(x) for x in range(torch.cuda.device_count())]
    n_gpus = len(available_gpus)
    procs_by_gpu = [None]*n_gpus

    while len(commands) > 0:
        for idx, gpu_idx in enumerate(available_gpus):
            proc = procs_by_gpu[idx]
            if (proc is None) or (proc.poll() is not None):
                # Nothing is running on this GPU; launch a command.
                cmd = commands.pop(0)
                new_proc = subprocess.Popen(
                    f'CUDA_VISIBLE_DEVICES={gpu_idx} {cmd}', shell=True)
                procs_by_gpu[idx] = new_proc
                break
        time.sleep(1)

    # Wait for the last few tasks to finish before returning
    for p in procs_by_gpu:
        if p is not None:
            p.wait()


#######################################################################################
#
#     MNIST 4 vs 9, CNN
#
# exp_name: fcmi-mnist-4vs9-CNN
#######################################################################################

@register_fn(local_functions, 'fcmi-mnist-4vs9-CNN')
def foo(deterministic=False, loo=False, **kwargs):
    config_file = 'configs/binary-mnist-4layer-CNN.json'
    batch_size = 128
    n_epochs = 200
    save_iter = 20
    exp_name = "fcmi-mnist-4vs9-CNN"
    if deterministic:
        exp_name = exp_name + '-deterministic'
    if loo:
        exp_name = exp_name + '-loo'
    dataset = 'mnist'
    which_labels = '4 9'

    command_prefix = f"python -um scripts.fcmi_train_classifier -c {config_file} -d cuda -b {batch_size} " \
                     f"-e {n_epochs} -s {save_iter} -v 10000 --exp_name {exp_name} -D {dataset} " \
                     f"--which_labels {which_labels} "

    if deterministic:
        command_prefix += "--deterministic "
    if loo:
        command_prefix += "--loo "

    n_seeds = 5
    n_S_seeds = 30
    ns = [75, 250, 1000, 4000]

    commands = []
    for n in ns:
        for seed in range(n_seeds):
            for S_seed in range(n_S_seeds):
                commands.append(command_prefix + f"--n {n} --seed {seed} --S_seed {S_seed};")
    return commands


@register_fn(local_functions, 'fcmi-mnist-4vs9-wide-CNN-deterministic')
def foo(**kwargs):
    config_file = 'configs/binary-mnist-4layer-wide-CNN.json'
    batch_size = 128
    n_epochs = 200
    save_iter = 200
    exp_name = "fcmi-mnist-4vs9-wide-CNN-deterministic"
    dataset = 'mnist'
    which_labels = '4 9'

    command_prefix = f"python -um scripts.fcmi_train_classifier -c {config_file} -d cuda -b {batch_size} " \
                     f"-e {n_epochs} -s {save_iter} -v 10000 --exp_name {exp_name} -D {dataset} " \
                     f"--which_labels {which_labels} --deterministic "

    n_seeds = 5
    n_S_seeds = 30
    ns = [75, 250, 1000, 4000]

    commands = []
    for n in ns:
        for seed in range(n_seeds):
            for S_seed in range(n_S_seeds):
                commands.append(command_prefix + f"--n {n} --seed {seed} --S_seed {S_seed};")
    return commands


@register_fn(local_functions, 'fcmi-mnist-4vs9-CNN-LD')
def foo(shuffle_train_only_after_first_epoch=False, loo=False, **kwargs):
    config_file = 'configs/binary-mnist-4layer-CNN.json'
    batch_size = 100
    n_epochs = 40
    save_iter = 4
    exp_name = "fcmi-mnist-4vs9-CNN-LD"
    if shuffle_train_only_after_first_epoch:
        exp_name = exp_name + '-shuffle_train_only_after_first_epoch'
    if loo:
        exp_name = exp_name + '-loo'
    dataset = 'mnist'
    which_labels = '4 9'
    ld_lr = 0.004
    ld_beta = 10.0

    command_prefix = f"python -um scripts.fcmi_train_classifier -c {config_file} -d cuda -b {batch_size} " \
                     f"-e {n_epochs} -s {save_iter} -v 10000 --exp_name {exp_name} -D {dataset} " \
                     f"--which_labels {which_labels} -m LangevinDynamics --ld_lr {ld_lr} "\
                     f"--ld_beta {ld_beta} "

    if shuffle_train_only_after_first_epoch:
        command_prefix += "--shuffle_train_only_after_first_epoch "
    if loo:
        command_prefix += "--loo "

    n_seeds = 5
    n_S_seeds = 30
    ns = [4000]

    commands = []
    for n in ns:
        for seed in range(n_seeds):
            for S_seed in range(n_S_seeds):
                command = command_prefix + f"--n {n} --seed {seed} --S_seed {S_seed}"
                if S_seed < 4:  # producing in total 20 runs with this flag
                    command += " --ld_track_grad_variance"
                command += ";"
                commands.append(command)
    return commands


@register_fn(local_functions, 'cifar10-pretrained-resnet50')
def foo(loo=False, **kwargs):
    config_file = 'configs/pretrained-resnet50-cifar10.json'
    batch_size = 64
    n_epochs = 40
    save_iter = n_epochs
    exp_name = "cifar10-pretrained-resnet50"
    if loo:
        exp_name = exp_name + '-loo'
    dataset = 'cifar10'
    optimizer = 'sgd'
    lr = 0.01
    momentum = 0.9

    command_prefix = f"python -um scripts.fcmi_train_classifier -c {config_file} -d cuda -b {batch_size} " \
                     f"-e {n_epochs} -s {save_iter} -v 10000 --exp_name {exp_name} -D {dataset} " \
                     f"-m StandardClassifier -A --resize_to_imagenet --optimizer {optimizer} "\
                     f"--lr {lr} --momentum {momentum} "

    if loo:
        command_prefix += "--loo "

    n_seeds = 1
    n_S_seeds = 40
    ns = [1000, 5000, 20000]

    commands = []
    for n in ns:
        for seed in range(n_seeds):
            for S_seed in range(n_S_seeds):
                commands.append(command_prefix + f"--n {n} --seed {seed} --S_seed {S_seed};")
    return commands


@register_fn(local_functions, 'cifar10-pretrained-resnet50-LD')
def foo(loo=False, **kwargs):
    config_file = 'configs/pretrained-resnet50-cifar10.json'
    batch_size = 64
    n_epochs = 16
    save_iter = 2
    exp_name = "cifar10-pretrained-resnet50-LD"
    if loo:
        exp_name = exp_name + '-loo'
    dataset = 'cifar10'
    ld_lr = 0.01
    ld_beta = 10.0

    command_prefix = f"python -um scripts.fcmi_train_classifier -c {config_file} -d cuda -b {batch_size} " \
                     f"-e {n_epochs} -s {save_iter} -v 10000 --exp_name {exp_name} -D {dataset} " \
                     f"-m LangevinDynamics -A --resize_to_imagenet  "\
                     f"--ld_lr {ld_lr} --ld_beta {ld_beta} "

    if loo:
        command_prefix += "--loo "

    n_seeds = 1
    n_S_seeds = 40
    ns = [20000]

    commands = []
    for n in ns:
        for seed in range(n_seeds):
            for S_seed in range(n_S_seeds):
                command = command_prefix + f"--n {n} --seed {seed} --S_seed {S_seed}"
                if S_seed < 4:  # producing in total 4 runs with this flag
                    command += " --ld_track_grad_variance --ld_track_every_iter 10"
                command += ";"
                commands.append(command)
    return commands


def ask_for_confirmation():
    response = input('Are you sure? (y/n) ')
    if not response.lower().strip()[:1] == "y":
        print('Nevermind!')
        exit(0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_names', '-E', type=str, nargs='+', required=True)
    parser.add_argument('--deterministic', action='store_true', dest='deterministic')
    parser.add_argument('--shuffle_train_only_after_first_epoch', action='store_true',
                        dest='shuffle_train_only_after_first_epoch')
    parser.add_argument('--loo', action='store_true', default=False)
    args = parser.parse_args()

    for exp_name in args.exp_names:
        assert exp_name in local_functions
        commands = local_functions[exp_name](**vars(args))
        for command in commands:
            print(command)
        # ask_for_confirmation()
        multi_gpu_launcher(commands)


if __name__ == '__main__':
    main()
