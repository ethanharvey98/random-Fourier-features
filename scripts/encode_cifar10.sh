#!/bin/bash
#SBATCH --array=0-11%10
#SBATCH --error=/cluster/tufts/hugheslab/eharve06/slurmlog/err/log_%j.err
#SBATCH --gres=gpu:rtx_6000:1
#SBATCH --mem=16g
#SBATCH --ntasks=4
#SBATCH --output=/cluster/tufts/hugheslab/eharve06/slurmlog/out/log_%j.out
#SBATCH --partition=hugheslab
#SBATCH --time=48:00:00

source ~/.bashrc
conda activate l3d_2024f_cuda12_1

# Define an array of commands
experiments=(
    'python ../src/encode_cifar10.py --batch_size=128 --cifar10_dir="/cluster/tufts/hugheslab/eharve06/CIFAR-10" --cifar101_v4_dir="/cluster/tufts/hugheslab/eharve06/CIFAR-10.1" --encoded_path="/cluster/tufts/hugheslab/eharve06/random-Fourier-features/datasets/CIFAR-10/n=100_random_state=1001.pth" --n=100 --num_workers=0 --random_state=1001'
    'python ../src/encode_cifar10.py --batch_size=128 --cifar10_dir="/cluster/tufts/hugheslab/eharve06/CIFAR-10" --cifar101_v4_dir="/cluster/tufts/hugheslab/eharve06/CIFAR-10.1" --encoded_path="/cluster/tufts/hugheslab/eharve06/random-Fourier-features/datasets/CIFAR-10/n=100_random_state=2001.pth" --n=100 --num_workers=0 --random_state=2001'
    'python ../src/encode_cifar10.py --batch_size=128 --cifar10_dir="/cluster/tufts/hugheslab/eharve06/CIFAR-10" --cifar101_v4_dir="/cluster/tufts/hugheslab/eharve06/CIFAR-10.1" --encoded_path="/cluster/tufts/hugheslab/eharve06/random-Fourier-features/datasets/CIFAR-10/n=100_random_state=3001.pth" --n=100 --num_workers=0 --random_state=3001'
    'python ../src/encode_cifar10.py --batch_size=128 --cifar10_dir="/cluster/tufts/hugheslab/eharve06/CIFAR-10" --cifar101_v4_dir="/cluster/tufts/hugheslab/eharve06/CIFAR-10.1" --encoded_path="/cluster/tufts/hugheslab/eharve06/random-Fourier-features/datasets/CIFAR-10/n=1000_random_state=1001.pth" --n=1000 --num_workers=0 --random_state=1001'
    'python ../src/encode_cifar10.py --batch_size=128 --cifar10_dir="/cluster/tufts/hugheslab/eharve06/CIFAR-10" --cifar101_v4_dir="/cluster/tufts/hugheslab/eharve06/CIFAR-10.1" --encoded_path="/cluster/tufts/hugheslab/eharve06/random-Fourier-features/datasets/CIFAR-10/n=1000_random_state=2001.pth" --n=1000 --num_workers=0 --random_state=2001'
    'python ../src/encode_cifar10.py --batch_size=128 --cifar10_dir="/cluster/tufts/hugheslab/eharve06/CIFAR-10" --cifar101_v4_dir="/cluster/tufts/hugheslab/eharve06/CIFAR-10.1" --encoded_path="/cluster/tufts/hugheslab/eharve06/random-Fourier-features/datasets/CIFAR-10/n=1000_random_state=3001.pth" --n=1000 --num_workers=0 --random_state=3001'
    'python ../src/encode_cifar10.py --batch_size=128 --cifar10_dir="/cluster/tufts/hugheslab/eharve06/CIFAR-10" --cifar101_v4_dir="/cluster/tufts/hugheslab/eharve06/CIFAR-10.1" --encoded_path="/cluster/tufts/hugheslab/eharve06/random-Fourier-features/datasets/CIFAR-10/n=10000_random_state=1001.pth" --n=10000 --num_workers=0 --random_state=1001'
    'python ../src/encode_cifar10.py --batch_size=128 --cifar10_dir="/cluster/tufts/hugheslab/eharve06/CIFAR-10" --cifar101_v4_dir="/cluster/tufts/hugheslab/eharve06/CIFAR-10.1" --encoded_path="/cluster/tufts/hugheslab/eharve06/random-Fourier-features/datasets/CIFAR-10/n=10000_random_state=2001.pth" --n=10000 --num_workers=0 --random_state=2001'
    'python ../src/encode_cifar10.py --batch_size=128 --cifar10_dir="/cluster/tufts/hugheslab/eharve06/CIFAR-10" --cifar101_v4_dir="/cluster/tufts/hugheslab/eharve06/CIFAR-10.1" --encoded_path="/cluster/tufts/hugheslab/eharve06/random-Fourier-features/datasets/CIFAR-10/n=10000_random_state=3001.pth" --n=10000 --num_workers=0 --random_state=3001'
    'python ../src/encode_cifar10.py --batch_size=128 --cifar10_dir="/cluster/tufts/hugheslab/eharve06/CIFAR-10" --cifar101_v4_dir="/cluster/tufts/hugheslab/eharve06/CIFAR-10.1" --encoded_path="/cluster/tufts/hugheslab/eharve06/random-Fourier-features/datasets/CIFAR-10/n=50000_random_state=1001.pth" --n=50000 --num_workers=0 --random_state=1001'
    'python ../src/encode_cifar10.py --batch_size=128 --cifar10_dir="/cluster/tufts/hugheslab/eharve06/CIFAR-10" --cifar101_v4_dir="/cluster/tufts/hugheslab/eharve06/CIFAR-10.1" --encoded_path="/cluster/tufts/hugheslab/eharve06/random-Fourier-features/datasets/CIFAR-10/n=50000_random_state=2001.pth" --n=50000 --num_workers=0 --random_state=2001'
    'python ../src/encode_cifar10.py --batch_size=128 --cifar10_dir="/cluster/tufts/hugheslab/eharve06/CIFAR-10" --cifar101_v4_dir="/cluster/tufts/hugheslab/eharve06/CIFAR-10.1" --encoded_path="/cluster/tufts/hugheslab/eharve06/random-Fourier-features/datasets/CIFAR-10/n=50000_random_state=3001.pth" --n=50000 --num_workers=0 --random_state=3001'
)

eval "${experiments[$SLURM_ARRAY_TASK_ID]}"

conda deactivate
