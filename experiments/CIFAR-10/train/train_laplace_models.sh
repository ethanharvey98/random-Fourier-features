#!/bin/bash
#
# Hyperparameter Search for RFF + Laplace Model on CIFAR-10
# Usage: bash run_rff_laplace_cifar10.sh ACTION_NAME
#
# ACTION_NAME options:
#   list     - Print experiment configurations without running (default)
#   submit   - Submit jobs to SLURM
#   run_here - Run experiments locally

if [[ -z $1 ]]; then
    ACTION_NAME='list'
else
    ACTION_NAME=$1
fi

################################################# OUTPUT DIRECTORY ##############################################################
export output_dir="../../../results/CIFAR-10/laplace_results"
export repo_dir="../../.."


########################### FIXED CONFIGURATION ################################
export script="./train_rff_laplace_cifar10.py"
export batch_size=128
export epochs=10000
export patience=50
export prediction_samples=10000


########################### HYPERPARAMETER SEARCH SPACE ################################

# Dataset sizes
declare -a ns=(50000)

# Random seeds
declare -a random_states=(1001 2001 3001)

# Kernel rank (number of random Fourier features)
declare -a ranks=(1024)

# Lengthscale
declare -a lengthscales=(20.0)

# Outputscale
declare -a outputscales=(1.0)

# Learning rate
declare -a learning_rates=(0.0001)

########################### EXPERIMENT EXECUTION ################################

# Calculate total number of experiments
total_experiments=0
for n in "${ns[@]}"; do
    for random_state in "${random_states[@]}"; do
        for rank in "${ranks[@]}"; do
            for lengthscale in "${lengthscales[@]}"; do
                for outputscale in "${outputscales[@]}"; do
                    for lr in "${learning_rates[@]}"; do
                        ((total_experiments++))
                    done
                done
            done
        done
    done
done

echo "=========================================="
echo "RFF + Laplace CIFAR-10 Hyperparameter Search"
echo "=========================================="
echo "Total experiments to run: $total_experiments"
echo ""

# Run the search
experiment_count=0
for n in "${ns[@]}"; do
    for random_state in "${random_states[@]}"; do
        for rank in "${ranks[@]}"; do
            for lengthscale in "${lengthscales[@]}"; do
                for outputscale in "${outputscales[@]}"; do
                    for lr in "${learning_rates[@]}"; do
                        ((experiment_count++))
                        
                        export n=$n
                        export random_state=$random_state
                        export rank=$rank
                        export lengthscale=$lengthscale
                        export outputscale=$outputscale
                        export lr=$lr
                        
                        export result_dir="${output_dir}/n_${n}/rs_${random_state}/rank_${rank}-ls_${lengthscale}-os_${outputscale}-lr_${lr}"
                        
                        echo "[$experiment_count/$total_experiments] n=$n, rs=$random_state, rank=$rank, ls=$lengthscale, os=$outputscale, lr=$lr"
                        
                        if [[ $ACTION_NAME == 'submit' ]]; then
                            mkdir -p $result_dir
                            sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=rff_n${n}_rs${random_state}
#SBATCH --output=${result_dir}/run_logs_%j.out
#SBATCH --error=${result_dir}/run_errs_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu
#SBATCH --partition=gpu

source ~/.bashrc
conda activate l3d_2024f_cuda12_1

python $script \
    --repo_dir=$repo_dir \
    --result_dir=$result_dir \
    --n=$n \
    --random_state=$random_state \
    --batch_size=$batch_size \
    --rank=$rank \
    --lengthscale=$lengthscale \
    --outputscale=$outputscale \
    --prediction_samples=$prediction_samples \
    --epochs=$epochs \
    --learning_rate=$lr \
    --patience=$patience
EOF
                                
                        elif [[ $ACTION_NAME == 'run_here' ]]; then
                            mkdir -p $result_dir
                            python $script \
                                --repo_dir=$repo_dir \
                                --result_dir=$result_dir \
                                --n=$n \
                                --random_state=$random_state \
                                --batch_size=$batch_size \
                                --rank=$rank \
                                --lengthscale=$lengthscale \
                                --outputscale=$outputscale \
                                --prediction_samples=$prediction_samples \
                                --epochs=$epochs \
                                --learning_rate=$lr \
                                --patience=$patience
                        
                        elif [[ $ACTION_NAME == 'list' ]]; then
                            echo "  -> $result_dir"
                        fi
                    done
                done
            done
        done
    done
done

echo ""
echo "=========================================="
echo "Search configuration complete!"
echo "=========================================="
echo ""
echo "Hyperparameter ranges:"
echo "  Dataset sizes (n): ${ns[@]}"
echo "  Random states: ${random_states[@]}"
echo "  Ranks: ${ranks[@]}"
echo "  Lengthscales: ${lengthscales[@]}"
echo "  Outputscales: ${outputscales[@]}"
echo "  Learning rates: ${learning_rates[@]}"