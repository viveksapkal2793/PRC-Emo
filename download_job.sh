#!/bin/bash
#SBATCH --job-name=download_iemocap
#SBATCH --output=/scratch/data/bikash_rs/Vivek/PRC-Emo/logs/%x_%j.out
#SBATCH --error=/scratch/data/bikash_rs/Vivek/PRC-Emo/logs/%x_%j.err
#SBATCH --partition=fat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH -D /scratch/data/bikash_rs/Vivek/PRC-Emo

# Create logs directory
mkdir -p logs

module purge
module load gcc
module load openssl
module load ca-certificates-mozilla

# export PATH=/scratch/data/bikash_rs/vivek/bin/aria2/bin:$PATH
export LD_LIBRARY_PATH=/scratch/apps/spack/opt/spack/linux-rhel8-zen2/gcc-12.2.0/openssl-3.1.3-v7ciwvw4qdckoayb6uzerjizxngkaovq/lib64:$LD_LIBRARY_PATH
export PATH=/scratch/data/bikash_rs/Vivek/aria2-local/bin:$PATH

aria2c -x 16 -s 16 --continue=true --check-certificate=false --dir="/scratch/data/bikash_rs/Vivek/dataset/" --http-user="iemocap" --http-passwd="sail_Qz>{GI8VLU{8x#L}" https://sail.usc.edu/databases/iemocap/IEMOCAP_full_release.tar.gz
