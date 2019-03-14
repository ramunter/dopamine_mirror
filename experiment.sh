#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1 
#SBATCH --time=72:00:00 
#SBATCH --partition=EPICALL 
#SBATCH --job-name="Bayesian DQN"
#SBATCH --output=bayesiandqn.out

HOMEDIR=${SLURM_SUBMIT_DIR}
cd ${HOMEDIR}
module load GCC/7.3.0-2.30  OpenMPI/3.1.1 TensorFlow/1.12.0-Python-3.6.6 
source venv/bin/activate
nice python -um dopamine.discrete_domains.train --base_dir=//lustre1/work/ramunter/bdqn --gin_files='dopamine/agents/bdqn/configs/dqn.gin'
