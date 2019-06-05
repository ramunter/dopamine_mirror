#!/usr/bin/env bash
source /home/shomec/r/ramunter/masters-thesis/venv/bin/activate

for ((i=0; i<10; i++)); do
	nice python -um dopamine.discrete_domains.train  --gin_files='dopamine/agents/bdqn/configs/dqn_cartpole.gin' --base_dir ../Desktop/cartpole_plot_data2/bdqn_cartpole_$i
	nice python -um dopamine.discrete_domains.train --gin_files='dopamine/agents/dqn/configs/dqn_cartpole.gin' --base_dir ../Desktop/cartpole_plot_data2/dqn_cartpole_$i
done

