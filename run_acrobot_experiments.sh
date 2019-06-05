#!/usr/bin/env bash
source /home/shomec/r/ramunter/masters-thesis/venv/bin/activate

for ((i=0; i<10; i++)); do
	nice python -um dopamine.discrete_domains.train  --gin_files='dopamine/agents/bdqn/configs/dqn_acrobot.gin' --base_dir ../Desktop/acrobot/bdqn_acrobot_$i
	#nice python -um dopamine.discrete_domains.train --gin_files='dopamine/agents/dqn/configs/dqn_acrobot.gin' --base_dir ../Desktop/acrobot/dqn_acrobot_$i
done

