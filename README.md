Description
===
This repository is the bunch of source codes for the ICAPS-21 
It includes an ASP code base that provides action knowledge, 
source code for the proposed method, and reinforcement learning algorithms for comparison.


How to run
---
Please create directories for logs, figures, and binary data of trained agents
at the root directory.
```bash
mkdir -p datas/images datas/logs
```
Install dependencies
```bash
pip install -r requirements.txt
```

For running simulation experiments, please execute the following command.
```bash
sh run.sh
```

There are 6 experiments in `run.sh`. 
The results of 6 experiments are included to the main paper.
The remaining experiments are auxiliary experiments where the agent navigates 
another initial state to another goal state.
You can change some parameters of the agents and simulation settings by modifying `run.sh`, `exe/exeutils.py`.
