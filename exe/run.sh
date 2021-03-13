#!/bin/bash
export PYTHONPATH=~/Documents/researches/ICAPS-2021/

ITERATION=20
EPISODES=2500
SEED=1

RMAX=20

START=14
GOAL=17
MDP=GRAPHWORLD_S${START}G${GOAL}E${EPISODES}I${ITERATION}_TEST
python exe/experiment_v1.py --start ${START} --goal ${GOAL} --mdpName ${MDP} -s ${SEED} -e ${EPISODES} -i ${ITERATION} -r ${RMAX}
python utils/graphics.py --mdp ${MDP} -w 25 -a 'Q-Learning Dyna-Q DARLING GDQ'

START=14
GOAL=18
MDP=GRAPHWORLD_S${START}G${GOAL}E${EPISODES}I${ITERATION}_TEST
python exe/experiment_v1.py --start ${START} --goal ${GOAL} --mdpName ${MDP} -s ${SEED} -e ${EPISODES} -i ${ITERATION} -r ${RMAX}
python utils/graphics.py --mdp ${MDP} -w 25 -a 'Q-Learning Dyna-Q DARLING GDQ'

START=0
GOAL=17
MDP=GRAPHWORLD_S${START}G${GOAL}E${EPISODES}I${ITERATION}_TEST
python exe/experiment_v1.py --start ${START} --goal ${GOAL} --mdpName ${MDP} -s ${SEED} -e ${EPISODES} -i ${ITERATION} -r ${RMAX}
python utils/graphics.py --mdp ${MDP} -w 25 -a 'Q-Learning Dyna-Q DARLING GDQ'

START=0
GOAL=18
MDP=GRAPHWORLD_S${START}G${GOAL}E${EPISODES}I${ITERATION}_TEST
python exe/experiment_v1.py --start ${START} --goal ${GOAL} --mdpName ${MDP} -s ${SEED} -e ${EPISODES} -i ${ITERATION} -r ${RMAX}
python utils/graphics.py --mdp ${MDP} -w 25 -a 'Q-Learning Dyna-Q DARLING GDQ'



ITERATION=20
EPISODES=5000
SEED=1

RMAX=20

START=0
GOAL=17
N_START=0
N_GOAL=18
MDP=GRAPHWORLD_S${START}G${GOAL}S${N_START}G${N_GOAL}E${EPISODES}I${ITERATION}_TEST
python exe/experiment_v2.py --start ${START} --goal ${GOAL} --next_start ${N_START} --next_goal ${N_GOAL} --mdpName ${MDP} -s ${SEED} -e ${EPISODES} -i ${ITERATION} -r ${RMAX}
python utils/graphics.py --mdp ${MDP} -w 25 -a 'Q-Learning Dyna-Q DARLING GDQ'

START=0
GOAL=17
N_START=17
N_GOAL=14
MDP=GRAPHWORLD_S${START}G${GOAL}S${N_START}G${N_GOAL}E${EPISODES}I${ITERATION}_TEST
python exe/experiment_v2.py --start ${START} --goal ${GOAL} --next_start ${N_START} --next_goal ${N_GOAL} --mdpName ${MDP} -s ${SEED} -e ${EPISODES} -i ${ITERATION} -r ${RMAX}
python utils/graphics.py --mdp ${MDP} -w 25 -a 'Q-Learning Dyna-Q DARLING GDQ'

