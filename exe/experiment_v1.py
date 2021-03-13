from mdp.graphworld.graphworld import GraphWorld
from mdp.graphworld.config import MAP_PATH

from agent.qlearning import QLearningAgent
from agent.dynaq import DynaQAgent
from agent.knowledge_base_rl.darling import DARLINGAgent
from agent.knowledge_base_rl.gdq import GDQAgent
import exe.exeutils


if __name__ == "__main__":
    opts = exe.exeutils.parse_options()
    ###########################
    # GET THE BLOCKWORLD
    ###########################
    env = GraphWorld(
        name=opts.mdpName,
        graphmap_path=MAP_PATH + "map2.json",
        init_node=opts.start,
        goal_node=opts.goal,
        step_cost=1.0,
        goal_reward=opts.rmax,
        stack_cost=opts.rmax
    )

    ###########################
    # GET THE AGENT
    ###########################
    qlearning = QLearningAgent(name="Q-Learning", gamma=opts.gamma, actions=env.get_executable_actions())
    dynaq = DynaQAgent(name="Dyna-Q", gamma=opts.gamma, actions=env.get_executable_actions(), lookahead=3)
    darling = DARLINGAgent(name="DARLING", gamma=opts.gamma, actions=env.get_executable_actions(),
                           goal_state=env.goal_state)
    gdq = GDQAgent(name="GDQ", gamma=opts.gamma, actions=env.get_executable_actions(), lookahead=5, rmax=opts.rmax,
                   goal_state=env.goal_state)

    ###########################
    # RUN
    ###########################
    exe.exeutils.runs_episodes_wo_switching(env, qlearning, step=opts.iters, episode=opts.episodes, seed=opts.seeds)
    exe.exeutils.runs_episodes_wo_switching(env, dynaq, step=opts.iters, episode=opts.episodes, seed=opts.seeds)
    exe.exeutils.runs_episodes_wo_switching(env, darling, step=opts.iters, episode=opts.episodes, seed=opts.seeds)
    exe.exeutils.runs_episodes_wo_switching(env, gdq, step=opts.iters, episode=opts.episodes, seed=opts.seeds)
