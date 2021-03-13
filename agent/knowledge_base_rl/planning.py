import subprocess
import os
from collections import defaultdict

DIR_ASP_FILES = os.path.dirname(__file__) + "/asp_navigation"
QUERY_FILE = DIR_ASP_FILES + "/query.asp"
STATE_FILES = DIR_ASP_FILES + "/state.asp"
ASP_FILES = DIR_ASP_FILES + "/*.asp "
SOLVER = "clingo "
OPTION_STEP = lambda x: "-c n={0} ".format(x)
OPTION_ANS = lambda x: "-n {0} ".format(x)  # 0: all answers
TOLERANCE = 1.3
MAX_STEPS = 20


class Planner:
    def __init__(self):
        self.plans_memory = defaultdict(list)
        self.reset_query()

    def planning(self, current_state, target_state):
        # Make file which plans paths.
        make_query(current_state, target_state)
        output = execute_planning()
        # print(output)
        plans_list = list()
        for i in parse_plans(output):
            plans_list.append(i)

        plans = list()
        for raw_plan in [item for sublist in plans_list for item in sublist]:
            plan = arrange_plan(raw_plan)
            plans.append(plan)
        if current_state == target_state:
            self.reset_query()
        self.plans_memory[(current_state, target_state)] = plans

    def get_plans(self, current_state, target_state):
        if (current_state, target_state) in self.plans_memory.keys():
            return self.plans_memory[(current_state, target_state)]
        else:
            self.planning(current_state, target_state)
            return self.plans_memory[(current_state, target_state)]

    @staticmethod
    def reset_query():
        f = open(STATE_FILES, "w")
        f.write("")
        f.close()

    def reset_memory(self):
        self.plans_memory = defaultdict(list)


def make_query(current_state, target_state):
    cur_s, cur_d, cur_ds = state2rule(current_state)
    tar_s, tar_d, tar_ds = state2rule(target_state)
    initial_at = "at(" + cur_s + ", 0)."
    final_at = ":- not at(" + tar_s + ", n-1)."
    show_text = """#show approach/2.
                    \n#show gothrough/2.
                    \n#show opendoor/2.
                    \n#show goto/2.
                    \n#show stay/2.
                    \n#show at/2.
                    \n#show hasdoor/2.
                    \n#show open/2.
                    \n#show facing/2.
                    \n%#show path/3."""

    query = initial_at + "\n" + final_at + "\n" + show_text
    f = open(QUERY_FILE, "w")
    f.write(query)
    f.close()

    facing, open_state = read_state_file()

    if cur_d is not None and cur_ds == "True":
        open_state.append("open({0}, 0).".format(cur_d))

    open_facts = list()
    for i, opens in enumerate(open_state):
        if cur_d is not None and opens.find(cur_d) != -1 and cur_ds == "False":
            continue
        else:
            open_facts.append(open_state[i])

    if cur_d is not None:
        facing = "facing({0}, 0).".format(cur_d)
    else:
        facing = ""

    f = open(STATE_FILES, "w")
    f.write("".join(open_facts) + "\n" + facing)
    f.close()
    read_state_file()


def read_state_file():
    f = open(STATE_FILES, "r")
    lines = f.readlines()
    facing = None
    open_state = list()
    # print("[", end="")
    for line in lines:
        # print(line, end=" ")
        if line.find("facing") != -1:
            facing = line
        if line.find("open") != -1:
            open_state.append(line)
    # print("]")
    return facing, open_state


def execute_planning():
    # Parse the output from clingo
    for step in range(MAX_STEPS):
        command = SOLVER + OPTION_STEP(step) + OPTION_ANS(0) + ASP_FILES
        p = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
        # print(command)
        (output, err) = p.communicate()
        p.wait()
        if "UNSATISFIABLE" not in str(output):
            tolerance_step = int(step * TOLERANCE)
            command = SOLVER + OPTION_STEP(tolerance_step) + OPTION_ANS(0) + ASP_FILES
            # print(command)
            p = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
            (output, err) = p.communicate()
            p.wait()
            # print(output)
            break
    else:
        raise Exception("Could not find satisfiable plans")

    return output


def parse_plans(output):
    lines = str(output).split("\\n")
    plans_list = []
    for i, line in enumerate(lines):
        if line.find("Answer") != -1:
            plans_list.append(lines[i + 1])

    plans_group = list()
    for trajectory in plans_list:
        plan = trajectory.split()
        at_list = []
        action_list = []
        hasdoor_list = []
        opendoor_list = []
        facing_list = []
        for p in plan:
            prefix = p[:p.find("(")]
            location_step_pair = p[p.find("(") + 1:p.find(")")]
            tmp = location_step_pair.split(",")
            if prefix == "at":
                at_list.append([prefix] + tmp)
            elif prefix == "hasdoor":
                hasdoor_list.append([prefix] + tmp)
            elif prefix == "open":
                opendoor_list.append([prefix] + tmp)
            elif prefix == "facing":
                facing_list.append([prefix] + tmp)
            else:
                action_list.append([prefix] + tmp)

        location_group = []
        """
            location: [timestep, state, door, door_state, action, next_state, next_door, next_door_state]
        """
        for _, s, t in at_list[:-1]:
            location = list()
            location.append(int(t))
            location.append(s)
            location.append(None)
            for d in hasdoor_list:
                if d[1] == at_list[int(t)][1]:
                    location[-1] = d[2]
                    break
            if ['open', location[-1], str(location[0])] in opendoor_list:
                location.append(True)
            else:
                location.append(False)
            for a in action_list:
                if a[2] == t:
                    location.append(a[0])
                    break
            location.append(at_list[int(t) + 1][1])
            location.append(None)
            for d in hasdoor_list:
                if d[1] == at_list[int(t) + 1][1]:
                    location[-1] = d[2]
                    break
            if ['open', location[-1], str(location[0] + 1)] in opendoor_list:
                location.append(True)
            else:
                location.append(False)

            location_group.append(location)
        location_group.sort(key=sort_tasks)
        plans_group.append(location_group)
    return plans_group


def arrange_plan(plan):
    t, s, d, ds, a, sp, nd, nds = plan
    if nd is None:
        if a == "goto" or a == "approach":
            a = (a, int(sp[1:]))
        else:
            a = (a, int(s[1:]))
        sp = "{0}".format(sp)
    else:
        if a == "goto" or a == "approach" or a == "stay":
            a = (a, int(sp[1:]))
        else:
            a = (a, int(s[1:]))
        sp = "{0}_{1}_{2}".format(sp, nd, nds)
    if d is None:
        s = "{0}".format(s)
    else:
        s = "{0}_{1}_{2}".format(s, d, ds)
    # return t, s, d, ds, a, sp, nd, nds
    return s, a, sp


def state2rule(state):
    if len(str(state).split("_")) == 3:
        s, d, ds = str(state).split("_")
        return s, d, ds
    else:
        s = str(state).split("_")[0]
        return s, None, None


def sort_tasks(current_task):
    return current_task[0]


if __name__ == '__main__':
    planner = Planner()
    # for s in ["s3_d0_True"]:
    for s in ["s0", "s14"]:
        plans = planner.get_plans(s, "s17")
        for plan in plans:
            print(plan)
        print(plans)
        print("+++++++++++++++++++++++++")

    # planner.reset_query()
