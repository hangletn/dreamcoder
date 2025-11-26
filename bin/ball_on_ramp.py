import datetime
import os
import random

import binutil
import numpy as np

from dreamcoder.ec import commandlineArguments, ecIterator
from dreamcoder.grammar import Grammar
from dreamcoder.program import Primitive
from dreamcoder.task import Task
from dreamcoder.type import arrow, tint, tlist, tbool, treal, tpair
from dreamcoder.utilities import numberOfCPUs

# Define primitives_map
def _always(f): return lambda x: all([f(i) for i in x])

def _eventually(f): return lambda x: any([f(i) for i in x])

def _until(phi):
    def with_psi(psi):
        def until_predicate(trace):
            n = len(trace)
            for i in range(n):
                if psi(trace[i]):
                    if all(phi(trace[j]) for j in range(i)):
                        return True
            return False
        return until_predicate
    return with_psi

def _eq(x): return lambda y: x == y

def _gt(x): return lambda y: x > y

def _and(x): return lambda y: x and y

def _or(x): return lambda y: x or y

def _not(x): return not x

def _eq0(x): return x == 0.0

def _gt0(x): return x > 0.0

def _pair(x):
    return lambda y: (x, y)

def _first(x):
    return x[0]

def _second(x):
    return x[1]

def get_primitives():
    return [
        Primitive("true", tbool, True),
        Primitive("not", arrow(tbool, tbool), _not),
        Primitive("and", arrow(tbool, tbool, tbool), _and),
        Primitive("or", arrow(tbool, tbool, tbool), _or),
        Primitive("eq_real", arrow(treal, treal, tbool), _eq),
        Primitive("gt_real", arrow(treal, treal, tbool), _gt),
        Primitive("eq0_real", arrow(treal, tbool), _eq0),
        Primitive("gt0_real", arrow(treal, tbool), _gt0),
        Primitive("always", arrow(arrow(treal, tbool), tlist(treal), tbool), _always),
        Primitive("eventually", arrow(arrow(treal, tbool), tlist(treal), tbool), _eventually),
        Primitive("until", arrow(arrow(treal, tbool), arrow(treal, tbool), tlist(treal), tbool), _until),
        Primitive("pair", arrow(tbool, tbool, tpair(tbool, tbool)), _pair),
        Primitive("pair_first", arrow(tpair(tbool, tbool), tbool), _first),
        Primitive("pair_second", arrow(tpair(tbool, tbool), tbool), _second),
    ]


# TYPE: (list[float], list[float], ) -> (bool, bool)
# TODO: Split into `move_x` and `move_y`
def get_ball_on_ramp_task(item, move="x"):
    if move == "x":
        return Task(
            item['name'] + "_x",
            arrow(
                tlist(treal), # x_pos
                tlist(treal), # y_pos
                treal, # x_obstacle
                treal, #y_obstacle
                tbool # (move_x)
            ),
            examples= [
                (
                    (
                        ex["ball_x"],
                        ex["ball_y"],
                        ex["obstacle_x"],
                        ex["obstacle_y"],
                    ),
                    ex["move_x"],
                )
                for ex in item["examples"]
            ]
        )
    elif move == "y":
        return Task(
            item['name'] + "_y",
            arrow(
                tlist(treal), # x_pos
                tlist(treal), # y_pos
                treal, # x_obstacle
                treal, #y_obstacle
                tbool # (move_y)
            ),
            examples= [
                (
                    (
                        ex["ball_x"],
                        ex["ball_y"],
                        ex["obstacle_x"],
                        ex["obstacle_y"],
                    ),
                    ex["move_y"],
                )
                for ex in item["examples"]
            ]
        )
    else:
        raise ValueError("Not implemented")

def generate_dummy_data(box_pos, trace_len=10):
    import numpy as np
    """
    Input: box_pos: float(0, 1), trace_len: int, num_examples: int
    Output: List([(pos_x: [float], pos_y[float], obstacle_x: float, obstacle_y: float), out: move_x: bool, move_y: bool])
    
    """
    seq_len = 100
    vert_pos = 25.0
    horizon_pos = 60.0
    ramp_coeff, ramp_offset = vert_pos / horizon_pos,  vert_pos
    box_pos_x = box_pos * seq_len
    box_pos_y = round(max(-ramp_coeff * box_pos_x + ramp_offset, 0.0),3)
    x_ramp = np.arange(horizon_pos)
    y_ramp = -ramp_coeff * x_ramp + ramp_offset
    x_horz = np.arange(horizon_pos, seq_len)
    y_horz = np.zeros(int(seq_len - horizon_pos))
    x_pos_no_obs = np.concatenate([x_ramp, x_horz])
    y_pos_no_obs = np.concatenate([y_ramp, y_horz])
    x_pos = np.zeros(seq_len)
    y_pos = np.zeros(seq_len)
    for i in range(seq_len):
        if i < box_pos_x:
            x_pos[i] = round(x_pos_no_obs[i], 3)
            y_pos[i] = round(y_pos_no_obs[i], 3)
        else:
            x_pos[i] = box_pos_x
            y_pos[i] = box_pos_y
    move_x = [True if x_pos[i] != x_pos[i+1] else False for i in range(len(x_pos)-1)] + [False]
    move_y = [True if y_pos[i] != y_pos[i+1] else False for i in range(len(y_pos)-1)] + [False]

    # Generate example list
    examples = []
    for i in range(0, seq_len-trace_len-1, 2):
        start_idx, end_idx = i, i + trace_len
        examples.append(
            {
                "ball_x": list(x_pos[start_idx : end_idx]),
                "ball_y": list(y_pos[start_idx : end_idx]),
                "obstacle_x": box_pos_x,
                "obstacle_y": box_pos_y,
                "move_x": move_x[end_idx-1],
                "move_y": move_y[end_idx-1],
            }
        )
    return examples

if __name__ == "__main__":

    args = commandlineArguments(
        enumerationTimeout=10, activation="tanh",
        iterations=5, recognitionTimeout=1000,
        a=3, maximumFrontier=10, topK=2, pseudoCounts=30.0,
        helmholtzRatio=0.5, structurePenalty=1.,
        CPUs=numberOfCPUs(),
    )

    timestamp = datetime.datetime.now().isoformat()
    outdir = 'experimentOutputs/ball_on_ramp/'
    os.makedirs(outdir, exist_ok=True)
    outprefix = outdir + timestamp
    args.update({"outputPrefix": outprefix})

    # Create list of primitives
    primitives = get_primitives()

    # Create grammar
    grammar = Grammar.uniform(primitives)
    box_ratios_train = [i/10.0 for i in range(10)]
    box_ratios_test = [i/10.0 for i in range(10) if i % 3 == 0]

    training_examples_list = []
    for i in box_ratios_train:
        training_examples_list += generate_dummy_data(i)
    training_examples = [
        {"name": "move", "examples": training_examples_list}
    ]

    training = [get_ball_on_ramp_task(item, move="x") for item in training_examples] + [get_ball_on_ramp_task(item, move="y") for item in training_examples]

    # Testing data

    testing_examples_list = []
    for i in box_ratios_test:
        testing_examples_list += generate_dummy_data(i)
    testing_examples = [
        {"name": "move", "examples": testing_examples_list}
    ]
    testing = [get_ball_on_ramp_task(item, move="x") for item in testing_examples] + [get_ball_on_ramp_task(item, move="y") for item in testing_examples]

    # EC iterate

    generator = ecIterator(grammar, training, testingTasks=testing, **args)
    for i, _ in enumerate(generator):
        print(f"ecIterator count {i}")
