import datetime
import os
import random

import binutil
import numpy as np

from dreamcoder.ec import commandlineArguments, ecIterator
from dreamcoder.grammar import Grammar
from dreamcoder.program import Primitive
from dreamcoder.task import Task
from dreamcoder.type import arrow, tint
from dreamcoder.utilities import numberOfCPUs
from makeListTasks import get_sim_info

# Define primitives
def _always(x, func): return lambda x, func: True if sum([func(i) for i in x]) == len(x) else False
def _eventually(x, func): return lambda x, func: True if sum([func(i) for i in x]) > 1 else False
def _until(x, func_1, func_2): return lambda x, func: True if sum(sum([func_1(x[j]) for j in range(i)] + [func_2(x[k] for k in range(i, len(x)))]) == len(x) for i in range(len(x))) > 1 else False

#TODO: Create a dataset with similar format
# TYPE: (list[float], list[float], ) -> (bool, bool)
def get_ball_on_ramp_task(item):
    return Task(
        item["name"],
        # TODO: find the compatible type
        [((ex["i"],), ex["o"]) for ex in item["examples"]]
    )

def generate_dummy_data(box_pos, trace_len=10):
    import numpy as np
    """
    Input: box_pos: float(0, 1), trace_len: int, num_examples: int
    Output: List([(pos_x: [float], pos_y[float], obstacle_x: float, obstacle_y: float), out: move_x: bool, move_y: bool])
    
    """
    seq_len = 100
    vert_pos = 25
    horizon_pos = 60
    ramp_coeff, ramp_offset = vert_pos / horizon_pos,  vert_pos
    box_pos_x = int(box_pos * seq_len)
    box_pos_y = -ramp_coeff * box_pos_x + ramp_offset
    x_ramp = np.arange(horizon_pos)
    y_ramp = -ramp_coeff * x_ramp + ramp_offset
    x_horz = np.arange(horizon_pos, seq_len)
    y_horz = np.zeros(seq_len - horizon_pos)
    x_pos_no_obs = np.concatenate([x_ramp, x_horz])
    y_pos_no_obs = np.concatenate([y_ramp, y_horz])
    x_pos = np.zeros(seq_len)
    y_pos = np.zeros(seq_len)
    for i in range(seq_len):
        if i < box_pos_x:
            x_pos[i] = x_pos_no_obs[i]
            y_pos[i] = y_pos_no_obs[i]
        else:
            x_pos[i] = box_pos_x
            y_pos[i] = box_pos_y
    move_x = [True if x_pos[i] != x_pos[i+1] else False for i in range(len(x_pos)-1)] + [False]
    move_y = [True if y_pos[i] != y_pos[i+1] else False for i in range(len(y_pos)-1)] + [False]

    # Generate example list
    examples = []
    for i in range(0, seq_len-trace_len-1, 2):
        start_idx, end_idx = i, i + trace_len
        examples.append({
            "i": (x_pos[start_idx: end_idx], y_pos[start_idx: end_idx], box_pos_x, box_pos_y),
            "o": (move_x[end_idx-1], move_y[end_idx-1]),
        })
    return examples, x_pos_no_obs, y_pos_no_obs, move_x, move_y

if __name__ == "__main__":

    args = commandlineArguments(
        enumerationTimeout=10, activation="tanh",
        iterations=10, recognitionTimeout=3600,
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
    # TODO: Check to see if we need other primitives too (AND, OR, NOT, NEXT, PREV, EQ, LEQ, GEQ, etc.)
    # Check whether there would be a set of default primitives,
    # or we need to define all the primitives here
    primitives = [
        Primitive("always", arrow(tint, tint), _always),
        Primitive("eventually", arrow(tint, tint), _eventually),
        Primitive("until", arrow(tint, tint), _until),
    ]

    # Create grammar
    grammar = Grammar.uniform(primitives)
    box_ratios_train = [i for i in range(10) if i % 3 != 0] / 10.0
    box_ratios_test = [i for i in range(10) if i % 3 == 0] / 10.0

    training_examples = [
        {"name": f"box_pos_{box_pos}", "examples": [generate_dummy_data(box_pos)]} for box_pos in box_ratios_train
    ] 

    training = [get_ball_on_ramp_task(item) for item in training_examples]

    # Testing data

    testing_examples = [
        {"name": f"box_pos_{box_pos}", "examples": [get_sim_data(box_pos)]} for box_pos in box_ratios_test
    ]
    testing = [get_ball_on_ramp_task(item) for item in testing_examples]

    # EC iterate

    generator = ecIterator(grammar, training, testingTasks=testing, **args)
    for i, _ in enumerate(generator):
        print(f"ecIterator count {i}")
