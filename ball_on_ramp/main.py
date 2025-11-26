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
from .makeListTasks import get_sim_info, get_available_positions
from .temporal_feature_extractor import TemporalFeatureExtractor

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
                    ex["move_y"],
                )
                for ex in item["examples"]
            ]
        )
    else:
        raise ValueError("Not implemented")

if __name__ == "__main__":

    # args = commandlineArguments(
    #     enumerationTimeout=10, activation="tanh",
    #     iterations=2, recognitionTimeout=1000,
    #     a=3, maximumFrontier=10, topK=2, pseudoCounts=30.0,
    #     helmholtzRatio=0.5, structurePenalty=1.,
    #     CPUs=numberOfCPUs(),
    #     cuda=False,  # Disable CUDA to avoid cuDNN errors
    # )
    args = commandlineArguments(
        enumerationTimeout=10, activation="tanh",
        iterations=2, recognitionTimeout=1000,
        a=3, maximumFrontier=10, topK=2, pseudoCounts=30.0,
        helmholtzRatio=0.2,  # lower: more weight on real replay once we get hits
        structurePenalty=1.,
        CPUs=numberOfCPUs(),
        cuda=False,
    )

    timestamp = datetime.datetime.now().isoformat()
    outdir = 'experimentOutputs/ball_on_ramp/'
    os.makedirs(outdir, exist_ok=True)
    outprefix = outdir + timestamp
    args.update({
        "outputPrefix": outprefix,
        "featureExtractor": TemporalFeatureExtractor,  # Re-enable recognition model
    })

    # Create list of primitives
    primitives = get_primitives()

    # Create grammar
    grammar = Grammar.uniform(primitives)

    available_positions = get_available_positions()
    if not available_positions:
        raise RuntimeError("No obstacle positions found in datasets/ramp_temporal.jsonl. Generate data first.")

    available_positions = sorted(available_positions)
    box_ratios_train = [t for idx, t in enumerate(available_positions) if idx % 3 != 0]
    box_ratios_test = [t for idx, t in enumerate(available_positions) if idx % 3 == 0]

    if not box_ratios_train:
        box_ratios_train = available_positions[:-1] or available_positions
        box_ratios_test = available_positions[-1:] if len(available_positions) > 1 else []

    def make_examples(positions):
        """
        Like other DreamCoder domains: each task gets all its examples.
        No subsampling—use the entire dataset generated for each position.
        Control dataset size via num_examples in get_ball_jsonl.py instead.
        """
        tasks = []
        for box_pos in positions:
            examples = get_sim_info(box_pos)
            if not examples:
                continue
            tasks.append({"name": f"box_pos_{box_pos:.3f}", "examples": examples})
        return tasks

    training_examples = make_examples(box_ratios_train)
    testing_examples = make_examples(box_ratios_test)

    training = (
        [get_ball_on_ramp_task(item, move="x") for item in training_examples]
        + [get_ball_on_ramp_task(item, move="y") for item in training_examples]
    )
    testing = (
        [get_ball_on_ramp_task(item, move="x") for item in testing_examples]
        + [get_ball_on_ramp_task(item, move="y") for item in testing_examples]
    )

    # EC iterate

    generator = ecIterator(grammar, training, testingTasks=testing, **args)
    for i, _ in enumerate(generator):
        print(f"ecIterator count {i}")