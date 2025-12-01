import datetime
import os
import random
import sys

_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
_bin_dir = os.path.join(_repo_root, 'bin')
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)
if _bin_dir not in sys.path:
    sys.path.insert(0, _bin_dir)

import binutil

import numpy as np

from dreamcoder.ec import commandlineArguments, ecIterator
from dreamcoder.grammar import Grammar
from dreamcoder.program import Primitive
from dreamcoder.task import Task
from dreamcoder.type import arrow, tint, tlist, tbool, treal, tpair, t0
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

# def _is_moving(trace, eps=1e-3):
#     """Check if values in trace are changing (ball is moving)."""
#     if len(trace) < 2:
#         return False
#     changes = [abs(trace[i+1] - trace[i]) for i in range(len(trace)-1)]
#     return max(changes) > eps

# def _reached_obstacle(pos, obstacle_pos):
#     """Check if ball position has reached or passed obstacle position."""
#     # Note: This is approximate - actual collision is pos + radius >= obstacle_pos
#     # But radius isn't available in program inputs, so we use pos >= obstacle as approximation
#     return pos >= obstacle_pos

# def _last(xs):
#     """Get last element of list."""
#     return xs[-1] if xs else 0.0

def _car(xs):
    """Get first element of list (same as heOCaml-compatible: car)."""
    return xs[0] if xs else 0.0

def _cdr(xs):
    """Get tail of list (same as the OCaml-compatible: cdr)."""
    return xs[1:] if len(xs) > 1 else []

def _cons(x):
    """Prepend element to list (same as the OCaml-compatible: cons)."""
    return lambda xs: [x] + xs

def _is_empty(xs):
    """Check if list is empty (same as the OCaml-compatible: empty?)."""
    return len(xs) == 0


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
        Primitive("car", arrow(tlist(t0), t0), _car),  # Get first element 
        Primitive("cdr", arrow(tlist(t0), tlist(t0)), _cdr),  # Get tail 
        Primitive("cons", arrow(t0, tlist(t0), tlist(t0)), _cons),  # Prepend element 
        Primitive("empty?", arrow(tlist(t0), tbool), _is_empty),  # Check if empty 
    ]


# TYPE: (list[float], list[float], ) -> (bool, bool)
# TODO: Split into `move_x` and `move_y`
def get_ball_on_ramp_task(item, move="x"):
    # Extract meta info from first example (all examples in a task share the same meta)
    # Default meta values
    meta = {"ball_radius": 25.0, "box_size": 50.0, "ramp_theta":  -0.2914567944778671}
    if item["examples"]:
        first_ex = item["examples"][0]
        meta = {
            "ball_radius": first_ex.get("ball_radius", 25.0),
            "box_size": first_ex.get("box_size", 50.0),
            "ramp_theta": first_ex.get("ramp_theta", -0.2914567944778671),
        }
    
    if move == "x":
        task = Task(
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
        # Store meta info as task attribute for feature extractor
        task.meta = meta
        return task
    elif move == "y":
        task = Task(
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
        task.meta = meta
        return task
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
        enumerationTimeout=300, activation="tanh",  # Increased to 5 minutes for deeper search
        iterations=10, recognitionTimeout=1200,  # Increased to 10 iterations and 20 min recognition training
        a=3, maximumFrontier=10, topK=2, pseudoCounts=30.0,
        helmholtzRatio=0.2,  # lower: more weight on real replay once we get hits
        structurePenalty=1.,
        CPUs=numberOfCPUs(),
        cuda=False,
        solver="ocaml",  
        compressor="ocaml", 
    )
    # args.update({
    #     "evaluationTimeout": 0.1,  # More time for complex evaluations
    # })

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
    
    # primitive_names = [p.name for p in primitives]
    # print(f"Total primitives: {len(primitives)}")

    # Create grammar
    grammar = Grammar.uniform(primitives)

    available_positions = get_available_positions()
    if not available_positions:
        raise RuntimeError("No obstacle positions found in datasets/ramp_temporal.jsonl. Generate data first.")

    available_positions = sorted(available_positions)
    
    MAX_TRAIN_POSITIONS = 6
    MAX_TEST_POSITIONS = 2
    MAX_EXAMPLES_PER_TASK = 100
    
    if len(available_positions) <= MAX_TRAIN_POSITIONS:
        box_ratios_train = available_positions
        box_ratios_test = []
    else:
        # Use first few positions for training, one more for testing
        box_ratios_train = available_positions[:MAX_TRAIN_POSITIONS]
        if len(available_positions) > MAX_TRAIN_POSITIONS:
            box_ratios_test = available_positions[MAX_TRAIN_POSITIONS:MAX_TRAIN_POSITIONS+MAX_TEST_POSITIONS]
        else:
            box_ratios_test = []
    
    print(f"Available positions: {available_positions}")
    print(f"Training positions: {box_ratios_train} ({len(box_ratios_train)} positions × 2 tasks = {len(box_ratios_train)*2} tasks)")
    print(f"Testing positions: {box_ratios_test} ({len(box_ratios_test)} positions × 2 tasks = {len(box_ratios_test)*2} tasks)")
    print(f"Max examples per task: {MAX_EXAMPLES_PER_TASK}")
    print(f"Max training positions: {MAX_TRAIN_POSITIONS} (actual: {len(box_ratios_train)})")
    print(f"Max testing positions: {MAX_TEST_POSITIONS} (actual: {len(box_ratios_test)})")

    def make_examples(positions, max_examples_per_task=40):
        """
        Create tasks with examples from JSONL data.
        
        NOTE: This controls how many examples each TASK gets (for evaluation).
        The feature extractor may subsample to 32 examples for CNN encoding efficiency,
        but that's only for feature computation, not task evaluation.
        
        Args:
            positions: List of obstacle positions
            max_examples_per_task: Maximum number of examples per task
        """
        tasks = []
        for box_pos in positions:
            examples = get_sim_info(box_pos)
            if not examples:
                continue
            
            simplified_examples = []
            true_x_examples = [ex for ex in examples if ex["move_x"]]
            false_x_examples = [ex for ex in examples if not ex["move_x"]]
            true_y_examples = [ex for ex in examples if ex["move_y"]]
            false_y_examples = [ex for ex in examples if not ex["move_y"]]
            
            if true_x_examples and false_x_examples and true_y_examples and false_y_examples:
                # Try to get examples covering all 4 combinations: (move_x, move_y)
                # (True, True), (True, False), (False, True), (False, False)
                combinations = [
                    (True, True), (True, False), (False, True), (False, False)
                ]
                for move_x_val, move_y_val in combinations:
                    matching = [ex for ex in examples 
                               if ex["move_x"] == move_x_val and ex["move_y"] == move_y_val]
                    if matching and len(simplified_examples) < max_examples_per_task:
                        simplified_examples.append(matching[0])
                
                if len(simplified_examples) < max_examples_per_task:
                    remaining = [ex for ex in examples if ex not in simplified_examples]
                    for ex in remaining:
                        if len(simplified_examples) >= max_examples_per_task:
                            break
                        current_x_vals = [e["move_x"] for e in simplified_examples]
                        current_y_vals = [e["move_y"] for e in simplified_examples]
                        adds_diversity = (ex["move_x"] not in current_x_vals or 
                                         ex["move_y"] not in current_y_vals)
                        if adds_diversity or len(simplified_examples) < 4:
                            simplified_examples.append(ex)
            elif true_x_examples and false_x_examples:
                simplified_examples.append(true_x_examples[0])
                simplified_examples.append(false_x_examples[0])
                if len(simplified_examples) < max_examples_per_task:
                    remaining = [ex for ex in examples if ex not in simplified_examples]
                    simplified_examples.extend(remaining[:max_examples_per_task - len(simplified_examples)])
            elif true_y_examples and false_y_examples:
                simplified_examples.append(true_y_examples[0])
                simplified_examples.append(false_y_examples[0])
                if len(simplified_examples) < max_examples_per_task:
                    remaining = [ex for ex in examples if ex not in simplified_examples]
                    simplified_examples.extend(remaining[:max_examples_per_task - len(simplified_examples)])
            else:
                simplified_examples = examples[:max_examples_per_task]
            
            if simplified_examples:
                move_x_vals = [ex["move_x"] for ex in simplified_examples]
                move_y_vals = [ex["move_y"] for ex in simplified_examples]
                if len(set(move_x_vals)) == 1:
                    print(f"Warning NIT: All {len(simplified_examples)} examples for {box_pos} have move_x={move_x_vals[0]} (task will be easier)")
                if len(set(move_y_vals)) == 1:
                    print(f"Warning NIT: All {len(simplified_examples)} examples for {box_pos} have move_y={move_y_vals[0]} (task will be easier)")
            
            tasks.append({"name": f"box_pos_{box_pos:.3f}", "examples": simplified_examples})
        return tasks

    training_examples = make_examples(box_ratios_train, max_examples_per_task=MAX_EXAMPLES_PER_TASK)
    testing_examples = make_examples(box_ratios_test, max_examples_per_task=MAX_EXAMPLES_PER_TASK)

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