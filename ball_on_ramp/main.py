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
        Primitive("car", arrow(tlist(t0), t0), _car),
        Primitive("cdr", arrow(tlist(t0), tlist(t0)), _cdr),
        Primitive("cons", arrow(t0, tlist(t0), tlist(t0)), _cons),
        Primitive("empty?", arrow(tlist(t0), tbool), _is_empty),
    ]


def get_ball_on_ramp_task(item, move="x"):
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
                tlist(treal),
                tlist(treal),
                treal,
                treal,
                tbool
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
        task.meta = meta
        return task
    elif move == "y":
        task = Task(
            item['name'] + "_y",
            arrow(
                tlist(treal),
                tlist(treal),
                treal,
                treal,
                tbool
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

    args = commandlineArguments(
        enumerationTimeout=300, activation="tanh",
        iterations=10, recognitionTimeout=1200,
        a=3, maximumFrontier=10, topK=2, pseudoCounts=30.0,
        helmholtzRatio=0.2,
        structurePenalty=1.,
        CPUs=numberOfCPUs(),
        cuda=False,
        solver="ocaml",  
        compressor="ocaml", 
    )

    timestamp = datetime.datetime.now().isoformat()
    outdir = 'experimentOutputs/ball_on_ramp/'
    os.makedirs(outdir, exist_ok=True)
    outprefix = outdir + timestamp
    args.update({
        "outputPrefix": outprefix,
        "featureExtractor": None,
    })

    primitives = get_primitives()
    grammar = Grammar.uniform(primitives)

    available_positions = get_available_positions()
    if not available_positions:
        raise RuntimeError("No obstacle positions found in datasets/ramp_temporal.jsonl. Generate data first.")

    available_positions = sorted(available_positions)
    
    MAX_TRAIN_POSITIONS = 6
    MAX_TEST_POSITIONS = 2
    
    if len(available_positions) <= MAX_TRAIN_POSITIONS:
        box_ratios_train = available_positions
        box_ratios_test = []
    else:
        box_ratios_train = available_positions[:MAX_TRAIN_POSITIONS]
        if len(available_positions) > MAX_TRAIN_POSITIONS:
            box_ratios_test = available_positions[MAX_TRAIN_POSITIONS:MAX_TRAIN_POSITIONS+MAX_TEST_POSITIONS]
        else:
            box_ratios_test = []
    
    print(f"Available positions: {available_positions}")
    print(f"Training positions: {box_ratios_train} ({len(box_ratios_train)} positions)")
    print(f"Testing positions: {box_ratios_test} ({len(box_ratios_test)} positions)")
    print(f"Creating 2 tasks (move_x, move_y) with examples from ALL positions combined")
    print(f"Max training positions: {MAX_TRAIN_POSITIONS} (actual: {len(box_ratios_train)})")
    print(f"Max testing positions: {MAX_TEST_POSITIONS} (actual: {len(box_ratios_test)})")

    def combine_all_examples(positions):
        """
        Combine examples from ALL positions into single lists for move_x and move_y.
        This creates 2 tasks instead of (num_positions × 2) tasks.
        
        Args:
            positions: List of obstacle positions
            
        Returns:
            tuple: (all_move_x_examples, all_move_y_examples)
        """
        all_move_x_examples = []
        all_move_y_examples = []
        
        for box_pos in positions:
            examples = get_sim_info(box_pos)
            if not examples:
                continue
            
            all_move_x_examples.extend(examples)
            all_move_y_examples.extend(examples)
        
        return all_move_x_examples, all_move_y_examples

    train_move_x_examples, train_move_y_examples = combine_all_examples(box_ratios_train)
    test_move_x_examples, test_move_y_examples = combine_all_examples(box_ratios_test)
    
    print(f"Combined training examples: {len(train_move_x_examples)} for move_x, {len(train_move_y_examples)} for move_y")
    print(f"Combined testing examples: {len(test_move_x_examples)} for move_x, {len(test_move_y_examples)} for move_y")
    
    combined_meta = None
    if train_move_x_examples:
        first_ex = train_move_x_examples[0]
        combined_meta = {
            "ball_radius": first_ex.get("ball_radius", 25.0),
            "box_size": first_ex.get("box_size", 50.0),
            "ramp_theta": first_ex.get("ramp_theta", 0.0),
        }
    
    training = []
    if train_move_x_examples:
        training.append(Task(
            "move_x_all_positions",
            arrow(tlist(treal), tlist(treal), treal, treal, tbool),
            examples=[
                ((ex["ball_x"], ex["ball_y"], ex["obstacle_x"], ex["obstacle_y"]), ex["move_x"])
                for ex in train_move_x_examples
            ]
        ))
        training[-1].meta = combined_meta
    
    if train_move_y_examples:
        training.append(Task(
            "move_y_all_positions",
            arrow(tlist(treal), tlist(treal), treal, treal, tbool),
            examples=[
                ((ex["ball_x"], ex["ball_y"], ex["obstacle_x"], ex["obstacle_y"]), ex["move_y"])
                for ex in train_move_y_examples
            ]
        ))
        training[-1].meta = combined_meta
    
    testing = []
    if test_move_x_examples:
        testing.append(Task(
            "move_x_all_positions_test",
            arrow(tlist(treal), tlist(treal), treal, treal, tbool),
            examples=[
                ((ex["ball_x"], ex["ball_y"], ex["obstacle_x"], ex["obstacle_y"]), ex["move_x"])
                for ex in test_move_x_examples
            ]
        ))
        testing[-1].meta = combined_meta
    
    if test_move_y_examples:
        testing.append(Task(
            "move_y_all_positions_test",
            arrow(tlist(treal), tlist(treal), treal, treal, tbool),
            examples=[
                ((ex["ball_x"], ex["ball_y"], ex["obstacle_x"], ex["obstacle_y"]), ex["move_y"])
                for ex in test_move_y_examples
            ]
        ))
        testing[-1].meta = combined_meta

    generator = ecIterator(grammar, training, testingTasks=testing, **args)
    for i, _ in enumerate(generator):
        print(f"ecIterator count {i}")