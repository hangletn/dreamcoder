import pymunk
import pymunk.pygame_util
import pygame
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import imageio
import math
import yaml
import os
from collections import defaultdict
import json
import os
from collections import defaultdict


_JSONL_CACHE = None


def _default_jsonl_path():
    """Return absolute path to ball_on_ramp/datasets/ramp_temporal.jsonl."""
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, "datasets", "ramp_temporal.jsonl")


def _load_jsonl(path=None):
    """
    Load ramp_temporal.jsonl and group records by meta.t.
    Each record is exactly what get_ball_jsonl.py writes.
    """
    if path is None:
        path = _default_jsonl_path()

    by_t = defaultdict(list)
    with open(path, "r") as f:
        for line in f:
            rec = json.loads(line)
            t = float(rec["meta"]["t"])
            by_t[t].append(rec)

    return by_t


def get_sim_info(box_pos, jsonl_path=None, tol=1e-6, filter_falling_ball=True):
    """
    Get examples for a specific obstacle position from JSONL.
    Used by main.py to load examples from each position, which are then
    combined into 2 tasks (move_x, move_y) with examples from all positions.
    """
    by_t = _load_jsonl(jsonl_path)

    matched_records = None
    for t_val, recs in by_t.items():
        if abs(t_val - box_pos) < tol:
            matched_records = recs
            break

    if matched_records is None:
        return []

    examples = []
    for rec in matched_records:
        xs = rec["pos_x"]
        ys = rec["pos_y"] 
        if isinstance(rec["obstacle_x"], list):
            obx = float(rec["obstacle_x"][0])
            oby = float(rec["obstacle_y"][0])
        else:
            obx = float(rec["obstacle_x"])
            oby = float(rec["obstacle_y"])

        move_x = bool(rec["out"]["move_x"])
        move_y = bool(rec["out"]["move_y"])
        meta = rec.get("meta", {})
        ball_radius = float(meta.get("ball_radius", 25.0))
        box_size = float(meta.get("box_size", 50.0))
        ramp_theta = float(meta.get("ramp_theta", -0.2914567944778671))
        if filter_falling_ball:
            if not move_x and move_y: # not account for ball falling down for now
                pass
            else:
                examples.append({
                    "ball_x": xs,
                    "ball_y": ys,
                    "obstacle_x": obx,
                    "obstacle_y": oby,
                    "move_x": move_x,
                    "move_y": move_y,
                    "ball_radius": ball_radius,
                    "box_size": box_size,
                    "ramp_theta": ramp_theta,
                })
        else:
            examples.append({
                "ball_x": xs,
                "ball_y": ys,
                "obstacle_x": obx,
                "obstacle_y": oby,
                "move_x": move_x,
                "move_y": move_y,
                "ball_radius": ball_radius,
                "box_size": box_size,
                "ramp_theta": ramp_theta,
            })

    return examples


def get_available_positions(jsonl_path=None):
    """
    Return the sorted list of obstacle positions (meta.t) present in the JSONL.
    """
    by_t = _load_jsonl(jsonl_path)
    return sorted(by_t.keys())