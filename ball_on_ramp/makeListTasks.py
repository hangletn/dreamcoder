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
    # Path is ball_on_ramp/datasets/ramp_temporal.jsonl (same dir as this file)
    return os.path.join(here, "datasets", "ramp_temporal.jsonl")


def _load_jsonl(path=None):
    """
    Load ramp_temporal.jsonl and group records by meta.t.
    Each record is exactly what get_ball_jsonl.py writes.
    
    NOTE: Cache disabled for debugging to ensure fresh data on each run.
    """
    # global _JSONL_CACHE
    # if _JSONL_CACHE is not None:
    #     return _JSONL_CACHE

    if path is None:
        path = _default_jsonl_path()

    by_t = defaultdict(list)
    with open(path, "r") as f:
        for line in f:
            rec = json.loads(line)
            t = float(rec["meta"]["t"])
            by_t[t].append(rec)

    # _JSONL_CACHE = by_t
    return by_t


def get_sim_info(box_pos, jsonl_path=None, tol=1e-6):

    by_t = _load_jsonl(jsonl_path)

    matched_records = None
    for t_val, recs in by_t.items():
        if abs(t_val - box_pos) < tol:
            matched_records = recs
            break

    if matched_records is None:
        # No data for this specific t; return empty list
        return []

    examples = []
    for rec in matched_records:
        xs = rec["pos_x"]
        ys = rec["pos_y"] 
        if isinstance(rec["obstacle_x"], list):
            obx = float(rec["obstacle_x"][0])  # Old format - take first value
            oby = float(rec["obstacle_y"][0])  # Old format
        else:
            obx = float(rec["obstacle_x"])  # New format - single value
            oby = float(rec["obstacle_y"])  # New format

        move_x = bool(rec["out"]["move_x"])
        move_y = bool(rec["out"]["move_y"])
        
        # Extract meta information
        meta = rec.get("meta", {})
        ball_radius = float(meta.get("ball_radius", 25.0))
        box_size = float(meta.get("box_size", 50.0))
        ramp_theta = float(meta.get("ramp_theta", -0.2914567944778671))

        examples.append({
            "ball_x": xs,
            "ball_y": ys,  # Raw pymunk coordinates
            "obstacle_x": obx,
            "obstacle_y": oby,  # Raw pymunk coordinates
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