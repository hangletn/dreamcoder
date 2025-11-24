"""
Generate a JSONL dataset of temporal windows from ball on ramp.

Each line in the output file is ONE example:
{
  "name": "ramp_window_t0.30",
  "input": {
    "body_x": [ ... h+1 floats ... ],
    "body_y": [ ... h+1 floats ... ],
    "obstacle_x": float,
    "obstacle_y": float,
    "ball_radius": float,
    "box_size": float, # square side length
    "ramp_theta": float # radians
  },
  "output": {
    "move_x": bool,
    "still_x": bool,
    "move_y": bool,
    "still_y": bool
  },
  "meta": {
    "t": float, # box placement param on ramp, 0..1
    "dt": float,
    "start_index": int, # window start index i
    "h": int, # window length
    "sim_steps": int # total steps in rollout
  }
}

Usage example:
  python get_jsonl.py \
    --config_filepath config.yaml \
    --out_jsonl datasets/ramp_temporal.jsonl \
    --positions 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9  (or just have one is also fine)\
    --num_examples 50 \
    --dt 0.02 \
    --simulation_time 2.0 \
    --window_h 10 \
    --eps 1e-3 \
    --jitter_px 2.0 --save_video --video_folder videos
"""

import argparse
import json
import math
import os
import random

import numpy as np
import pymunk
import yaml
from pymunk import Vec2d
import imageio
import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt
from pymunk.matplotlib_util import DrawOptions


def set_seed(s: int):
    random.seed(s)
    np.random.seed(s)

def ramp_point_and_angle(seg: pymunk.Segment, u: float):
    """
    Return (point, angle) on ramp segment for u in [0,1].
    Angle is the tangent angle in radians.
    """
    a, b = Vec2d(*seg.a), Vec2d(*seg.b)
    p = a + u * (b - a)
    angle = math.atan2((b - a).y, (b - a).x)
    return p, angle

def add_box_on_ramp(space: pymunk.Space, ramp: pymunk.Segment, ground: pymunk.Segment, t: float, size=(50.0, 50.0), friction=0.9, elasticity=0.05):
    """
    Place a static box centered on the ramp centerline at parameter t∈[0,1].
    t parameterizes from ramp start (a) to ground end (b).
    """
    ramp_a = ramp.a
    ramp_b = ramp.b
    ground_b = ground.b
    
    x = ramp_a.x + t * (ground_b.x - ramp_a.x)
    
    body = pymunk.Body(body_type=pymunk.Body.STATIC)
    if x <= ramp_b.x:
        ramp_length_x = ramp_b.x - ramp_a.x
        if ramp_length_x > 0:
            t_ramp = (x - ramp_a.x) / ramp_length_x
        else:
            t_ramp = 0.0
        t_ramp = max(0.0, min(1.0, t_ramp))
        p, ang = ramp_point_and_angle(ramp, t_ramp)
        

        normal_angle = ang - math.pi / 2.0  # Use -π/2 to get upward normal (negative y)
        offset_distance = size[1] / 2.0  # half box height
        offset_x = offset_distance * math.cos(normal_angle)
        offset_y = offset_distance * math.sin(normal_angle)
        
        body.position = (float(p.x - offset_x), float(p.y - offset_y))
        body.angle = ang
    else:
        angle = 0.0
        y = 0.0 + size[1] / 2.0  # box center y = half box height
        body.position = (float(x), float(y))
        body.angle = angle

    shape = pymunk.Poly.create_box(body, size)
    shape.friction = friction
    shape.elasticity = elasticity
    space.add(body, shape)
    return body, shape

def create_env(config_filepath: str, box_t: float, jitter_px: float = 0.0, seed=None):
    """
    Returns: (space, ball_body, params_dict)
    """
    if seed is not None:
        set_seed(seed)

    with open(config_filepath, "r") as f:
        cfg = yaml.safe_load(f)

    space = pymunk.Space()
    space.gravity = (0, -981)

    # Ramp segment
    r_seg = pymunk.Segment(
        space.static_body,
        tuple(cfg["ramp_segment"]["a"]),
        tuple(cfg["ramp_segment"]["b"]),
        cfg["ramp_segment"]["radius"],
    )
    r_seg.elasticity = cfg["ramp_segment"]["elasticity"]
    r_seg.friction = cfg["ramp_segment"]["friction"]

    # Ground segment
    g_seg = pymunk.Segment(space.static_body,
        tuple(cfg["horizontal_segment"]["a"]),
        tuple(cfg["horizontal_segment"]["b"]),
        cfg["horizontal_segment"]["radius"],
    )
    g_seg.elasticity = cfg["horizontal_segment"]["elasticity"]
    g_seg.friction = cfg["horizontal_segment"]["friction"]

    # Ball body/shape
    ball = cfg["ball"]
    body = pymunk.Body(mass=ball["mass"], moment=ball["moment"])
    bx, by = ball["position"]
    if jitter_px:
        bx += np.random.uniform(-jitter_px, jitter_px)
        by += np.random.uniform(-jitter_px, jitter_px)
    body.position = (bx, by)
    circle = pymunk.Circle(body, radius=ball["radius"])
    circle.elasticity = ball["elasticity"]
    circle.friction = ball["friction"]

    space.add(body, circle, r_seg, g_seg)

    box_cfg = cfg["box"]
    box_size = tuple(box_cfg["size"])
    box_body, _ = add_box_on_ramp(space, r_seg, g_seg, box_t, size=box_size, friction=box_cfg.get("friction", 0.9), elasticity=box_cfg.get("elasticity", 0.05),
    )

    ramp_theta = ramp_point_and_angle(r_seg, 0.5)[1]
    params = {
        "ball_radius": float(ball["radius"]),
        "box_size": float(box_size[0]), # assume square; use width
        "ramp_theta": float(ramp_theta),
        "obstacle_x": float(box_body.position.x),
        "obstacle_y": float(box_body.position.y),
    }
    return space, body, params

def render(space: pymunk.Space, size=(640, 480), xlim=(0, 640), ylim=(-10, 480)) -> np.ndarray:
    """
    Render the current Pymunk space to an RGB numpy array.
    Works on a headless machine because it uses matplotlib's Agg backend.
    """
    # Create off-screen figure
    dpi = 100
    fig_w = size[0] / dpi
    fig_h = size[1] / dpi
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.axis("off")
    plt.subplots_adjust(
        left=0, right=1, bottom=0, top=1,
        wspace=0, hspace=0
    )
    fig.patch.set_alpha(0.0) # Remove fig background (optional)
    ax.margins(0)

    options = DrawOptions(ax)
    space.debug_draw(options)

    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = buf.reshape((h, w, 3))

    plt.close(fig)
    return img

def save_video(frames: list, path: str, fps=60):
    """
    frames: list of np.ndarray objects
    path: output mp4 filename
    """
    imageio.mimsave(path, frames, fps=fps, codec="libx264")
    print(f"Saved video at {path}")

def simulate_positions(space: pymunk.Space, body: pymunk.Body, dt: float, steps: int, capture_frames: bool = False, size=(640, 480), xlim=(0, 640), ylim=(-10, 480)):
    xs, ys = [], []
    frames = [] if capture_frames else None
    for _ in range(steps):
        space.step(dt)
        xs.append(float(body.position.x))
        ys.append(float(body.position.y))
        if capture_frames:
            frames.append(render(space, size=size, xlim=xlim, ylim=ylim))
    if capture_frames:
        return xs, ys, frames
    return xs, ys

def make_windows(xs, ys, h):
    """
    Produce windows of length h+1 for features (t = i..i+h),
    ensuring we have t = i+h+1 for labels.
    Returns: list of (start_i, x_window_{h+1}, y_window_{h+1}, x_next, y_next)
    """
    out = []
    n = len(xs)
    # need i+h+1 < n  -> i <= n - h - 2
    for i in range(0, n - h - 1):
        xw = xs[i:i + h + 1]
        yw = ys[i:i + h + 1]
        x_next = xs[i + h + 1]
        y_next = ys[i + h + 1]
        out.append((i, xw, yw, x_next, y_next))
    return out

def next_step_bools(xw, yw, x_next, y_next, eps=1e-3):
    """
    Compute booleans for motion between the last point in the window (t=i+h)
    and the next point (t=i+h+1).
    """
    dx = x_next - xw[-1]
    dy = y_next - yw[-1]
    move_x  = dx > eps
    still_x = abs(dx) <= eps
    move_y  = dy > eps
    still_y = abs(dy) <= eps
    return {
        "move_x": bool(move_x),
        "still_x": bool(still_x),
        "move_y": bool(move_y),
        "still_y": bool(still_y),
    }


def main():
    ap = argparse.ArgumentParser(description="Generate JSONL dataset for ball-on-ramp temporal windows.")
    ap.add_argument("--config_filepath", type=str, default="config.yaml")
    ap.add_argument("--dt", type=float, default=0.02, help="simulation time step")
    ap.add_argument("--simulation_time", type=float, default=2.0, help="total simulated seconds")
    ap.add_argument("--positions", type=str, default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9",
                    help="comma-separated t values (0..1) for obstacle placement along ramp")
    ap.add_argument("--num_examples", type=int, default=50, help="rollouts per t")
    ap.add_argument("--window_h", type=int, default=10, help="window length h (we store h+1 points)")
    ap.add_argument("--eps", type=float, default=1e-3, help="tolerance for 'still' vs 'move'")
    ap.add_argument("--jitter_px", type=float, default=2.0, help="random start jitter for the ball")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--out_jsonl", type=str, default="datasets/ramp_temporal.jsonl")
    ap.add_argument("--save_video", action="store_true", help="save video of simulation")
    ap.add_argument("--video_folder", type=str, default="videos", help="folder to save videos")
    ap.add_argument("--video_fps", type=float, default=None, help="video fps (defaults to 1/dt)")

    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out_jsonl), exist_ok=True)
    set_seed(args.seed)

    steps_total = int(round(args.simulation_time / args.dt))
    t_vals = [float(s) for s in args.positions.split(",")]

    video_frames = None
    if args.save_video:
        os.makedirs(args.video_folder, exist_ok=True)
        video_fps = args.video_fps if args.video_fps is not None else int(1.0 / args.dt)
       
        first_t = t_vals[0]
        seed_first = (hash((round(first_t, 3), 0, args.seed)) & 0xFFFFFFFF)
        space_video, body_video, _ = create_env(
            args.config_filepath,
            box_t=first_t,
            jitter_px=args.jitter_px,
            seed=seed_first)
        _, _, video_frames = simulate_positions(space_video, body_video, args.dt, steps_total, capture_frames=True)

    n_written = 0
    with open(args.out_jsonl, "w") as wf:
        for t in t_vals:
            for e in range(args.num_examples):
                seed_e = (hash((round(t, 3), e, args.seed)) & 0xFFFFFFFF)
                space, body, params = create_env(
                    args.config_filepath,
                    box_t=t,
                    jitter_px=args.jitter_px,
                    seed=seed_e
                )
                xs, ys = simulate_positions(space, body, args.dt, steps_total)

                for start_i, xw, yw, x_next, y_next in make_windows(xs, ys, args.window_h):
                    labels = next_step_bools(xw, yw, x_next, y_next, eps=args.eps)
                    record = {
                        "name": f"ramp_window_t{t:.2f}",
                        "input": {
                            "body_x": xw,
                            "body_y": yw,
                            "obstacle_x": params["obstacle_x"],
                            "obstacle_y": params["obstacle_y"],
                            "ball_radius": params["ball_radius"],
                            "box_size": params["box_size"],
                            "ramp_theta": params["ramp_theta"],
                        },
                        "output": labels,
                        "meta": {
                            "t": t,
                            "dt": args.dt,
                            "start_index": start_i,
                            "h": args.window_h,
                            "sim_steps": steps_total,
                        },
                    }
                    wf.write(json.dumps(record) + "\n")
                    n_written += 1

    if args.save_video and video_frames is not None:
        video_path = os.path.join(args.video_folder, "ball_evolution.mp4")
        save_video(video_frames, video_path, fps=video_fps)

    print(f"Wrote {n_written} examples to {args.out_jsonl}")


if __name__ == "__main__":
    main()
