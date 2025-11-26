
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
    a, b = Vec2d(*seg.a), Vec2d(*seg.b)
    p = a + u * (b - a)
    angle = math.atan2((b - a).y, (b - a).x)
    return p, angle

def get_box_leftmost_corners(box_body: pymunk.Body, box_size: tuple):
    box_center_x = float(box_body.position.x)
    box_center_y = float(box_body.position.y)
    box_angle = float(box_body.angle)
    box_width = float(box_size[0])
    box_height = float(box_size[1])
    cos_a = math.cos(box_angle)
    sin_a = math.sin(box_angle)
    bottom_left_local = (-box_width/2, -box_height/2)
    top_left_local = (-box_width/2, box_height/2)
    def rotate_point(dx, dy):
        rotated_x = box_center_x + dx * cos_a - dy * sin_a
        rotated_y = box_center_y + dx * sin_a + dy * cos_a
        return (rotated_x, rotated_y)
    bottom_left = rotate_point(*bottom_left_local)
    top_left = rotate_point(*top_left_local)
    return bottom_left, top_left

def add_box_on_ramp(space: pymunk.Space, ramp: pymunk.Segment, ground: pymunk.Segment, t: float, size=(50.0, 50.0), friction=0.9, elasticity=0.00):
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
        normal_angle = ang - math.pi / 2.0
        offset_distance = size[1] / 2.0
        offset_x = offset_distance * math.cos(normal_angle)
        offset_y = offset_distance * math.sin(normal_angle)
        body.position = (float(p.x - offset_x), float(p.y - offset_y))
        body.angle = ang
    else:
        angle = 0.0
        y = 0.0 + size[1] / 2.0
        body.position = (float(x), float(y))
        body.angle = angle
    shape = pymunk.Poly.create_box(body, size)
    shape.friction = friction
    shape.elasticity = elasticity
    space.add(body, shape)
    return body, shape

def create_env(config_filepath: str, box_t: float, jitter_px: float = 0.0, seed=None):
    if seed is not None:
        set_seed(seed)
    with open(config_filepath, "r") as f:
        cfg = yaml.safe_load(f)
    space = pymunk.Space()
    space.gravity = (0, -981)
    r_seg = pymunk.Segment(space.static_body, tuple(cfg["ramp_segment"]["a"]), tuple(cfg["ramp_segment"]["b"]), cfg["ramp_segment"]["radius"])
    r_seg.elasticity = cfg["ramp_segment"]["elasticity"]
    r_seg.friction = cfg["ramp_segment"]["friction"]
    g_seg = pymunk.Segment(space.static_body, tuple(cfg["horizontal_segment"]["a"]), tuple(cfg["horizontal_segment"]["b"]), cfg["horizontal_segment"]["radius"])
    g_seg.elasticity = cfg["horizontal_segment"]["elasticity"]
    g_seg.friction = cfg["horizontal_segment"]["friction"]
    ball = cfg["ball"]
    body = pymunk.Body(mass=ball["mass"], moment=ball["moment"])
    bx, by = ball["position"]
    if jitter_px:
        bx += np.random.uniform(-jitter_px, jitter_px)
        by += np.random.uniform(-jitter_px, jitter_px)
    body.position = (bx, by)
    circle = pymunk.Circle(body, radius=ball["radius"])
    circle.elasticity = 0.0#ball["elasticity"]
    circle.friction = ball["friction"]
    space.add(body, circle, r_seg, g_seg)
    box_cfg = cfg["box"]
    box_size = tuple(box_cfg["size"])
    box_elasticity = 0.0
    box_body, _ = add_box_on_ramp(space, r_seg, g_seg, box_t, size=box_size, friction=box_cfg.get("friction", 0.9), elasticity=box_elasticity)
    ramp_theta = ramp_point_and_angle(r_seg, 0.5)[1]
    bottom_left, top_left = get_box_leftmost_corners(box_body, box_size)
    params = {"ball_radius": float(ball["radius"]), "box_size": float(box_size[0]), "ramp_theta": float(ramp_theta), "obstacle_bottom_left_x": bottom_left[0], "obstacle_bottom_left_y": bottom_left[1], "obstacle_top_left_x": top_left[0], "obstacle_top_left_y": top_left[1]}
    return space, body, params

def render(space: pymunk.Space, size=(640, 480), xlim=(0, 640), ylim=(-10, 480)) -> np.ndarray:
    dpi = 100
    fig_w = size[0] / dpi
    fig_h = size[1] / dpi
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.axis("off")
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)
    fig.patch.set_alpha(0.0)
    ax.margins(0)
    options = DrawOptions(ax)
    space.debug_draw(options)
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
    img_argb = buf.reshape((h, w, 4))
    img = img_argb[:, :, 1:4]
    plt.close(fig)
    return img

def save_video(frames: list, path: str, fps=60):
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

def make_windows(xs, ys, h, stride=2):
    """
    Create sliding windows matching bin/ball_on_ramp.py dummy data:
    - Windows of length h (we store h points, not h+1)
    - Stride of 2 (every 2nd window, like dummy data)
    - Label is about the NEXT step after window
    """
    out = []
    n = len(xs)
    # Match dummy: range(0, seq_len-trace_len-1, 2)
    # We need h points + 1 for next step label
    for i in range(0, n - h, stride):
        xw = xs[i:i + h]  # h points (not h+1)
        yw = ys[i:i + h]
        x_next = xs[i + h]  # next step for label
        y_next = ys[i + h]
        out.append((i, xw, yw, x_next, y_next))
    return out

def next_step_bools(xw, yw, x_next, y_next, eps=1e-3):
    """
    Label matches bin/ball_on_ramp.py dummy data:
    - move_x/y are about the NEXT step (after the window ends)
    - Checks if position changes between last point in window and next point
    
    This is what the dummy data does and what gets hits.
    """
    # Check if next step differs from last point in window (like dummy data)
    dx = x_next - xw[-1]
    dy = y_next - yw[-1]
    
    move_x = abs(dx) > eps
    move_y = abs(dy) > eps
    
    still_x = not move_x
    still_y = not move_y
    return {"move_x": bool(move_x), "still_x": bool(still_x), "move_y": bool(move_y), "still_y": bool(still_y)}


def main():
    ap = argparse.ArgumentParser(description="Generate JSONL dataset for ball-on-ramp temporal windows.")
    ap.add_argument("--config_filepath", type=str, default="config.yaml")
    ap.add_argument("--dt", type=float, default=0.02, help="simulation time step")
    ap.add_argument("--simulation_time", type=float, default=2.0, help="total simulated seconds")
    ap.add_argument("--positions", type=str, default="0.5", help="comma-separated t values (0..1) for obstacle placement along ramp")
    ap.add_argument("--num_examples", type=int, default=50, help="rollouts per t")
    ap.add_argument("--window_h", type=int, default=10, help="window length h (we store h+1 points)")
    ap.add_argument("--eps", type=float, default=1e-3, help="tolerance for 'still' vs 'move' (position change threshold)")
    ap.add_argument("--jitter_px", type=float, default=2.0, help="random start jitter for the ball")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--out_jsonl", type=str, default="datasets/ramp_temporal.jsonl")
    ap.add_argument("--save_video", action="store_true", help="save video of simulation")
    ap.add_argument("--video_folder", type=str, default="videos", help="folder to save videos")
    ap.add_argument("--video_fps", type=float, default=None, help="video fps (defaults to 1/dt)")
    ap.add_argument("--videos_per_position", type=int, default=1, help="max videos to save per obstacle position")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out_jsonl), exist_ok=True)
    set_seed(args.seed)
    steps_total = int(round(args.simulation_time / args.dt))
    t_vals = [float(s) for s in args.positions.split(",")]
    video_fps = None
    if args.save_video:
        os.makedirs(args.video_folder, exist_ok=True)
        video_fps = args.video_fps if args.video_fps is not None else int(1.0 / args.dt)
    n_written = 0
    with open(args.out_jsonl, "w") as wf:
        for t in t_vals:
            videos_recorded = 0
            for e in range(args.num_examples):
                seed_e = (hash((round(t, 3), e, args.seed)) & 0xFFFFFFFF)
                space, body, params = create_env(args.config_filepath, box_t=t, jitter_px=args.jitter_px, seed=seed_e)
                capture_video = args.save_video and videos_recorded < args.videos_per_position
                if capture_video:
                    xs, ys, frames = simulate_positions(space, body, args.dt, steps_total, capture_frames=True)
                    videos_recorded += 1
                else:
                    xs, ys = simulate_positions(space, body, args.dt, steps_total)
                    frames = None
                for start_i, xw, yw, x_next, y_next in make_windows(xs, ys, args.window_h):
                    labels = next_step_bools(xw, yw, x_next, y_next, eps=args.eps)
                    record = {"pos_x": xw, "pos_y": yw, "obstacle_x": [params["obstacle_bottom_left_x"], params["obstacle_top_left_x"]], "obstacle_y": [params["obstacle_bottom_left_y"], params["obstacle_top_left_y"]], "out": {"move_x": labels["move_x"], "move_y": labels["move_y"]}, "meta": {"t": t, "dt": args.dt, "start_index": start_i, "h": args.window_h, "sim_steps": steps_total, "example_id": e}}
                    wf.write(json.dumps(record) + "\n")
                    n_written += 1
                if frames is not None:
                    video_name = f"ball_evolution_t{t:.2f}_example{e:03d}.mp4"
                    video_path = os.path.join(args.video_folder, video_name)
                    save_video(frames, video_path, fps=video_fps)
    print(f"Wrote {n_written} examples to {args.out_jsonl}")


if __name__ == "__main__":
    main()
