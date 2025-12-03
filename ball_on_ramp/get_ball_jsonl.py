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
    circle.elasticity = ball["elasticity"]
    circle.friction = ball["friction"]
    space.add(body, circle, r_seg, g_seg)
    box_cfg = cfg["box"]
    box_size = tuple(box_cfg["size"])
    box_elasticity = box_cfg["elasticity"]
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

def simulate_positions(space: pymunk.Space, body: pymunk.Body, dt: float, steps: int, 
                        ramp_theta, obstacle_y: float = None,
                       obstacle_x: float = None, ball_radius: float = None, 
                       capture_frames: bool = False, size=(640, 480), xlim=(0, 640), ylim=(-10, 480)):
    """
    Simulate ball movement, clipping position when it hits obstacle.
    Like bin/ball_on_ramp.py, when ball hits obstacle, it stops and stays at that position.
    """
    xs, ys = [], []
    frames = [] if capture_frames else None
    collision_threshold = None
    collision_threshold_ramp = None
    if obstacle_x is not None and ball_radius is not None:
        collision_threshold = obstacle_x - ball_radius
        collision_threshold_ramp = obstacle_x - np.cos(ramp_theta)*ball_radius
    
    hit_obstacle = False
    last_x, last_y = float(body.position.x), float(body.position.y)
    
    for _ in range(steps):
        if not hit_obstacle:
            space.step(dt)
            current_x = float(body.position.x)
            current_y = float(body.position.y)
            if current_y - ball_radius > 0.01:
                if collision_threshold is not None and current_x >= collision_threshold_ramp:
                    hit_obstacle = True
                    current_x = collision_threshold_ramp
                    current_y = obstacle_y - np.sin(ramp_theta)*ball_radius
            else:
                if collision_threshold is not None and current_x >= collision_threshold:
                    hit_obstacle = True
                    current_x = collision_threshold
                    current_y = obstacle_y - ball_radius
            
            xs.append(current_x)
            ys.append(current_y)
            last_x, last_y = current_x, current_y
        else:
            xs.append(last_x)
            ys.append(last_y)
        
        if capture_frames:
            frames.append(render(space, size=size, xlim=xlim, ylim=ylim))
    
    if capture_frames:
        return xs, ys, frames
    return xs, ys

def make_windows(xs, ys, h, stride=2):
    """
    Create sliding windows matching bin/ball_on_ramp.py:
    - Windows of length h
    - Stride of 2 (every 2nd window, like dummy data)
    - Returns (start_idx, xw, yw, x_next, y_next) where next is the step after window
    - Matches: range(0, seq_len-trace_len-1, 2) and end_idx = start_idx + trace_len
    """
    out = []
    n = len(xs)
    for i in range(0, n - h - 1, stride):
        xw = xs[i:i + h]
        yw = ys[i:i + h]
        x_next = xs[i + h] if i + h < n else xs[-1]
        y_next = ys[i + h] if i + h < n else ys[-1]
        out.append((i, xw, yw, x_next, y_next))
    return out

def window_internal_bools(xw, yw, eps=1e-3, use_exact_equality=False):
    """
    Check if ball moved WITHIN the window.
    
    Args:
        xw: list of x positions in window
        yw: list of y positions in window
        eps: tolerance for floating point comparison
        use_exact_equality: if True, use exact equality (==) instead of eps threshold
    """
    if use_exact_equality:
        move_x = any(xw[i] != xw[i+1] for i in range(len(xw)-1))
        move_y = any(yw[i] != yw[i+1] for i in range(len(yw)-1))
    else:
        move_x = any(abs(xw[i+1] - xw[i]) > eps for i in range(len(xw)-1))
        move_y = any(abs(yw[i+1] - yw[i]) > eps for i in range(len(yw)-1))
    return {"move_x": bool(move_x), "still_x": bool(not move_x), "move_y": bool(move_y), "still_y": bool(not move_y)}


def next_step_bools(xw, yw, x_next, y_next, eps=1e-3):
    """
    Label matches bin/ball_on_ramp.py dummy data:
    - move_x/y are about the NEXT step (after the window ends)
    - Checks if position changes between last point in window and next point
    
    This is what the dummy data does and what gets hits.
    """
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
    ap.add_argument("--num_examples", type=int, default=1, help="rollouts per t")
    ap.add_argument("--window_h", type=int, default=10, help="window length h (we store h+1 points)")
    ap.add_argument("--eps", type=float, default=1e-3, help="tolerance for 'still' vs 'move' (position change threshold)")
    ap.add_argument("--round_decimals", type=int, default=2, help="round positions and obstacle coordinates to N decimal places")
    ap.add_argument("--use_exact_equality", action="store_true", help="use exact equality (==) for label generation instead of eps threshold")
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
                obstacle_x = (params["obstacle_bottom_left_x"] + params["obstacle_top_left_x"]) / 2.0  
                obstacle_y = (params["obstacle_bottom_left_y"] + params["obstacle_top_left_y"]) / 2.0 
                
                if capture_video:
                    xs, ys, frames = simulate_positions(space, body, args.dt, steps_total,
                                                        params["ramp_theta"], obstacle_y=obstacle_y,
                                                       obstacle_x=obstacle_x, 
                                                       ball_radius=params["ball_radius"],
                                                       capture_frames=True)
                    videos_recorded += 1
                else:
                    xs, ys = simulate_positions(space, body, args.dt, steps_total,
                                               obstacle_x=obstacle_x,
                                               ball_radius=params["ball_radius"])
                    frames = None
                
                if args.round_decimals is not None:
                    obstacle_x = round(obstacle_x, args.round_decimals)
                    obstacle_y = round(obstacle_y, args.round_decimals)
                
                for start_i, xw, yw, x_next, y_next in make_windows(xs, ys, args.window_h):
                    if args.round_decimals is not None:
                        xw = [round(x, args.round_decimals) for x in xw]
                        yw = [round(y, args.round_decimals) for y in yw]
                        x_next = round(x_next, args.round_decimals)
                        y_next = round(y_next, args.round_decimals)
                    add_radius = True
                    if add_radius:
                        not_on_ramp = np.array([True if (y - params["ball_radius"]/2 < 0.01) else False for y in yw])
                        theta_array = np.zeros(len(xw)) + params["ramp_theta"]
                        theta_array[not_on_ramp] = 0.0
                        xw += np.cos(theta_array)*params["ball_radius"]
                        yw += np.sin(theta_array)*params["ball_radius"]
                        xw = [round(x, args.round_decimals) for x in xw]
                        yw = [round(y, args.round_decimals) for y in yw]
                        
                        not_on_ramp_next = y_next - params["ball_radius"]/2 < 0.01
                        theta_next = 0.0 if not_on_ramp_next else params["ramp_theta"]
                        x_next += np.cos(theta_next) * params["ball_radius"]
                        y_next += np.sin(theta_next) * params["ball_radius"]
                        if args.round_decimals is not None:
                            x_next = round(x_next, args.round_decimals)
                            y_next = round(y_next, args.round_decimals)
                    
                    labels = next_step_bools(xw, yw, x_next, y_next, eps=args.eps)
                    
                    all_labels = {
                        "move_x": labels["move_x"],
                        "move_y": labels["move_y"],
                    }
                    
                    record = {
                        "pos_x": xw,
                        "pos_y": yw,
                        "obstacle_x": obstacle_x,
                        "obstacle_y": obstacle_y,
                        "out": all_labels,
                        "meta": {
                            "t": t,
                            "dt": args.dt,
                            "start_index": start_i,
                            "h": args.window_h,
                            "sim_steps": steps_total,
                            "example_id": e,
                            "ball_radius": params["ball_radius"],
                            "box_size": params["box_size"],
                            "ramp_theta": params["ramp_theta"]
                        }
                    }
                    wf.write(json.dumps(record) + "\n")
                    n_written += 1
                if frames is not None:
                    video_name = f"ball_evolution_t{t:.2f}_example{e:03d}.mp4"
                    video_path = os.path.join(args.video_folder, video_name)
                    save_video(frames, video_path, fps=video_fps)
    print(f"Wrote {n_written} examples to {args.out_jsonl}")


if __name__ == "__main__":
    main()
