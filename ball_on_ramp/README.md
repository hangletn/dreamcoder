## Ball-on-Ramp Dataset Generator

This is the script that simulates a ball rolling down a ramp, colliding with an obstacle, and exports the motion as temporal windows in JSONL format. 

### JSONL Layout at a Glance
Each line in the output file is a single window (length `h + 1` positions) plus the label for what happens at the next time step. This is the shape:

```
{
  "name": "ramp_window_t0.50",
  "input": {
    "body_x": [... h+1 floats ...], // ball center x positions
    "body_y": [... h+1 floats ...], // ball center y positions
    "obstacle_bottom_left_x": 296.05,
    "obstacle_bottom_left_y": 31.18,
    "obstacle_top_left_x": 310.42,
    "obstacle_top_left_y": 79.08,
    "ball_radius": 20.0,
    "box_size": 50.0,
    "ramp_theta": -0.291
  },
  "output": {
    "move_x": true,
    "still_x": false,
    "move_y": false,
    "still_y": true
  },
  "meta": {
    "t": 0.5, // obstacle position along the ramp (0=start, 1=end)
    "dt": 0.02,
    "start_index": 75,
    "h": 10,
    "sim_steps": 100,
    "example_id": 0 // which rollout this window came from (0 to num_examples-1)
  }
}
```

### How Temporal Windows Are Built
- Simulate the full rollout for `sim_steps` steps.
- Slide a window of length `h + 1` over the trajectory (`start_index` … `start_index + h`).
- Label movement from the last point in the window to the next point. If the center position changes more than `eps`, we mark `move_* = True`; otherwise `still_* = True`.
- Repeat for every example and every obstacle placement `t`.


### Position, Collision, and Labels
- `body_x`, `body_y` are the **center** of the ball (straight from `pymunk.Body.position`).
- Collision check: `body_x + ball_radius >= obstacle_bottom_left_x`. That’s when the ball’s right-most point overlaps the obstacle’s left edge.
- We store both bottom-left and top-left obstacle coordinates so you can reconstruct the contact edge exactly.
- Movement logic is intentionally simple and relies only on position deltas, not velocity.


### Running the Generator
Basic invocation (one obstacle position, 50 rollouts, save videos):

```bash
python get_jsonl.py --config_filepath config.yaml --out_jsonl datasets/ramp_temporal.jsonl --positions 0.3 --num_examples 50 --dt 0.02 --simulation_time 2.0 --window_h 10 --eps 1e-3 --jitter_px 2.0 --save_video --video_folder videos --videos_per_position 1
```

Say we want multiple obstacle placements, then we can pass them comma-separated to `--positions` (for example `0.3,0.5,0.7`). The script will still write a single JSONL but will now save one video per `t` using filenames like `videos/ball_evolution_t0.30_example000.mp4`. Reduce `--videos_per_position` (or turn off `--save_video`) if you don't want that many clips.

Each window in the JSONL includes an `example_id` in the `meta` field (0 to `num_examples-1`) so you can group windows by which rollout they came from. This makes it easy to filter or split the dataset by example.

Also note that even if the ball has already collided with the obstacle, the ball is still moving slowly because the ball deceltrates instead of stopping instantly. Like the ball may be slidining or rolling along the obstacle surface, and Pymunk (according to reading hte docs) applied friction forces over time so even with no bounce (elasticity=0.0), the ball doesn't stop immediately. 
