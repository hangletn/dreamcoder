Generate a JSONL dataset of temporal windows from ball on ramp simulation.

JSONL File Structure:
Each line in the output file is ONE training example (a temporal window).

Example JSON structure:
{
  "name": "ramp_window_t0.50",
  "input": {
    "body_x": [49.81, 49.81, 49.81, ..., 283.02],  // h+1 floats: ball center x positions
    "body_y": [151.69, 151.29, 150.51, ..., 56.91], // h+1 floats: ball center y positions
    "obstacle_bottom_left_x": 296.05, // Leftmost bottom corner x of obstacle
    "obstacle_bottom_left_y": 31.18, // Leftmost bottom corner y of obstacle
    "obstacle_top_left_x": 310.42, // Leftmost top corner x of obstacle
    "obstacle_top_left_y": 79.08, // Leftmost top corner y of obstacle
    "ball_radius": 20.0, // Radius of the ball
    "box_size": 50.0, // Side length of square obstacle
    "ramp_theta": -0.291 // Ramp angle in radians
  },
  "output": {
    "move_x": true,                                 // Ball is moving in x direction
    "still_x": false,                               // Ball is NOT still in x (complement of move_x)
    "move_y": false,                                // Ball is moving in y direction
    "still_y": true                                 // Ball is NOT still in y (complement of move_y)
  },
  "meta": {
    "t": 0.5,                                       // Obstacle placement param (0=ramp start, 1=ground end)
    "dt": 0.02,                                     // Simulation time step (seconds)
    "start_index": 75,                               // Starting index in full simulation sequence
    "h": 10,                                        // Window length (window has h+1 points)
    "sim_steps": 100                                // Total simulation steps in this rollout
  }
}

Temporal Windows:
The dataset uses sliding windows over the simulation trajectory:
- Each window contains h+1 consecutive time steps (indices i to i+h)
- The window starts at index `start_index` and ends at `start_index + h`
- `body_x` and `body_y` contain positions at times [i, i+1, ..., i+h]
- The output labels (`move_x`, `move_y`, etc.) predict movement from time i+h to i+h+1
- Windows are created for all valid indices: i = 0, 1, 2, ..., n-h-2 (where n = sim_steps)

Ball Position and Collision:
- `body_x` and `body_y` represent the center of the ball (Pymunk body.position)
- Ball bounds:
  - left edge = body_x - ball_radius
  - right edge = body_x + ball_radius
  - bottom edge = body_y - ball_radius
  - top edge = body_y + ball_radius
- Collision occurs when: ball_right_edge >= obstacle_bottom_left_x
  (i.e., when the ball's right edge touches or passes the obstacle's left edge)
- The obstacle is a rotated box, with `obstacle_bottom_left_x/y` and `obstacle_top_left_x/y`
  defining the leftmost edge (the contact surface)

Movement Detection:
- `move_x` = True if |x_next - x_last| > eps (ball moved in x direction)
- `move_y` = True if |y_next - y_last| > eps (ball moved in y direction)
- `still_x` = not move_x, `still_y` = not move_y
- `eps` is a threshold for detecting stillness (default 1e-3)

Usage example:
  `python get_jsonl.py \
    --config_filepath config.yaml \
    --out_jsonl datasets/ramp_temporal.jsonl \
    --positions 0.3 \
    --num_examples 1 \
    --dt 0.02 \
    --simulation_time 2.0 \
    --window_h 10 \
    --eps 1e-3 \
    --jitter_px 2.0 --save_video --video_folder videos`
