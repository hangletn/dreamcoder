# Do pytest here.
import pytest
import sys
import os
import json
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from get_jsonl import next_step_bools, create_env, simulate_positions, make_windows


def test_move_x_positive_direction():
    xw = [0.0, 1.0, 2.0, 3.0, 4.0]
    yw = [0.0, 0.0, 0.0, 0.0, 0.0]
    x_next = 5.0
    y_next = 0.0
    result = next_step_bools(xw, yw, x_next, y_next, eps=1e-3)
    assert result["move_x"] is True, "Should detect movement in positive x direction"
    assert result["still_x"] is False, "still_x should be False when moving"
    assert result["move_y"] is False, "Should not detect y movement"
    assert result["still_y"] is True, "Should detect no y movement"


def test_move_x_negative_direction():
    xw = [10.0, 9.0, 8.0, 7.0, 6.0]
    yw = [0.0, 0.0, 0.0, 0.0, 0.0]
    x_next = 5.0
    y_next = 0.0
    result = next_step_bools(xw, yw, x_next, y_next, eps=1e-3)
    assert result["move_x"] is True, "Should detect movement in negative x direction"
    assert result["still_x"] is False, "still_x should be False when moving"
    assert result["move_y"] is False, "Should not detect y movement"
    assert result["still_y"] is True, "Should detect no y movement"


def test_still_x_no_movement():
    xw = [5.0, 5.0, 5.0, 5.0, 5.0]
    yw = [0.0, 0.0, 0.0, 0.0, 0.0]
    x_next = 5.0 + 1e-4
    y_next = 0.0
    result = next_step_bools(xw, yw, x_next, y_next, eps=1e-3)
    assert result["move_x"] is False, "Should not detect movement for small change"
    assert result["still_x"] is True, "Should detect stillness for small change"
    assert result["move_y"] is False, "Should not detect y movement"
    assert result["still_y"] is True, "Should detect no y movement"


def test_move_y_positive_direction():
    xw = [0.0, 0.0, 0.0, 0.0, 0.0]
    yw = [0.0, 1.0, 2.0, 3.0, 4.0]
    x_next = 0.0
    y_next = 5.0
    result = next_step_bools(xw, yw, x_next, y_next, eps=1e-3)
    assert result["move_x"] is False, "Should not detect x movement"
    assert result["still_x"] is True, "Should detect no x movement"
    assert result["move_y"] is True, "Should detect movement in positive y direction"
    assert result["still_y"] is False, "still_y should be False when moving"


def test_move_y_negative_direction():
    xw = [0.0, 0.0, 0.0, 0.0, 0.0]
    yw = [10.0, 9.0, 8.0, 7.0, 6.0]
    x_next = 0.0
    y_next = 5.0
    result = next_step_bools(xw, yw, x_next, y_next, eps=1e-3)
    assert result["move_x"] is False, "Should not detect x movement"
    assert result["still_x"] is True, "Should detect no x movement"
    assert result["move_y"] is True, "Should detect movement in negative y direction"
    assert result["still_y"] is False, "still_y should be False when moving"


def test_both_move_x_and_y():
    xw = [0.0, 1.0, 2.0, 3.0, 4.0]
    yw = [0.0, 1.0, 2.0, 3.0, 4.0]
    x_next = 5.0
    y_next = 5.0
    result = next_step_bools(xw, yw, x_next, y_next, eps=1e-3)
    assert result["move_x"] is True, "Should detect x movement"
    assert result["still_x"] is False, "still_x should be False when moving"
    assert result["move_y"] is True, "Should detect y movement"
    assert result["still_y"] is False, "still_y should be False when moving"


def test_both_still_x_and_y():
    xw = [5.0, 5.0, 5.0, 5.0, 5.0]
    yw = [3.0, 3.0, 3.0, 3.0, 3.0]
    x_next = 5.0 + 1e-4
    y_next = 3.0 + 1e-4
    result = next_step_bools(xw, yw, x_next, y_next, eps=1e-3)
    assert result["move_x"] is False, "Should not detect x movement for small change"
    assert result["still_x"] is True, "Should detect x stillness"
    assert result["move_y"] is False, "Should not detect y movement for small change"
    assert result["still_y"] is True, "Should detect y stillness"


def test_mutually_exclusive_x():
    xw = [0.0, 1.0, 2.0, 3.0, 4.0]
    yw = [0.0, 0.0, 0.0, 0.0, 0.0]
    
    x_next = 5.0
    result = next_step_bools(xw, yw, x_next, 0.0, eps=1e-3)
    assert result["move_x"] != result["still_x"], "move_x and still_x should be mutually exclusive"
    
    x_next = 4.0 + 1e-4
    result = next_step_bools(xw, yw, x_next, 0.0, eps=1e-3)
    assert result["move_x"] != result["still_x"], "move_x and still_x should be mutually exclusive"


def test_mutually_exclusive_y():
    xw = [0.0, 0.0, 0.0, 0.0, 0.0]
    yw = [0.0, 1.0, 2.0, 3.0, 4.0]
    
    y_next = 5.0
    result = next_step_bools(xw, yw, 0.0, y_next, eps=1e-3)
    assert result["move_y"] != result["still_y"], "move_y and still_y should be mutually exclusive"
    
    y_next = 4.0 + 1e-4
    result = next_step_bools(xw, yw, 0.0, y_next, eps=1e-3)
    assert result["move_y"] != result["still_y"], "move_y and still_y should be mutually exclusive"


def test_eps_threshold():
    xw = [0.0, 1.0, 2.0, 3.0, 4.0]
    yw = [0.0, 0.0, 0.0, 0.0, 0.0]
    
    x_next = 4.0 + 0.5e-3
    result = next_step_bools(xw, yw, x_next, 0.0, eps=1e-3)
    assert result["move_x"] is False, "Below eps, should be considered still"
    assert result["still_x"] is True, "Below eps, should be considered still"
    
    x_next = 4.0 + 0.999e-3
    result = next_step_bools(xw, yw, x_next, 0.0, eps=1e-3)
    assert result["move_x"] is False, "Just below eps, should be considered still"
    assert result["still_x"] is True, "Just below eps, should be considered still"
    
    x_next = 4.0 + 2e-3
    result = next_step_bools(xw, yw, x_next, 0.0, eps=1e-3)
    assert result["move_x"] is True, "Above eps, should be considered movement"
    assert result["still_x"] is False, "Above eps, should not be considered still"


def test_ball_stops_after_hitting_obstacle():
    xw = [0.0, 10.0, 20.0, 30.0, 40.0]
    yw = [100.0, 90.0, 80.0, 70.0, 60.0]
    
    x_next = 40.0 + 1e-4
    y_next = 60.0 + 1e-4
    
    result = next_step_bools(xw, yw, x_next, y_next, eps=1e-3)
    assert result["move_x"] is False, "Ball should be stopped in x direction"
    assert result["still_x"] is True, "Ball should be still in x direction"
    assert result["move_y"] is False, "Ball should be stopped in y direction"
    assert result["still_y"] is True, "Ball should be still in y direction"


def test_position_history_detection():
    xw = [100.0, 100.01, 100.02, 100.03, 100.04, 100.05, 100.06, 100.07, 100.08, 100.09, 100.10]
    yw = [50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0]
    x_next = 100.11
    y_next = 50.0
    
    result = next_step_bools(xw, yw, x_next, y_next, eps=1e-3)
    assert result["move_x"] is True, "Should detect movement based on consistent position changes"
    assert result["still_x"] is False, "Should not be still when position is consistently changing"
    
    xw_stopped = [283.01, 283.01, 283.01, 283.01, 283.01, 283.01, 283.01, 283.01, 283.01, 283.01, 283.01]
    yw_stopped = [56.97, 56.97, 56.97, 56.97, 56.97, 56.97, 56.97, 56.97, 56.97, 56.97, 56.97]
    x_next_stopped = 283.01 + 1e-4
    y_next_stopped = 56.97 + 1e-4
    
    result_stopped = next_step_bools(xw_stopped, yw_stopped, x_next_stopped, y_next_stopped, eps=1e-3)
    assert result_stopped["move_x"] is False, "Should detect stillness when position is stable"
    assert result_stopped["still_x"] is True, "Should be still when position is stable"


def test_ball_center_plus_radius_gives_right_edge():
    body_x = 100.0
    ball_radius = 20.0
    ball_right_edge = body_x + ball_radius
    
    assert ball_right_edge == 120.0, "Right edge should be center + radius"
    
    body_x = 283.02
    ball_radius = 20.0
    ball_right_edge = body_x + ball_radius
    assert ball_right_edge == 303.02, "Right edge calculation should work for any values"


def test_ball_center_minus_radius_gives_left_edge():
    body_x = 100.0
    ball_radius = 20.0
    ball_left_edge = body_x - ball_radius
    
    assert ball_left_edge == 80.0, "Left edge should be center - radius"


def test_ball_vertical_bounds():
    body_y = 56.91
    ball_radius = 20.0
    ball_top = body_y + ball_radius
    ball_bottom = body_y - ball_radius
    
    assert ball_top == 76.91, "Top edge should be center + radius"
    assert ball_bottom == 36.91, "Bottom edge should be center - radius"


def test_collision_detection_logic():
    body_x = 276.0
    ball_radius = 20.0
    ball_right_edge = body_x + ball_radius
    obstacle_left_edge = 296.05
    
    has_collided = ball_right_edge >= obstacle_left_edge
    assert has_collided is False, "Ball should not have collided yet (296.0 < 296.05)"
    
    body_x = 276.05
    ball_right_edge = body_x + ball_radius
    has_collided = ball_right_edge >= obstacle_left_edge
    assert has_collided is True, "Ball should have collided (296.05 >= 296.05)"
    
    body_x = 283.0
    ball_right_edge = body_x + ball_radius
    has_collided = ball_right_edge >= obstacle_left_edge
    assert has_collided is True, "Ball should have collided (303.0 > 296.05)"


def test_collision_with_tolerance():
    body_x = 276.049999
    ball_radius = 20.0
    ball_right_edge = body_x + ball_radius
    obstacle_left_edge = 296.05
    
    distance = obstacle_left_edge - ball_right_edge
    has_collided = distance <= 0.1
    
    assert has_collided is True, "Should detect collision when very close (within tolerance)"


def test_jsonl_structure_from_file():
    jsonl_path = os.path.join(os.path.dirname(__file__), "datasets", "ramp_temporal.jsonl")
    
    if not os.path.exists(jsonl_path):
        pytest.skip(f"JSONL file not found at {jsonl_path}")
    
    with open(jsonl_path, 'r') as f:
        lines = f.readlines()
    
    assert len(lines) > 0, "JSONL file should contain at least one example"
    
    data = json.loads(lines[0])
    
    assert "name" in data, "Each example should have 'name'"
    assert "input" in data, "Each example should have 'input'"
    assert "output" in data, "Each example should have 'output'"
    assert "meta" in data, "Each example should have 'meta'"
    
    input_data = data["input"]
    assert "body_x" in input_data, "Input should have 'body_x'"
    assert "body_y" in input_data, "Input should have 'body_y'"
    assert "obstacle_bottom_left_x" in input_data, "Input should have 'obstacle_bottom_left_x'"
    assert "obstacle_bottom_left_y" in input_data, "Input should have 'obstacle_bottom_left_y'"
    assert "obstacle_top_left_x" in input_data, "Input should have 'obstacle_top_left_x'"
    assert "obstacle_top_left_y" in input_data, "Input should have 'obstacle_top_left_y'"
    assert "ball_radius" in input_data, "Input should have 'ball_radius'"
    assert "box_size" in input_data, "Input should have 'box_size'"
    assert "ramp_theta" in input_data, "Input should have 'ramp_theta'"
    
    output_data = data["output"]
    assert "move_x" in output_data, "Output should have 'move_x'"
    assert "still_x" in output_data, "Output should have 'still_x'"
    assert "move_y" in output_data, "Output should have 'move_y'"
    assert "still_y" in output_data, "Output should have 'still_y'"
    
    meta_data = data["meta"]
    assert "t" in meta_data, "Meta should have 't'"
    assert "dt" in meta_data, "Meta should have 'dt'"
    assert "start_index" in meta_data, "Meta should have 'start_index'"
    assert "h" in meta_data, "Meta should have 'h'"
    assert "sim_steps" in meta_data, "Meta should have 'sim_steps'"


def test_window_length_correctness():
    jsonl_path = os.path.join(os.path.dirname(__file__), "datasets", "ramp_temporal.jsonl")
    
    if not os.path.exists(jsonl_path):
        pytest.skip(f"JSONL file not found at {jsonl_path}")
    
    with open(jsonl_path, 'r') as f:
        lines = f.readlines()
    
    for line in lines[:10]:
        data = json.loads(line)
        h = data["meta"]["h"]
        body_x = data["input"]["body_x"]
        body_y = data["input"]["body_y"]
        
        assert len(body_x) == h + 1, f"body_x should have h+1={h+1} points, got {len(body_x)}"
        assert len(body_y) == h + 1, f"body_y should have h+1={h+1} points, got {len(body_y)}"


def test_collision_data_correctness():
    jsonl_path = os.path.join(os.path.dirname(__file__), "datasets", "ramp_temporal.jsonl")
    
    if not os.path.exists(jsonl_path):
        pytest.skip(f"JSONL file not found at {jsonl_path}")
    
    with open(jsonl_path, 'r') as f:
        lines = f.readlines()
    
    collision_examples = []
    for i, line in enumerate(lines):
        data = json.loads(line)
        body_x = data["input"]["body_x"][-1]
        body_y = data["input"]["body_y"][-1]
        ball_radius = data["input"]["ball_radius"]
        obstacle_left_x = data["input"]["obstacle_bottom_left_x"]
        
        ball_right_edge = body_x + ball_radius
        has_hit = ball_right_edge >= obstacle_left_x
        
        if has_hit:
            collision_examples.append({
                "line": i + 1,
                "body_x": body_x,
                "body_y": body_y,
                "ball_radius": ball_radius,
                "ball_right_edge": ball_right_edge,
                "obstacle_left_x": obstacle_left_x,
                "overlap": ball_right_edge - obstacle_left_x
            })
    
    assert len(collision_examples) > 0, "Should have at least one collision example"
    
    for ex in collision_examples[:5]:
        assert ex["ball_right_edge"] >= ex["obstacle_left_x"] - 0.1, \
            f"Line {ex['line']}: Ball should have hit obstacle (right_edge={ex['ball_right_edge']:.2f}, obstacle={ex['obstacle_left_x']:.2f})"
        
        assert ex["overlap"] < 50.0, \
            f"Line {ex['line']}: Overlap seems too large ({ex['overlap']:.2f}), possible data error"


def test_ball_obstacle_vertical_overlap():
    jsonl_path = os.path.join(os.path.dirname(__file__), "datasets", "ramp_temporal.jsonl")
    
    if not os.path.exists(jsonl_path):
        pytest.skip(f"JSONL file not found at {jsonl_path}")
    
    with open(jsonl_path, 'r') as f:
        lines = f.readlines()
    
    for i, line in enumerate(lines[:50]):
        data = json.loads(line)
        body_x = data["input"]["body_x"][-1]
        body_y = data["input"]["body_y"][-1]
        ball_radius = data["input"]["ball_radius"]
        obstacle_left_x = data["input"]["obstacle_bottom_left_x"]
        obstacle_bottom_y = data["input"]["obstacle_bottom_left_y"]
        obstacle_top_y = data["input"]["obstacle_top_left_y"]
        
        ball_right_edge = body_x + ball_radius
        ball_top = body_y + ball_radius
        ball_bottom = body_y - ball_radius
        
        if ball_right_edge >= obstacle_left_x:
            vertical_overlap = not (ball_top < obstacle_bottom_y or ball_bottom > obstacle_top_y)
            
            if not vertical_overlap:
                vertical_gap = min(abs(ball_top - obstacle_bottom_y), abs(ball_bottom - obstacle_top_y))
                assert vertical_gap < ball_radius * 2, \
                    f"Line {i+1}: Ball and obstacle should be close vertically when colliding (gap={vertical_gap:.2f})"


def test_window_continuity():
    jsonl_path = os.path.join(os.path.dirname(__file__), "datasets", "ramp_temporal.jsonl")
    
    if not os.path.exists(jsonl_path):
        pytest.skip(f"JSONL file not found at {jsonl_path}")
    
    with open(jsonl_path, 'r') as f:
        lines = f.readlines()
    
    prev_data = None
    for i, line in enumerate(lines[:100]):
        data = json.loads(line)
        
        if prev_data is not None and data["name"] == prev_data["name"]:
            prev_start = prev_data["meta"]["start_index"]
            prev_h = prev_data["meta"]["h"]
            curr_start = data["meta"]["start_index"]
            
            expected_start = prev_start + 1
            assert curr_start == expected_start, \
                f"Line {i+1}: Windows should be consecutive (prev starts at {prev_start}, curr starts at {curr_start})"
            
            prev_last_h_x = prev_data["input"]["body_x"][1:]
            curr_first_h_x = data["input"]["body_x"][:-1]
            prev_last_h_y = prev_data["input"]["body_y"][1:]
            curr_first_h_y = data["input"]["body_y"][:-1]
            
            for j, (px, cx) in enumerate(zip(prev_last_h_x, curr_first_h_x)):
                assert abs(px - cx) < 1e-6, \
                    f"Line {i+1}: Overlapping windows should share points (x[{j+1}]: {px} vs {cx})"
            
            for j, (py, cy) in enumerate(zip(prev_last_h_y, curr_first_h_y)):
                assert abs(py - cy) < 1e-6, \
                    f"Line {i+1}: Overlapping windows should share points (y[{j+1}]: {py} vs {cy})"
        
        prev_data = data


def test_movement_labels_consistency():
    jsonl_path = os.path.join(os.path.dirname(__file__), "datasets", "ramp_temporal.jsonl")
    
    if not os.path.exists(jsonl_path):
        pytest.skip(f"JSONL file not found at {jsonl_path}")
    
    with open(jsonl_path, 'r') as f:
        lines = f.readlines()
    
    for i, line in enumerate(lines):
        data = json.loads(line)
        output = data["output"]
        
        assert output["move_x"] != output["still_x"], \
            f"Line {i+1}: move_x and still_x should be mutually exclusive (both {output['move_x']})"
        
        assert output["move_y"] != output["still_y"], \
            f"Line {i+1}: move_y and still_y should be mutually exclusive (both {output['move_y']})"


def test_create_env_returns_correct_structure():
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    
    if not os.path.exists(config_path):
        pytest.skip(f"Config file not found at {config_path}")
    
    space, body, params = create_env(config_path, box_t=0.5, jitter_px=0.0, seed=123)
    
    assert "ball_radius" in params, "params should have 'ball_radius'"
    assert "box_size" in params, "params should have 'box_size'"
    assert "ramp_theta" in params, "params should have 'ramp_theta'"
    assert "obstacle_bottom_left_x" in params, "params should have 'obstacle_bottom_left_x'"
    assert "obstacle_bottom_left_y" in params, "params should have 'obstacle_bottom_left_y'"
    assert "obstacle_top_left_x" in params, "params should have 'obstacle_top_left_x'"
    assert "obstacle_top_left_y" in params, "params should have 'obstacle_top_left_y'"
    
    assert params["obstacle_bottom_left_x"] > 0, "Obstacle x should be positive"
    assert params["obstacle_bottom_left_y"] >= 0, "Obstacle y should be non-negative"
    assert params["obstacle_top_left_y"] > params["obstacle_bottom_left_y"], \
        "Top left y should be greater than bottom left y"
    
    assert params["ball_radius"] > 0, "Ball radius should be positive"


def test_make_windows_produces_correct_structure():
    xs = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    ys = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    h = 3
    
    windows = make_windows(xs, ys, h)
    
    assert len(windows) > 0, "Should produce at least one window"
    
    start_i, xw, yw, x_next, y_next = windows[0]
    assert start_i == 0, "First window should start at index 0"
    assert len(xw) == h + 1, f"Window should have h+1={h+1} points, got {len(xw)}"
    assert len(yw) == h + 1, f"Window should have h+1={h+1} points, got {len(yw)}"
    assert xw == xs[0:h+1], "Window should contain first h+1 points"
    assert yw == ys[0:h+1], "Window should contain first h+1 points"
    assert x_next == xs[h + 1], "x_next should be the point after the window"
    assert y_next == ys[h + 1], "y_next should be the point after the window"
    
    if len(windows) > 1:
        prev_start, prev_xw, prev_yw, _, _ = windows[0]
        curr_start, curr_xw, curr_yw, _, _ = windows[1]
        
        assert curr_start == prev_start + 1, "Windows should be consecutive (sliding by 1)"
        assert prev_xw[1:] == curr_xw[:-1], "Consecutive windows should overlap by h points (x)"
        assert prev_yw[1:] == curr_yw[:-1], "Consecutive windows should overlap by h points (y)"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
