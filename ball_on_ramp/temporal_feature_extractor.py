# ball_on_ramp/temporal_feature_extractor.py
import random
import time
import torch
import torch.nn as nn

from dreamcoder.task import Task
from dreamcoder.utilities import runWithTimeout


class TemporalFeatureExtractor(nn.Module):
    """
    Feature extractor for temporal ramp tasks.

    Expected task.examples:
      examples = [((pos_x, pos_y, obstacle_x, obstacle_y), label_bool), ...]
      where pos_x, pos_y are lists of floats of length T.
      Examples come from ALL obstacle positions combined (2 tasks: move_x, move_y).
    
    This extractor:
    - 1D CNN over [x, y, dx, dy] time series
    - Average embeddings across multiple windows per task (from all positions)
    - Do Helmholtz ie wake-sleep <- generate synthetic temporal tasks
    """

    special = None  # No special handling needed for Helmholtz enumeration

    def __init__(self, tasks, testingTasks=None, cuda=False, H=64, **kwargs):
        super().__init__()
        if testingTasks is None:
            testingTasks = []

        self.use_cuda = cuda
        self.H = H
        self.parallelTaskOfProgram = True

        # Infer typical window length from tasks
        lengths = []
        for t in tasks + testingTasks:
            if t.examples:
                (pos_x, pos_y, obx, oby), _ = t.examples[0]
                lengths.append(len(pos_x))
        self.window_length = max(lengths) if lengths else 11

        self.conv = nn.Sequential(nn.Conv1d(4, 32, kernel_size=5, padding=2), nn.ReLU(), nn.Conv1d(32, H, kernel_size=5, padding=2), nn.ReLU())
        # MLP to incorporate obstacle position and meta info (ball_radius, box_size, ramp_theta) into features
        # Input: H (from CNN) + 2 (obstacle x,y) + 3 (ball_radius, box_size, ramp_theta) = H + 5
        self.obstacle_mlp = nn.Sequential(nn.Linear(H + 5, H), nn.ReLU())

        self.outputDimensionality = H
        self.recomputeTasks = True

        self.helmholtzTimeout = 0.25
        self.helmholtzEvalTimeout = 0.01

        if self.use_cuda:
            self.cuda()

    def _encode_single_example(self, example, meta):
        """
        example: ((pos_x, pos_y, obx, oby), label_bool)
        meta: dict with ball_radius, box_size, ramp_theta (from task.meta)
        returns: feature vector (H,)
        """
        (pos_x, pos_y, obx, oby), label = example
        if meta is None or not isinstance(meta, dict):
            meta = {"ball_radius": 25.0, "box_size": 50.0, "ramp_theta": -0.2914567944778671}
        
        xs = torch.tensor(pos_x, dtype=torch.float32)
        ys = torch.tensor(pos_y, dtype=torch.float32)

        # Forward differences
        dx = torch.cat([torch.zeros(1), xs[1:] - xs[:-1]])
        dy = torch.cat([torch.zeros(1), ys[1:] - ys[:-1]])

        X = torch.stack([xs, ys, dx, dy], dim=0).unsqueeze(0)
        if self.use_cuda:
            X = X.cuda()

        z = self.conv(X)
        z = z.mean(dim=2).squeeze(0)
        ball_radius = float(meta.get("ball_radius", 25.0))
        box_size = float(meta.get("box_size", 50.0))
        ramp_theta = float(meta.get("ramp_theta", -0.2914567944778671)) 
        
        obstacle_pos = torch.tensor([obx, oby], dtype=torch.float32)
        meta_tensor = torch.tensor([ball_radius, box_size, ramp_theta], dtype=torch.float32)
        if self.use_cuda:
            obstacle_pos = obstacle_pos.cuda()
            meta_tensor = meta_tensor.cuda()
        z_with_obstacle_and_meta = torch.cat([z, obstacle_pos, meta_tensor], dim=0)
        z_final = self.obstacle_mlp(z_with_obstacle_and_meta)
        return z_final

    def _encode_examples(self, examples, meta=None):
        """
        Encode multiple examples and average their embeddings.
        Examples can come from multiple positions (all combined into 2 tasks: move_x, move_y).
        Each task contains windows from all obstacle positions.
        """
        if not examples:
            z = torch.zeros(self.H, dtype=torch.float32)
            return z.cuda() if self.use_cuda else z

        MAX_EXAMPLES = 32
        if len(examples) > MAX_EXAMPLES:
            examples = random.sample(examples, MAX_EXAMPLES)

        zs = []
        for ex in examples:
            zs.append(self._encode_single_example(ex, meta=meta))
        z = torch.stack(zs, dim=0).mean(dim=0)
        return z

    def forward(self, examples, meta=None):
        """
        For RecognitionModel compatibleness, forward(examples) 
        <- a feature vector (torch tensor of length H).
        """
        return self._encode_examples(examples, meta=meta)

    def featuresOfTask(self, task):
        meta = getattr(task, 'meta', None)
        if meta is None or not isinstance(meta, dict):
            meta = {"ball_radius": 25.0, "box_size": 50.0, "ramp_theta": -0.2914567944778671}
        return self(task.examples, meta=meta)

    def taskOfProgram(self, p, request):
        """
        Given a sampled program p : (list real, list real, real, real) -> bool
        construct a synthetic Task with random numeric traces.
        
        This is used for Helmholtz (dreaming) training.
        """
        examples = []
        start_time = time.time()

        target_examples = 5
        max_attempts = 50

        for _ in range(max_attempts):
            if time.time() - start_time > self.helmholtzTimeout:
                break

            T = self.window_length
            xs = [float(random.uniform(0.0, 640.0)) for _ in range(T)]
            ys = [float(random.uniform(0.0, 480.0)) for _ in range(T)]
            obx = float(random.uniform(0.0, 640.0))
            oby = float(random.uniform(0.0, 200.0))

            try:
                y = runWithTimeout(lambda: p.runWithArguments([xs, ys, obx, oby]), self.helmholtzEvalTimeout)
            except Exception:
                continue

            if not isinstance(y, bool):
                continue

            examples.append(((xs, ys, obx, oby), y))
            if len(examples) >= target_examples:
                break

        if not examples:
            return None

        task = Task("Helmholtz_temporal", request, examples)
        task.meta = {"ball_radius": 25.0, "box_size": 50.0, "ramp_theta": -0.2914567944778671}
        return task
