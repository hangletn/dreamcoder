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
    
    This extractor:
    - 1D CNN over [x, y, dx, dy] time series
    - Average embeddings across multiple windows per task
    - Do Helmholtz ie wake-sleep <- generate synthetic temporal tasks
    """

    def __init__(self, tasks, testingTasks=None, cuda=False, H=64, **kwargs):
        super().__init__()
        if testingTasks is None:
            testingTasks = []

        self.use_cuda = cuda
        self.H = H

        # Infer typical window length from tasks
        lengths = []
        for t in tasks + testingTasks:
            if t.examples:
                (pos_x, pos_y, obx, oby), _ = t.examples[0]
                lengths.append(len(pos_x))
        self.window_length = max(lengths) if lengths else 11

        # Simple 1D CNN over [x, y, dx, dy] channels
        self.conv = nn.Sequential(nn.Conv1d(4, 32, kernel_size=5, padding=2), nn.ReLU(), nn.Conv1d(32, H, kernel_size=5, padding=2), nn.ReLU())
        # MLP to incorporate obstacle position into features
        self.obstacle_mlp = nn.Sequential(nn.Linear(H + 2, H), nn.ReLU())

        self.outputDimensionality = H
        self.recomputeTasks = True

        # Helmholtz timeouts
        self.helmholtzTimeout = 0.25
        self.helmholtzEvalTimeout = 0.01

        if self.use_cuda:
            self.cuda()

    def _encode_single_example(self, example):
        """
        example: ((pos_x, pos_y, obx, oby), label_bool)
        returns: feature vector (H,)
        """
        (pos_x, pos_y, obx, oby), label = example
        xs = torch.tensor(pos_x, dtype=torch.float32)
        ys = torch.tensor(pos_y, dtype=torch.float32)

        # Forward differences
        dx = torch.cat([torch.zeros(1), xs[1:] - xs[:-1]])
        dy = torch.cat([torch.zeros(1), ys[1:] - ys[:-1]])

        X = torch.stack([xs, ys, dx, dy], dim=0).unsqueeze(0)  # (1, 4, T)
        if self.use_cuda:
            X = X.cuda()

        z = self.conv(X) # (1, H, T)
        z = z.mean(dim=2).squeeze(0) # (H,)
        # Incorporate obstacle position
        obstacle_pos = torch.tensor([obx, oby], dtype=torch.float32)
        if self.use_cuda:
            obstacle_pos = obstacle_pos.cuda()
        z_with_obstacle = torch.cat([z, obstacle_pos], dim=0) # (H+2,)
        z_final = self.obstacle_mlp(z_with_obstacle) # (H,)
        return z_final

    def _encode_examples(self, examples):
        """
        Encode multiple examples and average their embeddings.
        Deal with many windows per task from your JSONL.
        """
        if not examples:
            z = torch.zeros(self.H, dtype=torch.float32)
            return z.cuda() if self.use_cuda else z

        # Subsample if there are many windows (for faster)
        MAX_EXAMPLES = 32
        if len(examples) > MAX_EXAMPLES:
            examples = random.sample(examples, MAX_EXAMPLES)

        zs = []
        for ex in examples:
            zs.append(self._encode_single_example(ex))
        z = torch.stack(zs, dim=0).mean(dim=0)  # (H,)
        return z

    def forward(self, examples):
        """
        For RecognitionModel compatibleness, forward(examples) 
        <- a feature vector (torch tensor of length H).
        """
        return self._encode_examples(examples)

    def featuresOfTask(self, task):
        return self(task.examples)

    def taskOfProgram(self, p, request):
        """
        Given a sampled program p : (list real, list real, real, real) -> bool
        construct a synthetic Task with random numeric traces.
        
        This is used for Helmholtz (dreaming) training. 
        Do synthetic traces so we can learn what program in SDL look like (ie no need to match physics?)
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

        return Task("Helmholtz_temporal", request, examples)
