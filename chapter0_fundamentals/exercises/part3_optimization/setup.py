# All set up here cause otherwise it's annoying..

import os
import sys
from pathlib import Path

IN_COLAB = "google.colab" in sys.modules

chapter = "chapter0_fundamentals"
repo = "ARENA_3.0"
branch = "main"

# Get root directory, handling 3 different cases: (1) Colab, (2) notebook not in ARENA repo, (3) notebook in ARENA repo
root = (
    "/content"
    if IN_COLAB
    else "/root"
    if repo not in os.getcwd()
    else str(next(p for p in Path.cwd().parents if p.name == repo))
)




assert Path(f"{root}/{chapter}/exercises").exists(), "Unexpected error: please manually clone ARENA repo into `root`"

if f"{root}/{chapter}/exercises" not in sys.path:
    sys.path.append(f"{root}/{chapter}/exercises")


import importlib
import os
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Iterable, Literal

import numpy as np
import torch
import torch as t
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
import wandb
from IPython.core.display import HTML
from IPython.display import display
from jaxtyping import Float, Int
from torch import Tensor, optim
from torch.utils.data import DataLoader, DistributedSampler, Subset
from torchvision import datasets, transforms
from tqdm import tqdm

# Make sure exercises are in the path
chapter = "chapter0_fundamentals"
section = "part3_optimization"
root_dir = next(p for p in Path.cwd().parents if (p / chapter).exists())
exercises_dir = root_dir / chapter / "exercises"
section_dir = exercises_dir / section
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))


import part3_optimization.tests as tests
from part2_cnns.solutions import Linear, ResNet34, get_resnet_for_feature_extraction
from part3_optimization.utils import plot_fn, plot_fn_with_points
from plotly_utils import bar, imshow, line

device = t.device("mps" if t.backends.mps.is_available() else "cuda" if t.cuda.is_available() else "cpu")

MAIN = __name__ == "__main__"

print('setup')