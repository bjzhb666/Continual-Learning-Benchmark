import os
import sys
import argparse
import torch
import numpy as np
from random import shuffle
from collections import OrderedDict
import dataloaders.base
from torchvision import transforms
from dataloaders.datasetGen import SplitGen, PermutedGen, CORe50Gen
import agents
import wandb