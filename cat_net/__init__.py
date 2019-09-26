import torch

# This tells cudnn to search for the most efficient convolutional algorithms.
# Possibly faster. Definitely magic.
torch.backends.cudnn.benchmark = True
