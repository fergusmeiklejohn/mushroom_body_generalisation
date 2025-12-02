"""
Mushroom Body-Inspired Generalisation Network (MBGN)

A minimal neural network inspired by insect mushroom body architecture
for learning relational concepts that transfer to novel stimuli.
"""

from .model import MBGN
from .stimuli import StimulusGenerator
from .task import DMTSTask, DNMTSTask
from .training import Trainer
from .analysis import Analyzer

__version__ = "0.1.0"
__all__ = ["MBGN", "StimulusGenerator", "DMTSTask", "DNMTSTask", "Trainer", "Analyzer"]
