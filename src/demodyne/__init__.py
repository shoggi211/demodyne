"""demodyne: dynamics and data-integrity analysis from robot demonstration datasets."""
from .io import Dataset, Episode, load, from_arrays
from .model import ServoParams, simulate, one_step
from .ident import fit_joint, fit_dataset, JointFit

__version__ = "0.1.0"
