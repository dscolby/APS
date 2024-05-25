from dataclasses import dataclass

@dataclass
class ARSHyperparameters:
    alpha: float=0.02
    v: float=0.03
    num_rollouts: int=50
    b: int=16
    num_experiments: int=100
    horizon: int=100

@dataclass
class GAPSHyperparameters:
    """Class for keeping track of hyperparameters for APS experiments"""
    num_experiments: int=100
    num_rollouts: int=100
    horizon: int=100
    num_initial_points: int=100