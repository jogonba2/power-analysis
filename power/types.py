from dataclasses import dataclass


@dataclass
class PowerOutput:
    power: float
    mean_eff: float
    type_m: float
    type_s: float


@dataclass
class PowerBounds:
    upper: PowerOutput
    lower: PowerOutput
