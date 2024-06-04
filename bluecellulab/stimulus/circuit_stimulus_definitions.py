# Copyright 2012-2024 Blue Brain Project / EPFL

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Defines the expected data structures associated with the stimulus defined in
simulation configs.

Run-time validates the data via Pydantic.
"""
from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Optional
import warnings

from pydantic import field_validator, NonNegativeFloat, PositiveFloat
from pydantic.dataclasses import dataclass


# create an enum for StimulusMode with Current and Conductance values
class ClampMode(Enum):
    """Current clamp or conductance (dynamic) clamp."""
    CURRENT = "current_clamp"
    CONDUCTANCE = "conductance"


class Pattern(Enum):
    """Enum that defaults to SONATA values.

    Has blueconfig overload.
    """
    NOISE = "noise"
    HYPERPOLARIZING = "hyperpolarizing"
    PULSE = "pulse"
    RELATIVE_LINEAR = "relative_linear"
    SYNAPSE_REPLAY = "synapse_replay"
    SHOT_NOISE = "shot_noise"
    RELATIVE_SHOT_NOISE = "relative_shot_noise"
    ORNSTEIN_UHLENBECK = "ornstein_uhlenbeck"
    RELATIVE_ORNSTEIN_UHLENBECK = "relative_ornstein_uhlenbeck"

    @classmethod
    def from_blueconfig(cls, pattern: str) -> Pattern:
        if pattern == "Noise":
            return Pattern.NOISE
        elif pattern == "Hyperpolarizing":
            return Pattern.HYPERPOLARIZING
        elif pattern == "Pulse":
            return Pattern.PULSE
        elif pattern == "RelativeLinear":
            return Pattern.RELATIVE_LINEAR
        elif pattern == "SynapseReplay":
            return Pattern.SYNAPSE_REPLAY
        elif pattern == "ShotNoise":
            return Pattern.SHOT_NOISE
        elif pattern == "RelativeShotNoise":
            return Pattern.RELATIVE_SHOT_NOISE
        elif pattern == "OrnsteinUhlenbeck":
            return Pattern.ORNSTEIN_UHLENBECK
        elif pattern == "RelativeOrnsteinUhlenbeck":
            return Pattern.RELATIVE_ORNSTEIN_UHLENBECK
        else:
            raise ValueError(f"Unknown pattern {pattern}")

    @classmethod
    def from_sonata(cls, pattern: str) -> Pattern:
        if pattern == "noise":
            return Pattern.NOISE
        elif pattern == "hyperpolarizing":
            return Pattern.HYPERPOLARIZING
        elif pattern == "pulse":
            return Pattern.PULSE
        elif pattern == "relative_linear":
            return Pattern.RELATIVE_LINEAR
        elif pattern == "synapse_replay":
            return Pattern.SYNAPSE_REPLAY
        elif pattern == "shot_noise":
            return Pattern.SHOT_NOISE
        elif pattern == "relative_shot_noise":
            return Pattern.RELATIVE_SHOT_NOISE
        elif pattern == "ornstein_uhlenbeck":
            return Pattern.ORNSTEIN_UHLENBECK
        elif pattern == "relative_ornstein_uhlenbeck":
            return Pattern.RELATIVE_ORNSTEIN_UHLENBECK
        else:
            raise ValueError(f"Unknown pattern {pattern}")


@dataclass(frozen=True, config=dict(extra="forbid"))
class Stimulus:
    target: str
    delay: NonNegativeFloat
    duration: NonNegativeFloat

    @classmethod
    def from_blueconfig(cls, stimulus_entry: dict) -> Optional[Stimulus]:
        pattern = Pattern.from_blueconfig(stimulus_entry["Pattern"])
        mode_str = stimulus_entry.get("Mode", "Current").lower()
        if mode_str == "current":
            mode = ClampMode.CURRENT
        elif mode_str == "conductance":
            mode = ClampMode.CONDUCTANCE
        else:
            raise ValueError(f"Unknown clamp mode {mode_str}")
        if pattern == Pattern.NOISE:
            return Noise(
                target=stimulus_entry["Target"],
                delay=stimulus_entry["Delay"],
                duration=stimulus_entry["Duration"],
                mean_percent=stimulus_entry["MeanPercent"],
                variance=stimulus_entry["Variance"],
            )
        elif pattern == Pattern.HYPERPOLARIZING:
            return Hyperpolarizing(
                target=stimulus_entry["Target"],
                delay=stimulus_entry["Delay"],
                duration=stimulus_entry["Duration"],
            )
        elif pattern == Pattern.PULSE:
            return Pulse(
                target=stimulus_entry["Target"],
                delay=stimulus_entry["Delay"],
                duration=stimulus_entry["Duration"],
                amp_start=stimulus_entry["AmpStart"],
                width=stimulus_entry["Width"],
                frequency=stimulus_entry["Frequency"],
            )
        elif pattern == Pattern.RELATIVE_LINEAR:
            return RelativeLinear(
                target=stimulus_entry["Target"],
                delay=stimulus_entry["Delay"],
                duration=stimulus_entry["Duration"],
                percent_start=stimulus_entry["PercentStart"],
            )
        elif pattern == Pattern.SYNAPSE_REPLAY:
            warnings.warn("Ignoring syanpse replay stimulus as it is not supported")
            return None
        elif pattern == Pattern.SHOT_NOISE:
            return ShotNoise(
                target=stimulus_entry["Target"],
                delay=stimulus_entry["Delay"],
                duration=stimulus_entry["Duration"],
                dt=stimulus_entry.get("Dt", 0.25),
                rise_time=stimulus_entry["RiseTime"],
                decay_time=stimulus_entry["DecayTime"],
                rate=stimulus_entry["Rate"],
                amp_mean=stimulus_entry["AmpMean"],
                amp_var=stimulus_entry["AmpVar"],
                seed=stimulus_entry.get("Seed", None),
                mode=mode,
                reversal=stimulus_entry.get("Reversal", 0.0)
            )
        elif pattern == Pattern.RELATIVE_SHOT_NOISE:
            return RelativeShotNoise(
                target=stimulus_entry["Target"],
                delay=stimulus_entry["Delay"],
                duration=stimulus_entry["Duration"],
                dt=stimulus_entry.get("Dt", 0.25),
                rise_time=stimulus_entry["RiseTime"],
                decay_time=stimulus_entry["DecayTime"],
                mean_percent=stimulus_entry["MeanPercent"],
                sd_percent=stimulus_entry["SDPercent"],
                amp_cv=stimulus_entry["AmpCV"],
                seed=stimulus_entry.get("Seed", None),
                mode=mode,
                reversal=stimulus_entry.get("Reversal", 0.0)
            )
        elif pattern == Pattern.ORNSTEIN_UHLENBECK:
            return OrnsteinUhlenbeck(
                target=stimulus_entry["Target"],
                delay=stimulus_entry["Delay"],
                duration=stimulus_entry["Duration"],
                dt=stimulus_entry.get("Dt", 0.25),
                tau=stimulus_entry["Tau"],
                sigma=stimulus_entry["Sigma"],
                mean=stimulus_entry["Mean"],
                seed=stimulus_entry.get("Seed", None),
                mode=mode,
                reversal=stimulus_entry.get("Reversal", 0.0)
            )
        elif pattern == Pattern.RELATIVE_ORNSTEIN_UHLENBECK:
            return RelativeOrnsteinUhlenbeck(
                target=stimulus_entry["Target"],
                delay=stimulus_entry["Delay"],
                duration=stimulus_entry["Duration"],
                dt=stimulus_entry.get("Dt", 0.25),
                tau=stimulus_entry["Tau"],
                mean_percent=stimulus_entry["MeanPercent"],
                sd_percent=stimulus_entry["SDPercent"],
                seed=stimulus_entry.get("Seed", None),
                mode=mode,
                reversal=stimulus_entry.get("Reversal", 0.0)
            )
        else:
            raise ValueError(f"Unknown pattern {pattern}")

    @classmethod
    def from_sonata(cls, stimulus_entry: dict) -> Optional[Stimulus]:
        pattern = Pattern.from_sonata(stimulus_entry["module"])
        if pattern == Pattern.NOISE:
            return Noise(
                target=stimulus_entry["node_set"],
                delay=stimulus_entry["delay"],
                duration=stimulus_entry["duration"],
                mean_percent=stimulus_entry["mean_percent"],
                variance=stimulus_entry["variance"],
            )
        elif pattern == Pattern.HYPERPOLARIZING:
            return Hyperpolarizing(
                target=stimulus_entry["node_set"],
                delay=stimulus_entry["delay"],
                duration=stimulus_entry["duration"],
            )
        elif pattern == Pattern.PULSE:
            return Pulse(
                target=stimulus_entry["node_set"],
                delay=stimulus_entry["delay"],
                duration=stimulus_entry["duration"],
                amp_start=stimulus_entry["amp_start"],
                width=stimulus_entry["width"],
                frequency=stimulus_entry["frequency"],
            )
        elif pattern == Pattern.RELATIVE_LINEAR:
            return RelativeLinear(
                target=stimulus_entry["node_set"],
                delay=stimulus_entry["delay"],
                duration=stimulus_entry["duration"],
                percent_start=stimulus_entry["percent_start"],
            )
        elif pattern == Pattern.SYNAPSE_REPLAY:
            return SynapseReplay(
                target=stimulus_entry["node_set"],
                delay=stimulus_entry["delay"],
                duration=stimulus_entry["duration"],
                spike_file=stimulus_entry["spike_file"],
            )
        elif pattern == Pattern.SHOT_NOISE:
            return ShotNoise(
                target=stimulus_entry["node_set"],
                delay=stimulus_entry["delay"],
                duration=stimulus_entry["duration"],
                dt=stimulus_entry.get("dt", 0.25),
                rise_time=stimulus_entry["rise_time"],
                decay_time=stimulus_entry["decay_time"],
                rate=stimulus_entry["rate"],
                amp_mean=stimulus_entry["amp_mean"],
                amp_var=stimulus_entry["amp_var"],
                seed=stimulus_entry.get("random_seed", None),
                mode=ClampMode(stimulus_entry.get("input_type", "current_clamp").lower()),
                reversal=stimulus_entry.get("reversal", 0.0)
            )
        elif pattern == Pattern.RELATIVE_SHOT_NOISE:
            return RelativeShotNoise(
                target=stimulus_entry["node_set"],
                delay=stimulus_entry["delay"],
                duration=stimulus_entry["duration"],
                dt=stimulus_entry.get("dt", 0.25),
                rise_time=stimulus_entry["rise_time"],
                decay_time=stimulus_entry["decay_time"],
                mean_percent=stimulus_entry["mean_percent"],
                sd_percent=stimulus_entry["sd_percent"],
                amp_cv=stimulus_entry["amp_cv"],
                seed=stimulus_entry.get("random_seed", None),
                mode=ClampMode(stimulus_entry.get("input_type", "current_clamp").lower()),
                reversal=stimulus_entry.get("reversal", 0.0)
            )
        elif pattern == Pattern.ORNSTEIN_UHLENBECK:
            return OrnsteinUhlenbeck(
                target=stimulus_entry["node_set"],
                delay=stimulus_entry["delay"],
                duration=stimulus_entry["duration"],
                dt=stimulus_entry.get("dt", 0.25),
                tau=stimulus_entry["tau"],
                sigma=stimulus_entry["sigma"],
                mean=stimulus_entry["mean"],
                seed=stimulus_entry.get("random_seed", None),
                mode=ClampMode(stimulus_entry.get("input_type", "current_clamp").lower()),
                reversal=stimulus_entry.get("reversal", 0.0)
            )
        elif pattern == Pattern.RELATIVE_ORNSTEIN_UHLENBECK:
            return RelativeOrnsteinUhlenbeck(
                target=stimulus_entry["node_set"],
                delay=stimulus_entry["delay"],
                duration=stimulus_entry["duration"],
                dt=stimulus_entry.get("dt", 0.25),
                tau=stimulus_entry["tau"],
                mean_percent=stimulus_entry["mean_percent"],
                sd_percent=stimulus_entry["sd_percent"],
                seed=stimulus_entry.get("random_seed", None),
                mode=ClampMode(stimulus_entry.get("input_type", "current_clamp").lower()),
                reversal=stimulus_entry.get("reversal", 0.0)
            )
        else:
            raise ValueError(f"Unknown pattern {pattern}")


@dataclass(frozen=True, config=dict(extra="forbid"))
class Noise(Stimulus):
    mean_percent: float
    variance: float


@dataclass(frozen=True, config=dict(extra="forbid"))
class Hyperpolarizing(Stimulus):
    ...


@dataclass(frozen=True, config=dict(extra="forbid"))
class Pulse(Stimulus):
    amp_start: float
    width: float
    frequency: float


@dataclass(frozen=True, config=dict(extra="forbid"))
class RelativeLinear(Stimulus):
    percent_start: float


@dataclass(frozen=True, config=dict(extra="forbid"))
class SynapseReplay(Stimulus):
    spike_file: str

    @field_validator("spike_file")
    @classmethod
    def spike_file_exists(cls, v):
        if not Path(v).exists():
            raise ValueError(f"spike_file {v} does not exist")
        return v


@dataclass(frozen=True, config=dict(extra="forbid"))
class ShotNoise(Stimulus):
    rise_time: float
    decay_time: float
    rate: float
    amp_mean: float
    amp_var: float
    dt: float = 0.25
    seed: Optional[int] = None
    mode: ClampMode = ClampMode.CURRENT
    reversal: float = 0.0

    @field_validator("decay_time")
    @classmethod
    def decay_time_gt_rise_time(cls, v, values):
        if v <= values.data["rise_time"]:
            raise ValueError("decay_time must be greater than rise_time")
        return v


@dataclass(frozen=True, config=dict(extra="forbid"))
class RelativeShotNoise(Stimulus):
    rise_time: float
    decay_time: float
    mean_percent: float
    sd_percent: float
    amp_cv: float
    dt: float = 0.25
    seed: Optional[int] = None
    mode: ClampMode = ClampMode.CURRENT
    reversal: float = 0.0

    @field_validator("decay_time")
    @classmethod
    def decay_time_gt_rise_time(cls, v, values):
        if v <= values.data["rise_time"]:
            raise ValueError("decay_time must be greater than rise_time")
        return v


@dataclass(frozen=True, config=dict(extra="forbid"))
class OrnsteinUhlenbeck(Stimulus):
    tau: float
    sigma: PositiveFloat
    mean: float
    dt: float = 0.25
    seed: Optional[int] = None
    mode: ClampMode = ClampMode.CURRENT
    reversal: float = 0.0

    @field_validator("mean")
    @classmethod
    def mean_in_range(cls, v, values):
        if v < 0 and abs(v) > 2 * values.data["sigma"]:
            warnings.warn(
                "mean is outside of range [0, 2*sigma],",
                " ornstein uhlenbeck signal is mostly zero.",
            )
        return v


@dataclass(frozen=True, config=dict(extra="forbid"))
class RelativeOrnsteinUhlenbeck(Stimulus):
    tau: float
    mean_percent: float
    sd_percent: float
    dt: float = 0.25
    seed: Optional[int] = None
    mode: ClampMode = ClampMode.CURRENT
    reversal: float = 0.0
