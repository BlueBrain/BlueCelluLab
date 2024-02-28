from __future__ import annotations
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import numpy as np


class Stimulus(ABC):
    def __init__(self, dt) -> None:
        self.dt = dt

    @property
    @abstractmethod
    def time(self) -> np.ndarray:
        """Time values of the stimulus."""
        ...

    @property
    @abstractmethod
    def current(self) -> np.ndarray:
        """Current values of the stimulus."""
        ...

    def __len__(self) -> int:
        return len(self.time)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(dt={self.dt})"

    def plot(self, ax=None, **kwargs):
        if ax is None:
            ax = plt.gca()
        ax.plot(self.time, self.current, **kwargs)
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Current (nA)")
        ax.set_title(self.__class__.__name__)
        return ax

    def plot_during_simulation(self, duration: float, ax=None, **kwargs):
        if ax is None:
            ax = plt.gca()
        ax.plot(self.time, self.current, **kwargs)
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Current (nA)")
        ax.set_title(self.__class__.__name__)
        ax.set_xlim(0, duration)
        return ax


class Step(Stimulus):
    def __init__(self, dt: float, start: float, end: float, amplitude: float) -> None:
        super().__init__(dt)
        self.start = start
        self.end = end
        self.amplitude = amplitude

    @property
    def time(self) -> np.ndarray:
        return np.arange(self.start, self.end, self.dt)

    @property
    def current(self) -> np.ndarray:
        return np.full_like(self.time, self.amplitude)


class Ramp(Stimulus):
    def __init__(
        self,
        dt: float,
        start: float,
        end: float,
        amplitude_start: float,
        amplitude_end: float,
    ) -> None:
        super().__init__(dt)
        self.start = start
        self.end = end
        self.amplitude_start = amplitude_start
        self.amplitude_end = amplitude_end

    @property
    def time(self) -> np.ndarray:
        return np.arange(self.start, self.end, self.dt)

    @property
    def current(self) -> np.ndarray:
        return np.linspace(self.amplitude_start, self.amplitude_end, len(self.time))


class StimulusFactory:
    def __init__(self, dt: float):
        self.dt = dt

    def step(self, start: float, end: float, amplitude: float) -> Stimulus:
        return Step(self.dt, start, end, amplitude)

    def ramp(
        self, start: float, end: float, amplitude_start: float, amplitude_end: float
    ) -> Stimulus:
        return Ramp(self.dt, start, end, amplitude_start, amplitude_end)
