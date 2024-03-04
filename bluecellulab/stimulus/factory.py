from __future__ import annotations
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import numpy as np


class Stimulus(ABC):
    def __init__(self, dt: float) -> None:
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

    @property
    def stimulus_time(self) -> float:
        return len(self) * self.dt

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
        # Create an array for the entire duration
        full_time = np.arange(0, duration, self.dt)
        full_current = np.zeros_like(full_time)

        # Replace the corresponding values with self.time and self.current
        indices = (self.time / self.dt).astype(int)
        full_current[indices] = self.current

        ax.plot(full_time, full_current, **kwargs)
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Current (nA)")
        ax.set_title(self.__class__.__name__)
        ax.set_xlim(0, duration)
        return ax

    def __add__(self, other: Stimulus) -> CombinedStimulus:
        """Override + operator to concatenate Stimulus objects."""
        if self.dt != other.dt:
            raise ValueError("Stimulus objects must have the same dt to be concatenated")
        # shift other time
        other_time = other.time + self.time[-1] + self.dt
        combined_time = np.concatenate([self.time, other_time])
        # Concatenate the current arrays
        combined_current = np.concatenate([self.current, other.current])
        return CombinedStimulus(self.dt, combined_time, combined_current)


class CombinedStimulus(Stimulus):
    """Represents the Stimulus created by combining multiple stimuli."""
    def __init__(self, dt: float, time: np.ndarray, current: np.ndarray) -> None:
        super().__init__(dt)
        self._time = time
        self._current = current

    @property
    def time(self) -> np.ndarray:
        return self._time

    @property
    def current(self) -> np.ndarray:
        return self._current


class EmptyStimulus(Stimulus):
    """Represents empty stimulus (all zeros) that has no impact on the cell.

    This is required by some Stimuli that expect the cell to rest.
    """
    def __init__(self, dt: float, duration: float) -> None:
        super().__init__(dt)
        self.duration = duration

    @property
    def time(self) -> np.ndarray:
        return np.arange(0.0, self.duration, self.dt)

    @property
    def current(self) -> np.ndarray:
        return np.zeros_like(self.time)


class Step(Stimulus):
    def __init__(self, dt: float, start: float, end: float, amplitude: float) -> None:
        super().__init__(dt)
        self.start = start
        self.end = end
        self.amplitude = amplitude

    @classmethod
    def threshold_based(
        cls,
        dt: float,
        threshold_current: float,
        threshold_percentage: float,
        start: float,
        end: float,
        post_wait: float,
    ) -> CombinedStimulus:
        amplitude = threshold_current * threshold_percentage / 100
        res = cls(dt, start, end, amplitude) + EmptyStimulus(dt, duration=post_wait)
        return res

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

    def ap_waveform(
        self,
        threshold_current: float,
        threshold_percentage: float = 220.0,
        start: float = 250.0,
        end: float = 300.0,
        post_wait: float = 250.0,
    ) -> Stimulus:
        """Returns the APWaveform Stimulus object, a type of Step stimulus.

        Args:
            threshold_current: The threshold current of the Cell.
            threshold_percentage: Percentage of desired threshold_current amplification.
            start: The start time of the step.
            end: The end time of the step.
            post_wait: The time to wait after the end of the step.
        """
        return Step.threshold_based(
            self.dt, threshold_current, threshold_percentage, start, end, post_wait
        )

    def idrest(
        self,
        threshold_current: float,
        threshold_percentage: float = 200.0,
        start: float = 250.0,
        end: float = 1600.0,
        post_wait: float = 250.0,
    ) -> Stimulus:
        """Returns the IDRest Stimulus object, a type of Step stimulus.

        Args:
            threshold_current: The threshold current of the Cell.
            threshold_percentage: Percentage of desired threshold_current amplification.
            start: The start time of the step.
            end: The end time of the step.
            post_wait: The time to wait after the end of the step.
        """
        return Step.threshold_based(
            self.dt, threshold_current, threshold_percentage, start, end, post_wait
        )

    def iv(
        self,
        threshold_current: float,
        threshold_percentage: float = -40.0,
        start: float = 250.0,
        end: float = 3250.0,
        post_wait: float = 250.0,
    ) -> Stimulus:
        """Returns the IV Stimulus object, a type of Step stimulus.

        Args:
            threshold_current: The threshold current of the Cell.
            threshold_percentage: Percentage of desired threshold_current amplification.
            start: The start time of the step.
            end: The end time of the step.
            post_wait: The time to wait after the end of the step.
        """
        return Step.threshold_based(
            self.dt, threshold_current, threshold_percentage, start, end, post_wait
        )

    def fire_pattern(
        self,
        threshold_current: float,
        threshold_percentage: float = 200.0,
        start: float = 250.0,
        end: float = 3850.0,
        post_wait: float = 250.0,
    ) -> Stimulus:
        """Returns the FirePattern Stimulus object, a type of Step stimulus.

        Args:
            threshold_current: The threshold current of the Cell.
            threshold_percentage: Percentage of desired threshold_current amplification.
            start: The start time of the step.
            end: The end time of the step.
            post_wait: The time to wait after the end of the step.
        """
        return Step.threshold_based(
            self.dt, threshold_current, threshold_percentage, start, end, post_wait
        )
