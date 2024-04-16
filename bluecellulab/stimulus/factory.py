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

    def __add__(self, other: Stimulus) -> CombinedStimulus:
        """Override + operator to concatenate Stimulus objects."""
        if self.dt != other.dt:
            raise ValueError("Stimulus objects must have the same dt to be concatenated")
        if len(self.time) == 0:
            return CombinedStimulus(other.dt, other.time, other.current)
        elif len(other.time) == 0:
            return CombinedStimulus(self.dt, self.time, self.current)
        else:
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


class FlatStimulus(Stimulus):
    def __init__(self, dt: float, duration: float, amplitude: float) -> None:
        super().__init__(dt)
        self.duration = duration
        self.amplitude = amplitude

    @property
    def time(self) -> np.ndarray:
        return np.arange(0.0, self.duration, self.dt)

    @property
    def current(self) -> np.ndarray:
        return np.full_like(self.time, self.amplitude)


class Slope(Stimulus):
    def __init__(self, dt: float, duration: float, amplitude_start: float, amplitude_end: float) -> None:
        super().__init__(dt)
        self.duration = duration
        self.amplitude_start = amplitude_start
        self.amplitude_end = amplitude_end

    @property
    def time(self) -> np.ndarray:
        return np.arange(0.0, self.duration, self.dt)

    @property
    def current(self) -> np.ndarray:
        return np.linspace(self.amplitude_start, self.amplitude_end, len(self.time))


class Step(Stimulus):

    def __init__(self):
        raise NotImplementedError("This class cannot be instantiated directly. "
                                  "Please use the class methods 'amplitude_based' "
                                  "or 'threshold_based' to create objects.")

    @classmethod
    def amplitude_based(
        cls,
        dt: float,
        start: float,
        end: float,
        post_wait: float,
        amplitude: float,
    ) -> CombinedStimulus:
        """Create a Step stimulus from given time events and amplitude.

        Args:
            dt: The time step of the stimulus.
            start: The start time of the step.
            end: The end time of the step.
            post_wait: The time to wait after the end of the step.
            amplitude: The amplitude of the step.
        """
        return (
            EmptyStimulus(dt, duration=start)
            + FlatStimulus(dt, duration=end - start, amplitude=amplitude)
            + EmptyStimulus(dt, duration=post_wait)
        )

    @classmethod
    def threshold_based(
        cls,
        dt: float,
        start: float,
        end: float,
        post_wait: float,
        threshold_current: float,
        threshold_percentage: float,
    ) -> CombinedStimulus:
        """Creates a Step stimulus with respect to the threshold current.

        Args:

            dt: The time step of the stimulus.
            start: The start time of the step.
            end: The end time of the step.
            post_wait: The time to wait after the end of the step.
            threshold_current: The threshold current of the Cell.
            threshold_percentage: Percentage of desired threshold_current amplification.
        """
        amplitude = threshold_current * threshold_percentage / 100
        res = cls.amplitude_based(dt, amplitude, start, end, post_wait)
        return res


class Ramp(Stimulus):

    def __init__(self):
        raise NotImplementedError("This class cannot be instantiated directly. "
                                  "Please use the class methods 'amplitude_based' "
                                  "or 'threshold_based' to create objects.")

    @classmethod
    def amplitude_based(
        cls,
        dt: float,
        start: float,
        end: float,
        post_wait: float,
        amplitude: float,
    ) -> CombinedStimulus:
        """Create a Ramp stimulus from given time events and amplitudes.

        Args:
            dt: The time step of the stimulus.
            start: The start time of the ramp.
            end: The end time of the ramp.
            post_wait: The time to wait after the end of the ramp.
            amplitude: The final amplitude of the ramp.
        """
        return (
            EmptyStimulus(dt, duration=start)
            + Slope(dt, duration=end - start, amplitude_start=0.0, amplitude_end=amplitude)
            + EmptyStimulus(dt, duration=post_wait)
        )

    @classmethod
    def threshold_based(
        cls,
        dt: float,
        start: float,
        end: float,
        post_wait: float,
        threshold_current: float,
        threshold_percentage: float,
    ) -> CombinedStimulus:
        """Creates a Ramp stimulus with respect to the threshold current.

        Args:

            dt: The time step of the stimulus.
            start: The start time of the ramp.
            end: The end time of the ramp.
            post_wait: The time to wait after the end of the ramp.
            threshold_current: The threshold current of the Cell.
            threshold_percentage: Percentage of desired threshold_current amplification.
        """
        amplitude = threshold_current * threshold_percentage / 100
        res = cls.amplitude_based(dt, amplitude, start, end, post_wait)
        return res


class StimulusFactory:
    def __init__(self, dt: float):
        self.dt = dt

    def step(self, start: float, end: float, post_wait: float, amplitude: float) -> Stimulus:
        return Step.amplitude_based(self.dt, start, end, post_wait, amplitude)

    def ramp(
        self, start: float, end: float, post_wait: float, amplitude: float
    ) -> Stimulus:
        return Ramp(self.dt, start, end, amplitude)

    def ap_waveform(
        self,
        threshold_current: float,
        threshold_percentage: float = 220.0
    ) -> Stimulus:
        """Returns the APWaveform Stimulus object, a type of Step stimulus.

        Args:
            threshold_current: The threshold current of the Cell.
            threshold_percentage: Percentage of desired threshold_current amplification.
        """
        start = 250.0  # the start time of the step
        end = 300.0  # the end time of the step
        post_wait = 250.0  # the time to wait after the end of the step
        return Step.threshold_based(
            self.dt, threshold_current, threshold_percentage, start, end, post_wait
        )

    def idrest(
        self,
        threshold_current: float,
        threshold_percentage: float = 200.0,
    ) -> Stimulus:
        """Returns the IDRest Stimulus object, a type of Step stimulus.

        Args:
            threshold_current: The threshold current of the Cell.
            threshold_percentage: Percentage of desired threshold_current amplification.
        """
        start = 250.0  # the start time of the step
        end = 1600.0
        post_wait = 250.0  # the time to wait after the end of the step
        return Step.threshold_based(
            self.dt, threshold_current, threshold_percentage, start, end, post_wait
        )

    def iv(
        self,
        threshold_current: float,
        threshold_percentage: float = -40.0,
    ) -> Stimulus:
        """Returns the IV Stimulus object, a type of Step stimulus.

        Args:
            threshold_current: The threshold current of the Cell.
            threshold_percentage: Percentage of desired threshold_current amplification.
        """
        start = 250.0  # the start time of the step
        end = 3250.0  # the end time of the step
        post_wait = 250.0  # the time to wait after the end of the step
        return Step.threshold_based(
            self.dt, threshold_current, threshold_percentage, start, end, post_wait
        )

    def fire_pattern(
        self,
        threshold_current: float,
        threshold_percentage: float = 200.0,
    ) -> Stimulus:
        """Returns the FirePattern Stimulus object, a type of Step stimulus.

        Args:
            threshold_current: The threshold current of the Cell.
            threshold_percentage: Percentage of desired threshold_current amplification.
        """
        start = 250.0  # the start time of the step
        end = 3850.0  # the end time of the step
        post_wait = 250.0  # the time to wait after the end of the step
        return Step.threshold_based(
            self.dt, threshold_current, threshold_percentage, start, end, post_wait
        )
