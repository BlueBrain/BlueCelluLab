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
            raise ValueError(
                "Stimulus objects must have the same dt to be concatenated"
            )
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

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Stimulus):
            return NotImplemented
        else:
            return (
                np.allclose(self.time, other.time)
                and np.allclose(self.current, other.current)
                and self.dt == other.dt
            )


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


class Empty(Stimulus):
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


class Flat(Stimulus):
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
    def __init__(
        self, dt: float, duration: float, amplitude_start: float, amplitude_end: float
    ) -> None:
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
        raise NotImplementedError(
            "This class cannot be instantiated directly. "
            "Please use the class methods 'amplitude_based' "
            "or 'threshold_based' to create objects."
        )

    @classmethod
    def amplitude_based(
        cls,
        dt: float,
        pre_delay: float,
        duration: float,
        post_delay: float,
        amplitude: float,
    ) -> CombinedStimulus:
        """Create a Step stimulus from given time events and amplitude.

        Args:
            dt: The time step of the stimulus.
            pre_delay: The delay before the start of the step.
            duration: The duration of the step.
            post_delay: The time to wait after the end of the step.
            amplitude: The amplitude of the step.
        """
        return (
            Empty(dt, duration=pre_delay)
            + Flat(dt, duration=duration, amplitude=amplitude)
            + Empty(dt, duration=post_delay)
        )

    @classmethod
    def threshold_based(
        cls,
        dt: float,
        pre_delay: float,
        duration: float,
        post_delay: float,
        threshold_current: float,
        threshold_percentage: float,
    ) -> CombinedStimulus:
        """Creates a Step stimulus with respect to the threshold current.

        Args:

            dt: The time step of the stimulus.
            pre_delay: The delay before the start of the step.
            duration: The duration of the step.
            post_delay: The time to wait after the end of the step.
            threshold_current: The threshold current of the Cell.
            threshold_percentage: Percentage of desired threshold_current amplification.
        """
        amplitude = threshold_current * threshold_percentage / 100
        res = cls.amplitude_based(
            dt,
            pre_delay=pre_delay,
            duration=duration,
            post_delay=post_delay,
            amplitude=amplitude,
        )
        return res


class Ramp(Stimulus):

    def __init__(self):
        raise NotImplementedError(
            "This class cannot be instantiated directly. "
            "Please use the class methods 'amplitude_based' "
            "or 'threshold_based' to create objects."
        )

    @classmethod
    def amplitude_based(
        cls,
        dt: float,
        pre_delay: float,
        duration: float,
        post_delay: float,
        amplitude: float,
    ) -> CombinedStimulus:
        """Create a Ramp stimulus from given time events and amplitudes.

        Args:
            dt: The time step of the stimulus.
            pre_delay: The delay before the start of the ramp.
            duration: The duration of the ramp.
            post_delay: The time to wait after the end of the ramp.
            amplitude: The final amplitude of the ramp.
        """
        return (
            Empty(dt, duration=pre_delay)
            + Slope(
                dt,
                duration=duration,
                amplitude_start=0.0,
                amplitude_end=amplitude,
            )
            + Empty(dt, duration=post_delay)
        )

    @classmethod
    def threshold_based(
        cls,
        dt: float,
        pre_delay: float,
        duration: float,
        post_delay: float,
        threshold_current: float,
        threshold_percentage: float,
    ) -> CombinedStimulus:
        """Creates a Ramp stimulus with respect to the threshold current.

        Args:

            dt: The time step of the stimulus.
            pre_delay: The delay before the start of the ramp.
            duration: The duration of the ramp.
            post_delay: The time to wait after the end of the ramp.
            threshold_current: The threshold current of the Cell.
            threshold_percentage: Percentage of desired threshold_current amplification.
        """
        amplitude = threshold_current * threshold_percentage / 100
        res = cls.amplitude_based(
            dt,
            pre_delay=pre_delay,
            duration=duration,
            post_delay=post_delay,
            amplitude=amplitude,
        )
        return res


class StimulusFactory:
    def __init__(self, dt: float):
        self.dt = dt

    def step(
        self, pre_delay: float, duration: float, post_delay: float, amplitude: float
    ) -> Stimulus:
        return Step.amplitude_based(
            self.dt,
            pre_delay=pre_delay,
            duration=duration,
            post_delay=post_delay,
            amplitude=amplitude,
        )

    def ramp(
        self, pre_delay: float, duration: float, post_delay: float, amplitude: float
    ) -> Stimulus:
        return Ramp.amplitude_based(
            self.dt,
            pre_delay=pre_delay,
            duration=duration,
            post_delay=post_delay,
            amplitude=amplitude,
        )

    def ap_waveform(
        self, threshold_current: float, threshold_percentage: float = 220.0
    ) -> Stimulus:
        """Returns the APWaveform Stimulus object, a type of Step stimulus.

        Args:
            threshold_current: The threshold current of the Cell.
            threshold_percentage: Percentage of desired threshold_current amplification.
        """
        pre_delay = 250.0
        duration = 50.0
        post_delay = 250.0
        return Step.threshold_based(
            self.dt,
            pre_delay=pre_delay,
            duration=duration,
            post_delay=post_delay,
            threshold_current=threshold_current,
            threshold_percentage=threshold_percentage,
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
        pre_delay = 250.0
        duration = 1350.0
        post_delay = 250.0
        return Step.threshold_based(
            self.dt,
            pre_delay=pre_delay,
            duration=duration,
            post_delay=post_delay,
            threshold_current=threshold_current,
            threshold_percentage=threshold_percentage,
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
        pre_delay = 250.0
        duration = 3000.0
        post_delay = 250.0
        return Step.threshold_based(
            self.dt,
            pre_delay=pre_delay,
            duration=duration,
            post_delay=post_delay,
            threshold_current=threshold_current,
            threshold_percentage=threshold_percentage,
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
        pre_delay = 250.0
        duration = 3600.0
        post_delay = 250.0
        return Step.threshold_based(
            self.dt,
            pre_delay=pre_delay,
            duration=duration,
            post_delay=post_delay,
            threshold_current=threshold_current,
            threshold_percentage=threshold_percentage,
        )
