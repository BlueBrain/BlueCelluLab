"""Back Propagating Action Potential."""


from itertools import islice
from bluecellulab import Cell, neuron
from bluecellulab.simulation import Simulation
from bluecellulab.stimuli import Hyperpolarizing


def get_peak_voltage(voltage, stim_start: int, stim_end: int) -> float:
    """Get the peak voltage from a trace."""
    return max(voltage[stim_start:stim_end])


def get_peak_index(voltage, stim_start: int, stim_end: int) -> int:
    """Get the peak voltage from a trace."""
    return voltage[stim_start:stim_end].argmax() + stim_start


class BPAP:

    def __init__(self, cell: Cell) -> None:
        self.cell = cell
        self.dt = 0.025
        self.stim_start = 1000
        self.stim_duration = 1

    @property
    def start_index(self) -> int:
        """Get the index of the start of the stimulus."""
        return int(self.stim_start / self.dt)

    @property
    def end_index(self) -> int:
        """Get the index of the end of the stimulus."""
        return int((self.stim_start + self.stim_duration) / self.dt)

    def select_dendritic_recordings(self, all_recordings : dict[str, float]) -> dict[str, float]:
        """Select dendritic recordings from all recordings."""
        res = {}
        for key, value in all_recordings.items():
            if "myelin" in key:
                continue
            elif "axon" in key:
                continue
            else:
                res[key] = value
        return res

    def run(self, duration: float, amplitude: float) -> None:
        """Apply depolarization and hyperpolarization at the same time."""
        sim = Simulation()
        self.cell.record_dt
        sim.run(10)
        sim.add_cell(self.cell)
        self.cell.add_allsections_voltagerecordings()
        self.cell.add_step(start_time=self.stim_start, stop_time=self.stim_start+self.stim_duration, level=amplitude)
        hyperpolarizing = Hyperpolarizing("single-cell", delay=self.stim_start, duration=self.stim_duration)
        self.cell.add_replay_hypamp(hyperpolarizing)
        sim.run(duration, dt=self.dt, cvode=False)

    def voltage_attenuation(self) -> dict[str, float]:
        """Return soma peak voltage across all sections."""
        all_recordings = self.cell.get_allsections_voltagerecordings()
        dendritic_recordings = self.select_dendritic_recordings(all_recordings)
        soma_key = [key for key in dendritic_recordings.keys() if key.endswith("soma[0]")][0]
        soma_voltage = dendritic_recordings[soma_key]
        soma_peak_index = get_peak_index(soma_voltage, self.start_index, self.end_index)
        res = {}
        for key, voltage in dendritic_recordings.items():
            peak_index_volt = voltage[soma_peak_index]
            res[key] = peak_index_volt
        return res

    def peak_delays(self) -> dict[str, float]:
        """Return the peak delays in each section."""
        all_recordings = self.cell.get_allsections_voltagerecordings()
        dendritic_recordings = self.select_dendritic_recordings(all_recordings)
        soma_key = [key for key in dendritic_recordings.keys() if key.endswith("soma[0]")][0]
        soma_voltage = dendritic_recordings[soma_key]
        soma_peak_index = get_peak_index(soma_voltage, self.start_index, self.end_index)
        res = {}
        for key, voltage in dendritic_recordings.items():
            peak_index = get_peak_index(voltage, self.start_index, self.end_index)
            index_delay = peak_index - soma_peak_index
            time_delay = index_delay * self.dt
            res[key] = time_delay
        return res

    def distances_to_soma(self) -> dict[str, float]:
        """Return the distance to the soma for each section."""
        res = {}
        all_recordings = self.cell.get_allsections_voltagerecordings()
        dendritic_recordings = self.select_dendritic_recordings(all_recordings)
        soma = self.cell.soma
        for key in dendritic_recordings.keys():
            section_name = key.rsplit(".")[-1].split("[")[0]  # e.g. "dend"
            section_idx = int(key.rsplit(".")[-1].split("[")[1].split("]")[0])  # e.g. 0
            attribute_value = getattr(self.cell.cell.getCell(), section_name)
            section = next(islice(attribute_value, section_idx, None))
            # section e.g. cADpyr_L2TPC_bluecellulab_x[0].dend[0]
            res[key] = neuron.h.distance(soma(0.5),section(0.5))
        return res
