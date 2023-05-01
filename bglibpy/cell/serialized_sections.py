"""Module that allows morphology sections to be accessed from an array by index."""
import warnings
import neuron

warnings.filterwarnings("once", category=UserWarning, module=__name__)


class SerializedSections:

    def __init__(self, cell):
        self.isec2sec = {}
        n = cell.nSecAll

        for index, sec in enumerate(cell.all, start=1):
            v_value = sec(0.0001).v
            if v_value >= n:
                print(f"{sec.name()} v(1)={sec(1).v} n3d()={sec.n3d()}")
                raise ValueError("Error: failure in mk2_isec2sec()")

            if v_value < 0:
                warnings.warn(
                    f"[Warning] SerializedSections: v(0.0001) < 0. index={index} v()={v_value}")
            else:
                self.isec2sec[int(v_value)] = neuron.h.SectionRef(sec=sec)
