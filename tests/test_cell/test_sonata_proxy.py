"""Unit tests for cell.sonata_proxy."""
from pathlib import Path

from bluecellulab.cell import SonataProxy
from bluecellulab.circuit import SonataCircuitAccess
from bluecellulab.circuit.node_id import CellId
from bluecellulab.exceptions import MissingSonataPropertyError

from pytest import approx, raises

parent_dir = Path(__file__).resolve().parent.parent


hipp_circuit_with_projections = (
    parent_dir
    / "examples"
    / "sonata_unit_test_sims"
    / "projections"
    / "simulation_config.json"
)


class TestSonataProxy:

    def setup(self):
        circuit_access = SonataCircuitAccess(str(hipp_circuit_with_projections))
        cell_id = CellId("hippocampus_neurons", 1)
        self.sonata_proxy = SonataProxy(cell_id, circuit_access)

    def test_get_property(self):
        holding_current = self.sonata_proxy.get_property("@dynamics:holding_current")
        assert holding_current.iloc[0] == approx(-0.116351)

        threshold_current = self.sonata_proxy.get_property("@dynamics:threshold_current")
        assert threshold_current.iloc[0] == approx(0.332031)

    def test_get_input_resistance(self):
        with raises(MissingSonataPropertyError):
            self.sonata_proxy.get_input_resistance()
