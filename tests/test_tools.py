"""Unit tests for tools.py"""

import json
from pathlib import Path

import numpy as np
import pytest

import bluecellulab
from bluecellulab.circuit.circuit_access import EmodelProperties
from bluecellulab.tools import NumpyEncoder, Singleton, template_accepts_cvode

script_dir = Path(__file__).parent


@pytest.mark.v5
def test_calculate_SS_voltage_subprocess():
    """Tools: Test calculate_SS_voltage"""
    SS_voltage = bluecellulab.calculate_SS_voltage_subprocess(
        template_path=script_dir / "examples/cell_example1/test_cell.hoc",
        morphology_path=script_dir / "examples/cell_example1",
        template_format="v5",
        emodel_properties=None,
        step_level=0)
    assert abs(SS_voltage - -73.9235504304) < 0.001

    SS_voltage_stoch = bluecellulab.calculate_SS_voltage_subprocess(
        template_path=script_dir / "examples/cell_example2/test_cell.hoc",
        morphology_path=script_dir / "examples/cell_example2",
        template_format="v5",
        emodel_properties=None,
        step_level=0)
    assert abs(SS_voltage_stoch - -73.9235504304) < 0.001


def test_singleton():
    """Make sure only 1 object gets created in a singleton."""

    class TestClass(metaclass=Singleton):
        """Class to test Singleton object creation."""

        n_init_calls = 0

        def __init__(self):
            print("I'm called but not re-instantiated")
            TestClass.n_init_calls += 1

    test_obj1 = TestClass()
    test_obj2 = TestClass()
    test_objs = [TestClass() for _ in range(10)]

    assert test_obj1 is test_obj2
    assert id(test_obj1) == id(test_obj2)

    assert len(set(test_objs)) == 1

    assert TestClass.n_init_calls == 12


def test_numpy_encoder():
    """Tools: Test NumpyEncoder"""
    assert json.dumps(np.int32(1), cls=NumpyEncoder) == "1"
    assert json.dumps(np.float32(1.2), cls=NumpyEncoder)[0:3] == "1.2"
    assert json.dumps(np.array([1, 2, 3]), cls=NumpyEncoder) == "[1, 2, 3]"
    assert json.dumps(np.array([1.2, 2.3, 3.4]), cls=NumpyEncoder) == "[1.2, 2.3, 3.4]"
    assert (
        json.dumps(np.array([True, False, True]), cls=NumpyEncoder)
        == "[true, false, true]"
    )


def test_detect_spike():
    """Unit test for detect_spike."""

    # Case where there is a spike
    voltage_with_spike = np.array([-80, -70, -50, -10, -60, -80])
    assert bluecellulab.detect_spike(voltage_with_spike) is True

    # Case where there is no spike
    voltage_without_spike = np.array([-80, -70, -60, -50, -60, -80])
    assert bluecellulab.detect_spike(voltage_without_spike) is False

    # Edge case where the voltage reaches exactly -20 mV but does not surpass it
    voltage_at_edge = np.array([-80, -70, -60, -20, -60, -80])
    assert bluecellulab.detect_spike(voltage_at_edge) is False

    # Test with an empty array
    voltage_empty = np.array([])
    assert bluecellulab.detect_spike(voltage_empty) is False


@pytest.mark.v6
class TestOnSonataCell:
    def setup_method(self):
        self.template_name = (
            script_dir
            / "examples/circuit_sonata_quick_scx/components/hoc/cADpyr_L2TPC.hoc"
        )
        self.morphology_path = (
            script_dir
            / "examples/circuit_sonata_quick_scx/components/morphologies/asc/rr110330_C3_idA.asc"
        )
        self.template_format = "v6"
        self.emodel_properties = EmodelProperties(
            threshold_current=0.03203125, holding_current=-0.11, AIS_scaler=1.11
        )

    def test_detect_hyp_current(self):
        """Unit test detect_hyp_current."""
        target_voltage = -85
        hyp_current = bluecellulab.detect_hyp_current(
            template_path=self.template_name,
            morphology_path=self.morphology_path,
            template_format=self.template_format,
            emodel_properties=self.emodel_properties,
            target_voltage=target_voltage,
        )
        assert hyp_current == pytest.approx(-0.0009765625)

    def test_calculate_input_resistance(self):
        """Unit test calculate_input_resistance."""
        input_resistance = bluecellulab.calculate_input_resistance(
            template_path=self.template_name,
            morphology_path=self.morphology_path,
            template_format=self.template_format,
            emodel_properties=self.emodel_properties,
        )

        assert input_resistance == pytest.approx(375.8880669033445)

    def test_calculate_SS_voltage(self):
        """Unit test calculate_SS_voltage."""
        step_level = 0
        SS_voltage = bluecellulab.calculate_SS_voltage(
            template_path=self.template_name,
            morphology_path=self.morphology_path,
            template_format=self.template_format,
            emodel_properties=self.emodel_properties,
            step_level=step_level,
        )
        assert SS_voltage == pytest.approx(-85.13127127240696)

    def test_calculate_SS_voltage_subprocess_exception(self):
        """Unit test calculate_SS_voltage_subprocess."""
        step_level = 2
        with pytest.raises(bluecellulab.UnsteadyCellError):
            _ = bluecellulab.calculate_SS_voltage_subprocess(
                template_path=self.template_name,
                morphology_path=self.morphology_path,
                template_format=self.template_format,
                emodel_properties=self.emodel_properties,
                step_level=step_level,
                check_for_spiking=True,
            )

    def test_template_accepts_cvode(self):
        """Unit test for template_accepts_cvode."""
        assert template_accepts_cvode(self.template_name) is True

        mock_template_with_stochkv = script_dir / "examples/cell_example2/test_cell.hoc"
        assert template_accepts_cvode(mock_template_with_stochkv) is False

    def test_detect_spike_step(self):
        """Unit test for detect_spike_step."""
        hyp_level = -2
        inj_start = 100  # some start time for the current injection
        inj_stop = 200  # some stop time for the current injection
        step_level = 4  # some current level for the step

        spike_occurred = bluecellulab.detect_spike_step(
            template_path=self.template_name,
            morphology_path=self.morphology_path,
            template_format=self.template_format,
            emodel_properties=self.emodel_properties,
            hyp_level=hyp_level,
            inj_start=inj_start,
            inj_stop=inj_stop,
            step_level=step_level,
        )
        assert spike_occurred is True

    def test_detect_spike_step_subprocess(self):
        """Unit test for detect_spike_step_subprocess."""
        hyp_level = -2
        inj_start = 100
        inj_stop = 200
        step_level = 4

        spike_occurred = bluecellulab.detect_spike_step_subprocess(
            template_path=self.template_name,
            morphology_path=self.morphology_path,
            template_format=self.template_format,
            emodel_properties=self.emodel_properties,
            hyp_level=hyp_level,
            inj_start=inj_start,
            inj_stop=inj_stop,
            step_level=step_level,
        )
        assert spike_occurred is True


@pytest.mark.v6
class TestOnSonataCircuit:
    def setup_method(self):
        self.circuit_path = (
            script_dir
            / "examples/sim_quick_scx_sonata_multicircuit/simulation_config_noinput.json"
        )
        self.cell_id = ("NodeA", 0)

    def test_holding_current(self):
        """Unit test for holding_current on a SONATA circuit."""
        v_hold = -70  # arbitrary holding voltage for the test
        i_hold, v_control = bluecellulab.holding_current(
            v_hold, cell_id=self.cell_id, circuit_path=self.circuit_path
        )

        assert i_hold == pytest.approx(0.0020779234)
        assert v_control == pytest.approx(v_hold)

    def test_holding_current_subprocess(self):
        """Unit test for holding_current_subprocess on a SONATA circuit."""
        v_hold = -80
        ssim = bluecellulab.SSim(self.circuit_path)
        cell_id = bluecellulab.circuit.node_id.create_cell_id(self.cell_id)
        cell_kwargs = ssim.fetch_cell_kwargs(cell_id)
        i_hold, v_control = bluecellulab.holding_current_subprocess(
            v_hold, enable_ttx=True, cell_kwargs=cell_kwargs
        )
        assert i_hold == pytest.approx(-0.03160848349)
        assert v_control == pytest.approx(v_hold)
