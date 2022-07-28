"""Unit tests for the circuit_access module."""

from pathlib import Path

import pytest

import bglibpy
from bglibpy.circuit import CircuitAccess
from bglibpy.exceptions import BGLibPyError


parent_dir = Path(__file__).resolve().parent.parent


def test_non_existing_circuit_config():
    with pytest.raises(FileNotFoundError):
        CircuitAccess(str(parent_dir / "examples" / "non_existing_circuit_config"))


class TestCircuitAccess:

    def setup(self):
        conf_pre_path = parent_dir / "examples" / "sim_twocell_all"

        # make the paths absolute
        modified_conf = bglibpy.tools.blueconfig_append_path(
            conf_pre_path / "BlueConfigWithConditions", conf_pre_path
        )
        self.circuit_access = CircuitAccess(modified_conf)

    def test_use_mecombo_tsv(self):
        assert not self.circuit_access.use_mecombo_tsv

    def test_is_cell_target(self):
        assert self.circuit_access.is_cell_target(target="a1", gid=1)

    def test_is_group_target(self):
        assert self.circuit_access.is_group_target(target="Mosaic")

    def test_get_target_gids(self):
        assert self.circuit_access._get_target_gids(target="Mosaic") == {1, 2}

    def test_target_has_gid(self):
        assert self.circuit_access.target_has_gid(target="Mosaic", gid=1)

    def test_fetch_gid_cell_info(self):
        res = self.circuit_access.fetch_gid_cell_info(gid=1)
        assert res.etype == "cADpyr"

        with pytest.raises(BGLibPyError):
            self.circuit_access.fetch_gid_cell_info(gid=99999999)

    def test_fetch_mecombo_name(self):
        res = self.circuit_access.fetch_mecombo_name(gid=1)
        assert res == "cADpyr232_L5_TTPC1_5_dend-C220197A-P2_axon-C060110A3_-_Clone_2"

    def test_fetch_emodel_name(self):
        res = self.circuit_access.fetch_emodel_name(gid=1)
        assert res == "cADpyr232_L5_TTPC1_5_dend-C220197A-P2_axon-C060110A3_-_Clone_2"

    def test_fetch_morph_name(self):
        res = self.circuit_access.fetch_morph_name(gid=1)
        assert res == "dend-C220197A-P2_axon-C060110A3_-_Clone_2"

    def test_fetch_mini_frequencies(self):
        res = self.circuit_access.fetch_mini_frequencies(gid=1)
        assert res == (None, None)

    def test_node_properties_available(self):
        assert not self.circuit_access.node_properties_available

    def test_condition_parameters_dict(self):
        res = self.circuit_access.condition_parameters_dict()
        assert res["cao_CR_GluSynapse"] == 1.25

    def test_connection_entries(self):
        assert self.circuit_access.connection_entries == []

    def test_is_glusynapse_used(self):
        assert not self.circuit_access.is_glusynapse_used

    def test_get_gids_of_mtypes(self):
        res = self.circuit_access.get_gids_of_mtypes(mtypes=["L5_TTPC1", "L6_TPC_L1"])
        assert res == {1, 2}

    def test_get_gids_of_targets(self):
        res = self.circuit_access.get_gids_of_targets(targets=["Mosaic", "Excitatory"])
        assert res == {1, 2}

    def test_get_morph_dir_and_extension(self):
        f_dir, f_ext = self.circuit_access.get_morph_dir_and_extension()
        assert Path(f_dir).stem == "ascii"
        assert f_ext == "asc"

    def test_morph_dir(self):
        res = self.circuit_access.morph_dir
        assert Path(res).stem == "ascii"

    def test_morph_extension(self):
        res = self.circuit_access.morph_extension
        assert res == "asc"

    def test_morph_filename(self):
        res = self.circuit_access.morph_filename(gid=1)
        assert res == "dend-C220197A-P2_axon-C060110A3_-_Clone_2.asc"

    def test_emodels_dir(self):
        res = self.circuit_access.emodels_dir
        assert Path(res).stem == "ccells"

    def test_emodel_path(self):
        res = self.circuit_access.emodel_path(gid=1)
        fname = Path(res).name
        assert fname == "cADpyr232_L5_TTPC1_5_dend-C220197A-P2_axon-C060110A3_-_Clone_2.hoc"

    def test_extracellular_calcium(self):
        assert self.circuit_access.extracellular_calcium == 1.25
