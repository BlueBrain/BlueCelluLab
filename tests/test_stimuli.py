"""Unit tests for the stimuli module."""

from pydantic import ValidationError
import pytest
from bluecellulab.stimuli import SynapseReplay


def test_synapse_replay_validator():
    """Assures synapse replay's validator fails."""
    with pytest.raises(ValidationError):
        synapse_replay = SynapseReplay(
            target="target1",
            source="source1",
            delay=0,
            duration=3000,
            spike_file="file_that_does_not_exist.dat",
        )
