"""Type aliases used within the package."""

from typing_extensions import TypeAlias
from neuron import h as hoc_type

HocObjectType: TypeAlias = hoc_type   # until NEURON is typed, most NEURON types are this
NeuronRNG: TypeAlias = hoc_type
NeuronVector: TypeAlias = hoc_type
NeuronSection: TypeAlias = hoc_type
