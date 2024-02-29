
List of Stimuli
===============

.. plot::
   :context: close-figs
   :include-source: False

   import seaborn as sns
   sns.set_style("darkgrid")
   sns.set(rc={'figure.figsize':(10, 6)})


This section provides an overview of the various stimuli available for use within BlueCelluLab. We showcase how each stimulus can be generated and visualized.

Setting Up
----------

The stimuli visualization is controlled through a `StimulusFactory` object, which is configured with a time step (`dt`) parameter. This allows for precise temporal resolution in stimuli generation.

.. plot::
   :context: close-figs
   :include-source: True

   from bluecellulab.stimulus import StimulusFactory
   stim_factory = StimulusFactory(dt=0.1)

Stimulus Types
--------------

Step Stimulus
~~~~~~~~~~~~~

**Definition**: The step stimulus represents a sudden change in value at a specified point in time, maintaining this new value until the end of the stimulus period.

.. plot::
   :context: close-figs
   :include-source: True

   step = stim_factory.step(start=20, end=180, amplitude=70)
   step.plot()


Ramp Stimulus
~~~~~~~~~~~~~

**Definition**: A ramp stimulus gradually changes from an initial value to a final value over a defined period. This allows for the examination of systems' responses to gradual input changes.

.. plot::
   :context: close-figs
   :include-source: True

   ramp = stim_factory.ramp(start=20, end=180, amplitude_start=0.25, amplitude_end=0.5)

   ramp.plot()


.. plot::
   :context: close-figs
   :include-source: False

   # Testing
   assert len(ramp.current) > 0
   assert len(step.current) > 0
