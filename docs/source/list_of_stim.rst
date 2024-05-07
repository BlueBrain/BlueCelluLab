
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

   step = stim_factory.step(pre_delay=20, duration=80, post_delay=20, amplitude=70)
   step.plot()


Multiple step stimuli with different amplitudes.

.. plot::
   :context: close-figs
   :include-source: True

   step1 = stim_factory.step(pre_delay=50, duration=100, post_delay=50, amplitude=60)
   step2 = stim_factory.step(pre_delay=50, duration=100, post_delay=50, amplitude=80)
   step3 = stim_factory.step(pre_delay=50, duration=100, post_delay=50, amplitude=100)
   step4 = stim_factory.step(pre_delay=50, duration=100, post_delay=50, amplitude=120)

   # Create a figure and an Axes object
   fig, ax = plt.subplots()

   # Plot the step functions on the same axes
   step1.plot(ax=ax, color='red')
   step2.plot(ax=ax, color='green')
   step3.plot(ax=ax, color='blue')
   step4.plot(ax=ax, color='orange')

   plt.show()


Ramp Stimulus
~~~~~~~~~~~~~

**Definition**: A ramp stimulus gradually changes from an initial value to a final value over a defined period. This allows for the examination of systems' responses to gradual input changes.

.. plot::
   :context: close-figs
   :include-source: True

   ramp = stim_factory.ramp(pre_delay=20, duration=160, post_delay=70, amplitude=0.5)

   ramp.plot()



APWaveform
~~~~~~~~~~

**Definition**: The action potential waveform is a step stimulus with a defined amplitude and duration.

.. plot::
   :context: close-figs
   :include-source: True

   ap_waveform = stim_factory.ap_waveform(threshold_current=0.5)
   ap_waveform.plot()


IDRest
~~~~~~

**Definition**: The IDRest stimulus is a step stimulus with a defined amplitude and duration.

.. plot::
   :context: close-figs
   :include-source: True

   id_rest = stim_factory.idrest(threshold_current=0.5)
   id_rest.plot()

IV
~~

**Definition**: The IV stimulus is a step stimulus with a defined amplitude (negative value) and duration.

.. plot::
   :context: close-figs
   :include-source: True

   iv = stim_factory.iv(threshold_current=0.5)
   iv.plot()

FirePattern
~~~~~~~~~~~~~

**Definition**: The fire pattern stimulus is a step stimulus with a defined amplitude and duration.

.. plot::
   :context: close-figs
   :include-source: True

   fire_pattern = stim_factory.fire_pattern(threshold_current=0.5)
   fire_pattern.plot()


.. plot::
   :context: close-figs
   :include-source: False

   # Testing
   assert len(ramp.current) > 0
   assert len(step.current) > 0

PosCheops
~~~~~~~~~

**Definition**: The PosCheops stimulus is a sequence of increasing and decreasing ramps with a positive amplitude.

.. plot::
   :context: close-figs
   :include-source: True

   pos_cheops = stim_factory.pos_cheops(threshold_current=0.5)
   pos_cheops.plot()


NegCheops
~~~~~~~~~

**Definition**: The NegCheops stimulus is a sequence of increasing and decreasing ramps with a negative amplitude.

.. plot::
   :context: close-figs
   :include-source: True

   neg_cheops = stim_factory.neg_cheops(threshold_current=0.5)
   neg_cheops.plot()
