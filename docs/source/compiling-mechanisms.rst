Compiling Mechanisms
====================================

Welcome to a brief tutorial on compiling mechanisms in BlueCelluLab!

In order to facilitate smooth simulations with this tool, one must navigate through specific steps, especially given its dependency on the NEURON simulator. This guide aims to detail the necessary steps and considerations for compiling neuron mechanisms in BlueCelluLab.

Importance of the Working Directory
-----------------------------------

It's vital to underscore that, unlike other Python packages, the working directory you are in when you import BlueCelluLab significantly influences the output. This peculiar behavior is attributed to its utilization of the NEURON simulator.

When importing NEURON or when importing bluecellulab (that imports NEURON), ensure to:

- Navigate (``cd``) into the appropriate directory that is contaning one of the architecture-specific folders ("i686", "x86_64", "powerpc", or "umac") containing your compiled mechanisms.
- Confirm the potential impact on results that may stem from being in different directories when invoking NEURON.

.. code-block:: python

   import bluecellulab

Compile Mechanisms Using nrnivmodl
----------------------------------

To compile the neuron mechanisms, utilize the ``nrnivmodl`` command. This command should generate a folder containing compiled mechanisms, and the name of this folder will vary depending on your machine's architecture. The typical folder names to expect include "i686", "x86_64", "powerpc", or "umac".

Usage:

.. code-block:: shell

   nrnivmodl <path_to_mod_files>

Ensure to:

1. Replace ``<path_to_mod_files>`` with the path to your ``.mod`` files.
2. Check for the creation of one of the aforementioned folders upon successful compilation.

Working with Compiled Mechanisms
--------------------------------

BlueCelluLab will automatically load the compiled mechanisms if one of the architecture-specific folders ("i686", "x86_64", "powerpc", or "umac") is present in the current working directory. It is worth noting that after utilizing ``nrnivmodl``, only one of these folders should be present.

Customizing Mechanism Path with Environment Variable
----------------------------------------------------

In scenarios where you desire to point BlueCelluLab to another directory containing the compiled mechanisms, utilize the ``BLUECELLULAB_MOD_LIBRARY_PATH`` environment variable. Set it to point to the desired folder containing the compiled mechanisms.

Example:

.. code-block:: shell

   export BLUECELLULAB_MOD_LIBRARY_PATH="YOUR/DIRECTORY/x86_64"

Replace ``"YOUR/DIRECTORY/x86_64"`` with the path to your specific compiled mechanism directory.

Important Note on Path Specification
------------------------------------

Be mindful to adhere to the condition that **either** the current working directory should contain the compiled mechanisms **or** the ``BLUECELLULAB_MOD_LIBRARY_PATH`` environment variable should be set—**not both**. Setting the environment variable and importing BlueCelluLab from a directory containing (e.g.) an "x86_64" folder results in an error.

In summary:

- Ensure your working directory is aptly considered when utilizing BlueCelluLab and NEURON.
- Employ ``nrnivmodl`` for mechanism compilation and verify the resultant architecture-specific folder.
- Opt between utilizing the working directory or the ``BLUECELLULAB_MOD_LIBRARY_PATH`` for mechanism location, observing the necessity to avoid using both simultaneously.

May your simulations run smoothly with BlueCelluLab!
