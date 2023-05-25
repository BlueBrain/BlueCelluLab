****
FAQ
****

Here are the frequently asked questions on bluecellulab.


MPT ERROR: PMI2_Init
====================
This error is probably caused by an unresolved 
issue between the neuron simulator and the MPI 
compilation on BB5. 
Try to run the following command before loading 
bluecellulab to solve this issue.

.. code-block:: bash

        unset PMI_RANK
