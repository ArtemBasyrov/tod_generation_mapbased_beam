TOD Generation from Beam Convolution
=====================================

Sample-based Time-Ordered Data (TOD) generation for CMB experiments.
Convolves polarised sky maps (I, Q, U) with pixelated beam patterns over a
boresight scan trajectory, producing one output per observation day: a TOAST
HDF5 observation for the furax mapmaker (default) or a raw ``.npy`` TOD.

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   getting_started
   configuration
   data_formats
   beam_creation
   beam_interpolation_accuracy
   beam_cluster_calibration

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/index

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
