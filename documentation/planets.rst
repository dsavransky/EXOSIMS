.. _planetphysicalmodel:

PlanetPhysicalModel
======================

Planet physical model modules provide methods for calculating planet properties as functions of planet parameters.  These include density models (for converting between planet mass and radius) and any photometric models (for evaluating albedo/phase as a function of orbit size/location).

Mass-Radius Forecaster
---------------------

We use the mass-radius prescription from [Sousa2024]_ (shown in :numref:`fig:gauss_box_bandpasses`) which empirically derived a piecewise mass-radius function for planets with a homogeneous set of stellar parameters:

    .. math::

        \text{for } M < 159M_{\oplus} \text{: } \log_{10}\left(\frac{R}{R_{\oplus}}\right) = 0.59^{+0.01}_{-0.01}\log_{10} \frac{M}{M_{\oplus}} - 0.15^{+0.02}_{-0.02}  
        \text{for } M > 159M_{\oplus} \text{: } \log_{10}\left(\frac{R}{R_{\oplus}}\right) = 0.00^{+0.02}_{-0.02}\log_{10} \frac{M}{M_{\oplus}} + 1.15^{+0.06}_{-0.06}


.. _fig:sousamrrelation:
.. figure:: sousa2024relation.png
   :width: 100.0%
   :alt: the mass-radius relation from Sousa et al. 2024
    
   The mass-radius relation from [Sousa2024]_ overplotted onto the planet population they derived the relation from with homogeneously derived stellar parameters from SWEET-Cat, additionally with their :math:`T_{eq}` color coded.


.. _planetpopulation:
   
PlanetPopulation
====================

Planet population modules encode the distributions defining a planet population, and provide methods for sampling from these distributions.


