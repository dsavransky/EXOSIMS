Fundamental Concepts
========================

This is a brief summary of fundamental physical concepts underlying the code, and how they are treated in the code.  Many more details are available in the :ref:`refs`.


Planetary Parameters
------------------------

An exoplanet in EXOSIMS is defined via a set of scalar orbital and physical parameters. 

.. _fig:orbit_diagram:
.. figure:: orbit_diagram.png
   :width: 100.0%
   :alt: Orbit Diagram

   Exoplanetary system orbit diagram.

The planet's orbit is defined via Keplerian orbital elements, as in :numref:`fig:orbit_diagram`, where :math:`a` is the semi-major axis, :math:`e` is the eccentricity, and the orbit's orientation with respect to the observer is given by  3-1-3 :math:`(\Omega,I,\omega)` Euler angle set (the longitude of the ascending node, the inclination, and the argument of periapsis, respectively).  By default, all of these quantities are considered to be constant (i.e., no orbital evolution due to perturbations or mutual gravitational effects in multi-planet systems), but the code may be extended to account for these effects, in which case they should be treated as the osculating values at epoch.



