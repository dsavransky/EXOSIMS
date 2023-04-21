.. _starcatalog:
   
StarCatalog
==============

The star catalog modules are intended to be primarily static, providing only an interface between
external catalog data and ``EXOSIMS`` standards.  Any processing or augmentation of catalog data
should be done in the ``TargetList`` module.

``StarCatalog`` objects must contain equally sized arrays of stellar attributes (stored as :py:class:`numpy.ndarray` attributes, or :py:class:`astropy.units.quantity.Quantity` arrays, as appropriate).  The prototype will generate dummy values for all attributes, with the catalog size set by input ``ntargs``.

While different catalogs may contain different information, the :ref:`TargetList` will impose a minimum set of attributes (those listed in the target list's :py:attr:`~EXOSIMS.Prototypes.TargetList.TargetList.required_catalog_atts` attribute).  The contents of this list will vary depending on the particular implementation, but, for the prototype, include:

* Name:
  The star's name (must be unique).  For Hipparcos strings referring to multiple stars, the  component letter should be appended (as in CCDM, WDS, or SIMBAD).
* Vmag:
  Johnson V-band apparent magnitude
* BV:
  Johnson B-V color
* MV:
  Johnson V-band absolute magnitude
* BC:
  Bolometric correction
* L:
  Luminosity in solar luminosities  

  .. important::

    This must be the actual luinosity.  *Not* :math:`\log(L)`!

* coords:
  Target coordinates, encoded as a :py:class:`astropy.coordinates.SkyCoord` array
* dist:
  Distances
* parx:
  Parallaxes
* pmra:
  Proper motions in RA
* pmdec:
  Proper motions in DEC
* rv:
  Proper motions along the line of sight
* Binary_Cut:
  Boolean array. True indicates that there is a star within 10 arcsec of the target.
* Spec:
  Spectral type string.  Must be parsable by MeanStars.


In addition to these attributes, the prototype will also generate (all-zero) arrays for other band magnitudes.  All of these have the same form as Vmag (e.g. Imag and Bmag) and include the UBRIJHK bands.




