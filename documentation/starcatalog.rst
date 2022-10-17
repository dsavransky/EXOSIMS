.. _starcatalog:
   
StarCatalog
==============

The star catalog modules are intended to be primarily static, providing only an interface between
external catalog data and ``EXOSIMS`` standards.  Any processing or augmentation of catalog data
should be done in the ``TargetList`` module.

``StarCatalog`` objects must contain equally sized arrays of stellar attributes (stored as :py:class:`numpy.ndarray` attributes).  While 
different catalogs may contain different information, a minimum set of attributes (the ones listed in the :py:class:`~EXOSIMS.Prototypes.StarCatalog` prototype) is required to run a full survey simulation. 



