.. _targetlist:

TargetList
=============

The target list modules take catalog data from a :ref:`StarCatalog` object and, using additional information and methods from :ref:`Completeness`, :ref:`OpticalSystem`, and :ref:`ZodiacalLight` objects, generate an input target list for a survey simulation.  The basic functionality is fully implemented by the :py:class:`~EXOSIMS.Prototypes.TargetList` prototype, with other implementations focused on special cases, such as the :py:class:`~EXOSIMS.TargetList.KnownRVPlanetsTargetList`.  The end result is an object analogous to the :ref:`StarCatalog`, with target attributes stored in equally sized :py:class:`numpy.ndarray`\s. :numref:`fig:TL_init_flowchart` shows the initialization of the  :py:class:`~EXOSIMS.Prototypes.TargetList` prototype.


.. _fig:TL_init_flowchart:
.. mermaid:: targetlist_init.mmd
   :caption: TargetList Prototype ``__init__``.


After parsing keyword inputs and instantiating objects of :ref:`StarCatalog`, :ref:`OpticalSystem`, :ref:`PostProcessing`,  :ref:`ZodiacalLight`, and :ref:`Completeness`, the prototype :py:class:`~EXOSIMS.Prototypes.TargetList` initialization calls :py:meth:`~EXOSIMS.TargetList.TargetList.populate_target_list`, which makes :ref:`StarCatalog` attribute property arrays attributes of the ``TargetList`` object, fills in missing photometric data (if the ``fillPhotometry`` keyword input is set to True), and assigns each target system additional, computed attributes: 

* The true and approximate stellar mass (attributes :py:attr:`~EXOSIMS.Prototypes.TargetList.TargetList.MsEst` and :py:attr:`~EXOSIMS.Prototypes.TargetList.TargetList.MsTrue` , respectively), calculated in :py:meth:`~EXOSIMS.Prototypes.TargetList.TargetList.stellar_mass`.   The estimated mass is based on the Mass-Luminosity relationship from [Henry1993]_ and the 'true' mass is equal to the estimated mass of each star plus a randomly generated Gaussian value with mean 0 and standard deviation of 7% (the error associated with the fit in that publication). 
* The inclination of the target system's orbital plane (attribute :py:attr:`~EXOSIMS.Prototypes.TargetList.TargetList.I`),  calculated in  :py:meth:`~EXOSIMS.Prototypes.TargetList.TargetList.gen_inclinations`. This is used only if the ``commonSystemInclinations`` keyword input to the :ref:`SimulatedUniverse` is set to True. The inclinations are sinusoidally distributed, within the bounds set by the :ref:`PlanetPopulation` attribute ``Irange``.
* The :math:`\Delta\textrm{mag}` and completeness values associated with the integration cutoff time set in the :ref:`OpticalSystem` and the saturation integration time (i.e., the point at which these values stop changing).  For optical systems where there is no fundamental noise floor (i.e., where :term:`SNR` can always be increased with additional integration time) the saturation :math:`\Delta\textrm{mag}`  is effectively infinite, but the saturation completeness is limited to the maximum :term:`obscurational completeness` for that system (see [Brown2005]_ for details). These values, along with the user-selectable :math:`\Delta\textrm{mag}_\textrm{int}` and :math:`WA_\textrm{int}` are calculated in  :py:meth:`~EXOSIMS.Prototypes.TargetList.TargetList.calc_saturation_and_intCutoff_vals`, which calls helper methods :py:meth:`~EXOSIMS.Prototypes.TargetList.TargetList.calc_saturation_dMag` and :py:meth:`~EXOSIMS.Prototypes.TargetList.TargetList.calc_intCutoff_dMag`. 
* The single-visit :ref:`Completeness` (attribute :py:attr:`~EXOSIMS.Prototypes.TargetList.TargetList.int_comp`) based on :math:`\Delta\textrm{mag}_\textrm{int}`.

Finally, the whole target list is filtered by :py:meth:`~EXOSIMS.Prototypes.TargetList.TargetList.filter_target_list`, based on filters selected by input keywords.  The default filter set removes binary stars (or stars with close companions), systems where :term:`obscurational completeness` is zero (i.e., all planets are inside the :term:`IWA` or outside the :term:`OWA`), and systems for which the integration cutoff completeness is less than the ``minComp`` input value. 


