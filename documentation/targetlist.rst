.. _targetlist:

TargetList
=============

The target list modules take catalog data from a :ref:`StarCatalog` object and, using additional information and methods from :ref:`Completeness`, :ref:`OpticalSystem`, and :ref:`ZodiacalLight` objects, generate an input target list for a survey simulation.  The basic functionality is fully implemented by the :py:class:`~EXOSIMS.Prototypes.TargetList` prototype, with other implementations focused on special cases, such as the :py:class:`~EXOSIMS.TargetList.KnownRVPlanetsTargetList`.  The end result is an object analogous to the :ref:`StarCatalog`, with target attributes stored in equally sized :py:class:`numpy.ndarray`\s. :numref:`fig:TL_init_flowchart` shows the initialization of the  :py:class:`~EXOSIMS.Prototypes.TargetList` prototype.


.. _fig:TL_init_flowchart:
.. mermaid:: targetlist_init.mmd
   :caption: TargetList Prototype ``__init__``.


After parsing keyword inputs and instantiating objects of :ref:`StarCatalog`, :ref:`OpticalSystem`, :ref:`PostProcessing`,  :ref:`ZodiacalLight`, and :ref:`Completeness`, the prototype :py:class:`~EXOSIMS.Prototypes.TargetList` initialization calls :py:meth:`~EXOSIMS.TargetList.TargetList.populate_target_list`, which makes :ref:`StarCatalog` attribute property arrays attributes of the ``TargetList`` object, fills in missing photometric data (if the ``fillPhotometry`` keyword input is set to True), and assigns each target system additional, computed attributes: 

* The true and approximate stellar mass (attributes :py:attr:`~EXOSIMS.Prototypes.TargetList.TargetList.MsEst` and :py:attr:`~EXOSIMS.Prototypes.TargetList.TargetList.MsTrue` , respectively) is calculated in :py:meth:`~EXOSIMS.Prototypes.TargetList.TargetList.stellar_mass`.   The estimated mass is based on the Mass-Luminosity relationship defined by the user, defaulting to [Henry1993]_. 
* The inclination of the target system's orbital plane (attribute :py:attr:`~EXOSIMS.Prototypes.TargetList.TargetList.I`),  calculated in  :py:meth:`~EXOSIMS.Prototypes.TargetList.TargetList.gen_inclinations`. This is used only if the ``commonSystemInclinations`` keyword input to the :ref:`SimulatedUniverse` is set to True. The inclinations are sinusoidally distributed, within the bounds set by the :ref:`PlanetPopulation` attribute ``Irange``.
* The :math:`\Delta\textrm{mag}` and completeness values associated with the integration cutoff time set in the :ref:`OpticalSystem` and the saturation integration time (i.e., the point at which these values stop changing).  For optical systems where there is no fundamental noise floor (i.e., where :term:`SNR` can always be increased with additional integration time) the saturation :math:`\Delta\textrm{mag}`  is effectively infinite, but the saturation completeness is limited to the maximum :term:`obscurational completeness` for that system (see [Brown2005]_ for details). These values, along with the user-selectable :math:`\Delta\textrm{mag}_\textrm{int}` and :math:`WA_\textrm{int}` are calculated in  :py:meth:`~EXOSIMS.Prototypes.TargetList.TargetList.calc_saturation_and_intCutoff_vals`, which calls helper methods :py:meth:`~EXOSIMS.Prototypes.TargetList.TargetList.calc_saturation_dMag` and :py:meth:`~EXOSIMS.Prototypes.TargetList.TargetList.calc_intCutoff_dMag`. 
* The single-visit :ref:`Completeness` (attribute :py:attr:`~EXOSIMS.Prototypes.TargetList.TargetList.int_comp`) based on :math:`\Delta\textrm{mag}_\textrm{int}`.

Finally, the whole target list is filtered by :py:meth:`~EXOSIMS.Prototypes.TargetList.TargetList.filter_target_list`, based on filters selected by input keywords.  The default filter set removes binary stars (or stars with close companions), systems where :term:`obscurational completeness` is zero (i.e., all planets are inside the :term:`IWA` or outside the :term:`OWA`), and systems for which the integration cutoff completeness is less than the ``minComp`` input value. 

Mass-Luminosity Relationship
--------------------------------------------------
The Mass Luminsoity Relationship (MLR) gives us an estimate for the mass of stars, which becomes extremely important for later caclulations. When the mass is not given by the catalog, we can estimate it using a variety of different models pre-programmed. Henry1993 is the default model, however there are more options and the ability to framework to add new models is in place.

The first calculation is of the 'Estimated' mass, which is the approximate stellar mass according to the model. We impose the standard deviation error described in the publication onto this value to create the 'True' mass.

All graphs shown below are using the HWO mission stars catalog with filtering to test the frameworks.

* **Henry 1993** - This is a great generalist and will output reliable data. The 'true' mass is equal to the estimated mass of each star plus a randomly generated Gaussian value with mean 0 and standard deviation of 7% (the error associated with the fit in that publication). (See [Henry1993]_ for details.)

  .. image:: henry1993chart.png

* **Fernandes 2021** - This model is much more up to date, but should only be used for FGK stars. Not reliable on other spectral classes. If your catalog only includes these stars or other stars are filtered out at this point, this model is highly suggested. Erorr is described in publication as 3%. (See [Fernandes2021]_ for details.)

  .. image:: fernandes2021chart.png

* **Henry 1993 + 1999** - This includes the paper from Henry 1999, where the caculations for lower mass stars are considered as well. Best generalist when working with a large variety of masses, especially into M dwarf territory. (See [Henry1999]_ for details.)
  
  .. image:: henry1993+1999chart.png

* **Fang 2010** - Great generalist, includes more specific data with a slightly smaller error. Good for all main sequence stars. (See [Fang2010]_ for details.)

  .. image:: fang2010chart.png


