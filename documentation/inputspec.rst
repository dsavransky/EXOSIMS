.. _sec:inputspec:

Input Specification
========================

A simulation specification is a single dictionary, typically
stored on disk in JSON-formatted (http://json.org/)
file that encodes user-settable parameters and the names of specific modules to use in the simulation.

Both the :py:mod:`~EXOSIMS.MissionSim` and prototype :py:mod:`~EXOSIMS.Prototypes.SurveySimulation`
``__init__`` can accept a string input (keyword ``scriptfile``, or as the first positional argument)
describing the full path on disk to a JSON-formatted script file.  Alternatively, a user can manually
read in such a file (see :ref:`quickstart` for sample code) and then pass its contents to either of these (or any other EXOSIMS
module ``__init__``) as a keyword dictionary (see :ref:`modinit`). 


Module Specification
---------------------

The input specification must contain a dictionary called ``modules`` specifying
which module implementations to use. The order
of the modules in the dictionary is arbitrary, but they must **all** be present.

The prototype module can be specified by setting the relevant dictionary entry to the module type
(i.e., entry ``"BackgroundSources"`` is set to ``"BackgroundSources"``) or to a string 
with a single space (i.e., ``" "``).

.. warning::

    Setting the module to an empty string will **not** work.

If the module implementation you wish to use is located in the appropriate subfolder in the
EXOSIMS tree (and EXOSIMS was installed in editable/developer mode - see :ref:`install`), then it can be specified 
by the module name (i.e., the name of the module class, which must match the filename). However, if
you wish to use an implemented module outside of the EXOSIMS directory,
then you need to specify it via its full path in the input
specification.

.. note::

    Full path specifications in the input specification may include environment variables.

Here is an example of a module specification, including all 3 types of allowable values (default Prototypes, modules within the
EXOSIMS code tree, and modules elsewhere on disk):

::

   {
       "modules": {
           "BackgroundSources": " ",
           "Completeness": "GarrettCompleteness",
           "Observatory": "WFIRSTObservatoryL2",
           "OpticalSystem": "Nemati",
           "PlanetPhysicalModel": "Forecaster",
           "PlanetPopulation": "KeplerLike2",
           "PostProcessing": "PostProcessing",
           "SimulatedUniverse": "KeplerLikeUniverse",
           "StarCatalog": "EXOCAT1",
           "SurveyEnsemble": "SurveyEnsemble",
           "SurveySimulation": "SurveySimulation",
           "TargetList": "TargetList",
           "TimeKeeping": "TimeKeeping",
           "ZodiacalLight": "$HOME/myEXOSIMSmodules/myZodicalLightModule.py"
       },
   }


Keyword Inputs
-----------------

Essentially all inputs to all EXOSIMS prototype module ``__init__`` methods are named keywords (so that a user does not need to remember
or constantly look up positional argument order, and so that default values can be assigned to all possible inputs). 
Python syntax is such that any named keywords present in the
syntax declaration of a method, which are also present within an input keyword dictionary, will automatically have values copied
from the keyword dictionary and assigned to the named keyword.  As an example, consider this simplified stub of a class implementation:

.. code-block:: python

    class EXOSIMSmod1(object):

        def __init__(self, arg1 = 0, **specs):

            self.arg1 = arg1

We have a keyword argument (``arg1``) as well as our usual ``**specs`` keyword argument dictionary. We now try to instantiate three
different objects of this class:

.. code-block:: python

    specs1 = {'arg1': 1}
    mod1a = EXOSIMSmod1()
    mod1b = EXOSIMSmod1(**specs1)
    mod1c = EXOSIMSmod1(arg1=2,**specs1)

``mod1a.arg1`` will be 0, as that is the default value for this keyword.  ``mod1b.arg1`` will be 1, as that is the value in the ``specs1`` dictionary. The attempt to initialize ``mod1c``, however, will raise a :py:exc:`TypeError` as there will be multiple available values for ``arg1``, and Python does not make any choices as to which to use in such instances. 

This behavior means that EXOSIMS allows for fairly great flexibility in mixing input specifications stored on disk with additional input values set at runtime. For example, it may be that a user wishes to evaluate the effects of changing a single parameter, say the initial spacecraft mass, which is encoded by input ``scMass`` to the ``Observatory`` module, while keeping all other inputs constant.  We could create a detailed input specification file on disk (say called ``specfile1.json``), which would include values for everything we cared about *except for* ``scMass`` and then generate multiple different ``MissionSim`` objects as:

.. code-block:: python

    sim = EXOSIMS.MissionSim.MissionSim('/path/to/specfile1.json', scMass=5000)

with different values for the ``scMass`` keyword.


Keywords and Nested Module Initializations
-----------------------------------------------

When EXOSIMS modules create objects of other modules as part of their initialization (see :numref:`fig:instantiation_tree`), 
the same input specification is passed to every ``__init__`` called.  Therefore, in theory, every single input is available to
every module constructed in a single object instantiation.  However, the behavior of Python is such that when keyword dictionary values are 
copied to named keywords, the associated keys are popped from the dictionary, meaning that the keyword information will be gone from ``**specs``
within the very first ``__init__`` where that keyword explicitly appears.  To illustrate: let's consider the addition of a second class to our previous example:

.. code-block:: python

    class EXOSIMSmod2(object):

        def __init__(self, arg1 = 2, **specs):

            self.arg1 = arg1
            self.mod1 = EXOSIMSmod1(**specs)

That is, our ``EXOSIMSmod2`` will have its own ``arg1`` attribute (with a different default value from ``EXOSIMSmod1``), and will also have an 
attribute containing an object instance of ``EXOSIMSmod1``, with the same keyword dictionary passed to both ``__init__`` methods.  We again have two possible instantiations:

.. code-block:: python

    specs1 = {'arg1': 1}
    mod2a = EXOSIMSmod2()
    mod2b = EXOSIMSmod2(**specs1)

``mod2a.arg1`` and ``mod2a.mod1.arg1`` will be 2 and 0, respectively, as these are the default values for each.  However, ``mod2b.arg1`` and ``mod2b.mod1.arg1`` will be 1 and 0, respectively.  The value of ``arg1`` in ``specs1`` will be applied in the ``__init__`` of the ``EXOSIMSmod2`` object, and removed from the dictionary, meaning that the ``**specs`` passed to ``EXOSIMSmod1.__init__`` will be empty, causing the resulting object to use its default value for ``arg1``. 

It's fairly straightforward to get around this issue.  If we know that a keyword needs to be reused in a downstream ``__init__``, we can just pass it explicitly.  Consider a modification to our second class, and the same two instantiations:

.. code-block:: python

    class EXOSIMSmod3(object):

        def __init__(self, arg1 = 2, **specs):

            self.arg1 = arg1
            self.mod1 = EXOSIMSmod1(arg1=self.arg1, **specs)

    specs1 = {'arg1': 1}
    mod3a = EXOSIMSmod3()
    mod3b = EXOSIMSmod3(**specs1)

In this case, the ``arg1`` and ``mod1.arg1`` attributes of any ``EXOSIMSmod3`` object will always be equal.  For ``mod3a`` they will both be 2 (the default) and for ``mod3b`` they will both be 1 (the value from ``specs1``.  

It should also be noted that the popping of keyword values only occurs within the scope of the relevant ``__init__`` (or any other method the keyword dictionary is passed to).  In all of these examples, ``specs1`` remains unmodified in whatever scope it was originally defined.  Thus, if a module does not have a particular keyword in its own ``__init__``, but instantiates two modules in series that both use the same keyword input, then both will get the value for this keyword (if present) in the ``**specs`` input.  One final illustration:

.. code-block:: python

    class EXOSIMSmod4(EXOSIMSmod1):

        pass

    class EXOSIMSmod5(object):

        def __init__(self, **specs):

            self.mod1 = EXOSIMSmod1(**specs)
            self.mod4 = EXOSIMSmod4(**specs)


    specs1 = {'arg1': 1}
    mod5a = EXOSIMSmod5()
    mod5b = EXOSIMSmod5(**specs1)

``EXOSIMSmod4`` is an exact copy of ``EXOSIMSmod1`` (since the ``__init__`` is directly inherited).  Objects of ``EXOSIMSmod5`` will not have their own `arg1` attributes, but will have two attributes storing object instances of ``EXOSIMSmod1`` and ``EXOSIMSmod4``.  The attributes ``mod1.arg1`` and ``mod4.arg1`` will always be identical (as both have the same default values.  In the case of ``mod5a``, both will be the default (0) and for ``mod5b`` both will have the value from ``specs1`` (1). 

The upshot is that EXOSIMS modules built at the same level can have access to the same entries in the input specification.
An example of this is the ``missionStart`` input (representing the absolute time at the beginning 
of the mission).  This is nominally intended for use by the :py:mod:`~EXOSIMS.Prototypes.TimeKeeping`
module, but can also be used by any :py:mod:`~EXOSIMS.Observatory` implementations
that need to know absolute times at initialization (before they could potentially have
access to a ``TimeKeeping`` object) for orbital calculations.


.. _sec:outspec:

Output Specification
-----------------------------------------------

Every EXOSIMS module contains a private dictionary attribute called
``_outspec``. This dictionary includes a key for **every** user-settable
parameter for that module implementation, along with their values once
the object has been instantiated and initialized.  This means that a given
module object's ``_outspec`` will contain a mixture of user-set values from the input
specification, along with default values from the class's ``__init__`` declaration.

Taken together, all of the 14 module ``_outspec`` dictionaries will define a complete
specification for a given simulation.  The ``MissionSim``/``SurveySimulation`` method :py:meth:`~EXOSIMS.MissionSim.MissionSim.genOutSpec`
provides functionality for assembling a complete specification dictionary out of the 14 independent 
``_outspec`` dictionaries, and (optionally) writing the complete specification to disk in JSON format.  This JSON
file can then be used as an input specification to another simulation, which will have the exact same parameters
as the original simulation. ``genOutSpec`` also adds code version information to the 
output, and, in the case where EXOSIMS is installed in developer mode from a git 
repository (see :ref:`install`), the commit hash.

.. warning::

    If there are code changes in the modules being used that have not been checked in at the time
    when the output specification is generated, then this will not be captured by the versioning information
    and may lead to irreproducible results. 

.. important::
    Any new user inputs added to the ``__init__`` of a new module implementation
    **must** also be added to that implementation's ``_outspec`` dictionary attribute.

Input/Output Checking
--------------------------------

By default, at the end of instantiation of a :py:mod:`~EXOSIMS.MissionSim` object, a check is performed on both the input and output specifications (this can be toggled off via the boolean input keyword ``inputCheck``). The check consists of compiling a list of all arguments to all ``__init__`` methods of every module (including the :py:mod:`~EXOSIMS.MissionSim`), as well as all of the ``__init__`` methods of any base classes those modules might inherit.  This list is then compared against the original input specification, as well as the output specification generated by :py:meth:`~EXOSIMS.MissionSim.MissionSim.genOutSpec`.  Warnings are raised in the event that the input specification includes keywords not appearing in the arguments list, or if the argument list contains entries that aren't in the output specification. 

The argument list for all Prototype modules is given in :ref:`arglist`.
