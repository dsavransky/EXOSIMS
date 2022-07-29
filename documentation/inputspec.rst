.. _sec:inputspec:

Input Specification
========================

A simulation specification is a single dictionary, typically
stored on disk in JSON-formatted (http://json.org/)
file that encodes user-settable parameters and module names.
SurveySimulation will contain a reference specification with *all*
parameters and modules set via defaults in the constructors of each of
the modules. In the initial parsing of the user-supplied specification,
it will be merged with the reference specification such that any fields
not set by the user will be assigned to their reference (default)
values. Each instantiated module object will contain a dictionary called
``_outspec``, which, taken together, will form the full specification
for the current run (as defined by the loaded modules). This
specification will be written out to a json file associated with the
output of every run. *Any specification added by a user implementation
of any module must also be added to the \_outspec dictionary*. The
assembly of the full output specification is provided by MissionSim
method ``genOutSpec``.

For every simulation (or ensemble), an output specification will be
written to disk along with the simulation results with all defaults used
filled in.


