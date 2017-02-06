.. _documentation:
Documentation Guide
######################

This documentation is written using `Sphinx <http://sphinx-doc.org/>`_.


Installing Sphinx
------------------

You will need the `sphinx documentation tool <http://sphinx-doc.org>`_ and 
its `numpydoc extension <https://pypi.python.org/pypi/numpydoc>`_ and `napoleon extension <https://pypi.python.org/pypi/sphinxcontrib-napoleon>`_. 

Assuming you have the `pip <http://www.pip-installer.org/en/latest/installing.html>`_ python package installer, 
you can easily install the required tools from `PyPI <https://pypi.python.org/pypi>`_::

   >>> pip install Sphinx
   >>> pip install numpydoc
   >>> pip install sphinxcontrib-napoleon

Building the docs
------------------

cd to the root directory ``documentation``. If Sphinx is propery installed, you should be able to just run ``make html``::

    >>> make html
    [huge printout of compiling status informational messages]

    Build finished. The HTML pages are in _build/html

The documentation will be built to HTML and you can view it locally by opening ``documentation/_build/html/index.html`` 

You can also similarly build the documentation to PDF, LaTeX, Windows Help, or several other formats as desired. Use ``make help`` to see the available output formats. 

If there are structural changes made to the EXOSIMS directory (i.e., file/folder additions or deletions) you will need to run ``sphinx-apidoc``.  From the ``documentation`` directory run::

    sphinx-apidoc -f -o . ../EXOSIMS/

This will rebuild all of the ``EXOSIMS.*`` rst files, after which you can run ``make html`` to rebuild the html documentation. Note that this command also generates a ``modules.rst`` file that we do not use, and which can be safely deleted (but should not be added to the repository).


This entire procedure is also packaged into a bash script: ``builddocs.sh``.  On POSIX systems, you can just run this script from the documentation directory to rebuild and recompile everything and move the resulting html directory to the top level documentation directory. 


