Fundamental Concepts
========================

This is a brief summary of fundamental physical concepts underlying the code, and how they are treated in the code.  Many more details are available in the :ref:`refs`.


Orbit Geometry
------------------------

An exoplanet in ``EXOSIMS`` is defined via a set of scalar orbital and physical parameters. For each target star :math:`S`, we define a reference frame  :math:`\mathcal{S} = (\mathbf{\hat s}_1, \mathbf{\hat s}_2, \mathbf{\hat s}_3)`, with :math:`\mathbf{\hat s}_3` pointing along the vector from the observer to the star (:math:`\mathbf{\hat{r}}_{S/\textrm{observer}} \equiv -\mathbf{\hat{r}}_{\textrm{observer}/S}`) such that the plane of the sky (the plane orthogonal to the this vector lies in the :math:`\mathbf{\hat s}_1-\mathbf{\hat s}_2` plane, as in :numref:`fig:orbit_diagram`.  The :math:`\mathcal{S}` is fixed at the time of mission start, and does not evolve throughout the mission simulation, making :math:`\mathcal{S}` a true inertial frame. While the orientation of :math:`\mathbf{\hat s}_3)` is arbitrary, we take it to be the same inertially fixed direction for all targets (by default equivalent to celestial north). 

.. _fig:orbit_diagram:
.. figure:: orbit_diagram.png
   :width: 100.0%
   :alt: Orbit Diagram

   Exoplanetary system orbit diagram.

The planet's orbit is defined via Keplerian orbital elements, where :math:`a` is the semi-major axis, :math:`e` is the eccentricity, and the orbit's orientation in the  :math:`\mathcal{S}` frame is given by  3-1-3 :math:`(\Omega,I,\omega)` Euler angle set (the longitude of the ascending node, the inclination, and the argument of periapsis, respectively).  By default, all of these quantities are considered to be constant (i.e., no orbital evolution due to perturbations or mutual gravitational effects in multi-planet systems), but the code may be extended to account for these effects, in which case they should be treated as the osculating values at epoch. 

The planet's instantaneous location at time :math:`t` is given by the true anomaly :math:`\nu(t)`.  The orbit (or osculating orbit, in cases where perturbations are allowed) is fully characterized by a simultaneous measurement of the orbital radius and velocity vectors.  The orbital radius vector is given by:

   .. math::
        
        \mathbf{r}_{P/S} = \left[\begin{matrix}- \sin{\left(\Omega \right)} \sin{\left(\theta \right)} \cos{\left(I \right)} + \cos{\left(\Omega \right)} \cos{\left(\theta \right)}\\\sin{\left(\Omega \right)} \cos{\left(\theta \right)} + \sin{\left(\theta \right)} \cos{\left(I \right)} \cos{\left(\Omega \right)}\\\sin{\left(I \right)} \sin{\left(\theta \right)}\end{matrix}\right]

where :math:`r` is the orbital radius magnitude:

   .. math::
        
        r \equiv \Vert  \mathbf{r}_{P/S} \Vert =  \frac{a(1 - e^2)}{1 + e\cos\nu}

and :math:`\theta` is the argument of latitude, :math:`\theta \triangleq \nu + \omega`. The orbital velocity vector is given by:

   .. math::
        
        \mathbf{v}_{P/S} = \sqrt{ \frac{\mu}{a}} \sqrt{\frac{1}{1 - e^{2}}} \left[\begin{matrix}- e \sin{\left(\Omega \right)} \cos{\left(I \right)} \cos{\left(\omega \right)} - e \sin{\left(\omega \right)} \cos{\left(\Omega \right)} - \sin{\left(\Omega \right)} \cos{\left(I \right)} \cos{\left(\theta \right)} - \sin{\left(\theta \right)} \cos{\left(\Omega \right)}\\- e \sin{\left(\Omega \right)} \sin{\left(\omega \right)} + e \cos{\left(I \right)} \cos{\left(\Omega \right)} \cos{\left(\omega \right)} - \sin{\left(\Omega \right)} \sin{\left(\theta \right)} + \cos{\left(I \right)} \cos{\left(\Omega \right)} \cos{\left(\theta \right)}\\\left(e \cos{\left(\omega \right)} + \cos{\left(\theta \right)}\right) \sin{\left(I \right)}\end{matrix}\right]

where :math:`\mu` is the gravitational parameter: :math:`\mu \triangleq G(m_S + m_P)` for gravitational constant :math:`G` and star and planet masses  :math:`m_S` and :math:`m_P`, respectively.  Internally, ``EXOSIMS`` stores the standard gravitational parameters of the stars and planets: :math:`\mu_S = G m_S` and :math:`\mu_P = G m_P`, respectively.  Each planet has only a 'true' mass, whereas for each target star, we generate a 'true' and 'estimated' mass, based on a fit to the star's luminosity, and the known error statistics of that fit.

An imaging detection measures the projection of the orbital radius onto the plane of the sky, which is known as the projected separation vector, :math:`\mathbf{s} = \mathbf{r}_{P/O} - \mathbf{r}_{P/O} \cdot \mathbf{\hat e}_3`.  The projected separation is the magnitude of this vector, and is given by:

    .. math::
        
        s \triangleq \Vert\mathbf{s}\Vert = \frac{r}{4} \sqrt{4 \cos{\left (2 I \right )} + 4 \cos{\left (2 \theta \right )} - 2 \cos{\left (2 I - 2 \theta \right )} - 2 \cos{\left (2 I + 2 \theta \right )} + 12}

The angular separation can be calculated as 
     
   .. math::

      \alpha = \tan^{-1}\left( \frac{s}{d} \right)

where :math:`d` is the distance between the observer and the target star.  In the small angle approximation (which applies in all cases) this can be simplified to :math:`s/d`.

Planet Photometry
------------------------

The second quantity observed by direct imaging is the flux ratio between the planet and star: :math:`\frac{F_P}{F_S}`.  This is typically reported in astronomical magnitudes, as the difference in magnitude between star and planet:

    .. math::
        
        \Delta{\textrm{mag}} \triangleq -2.5\log_{10}\left(\frac{F_P}{F_S}\right) =  -2.5\log_{10}\left(p\Phi(\beta) \left(\frac{R_P}{r}\right)^2 \right)

where :math:`p` is the planet's geometric albedo, :math:`R_P` is the planet's (equatorial) radius, and :math:`\Phi` is the planet's phase function, which is parameterized by phase angle :math:`\beta`.

The phase angle is the illuminant-object-observer angle, and therefore the angle between the planet-star vector ( :math:`\mathbf{r}_{S/P} \equiv -\mathbf{r}_{P/S}`) and the planet-observer vector :math:`\mathbf{r}_{\textrm{observer}/P}`, which is given by:

    .. math::
        
        \mathbf{r}_{\textrm{observer}/P} = \mathbf{r}_{\textrm{observer}/S} - \mathbf{r}_{P/S} = -d \mathbf{\hat s}_3 -  \mathbf{r}_{P/S} 


Thus, the phase angle can be evaluated as:
 
   .. math::

      \cos\beta = \frac{-\mathbf{r}_{P/S} \cdot (-d\mathbf{\hat s}_3 - \mathbf{r}_{P/S} )}{r \Vert -d\mathbf{\hat s}_3 - \mathbf{r}_{P/S} \Vert}

If we assume that :math:`d \gg r` (the observer-target distance is much larger than the orbital radius, a safe assumption for all cases), then the planet-observer and star-observer vectors become nearly parallel, and we can approximate :math:`-d\mathbf{\hat s}_3 - \mathbf{r}_{P/S} \approx  -d\mathbf{\hat s}_3`.  In this case, the phase angle equation simplifies to:

   .. math::

      \cos\beta \approx \frac{-\mathbf{r}_{P/S} \cdot -d\mathbf{\hat s}_3}{rd} = \frac{\mathbf{r}_{P/S}}{r} \cdot \mathbf{\hat s}_3

If we evaluate this expression in terms of the components of the orbital radius vector as a function of the Euler angles defined above, we find:

.. _betacalcref:

   .. math::
      
      \cos\beta = \sin I \sin\theta


.. important::

    ``EXOSIMS`` adpots the convention that the observer is *below* the planet of the sky, looking up (i.e., along the positive :math:`\mathbf{\hat s}_3` direction in  :numref:`fig:orbit_diagram`).  This is different from the convention used elsewhere, and especially the convention adopted by the Exoplanet Archive, where the observer is located *above* the planet of the sky, and looking down (i.e., along the negative :math:`\mathbf{\hat e}_3` axis).  Switching conventions has no effect on the calculation of the projected separation, but does flip the sign of the phase angle, such that :math:`\cos\beta = -\sin I \sin\theta`.

It is important to note that not every orbit admits the full range of possible phase angles.  As :math:`theta` always varies between 0 and :math:`2\pi` for every closed orbit, from the :ref:`equation<betacalcref>`, we see that the phase angle is bounded by the value of the inclination, such that the maximum phase angle falls within the range :math:`\left[\frac{\pi}{2} - I, \frac{\pi}{2} + I\right]`, as shown in :numref:`fig:beta_plot`.  For a face-on orbit (:math:`I = 0`), the only possible phase angle is :math:`\frac{\pi}{2}` (the observer is always at a right angle from the star-planet vector), while an edge-on orbit (:math:`I = `\frac{\pi}{2}`), admits the full range of phase angles, :math:`\beta \in [0, \pi]`.

.. _fig:beta_plot:
.. figure:: beta_plot.png
   :width: 100.0%
   :alt: Phase angle as a function of argument of latitude for different orbit inclinations. 

   The range of phase angles that can occur within a given orbit are strictly bounded by the orbit's inclination. 


Phase Functions
------------------

The phase function of a planet depends on the composition of its surface and atmosphere (including any potential clouds), and can be arbitrarily difficult to model.  The simplest possible approximation to the phase function is given by the Lambert phase function, which describes a spherical, ideally isotropic, scattering body (none of which are good assumptions for planets.  The Lambert phase function is given by (see [Sobolev1975]_ for a full derivation):

    .. math::

        \pi\Phi_L(\beta) = \sin\beta + (\pi - \beta)\cos\beta

While not strictly correct for any physical planet, the Lambert phase function has the benefits of being very simple to evaluate. In particular, if assuming this phase function, we can strictly bound the :math:`\Delta{\textrm{mag}}`.  Following [Brown2004]_, the flux ratio (and therefore :math:`\Delta{\textrm{mag}}`) extrema for any phase function can be found by solving for the zeros of the derivative of the flux ratio with respect to the phase angle:

    .. math::
        
        \frac{\partial}{\partial \beta} \left(\frac{F_P}{F_S}\right) = \frac{2 \Phi{\left(\beta \right)} \sin{\left(\beta \right)} \cos{\left(\beta \right)}}{s^{2}} + \frac{\sin^{2}{\left(\beta \right)} \frac{d}{d \beta} \Phi{\left(\beta \right)}}{s^{2}} = 0

where we have substituted :math:`r = s/\sin(\beta)` and assumed that both planet radius and geometric albedo are constants. This simplifies to:

    .. math::
        
        2 \Phi{\left(\beta \right)} \cos{\left(\beta \right)} + \sin{\left(\beta \right)} \frac{d}{d \beta} \Phi{\left(\beta \right)} = 0

Substituting the Lambert phase function, we find the extrema-generating phase angle to be given by:

    .. math::
        
        - 3 \beta \cos{\left(2 \beta \right)} - \beta + 2 \sin{\left(2 \beta \right)} + 3 \pi \cos{\left(2 \beta \right)} + \pi = 0

which, as shown in :numref:`fig:lambert_extrema`, has a single non-trivial value at :math:`\beta \approx 1.10472882` rad (or 63.2963 degrees).
This is the value shown by the black dashed line in :numref:`fig:beta_plot`.
 
.. _fig:lambert_extrema:
.. figure:: lambert_extrema.png
   :width: 100.0%
   :alt: Flux ratio extrema for Lambert phase function. 
    
   The zeros of this function are the :math:`\beta` values corresponding to flux ratio exterma. 
    

A drawback of the Lambert phase function, however, is that it is not analytically invertible.  An alternative, suggested in [Agol2007]_ is the quasi-Lambert function, which, while not physically motivated, approximates the Lambert phase function relatively well, and has the benefit of analytical invertibility:

    .. math::

        \Phi_{QL}(\beta) = \cos^4\left(\frac{\beta}{2}\right)

For further discussion and other phase functions built into ``EXOSIMS`` see [Keithly2021]_.  All phase functions are provided by methods in :py:mod:`~EXOSIMS.util.phaseFunctions`.


Completeness, Integration Time, and :math:`\Delta{\textrm{mag}}`
----------------------------------------------------------------------

Photometric and obscurational completeness, as defined originally in [Brown2005]_, is the probability of detecting a planet from some population (given that one exists), about a particular star, with a particular instrument, upon the first observation of that target (this is also known as the single-visit completeness).  Completeness is evaluated as the double integral over the joint probability density function of projected separation and :math:`\Delta{\textrm{mag}}` associated with the planet population:

    .. math::
        
        c = \int_{0}^{\Delta\mathrm{mag}_\mathrm{max}(s, t_\mathrm{int})} \int_{s_{\mathrm{min}}}^{s_
        \mathrm{max}} f_{\bar{s},\overline{\Delta\mathrm{mag}}}\left(s,\Delta\mathrm{mag}\right) \intd{s} \intd{\Delta\mathrm{mag}}.

The limits on the projected separation are given by the starlight suppression system's inner and outer working angles (:term:`IWA` and :term:`OWA`):

    .. math::
        
        s_\mathrm{min} = \tan\left(\mathrm{IWA}\right) d \qquad  s_\mathrm{max} = \tan\left(\mathrm{OWA}\right) d 

In the small-angle approximation (essentially always appropriate for feasibly starlight suppression systems), these are just :math:`s_\mathrm{min} = \mathrm{IWA} d` and :math:`s_\mathrm{max} = \mathrm{OWA} d`.  For angles given in arcseconds and distances in parsecs, these evaluate to projected separations in AU. 

The lower limit on :math:`\Delta\mathrm{mag}` technically depends on the assumed planet population, but as the density function will be uniformly zero below this limit, it can be taken to be zero for all separations, without loss of generality. The upper limit on :math:`\Delta\mathrm{mag}`, however, is a function of the instrument *and* the integration time (:math:`t_\mathrm{int}`). 

The integration time is typically calculated as the amount of time needed to reach a particular :term:`SNR` with some optical system for a particular :math:`\Delta\mathrm{mag}`.  We can invert this relationship (either analytically or numerically, depending on the optical system model), to compute the largest possible :math:`\Delta\mathrm{mag}` that can be achieved by our instrument on a given star for a given integration time. Since the instrument's performance typically varies with angular separation, we end up with a different :math:`\Delta\mathrm{mag}_\mathrm{max}` for every angular separation even if using a single integration time.

Thus, single-visit completeness is directly a function of integration time.  The relationship is not necessarily invertible, as completeness is strictly bounded (by unity), meaning that completeness will saturate for some value of integration time.  Completeness is also not guaranteed to saturate at unity, for two possible reasons:

#. The projected :term:`IWA` and/or :term:`OWA` for a given star may lie within the bounds of all possible orbit geometries for the selected planet population, such that the maximum obscurational completeness is less than 1.
#. The optical system model may include a noise floor, such that SNR stops increasing with additional integration time past some point.  In this case, :math:`\Delta\mathrm{mag}_\mathrm{max}` will saturate at the noise floor integration time, leading to a maximum photometric completeness of less than 1.

All of this is illustrated in :numref:`fig:compgrid_w_contrast`.  The heatmap shows the joint PDF of the assumed planet population (in log scale) and the three black curves represent  :math:`\Delta\mathrm{mag}_\mathrm{max}(s)` for three different integration times.  All three of the curves have the same limits in :math:`s`, set by the assumed instrument's inner and outer working angles, projected onto one particular target star.  Even though the integration times are logarithmically spaced, we can see that the growth of :math:`\Delta\mathrm{mag}_\mathrm{max}(s)` is not linear on the logarithmic scale of the figure.  In this case, this is due to the particular optical system model employed to generate this data.  This model assumes that SNR increases as approximately :math:`\sqrt{t_\mathrm{int}}`, and that there exists an absolute noise floor.  In this specific case, the noise floor corresponds to an integration time of about 6 days, meaning that any integration time larger than this (including the displayed 10 day curve) will produce exactly the same :math:`\Delta\mathrm{mag}_\mathrm{max}(s)` curve and therefor the same completeness value.

.. _fig:compgrid_w_contrast:
.. figure:: compgrid_w_contrast.png
   :width: 100.0%
   :alt: completeness visualization. 

   Joint PDF of projected separation and :math:`\Delta\mathrm{mag}` with  :math:`\Delta\mathrm{mag}_\mathrm{max}` curves for various integration times.

All of this can get very complicated very quickly, and all of these calculations depend on having high-fidelity models of the instrument and the numerical machinery to invert the calculation of :math:`\Delta\mathrm{mag}_\mathrm{max}` as a function of integration time.  It is typical (especially with well developed instrument models) to make the simplifying assumption (as in [Brown2005]_ and others) that :math:`\Delta\mathrm{mag}` is a constant value (sometimes called :math:`\Delta\mathrm{mag}_0` or :math:`\Delta\mathrm{mag}_\mathrm{lim}` in the literature) for all angular separations and for all targets.  In this case, the calculation of completeness is greatly simplified.  This simplification is made in ``EXOSIMS`` by default, but the full calculation is also available. 

``EXOSIMS`` actually keeps track of 3 sets of completeness, integration time, and :math:`\Delta\mathrm{mag}` values:

#. The integration time and completeness corresponding to user selected :math:`\Delta\mathrm{mag}_\textrm{max}` at a particular angular separation from the target (controlled by inputs ``int_dMag`` and ``WAint`` which can be target-specific or global. This is the default integration time and completeness used in mission scheduling (or as an initial guess for further optimization of integration time allocation between targets).
#. The :math:`\Delta\mathrm{mag}_\textrm{max}` and completeness associated with infinite integration times.  These are the saturation values described above.  In certain cases, the saturation :math:`\Delta\mathrm{mag}_\textrm{max}` may be infinite, but the saturation completeness is always strictly bounded by 1. These values are useful in comparing mission simulation results to theoretically maximum yields. 
#. The :math:`\Delta\mathrm{mag}_\textrm{max}` and completeness associated with the maximum allowable integration time on any target by the mission rules (input variable ``intCutoff``).  In cases where the mission rules do not dictate a cutoff time, these values will be equivalent to the saturation values.  These are used to filter out target stars where no detections are likely for a particular mission setup. 

See :ref:`TargetList` for further details. 
