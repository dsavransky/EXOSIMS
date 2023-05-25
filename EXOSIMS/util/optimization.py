import astropy.units as u
import numpy as np
from scipy.optimize import minimize, milp, Bounds,  LinearConstraint


def maximize_sumcomp_fixed_time(comp, t, tmax):

    res = milp(
        -comp,
        integrality=np.ones(len(comp)),
        bounds=(np.zeros(len(comp)), np.ones(len(comp))),
        constraints=(t.to(u.d).value, 0, tmax.to(u.d).value),
    )

    return res.x


def maximize_sumcomp_no_overhead2(
    TL, sInds, fZ, t0, tmax, maxiter=60, ftol=1e-3, disp=True
):
    """
    Maximize summed completeness without overhead

    Args:
        TL (:ref:`TargetList`):
            TargetList object
        sInds (numpy.ndarray(int)):
            Target star indices (of same size as t)
        fZ (astropy.units.Quantity):
            Surface brightness of local zodiacal light in units of 1/arcsec2
            Same size as sInds
        t0 ((~astropy.units.Quantity):
            Array of initial guess integration times.
        tmax (~astropy.units.Quantity):
            Maximum total integration time
        maxiter (int):
            Maximum number of iterations. Defaults to 60
        ftol (float):
            Convergence tolerance. Defaults to 1e-3.
        disp (bool):
            Print convergence messages. Defaults True.

    Returns:
        tuple:
            t (astropy.units.Quantity(numpy.ndarray(float))):
                Optimized integration times
            scomp (float):
                Optimized summed completeness

    """

    bounds = Bounds(lb=np.zeros(len(sInds)), ub=[tmax.to(u.d).value]*len(sInds))
    linear_constraint = LinearConstraint(np.array(np.ones(len(sInds)),ndmin=2), [0], [tmax.to(u.d).value])

    ires = minimize(
        sumcomp_objfun,
        t0.to(u.d).value,
        jac=sumcomp_objfun_deriv,
        args=(TL, sInds, fZ),
        constraints=[linear_constraint],
        method="trust-constr",
        bounds=bounds,
        options={"maxiter": maxiter, "ftol": ftol, "disp": disp},
    )

    assert ires["success"], "Summed completeness optimization failed."

    t = ires["x"] * u.d
    t[t < 1 * u.s] = 0 * u.d
    scomp = -ires["fun"]

    return t, scomp


def maximize_sumcomp_no_overhead(
    TL, sInds, fZ, t0, tmax, maxiter=60, ftol=1e-3, disp=True
):
    """
    Maximize summed completeness without overhead

    Args:
        TL (:ref:`TargetList`):
            TargetList object
        sInds (numpy.ndarray(int)):
            Target star indices (of same size as t)
        fZ (astropy.units.Quantity):
            Surface brightness of local zodiacal light in units of 1/arcsec2
            Same size as sInds
        t0 ((~astropy.units.Quantity):
            Array of initial guess integration times.
        tmax (~astropy.units.Quantity):
            Maximum total integration time
        maxiter (int):
            Maximum number of iterations. Defaults to 60
        ftol (float):
            Convergence tolerance. Defaults to 1e-3.
        disp (bool):
            Print convergence messages. Defaults True.

    Returns:
        tuple:
            t (astropy.units.Quantity(numpy.ndarray(float))):
                Optimized integration times
            scomp (float):
                Optimized summed completeness

    """

    constraints = {
        "type": "ineq",
        "fun": lambda x: tmax.to(u.d).value - np.sum(x),
        "jac": lambda x: np.ones(len(x)) * -1.0,
    }
    bounds = [(0, tmax.to(u.d).value) for i in np.arange(len(sInds))]

    ires = minimize(
        sumcomp_objfun,
        t0.to(u.d).value,
        jac=sumcomp_objfun_deriv,
        args=(TL, sInds, fZ),
        constraints=constraints,
        method="SLSQP",
        bounds=bounds,
        options={"maxiter": maxiter, "ftol": ftol, "disp": disp},
    )

    assert ires["success"], "Summed completeness optimization failed."

    t = ires["x"] * u.d
    t[t < 1 * u.s] = 0 * u.d
    scomp = -ires["fun"]

    return t, scomp


def sumcomp_objfun(t, TargetList, sInds, fZ):
    """
    Objective Function to maximize summed completeness

    Args:
        t (numpy.ndarray(float)):
            Integration times in days. NB: NOT an astropy quantity.
        TargetList (:ref:`TargetList`):
            TargetList object
        sInds (numpy.ndarray(int)):
            Target star indices (of same size as t)
        fZ (astropy.units.Quantity):
            Surface brightness of local zodiacal light in units of 1/arcsec2
            Same size as t

    """
    good = np.isfinite(t) & (t * u.d >= 0.1 * u.s)

    comp = TargetList.Completeness.comp_per_intTime(
        t[good] * u.d,
        TargetList,
        sInds[good],
        fZ[good],
        TargetList.ZodiacalLight.fEZ0,
        TargetList.int_WA[sInds][good],
        TargetList.default_mode,
    )
    return -comp.sum()


def sumcomp_objfun_deriv(t, TargetList, sInds, fZ):
    """
    Jacobian of maximized summed completeness objective function.

    Args:
        t (astropy Quantity):
            Integration times in days. NB: NOT an astropy quantity.
        sInds (ndarray):
            Target star indices (of same size as t)
        fZ (astropy.units.Quantity):
            Surface brightness of local zodiacal light in units of 1/arcsec2
            Same size as t

    """
    good = np.isfinite(t) & (t * u.d >= 0.1 * u.s)

    tmp = (
        TargetList.Completeness.dcomp_dt(
            t[good] * u.d,
            TargetList,
            sInds[good],
            fZ[good],
            TargetList.ZodiacalLight.fEZ0,
            TargetList.int_WA[sInds][good],
            TargetList.default_mode,
        )
        .to("1/d")
        .value
    )

    jac = np.zeros(len(t))
    jac[good] = tmp
    return -jac
