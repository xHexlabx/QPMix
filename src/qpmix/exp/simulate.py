"""Synthetic measurements, for testing the analysis end to end.

Real measurements come with no ground truth: you cannot check that an
analysis recovered the right gap voltage when nobody knows what it was.
These generators build data from *known* parameters and then add the
distortions a real measurement has -- an offset, a series resistance, gain
errors, a hysteretic double sweep, noise -- so the analysis in
:mod:`qpmix.exp` can be checked by asking whether it recovers the numbers
that went in.

That makes the test suite a round-trip test rather than a comparison
against another implementation, and it is what lets QPMix be tested without
shipping any measured data files.

Examples:

    >>> import numpy as np
    >>> from qpmix.exp.simulate import simulate_dciv
    >>> raw = simulate_dciv(vgap=2.8e-3, rn=14.0, noise=0.0)
    >>> raw.shape[1]
    2

"""

from __future__ import annotations

import numpy as np

from qpmix.exp.tucker import pumped_iv_curve
from qpmix.mathfn.ivcurve_models import expanded, exponential
from qpmix.respfn import RespFnFromIVData

__all__ = [
    "simulate_dciv",
    "simulate_embedded_iv",
    "simulate_if_power",
    "simulate_pumped_iv",
    "simulate_response",
]


def _ideal_iv(volt_v, vgap, rn, rsg, agap, model="expanded", ileak=5e-6):
    """The noise-free I-V curve, in SI units.

    Note:
        ``model="expanded"`` is the default because it includes a leakage
        current that is *non-linear* near zero bias.  That matters for
        testing :func:`qpmix.exp.iv_data.dciv_curve`: if the subgap region
        is perfectly ohmic, a voltage offset and a current offset produce
        exactly the same distortion, so no analysis can separate them.  Real
        junctions have curvature there, and so does this model.

    Args:
        volt_v (ndarray): Bias voltage in volts.
        vgap (float): Gap voltage in volts.
        rn (float): Normal resistance in ohms.
        rsg (float): Subgap resistance in ohms.
        agap (float): Gap sharpness.
        model (str, optional): ``"expanded"`` or ``"exponential"``.  Default
            is ``"expanded"``.
        ileak (float, optional): Leakage current amplitude in amps, for the
            expanded model.  Default is 5e-6.

    Returns:
        ndarray: Tunneling current in amps.

    Raises:
        ValueError: If the model name is not recognised.

    """
    vnorm = volt_v / vgap
    if model == "exponential":
        inorm = exponential(vnorm, vgap=vgap, rn=rn, rsg=rsg, agap=agap)
    elif model == "expanded":
        inorm = expanded(
            vnorm,
            vgap=vgap,
            rn=rn,
            rsg=rsg,
            agap=agap,
            ileak=ileak,
            inot=0.0,
            ioff=0.0,
        )
    else:
        raise ValueError(f"Unknown I-V model: {model!r}")
    return inorm * vgap / rn


def simulate_dciv(
    vgap: float = 2.8e-3,
    rn: float = 14.0,
    rsg: float = 800.0,
    agap: float = 4e4,
    vmax: float = 6e-3,
    npts: int = 3001,
    voffset: float = 0.0,
    ioffset: float = 0.0,
    noise: float = 0.0,
    double_sweep: bool = False,
    model: str = "expanded",
    v_fmt: str = "mV",
    i_fmt: str = "mA",
    seed: int = 0,
) -> np.ndarray:
    """Generate a synthetic unpumped DC I-V measurement.

    Args:
        vgap (float, optional): Gap voltage in volts.  Default is 2.8e-3.
        rn (float, optional): Normal resistance in ohms.  Default is 14.
        rsg (float, optional): Subgap resistance in ohms.  Default is 800.
        agap (float, optional): Gap sharpness.  Default is 4e4.
        vmax (float, optional): Sweep limit in volts.  Default is 6e-3.
        npts (int, optional): Points in the sweep.  Default is 3001.
        voffset (float, optional): Voltage offset to add, in volts.  Default
            is 0.
        ioffset (float, optional): Current offset to add, in amps.  Default
            is 0.
        noise (float, optional): Gaussian current noise, as a fraction of
            the gap current.  Default is 0.
        double_sweep (bool, optional): Sweep up, down, then up again, as a
            real measurement does.  Default is False.
        model (str, optional): ``"expanded"`` (default) or
            ``"exponential"``.  See :func:`_ideal_iv` for why the default
            matters when testing offset recovery.
        v_fmt (str, optional): Units for the returned voltage column.
            Default is ``'mV'``.
        i_fmt (str, optional): Units for the returned current column.
            Default is ``'mA'``.
        seed (int, optional): Seed for the noise.  Default is 0.

    Returns:
        ndarray: Two-column array of voltage and current, in the requested
        units.

    """
    from qpmix.exp.iv_data import CURR_UNITS, VOLT_UNITS

    volt_v = np.linspace(-vmax, vmax, npts)
    if double_sweep:
        volt_v = np.r_[
            np.linspace(0, vmax, npts // 2),
            np.linspace(vmax, -vmax, npts),
            np.linspace(-vmax, 0, npts // 2),
        ]

    curr_a = _ideal_iv(volt_v, vgap, rn, rsg, agap, model=model)
    if noise:
        rng = np.random.default_rng(seed)
        curr_a = curr_a + rng.normal(0, noise * vgap / rn, curr_a.shape)

    return np.column_stack(
        (
            (volt_v + voffset) / VOLT_UNITS[v_fmt],
            (curr_a + ioffset) / CURR_UNITS[i_fmt],
        )
    )


def simulate_response(
    vgap: float = 2.8e-3,
    rn: float = 14.0,
    rsg: float = 800.0,
    agap: float = 4e4,
    vmax: float = 35.0,
    npts: int = 20001,
    model: str = "expanded",
):
    """Build the response function matching :func:`simulate_dciv`.

    Args:
        vgap (float, optional): Gap voltage in volts.  Default is 2.8e-3.
        rn (float, optional): Normal resistance in ohms.  Default is 14.
        rsg (float, optional): Subgap resistance in ohms.  Default is 800.
        agap (float, optional): Gap sharpness.  Default is 4e4.
        vmax (float, optional): Normalized voltage limit.  Default is 35.
        npts (int, optional): Points in the source curve.  Default is 20001.
        model (str, optional): Must match what :func:`simulate_dciv` used.
            Default is ``"expanded"``.

    Returns:
        qpmix.respfn.RespFn: The matching response function.

    """
    v = np.linspace(0, vmax, npts)
    i = _ideal_iv(v * vgap, vgap, rn, rsg, agap, model=model) * rn / vgap
    return RespFnFromIVData(v, i, verbose=False)


def simulate_pumped_iv(
    alpha: float = 0.8,
    freq: float = 230.0,
    vgap: float = 2.8e-3,
    rn: float = 14.0,
    rsg: float = 800.0,
    agap: float = 4e4,
    vmax: float = 6e-3,
    npts: int = 3001,
    voffset: float = 0.0,
    ioffset: float = 0.0,
    noise: float = 0.0,
    num_b: int = 20,
    v_fmt: str = "mV",
    i_fmt: str = "mA",
    seed: int = 0,
    resp=None,
) -> np.ndarray:
    """Generate a synthetic pumped I-V measurement at a known drive level.

    The curve comes from Tucker theory, so the drive level that produced it
    is known exactly and can be compared against what the analysis recovers.

    Args:
        alpha (float, optional): Drive level.  Default is 0.8.
        freq (float, optional): LO frequency in GHz.  Default is 230.
        vgap (float, optional): Gap voltage in volts.  Default is 2.8e-3.
        rn (float, optional): Normal resistance in ohms.  Default is 14.
        rsg (float, optional): Subgap resistance in ohms.  Default is 800.
        agap (float, optional): Gap sharpness.  Default is 4e4.
        vmax (float, optional): Sweep limit in volts.  Default is 6e-3.
        npts (int, optional): Points in the sweep.  Default is 3001.
        voffset (float, optional): Voltage offset in volts.  Default is 0.
        ioffset (float, optional): Current offset in amps.  Default is 0.
        noise (float, optional): Gaussian current noise, as a fraction of
            the gap current.  Default is 0.
        num_b (int, optional): Bessel summation limit.  Default is 20.
        v_fmt (str, optional): Units for the voltage column.  Default is
            ``'mV'``.
        i_fmt (str, optional): Units for the current column.  Default is
            ``'mA'``.
        seed (int, optional): Seed for the noise.  Default is 0.
        resp (qpmix.respfn.RespFn, optional): Reuse a response function
            instead of building one.  Default is None.

    Returns:
        ndarray: Two-column array of voltage and current.

    """
    import scipy.constants as sc

    from qpmix.exp.iv_data import CURR_UNITS, VOLT_UNITS

    if resp is None:
        resp = simulate_response(vgap=vgap, rn=rn, rsg=rsg, agap=agap)

    fgap = sc.e * vgap / sc.h
    vph = freq * 1e9 / fgap

    volt_v = np.linspace(-vmax, vmax, npts)
    inorm = pumped_iv_curve(resp, volt_v / vgap, vph, alpha, num_b=num_b)
    curr_a = inorm * vgap / rn
    if noise:
        rng = np.random.default_rng(seed)
        curr_a = curr_a + rng.normal(0, noise * vgap / rn, curr_a.shape)

    return np.column_stack(
        (
            (volt_v + voffset) / VOLT_UNITS[v_fmt],
            (curr_a + ioffset) / CURR_UNITS[i_fmt],
        )
    )


def simulate_if_power(
    ivdata: np.ndarray,
    shot_slope: float = 5.8,
    if_noise: float = 8.0,
    gain: float = 1.0,
    vgap: float = 2.8e-3,
    vint: float = 2.65e-3,
    npts: int = 2001,
    vmax: float = 6e-3,
    noise: float = 0.0,
    v_fmt: str = "mV",
    seed: int = 0,
) -> np.ndarray:
    """Generate synthetic IF output power versus bias voltage.

    Above the gap the IF power is dominated by shot noise, which rises
    linearly with bias voltage.  Extrapolating that line back to ``vint``
    gives the IF noise contribution -- the basis of Woody's method, which
    :func:`qpmix.exp.if_data.dcif_data` implements.  This generator builds
    data with a known slope and known intercept so that recovery can be
    checked.

    Args:
        ivdata (ndarray): Unused placeholder kept for symmetry with the
            other generators; pass ``None``.
        shot_slope (float, optional): Shot-noise slope in K/mV.  Default is
            5.8.
        if_noise (float, optional): IF noise contribution in K.  Default is
            8.
        gain (float, optional): Arbitrary power scaling, to mimic a
            measurement in arbitrary units.  Default is 1.
        vgap (float, optional): Gap voltage in volts.  Default is 2.8e-3.
        vint (float, optional): Intercept voltage in volts.  Default is
            2.65e-3.
        npts (int, optional): Points in the sweep.  Default is 2001.
        vmax (float, optional): Sweep limit in volts.  Default is 6e-3.
        noise (float, optional): Gaussian power noise in K.  Default is 0.
        v_fmt (str, optional): Units for the voltage column.  Default is
            ``'mV'``.
        seed (int, optional): Seed for the noise.  Default is 0.

    Returns:
        ndarray: Two-column array of voltage and IF power.

    """
    from qpmix.exp.iv_data import VOLT_UNITS

    volt_v = np.linspace(0.0, vmax, npts)
    # Shot noise above the gap, extrapolating back to if_noise at vint.
    power_k = if_noise + shot_slope * (volt_v - vint) * 1e3
    power_k = np.where(volt_v > vgap, power_k, if_noise)
    if noise:
        rng = np.random.default_rng(seed)
        power_k = power_k + rng.normal(0, noise, power_k.shape)

    return np.column_stack((volt_v / VOLT_UNITS[v_fmt], power_k * gain))


def simulate_embedded_iv(
    vt: float = 0.35,
    zt: complex = 0.4 - 0.3j,
    freq: float = 230.0,
    vgap: float = 2.8e-3,
    rn: float = 14.0,
    rsg: float = 800.0,
    agap: float = 4e4,
    vmax: float = 6e-3,
    npts: int = 1201,
    num_b: int = 20,
    noise: float = 0.0,
    v_fmt: str = "mV",
    i_fmt: str = "mA",
    seed: int = 0,
    resp=None,
) -> np.ndarray:
    """Generate a pumped I-V curve for a junction fed by a Thevenin source.

    Unlike :func:`simulate_pumped_iv`, which drives the junction at a fixed
    drive level (an ideal voltage source, so the recovered embedding
    impedance is zero by construction), this solves

        ``vj = vt - zt * Iac(vj)``

    at every bias point.  The junction therefore sees a real source
    impedance, and :class:`qpmix.exp.PumpedData` has something non-trivial
    to recover.  The solution uses the same Tucker-theory AC current that
    the recovery does, so the test is a genuine round trip rather than a
    tautology.

    Args:
        vt (float, optional): Thevenin voltage, normalized to the gap
            voltage.  Default is 0.35.
        zt (complex, optional): Thevenin impedance, normalized to ``rn``.
            Default is ``0.4 - 0.3j``.
        freq (float, optional): LO frequency in GHz.  Default is 230.
        vgap (float, optional): Gap voltage in volts.  Default is 2.8e-3.
        rn (float, optional): Normal resistance in ohms.  Default is 14.
        rsg (float, optional): Subgap resistance in ohms.  Default is 800.
        agap (float, optional): Gap sharpness.  Default is 4e4.
        vmax (float, optional): Sweep limit in volts.  Default is 6e-3.
        npts (int, optional): Points in the sweep.  Default is 1201.
        num_b (int, optional): Bessel summation limit.  Default is 20.
        noise (float, optional): Gaussian current noise, as a fraction of
            the gap current.  Default is 0.
        v_fmt (str, optional): Units for the voltage column.  Default is
            ``'mV'``.
        i_fmt (str, optional): Units for the current column.  Default is
            ``'mA'``.
        seed (int, optional): Seed for the noise.  Default is 0.
        resp (qpmix.respfn.RespFn, optional): Reuse a response function.
            Default is None.

    Returns:
        ndarray: Two-column array of voltage and current.

    """
    import scipy.constants as sc

    from qpmix.exp.iv_data import CURR_UNITS, VOLT_UNITS
    from qpmix.exp.tucker import ac_current

    if resp is None:
        resp = simulate_response(vgap=vgap, rn=rn, rsg=rsg, agap=agap)

    fgap = sc.e * vgap / sc.h
    vph = freq * 1e9 / fgap
    volt_v = np.linspace(-vmax, vmax, npts)
    vb = volt_v / vgap

    # The junction voltage is real by choice of time origin, so the source
    # relation Vt = Vj + Zt * Iac(Vj) fixes |Vt| but leaves its phase free.
    # Solve |Vj + Zt * Iac(Vj)| = |Vt| for Vj >= 0 at every bias point, by
    # bisection -- the left-hand side rises monotonically with Vj.
    def source_magnitude(vj):
        """|Vt| required to drive the junction to this AC voltage."""
        iac = ac_current(resp, vb, vph, vj / vph, num_b=num_b)
        return np.abs(vj + zt * iac)

    lo = np.zeros(vb.shape)
    hi = np.full(vb.shape, max(abs(vt) * 4.0, 1e-3))
    for _ in range(60):
        if (source_magnitude(hi) >= abs(vt)).all():
            break
        hi *= 2.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        too_small = source_magnitude(mid) < abs(vt)
        lo = np.where(too_small, mid, lo)
        hi = np.where(too_small, hi, mid)
    vj = 0.5 * (lo + hi)

    alpha = vj / vph
    inorm = pumped_iv_curve(resp, vb, vph, alpha, num_b=num_b)
    curr_a = inorm * vgap / rn
    if noise:
        rng = np.random.default_rng(seed)
        curr_a = curr_a + rng.normal(0, noise * vgap / rn, curr_a.shape)

    return np.column_stack((volt_v / VOLT_UNITS[v_fmt], curr_a / CURR_UNITS[i_fmt]))
