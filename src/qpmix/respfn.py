"""The SIS junction response function.

The response function is the complex quantity

    ``resp(V) = Ikk(V) + 1j * Idc(V)``

where ``Idc`` is the DC I-V curve and ``Ikk`` is its Kramers-Kronig
transform.  It is what :func:`qpmix.qtcurrent.qtcurrent` evaluates, over and
over, at every photon-step-shifted bias voltage.

Classes
-------

============================  ==================================================
:class:`RespFn`               From pre-sampled DC I-V data.
:class:`RespFnFromIVData`     From arbitrary DC I-V data (resampled first).
:class:`RespFnPerfect`        From the ideal step I-V curve.
:class:`RespFnPolynomial`     From the Kennedy (1999) polynomial model.
:class:`RespFnExponential`    From the Rashid *et al.* (2016) model.
============================  ==================================================

How this differs from upstream QMix
-----------------------------------

QMix subsamples the curve down to ~100 knots chosen by curvature and fits a
cubic spline through them, trading accuracy for evaluation speed, because
every spline evaluation costs a binary search.

QPMix removes that trade-off.  The curve is tabulated on a *dense uniform*
grid and evaluated by index arithmetic (:class:`qpmix.interp.
UniformInterpolator`), which is O(1) per point regardless of how many
points the table holds.  So the table can be ~500x denser than QMix's knot
set and still be faster to evaluate — accuracy and speed both improve.

Examples:

    >>> import numpy as np
    >>> resp = RespFnPolynomial(50, verbose=False)
    >>> np.round(resp.idc(np.array([0.5, 1.0, 2.0])), 1)
    array([0. , 0.5, 2. ])
    >>> np.round(resp.ikk(np.array([0.5, 1.0, 2.0])), 1)
    array([-0.5,  1.1,  0.1])
    >>> np.round(resp(np.array([0.5, 1.0, 2.0])), 1)
    array([-0.5+0.j ,  1.1+0.5j,  0.1+2.j ])

"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline

import qpmix.mathfn.ivcurve_models as iv
from qpmix.interp import UniformInterpolator
from qpmix.mathfn.filters import gauss_conv
from qpmix.mathfn.kktrans import kk_trans

__all__ = [
    "RespFn",
    "RespFnExponential",
    "RespFnFromIVData",
    "RespFnPerfect",
    "RespFnPolynomial",
]

#: The response function is tabulated over ``[-VRANGE, VRANGE]``, normalized
#: to the gap voltage.  This must cover ``max(vb) + num_b * sum(freq)``.
VRANGE = 35.0

#: Default grid spacing of the tabulated response function.  1/DV_DEFAULT is
#: an integer, so the gap at ``v = +/-1`` lands exactly on a grid node.
DV_DEFAULT = 5e-4

#: Default zero-padding factor for the Hilbert transform.  The padded range
#: is ``kk_n`` times the voltage span, *independent of the grid spacing*, so
#: this controls physical padding, not resolution.  QMix defaults to 50;
#: measured against a factor-200 reference, dropping to 20 changes the
#: transform by under 4e-7 -- two orders of magnitude below the
#: interpolation error -- while halving the one-off construction cost.
KK_N_DEFAULT = 20

#: Default parameters, matching the keyword arguments of :class:`RespFn`.
DEFAULT_PARAMS: dict[str, Any] = {
    "verbose": True,
    "v_smear": None,
    "kk_n": KK_N_DEFAULT,
    "dv": DV_DEFAULT,
    "vrange": VRANGE,
    "vlimit": None,
}

#: A measured curve handed to :class:`RespFnFromIVData` must reach at least
#: this normalized voltage, so that the ohmic continuation starts on the
#: normal-state branch rather than in the transition.
VLIMIT_MIN = 1.8


def _default_params(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Merge user keyword arguments over :data:`DEFAULT_PARAMS`.

    Args:
        kwargs (dict): User-supplied keyword arguments.

    Returns:
        dict: The merged parameters.

    Raises:
        TypeError: If an unknown keyword argument is given.

    """
    unknown = set(kwargs) - set(DEFAULT_PARAMS)
    if unknown:
        raise TypeError(f"Unexpected keyword argument(s): {sorted(unknown)}")
    params = dict(DEFAULT_PARAMS)
    params.update(kwargs)
    return params


def _uniform_grid(vrange: float, dv: float) -> np.ndarray:
    """Build the symmetric uniform voltage grid the response is tabulated on.

    Args:
        vrange (float): Half-width of the grid.
        dv (float): Target grid spacing; adjusted so that ``v = 0`` and
            ``v = +/-vrange`` are both grid nodes.

    Returns:
        ndarray: The grid, from ``-vrange`` to ``+vrange``.

    """
    half = round(vrange / dv)
    return np.linspace(-vrange, vrange, 2 * half + 1)


def _perfect_kk_tabulated(voltage: np.ndarray) -> np.ndarray:
    """Tabulate the analytic KK transform of the perfect I-V curve.

    ``perfect_kk`` diverges logarithmically at ``v = +/-1``.  QMix
    substitutes a large sentinel (100) there, which is fine when the
    function is called directly but would poison a neighbouring
    interpolation stencil.  Here the singular nodes are filled by linear
    interpolation from their neighbours instead, so the tabulated curve
    stays smooth and bounded.

    Args:
        voltage (ndarray): Uniform voltage grid.

    Returns:
        ndarray: The tabulated KK transform.

    """
    kk = iv.perfect_kk(voltage, max_kk=np.nan)
    bad = ~np.isfinite(kk)
    if bad.any():
        kk[bad] = np.interp(voltage[bad], voltage[~bad], kk[~bad])
    return kk


class RespFn:
    """Response function built from pre-sampled DC I-V data.

    The input data must start at ``voltage[0] == 0``, be uniformly spaced,
    and extend past ``v = 5``.  It is reflected about the origin, optionally
    smeared, Kramers-Kronig transformed, and resampled onto the uniform
    interpolation grid.

    If your data is not uniformly spaced, use :class:`RespFnFromIVData`.

    Args:
        voltage (ndarray): Normalized DC bias voltage, starting at 0.
        current (ndarray): Normalized DC tunneling current.

    Keyword Args:
        verbose (bool): Print progress to the terminal.  Default is True.
        v_smear (float): Smear the DC I-V curve by convolving with a
            Gaussian of this standard deviation.  Default is None.
        kk_n (int): Zero-padding factor for the Hilbert transform.  Default
            is 20 (see :data:`KK_N_DEFAULT`).
        dv (float): Grid spacing of the interpolation table.  Default is
            5e-4.
        vrange (float): The table spans ``[-vrange, vrange]``.  Default is
            35.

    Attributes:
        voltage (ndarray): DC bias voltage of the tabulated curve.
        current (ndarray): DC tunneling current, tabulated.
        voltage_kk (ndarray): Same as ``voltage``; kept for QMix
            compatibility.
        current_kk (ndarray): KK transform of the DC I-V curve, tabulated.
        table (qpmix.interp.UniformInterpolator): The complex response
            function, ``ikk + 1j * idc``.

    Raises:
        ValueError: If the input data does not meet the requirements above.

    """

    def __init__(self, voltage: np.ndarray, current: np.ndarray, **params: Any) -> None:
        opts = _default_params(params)
        if opts["verbose"]:
            print("Generating response function:")

        voltage = np.asarray(voltage, dtype=float)
        current = np.asarray(current, dtype=float)
        if voltage.shape != current.shape:
            raise ValueError("voltage and current must have the same shape.")
        if voltage[0] != 0.0:
            raise ValueError("First voltage value must be zero.")
        if voltage[-1] <= 5:
            raise ValueError("Voltage must extend to at least 5.")

        # Reflect about the origin: the I-V curve is odd.
        voltage = np.r_[-voltage[::-1][:-1], voltage]
        current = np.r_[-current[::-1][:-1], current]

        # Resample onto the uniform interpolation grid.  Cubic through the
        # supplied points; this is the one and only spline evaluation, paid
        # once at construction instead of millions of times per simulation.
        grid = _uniform_grid(opts["vrange"], opts["dv"])
        spline = InterpolatedUnivariateSpline(voltage, current, k=3)
        current = np.asarray(spline(grid), dtype=float)
        voltage = grid
        v_step = float(grid[1] - grid[0])

        # Outside the supplied data range the curve is ohmic.
        outside = np.abs(voltage) > np.abs(spline.get_knots()[-1])
        current[outside] = voltage[outside]

        if opts["v_smear"] is not None:
            current = (
                gauss_conv(current - voltage, sigma=opts["v_smear"] / v_step) + voltage
            )
            if opts["verbose"]:
                print(f" - Voltage smear: {opts['v_smear']:.4f}")

        current_kk = kk_trans(voltage, current, opts["kk_n"])

        self.voltage = voltage
        self.current = current
        self.voltage_kk = voltage
        self.current_kk = current_kk
        self._build_tables(voltage, current, current_kk)

        if opts["verbose"]:
            print(f" - Tabulated on {voltage.size} points, dv = {v_step:.2e}")
            print("")

    def _build_tables(
        self, voltage: np.ndarray, current: np.ndarray, current_kk: np.ndarray
    ) -> None:
        """Build the interpolation tables from the tabulated curves.

        Args:
            voltage (ndarray): Uniform voltage grid.
            current (ndarray): DC tunneling current on that grid.
            current_kk (ndarray): KK transform on that grid.

        """
        dv = float(voltage[1] - voltage[0])
        self.table = UniformInterpolator(
            voltage[0], dv, np.ascontiguousarray(current_kk + 1j * current)
        )
        # Derivatives share the grid, so they are just another table.
        didc = np.gradient(current, dv)
        dikk = np.gradient(current_kk, dv)
        self._d_table = UniformInterpolator(
            voltage[0], dv, np.ascontiguousarray(dikk + 1j * didc)
        )

    def __str__(self) -> str:  # pragma: no cover - debugging aid
        return "Response function object: RespFn"

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return self.__str__()

    def __call__(self, vbias: np.ndarray | float) -> np.ndarray | complex:
        """Interpolate the complex response function.

        Args:
            vbias (ndarray or float): Normalized bias voltage.

        Returns:
            ndarray or complex: ``ikk + 1j * idc``.

        """
        return self.table(vbias)

    def resp(self, vbias: np.ndarray | float) -> np.ndarray | complex:
        """Interpolate the complex response function.

        Args:
            vbias (ndarray or float): Normalized bias voltage.

        Returns:
            ndarray or complex: ``ikk + 1j * idc``.

        """
        return self.table(vbias)

    def resp_conj(self, vbias: np.ndarray | float) -> np.ndarray | complex:
        """The complex conjugate of the response function.

        Equivalent to ``np.conj(resp(vb))`` but computed in one pass.

        Args:
            vbias (ndarray or float): Normalized bias voltage.

        Returns:
            ndarray or complex: ``ikk - 1j * idc``.

        """
        return np.conj(self.table(vbias))

    def resp_swap(self, vbias: np.ndarray | float) -> np.ndarray | complex:
        """The response function with real and imaginary parts swapped.

        Equivalent to ``1j * np.conj(resp(vb))``.

        Args:
            vbias (ndarray or float): Normalized bias voltage.

        Returns:
            ndarray or complex: ``idc + 1j * ikk``.

        """
        return 1j * np.conj(self.table(vbias))

    def idc(self, vbias: np.ndarray | float) -> np.ndarray | float:
        """Interpolate the DC I-V curve.

        This is the imaginary part of the response function.

        Args:
            vbias (ndarray or float): Normalized bias voltage.

        Returns:
            ndarray or float: DC tunneling current.

        """
        return np.imag(self.table(vbias))

    def ikk(self, vbias: np.ndarray | float) -> np.ndarray | float:
        """Interpolate the Kramers-Kronig transform of the DC I-V curve.

        This is the real part of the response function.

        Args:
            vbias (ndarray or float): Normalized bias voltage.

        Returns:
            ndarray or float: The KK transform.

        """
        return np.real(self.table(vbias))

    def didc(self, vbias: np.ndarray | float) -> np.ndarray | float:
        """Interpolate ``d(idc)/d(vb)``.

        Not used by QPMix itself, but needed to evaluate Tucker theory
        (Tucker and Feldman, 1985).

        Args:
            vbias (ndarray or float): Normalized bias voltage.

        Returns:
            ndarray or float: Derivative of the DC tunneling current.

        """
        return np.imag(self._d_table(vbias))

    def dikk(self, vbias: np.ndarray | float) -> np.ndarray | float:
        """Interpolate ``d(ikk)/d(vb)``.

        Args:
            vbias (ndarray or float): Normalized bias voltage.

        Returns:
            ndarray or float: Derivative of the KK transform.

        """
        return np.real(self._d_table(vbias))

    def plot(self, ax=None, fig_name=None):  # pragma: no cover - plotting
        """Plot the DC I-V curve and its KK transform.

        Args:
            ax (matplotlib.axes.Axes, optional): Axis to draw on.
            fig_name (str, optional): If given, save to this path and close
                the figure instead of returning the axis.

        Returns:
            matplotlib.axes.Axes: The axis, unless ``fig_name`` was given.

        """
        import matplotlib.pyplot as plt

        fig = None
        if ax is None:
            fig, ax = plt.subplots()
        ax.plot(self.voltage, self.current, "k", label=r"$I_\mathrm{dc}^0(V_0)$")
        ax.plot(
            self.voltage_kk, self.current_kk, "r--", label=r"$I_\mathrm{kk}^0(V_0)$"
        )
        ax.set_xlabel("Bias Voltage (normalized)")
        ax.set_ylabel("Current (normalized)")
        ax.set_xlim([0, 2])
        ax.set_ylim([-1.2, 2])
        ax.legend(loc=0, frameon=True)
        ax.grid()
        if fig_name is not None and fig is not None:
            fig.savefig(fig_name, bbox_inches="tight")
            plt.close(fig)
            return None
        return ax


class RespFnFromIVData(RespFn):
    """Response function built from measured DC I-V data.

    Unlike :class:`RespFn`, the input need not be uniformly spaced, start at
    zero, or extend far past the gap: it is cleaned and continued ohmically
    beyond the last measured point.  Measured curves stop a few millivolts
    above the gap, while the response function has to be evaluated out to
    tens of gap voltages, so that continuation is what makes measured data
    usable at all.

    The continuation has slope one (that is what normalizing by ``Rn``
    means) and an offset ``i - v`` fitted over the top tenth of the measured
    range.  The tail matters more than it looks: the Kramers-Kronig
    transform weights it logarithmically, and on a real junction the offset
    keeps drifting well above two gap voltages, so cutting the curve at 1.8
    (QMix's rule, and this class's former default) instead of using a sweep
    that reaches 3.4 shifts ``ikk`` across the whole sub-gap region by a few
    percent of ``I_gap``.  The shift is nearly uniform, which the tunnelling
    currents cancel, but anything reading ``ikk`` directly sees all of it.
    Every measured point is therefore used unless ``vlimit`` says otherwise.

    Args:
        voltage (ndarray): Normalized DC bias voltage.
        current (ndarray): Normalized DC tunneling current.

    Keyword Args:
        vlimit (float or None): Use the measured data only up to this
            normalized bias voltage.  Default is None, meaning all of it.
            Pass ``1.8`` to reproduce QMix.
        **kwargs: See :class:`RespFn`.

    Raises:
        ValueError: If the data does not reach ``vlimit``, or
            :data:`VLIMIT_MIN` when ``vlimit`` is None.

    """

    def __init__(self, voltage: np.ndarray, current: np.ndarray, **kwargs: Any) -> None:
        voltage = np.asarray(voltage, dtype=float)
        current = np.asarray(current, dtype=float)

        order = np.argsort(voltage)
        voltage, current = voltage[order], current[order]
        keep = np.r_[True, np.diff(voltage) > 0]
        voltage, current = voltage[keep], current[keep]

        opts = _default_params(kwargs)
        vlimit = opts["vlimit"]
        vtop = float(voltage.max()) if vlimit is None else float(vlimit)
        if voltage.max() < vtop or vtop < VLIMIT_MIN:
            raise ValueError(
                f"I-V data only reaches v={voltage.max():.2f}, but it must "
                f"reach at least {VLIMIT_MIN} (vlimit={vlimit}). Supply more "
                "data, or lower 'vlimit' if it was set."
            )

        # Keep the measured curve up to vtop.  Beyond it the curve is ohmic
        # with slope one; the offset comes from a straight-line fit over the
        # top tenth of the used range, evaluated at its end, so that one
        # noisy last point does not set the whole tail.
        mask = (voltage > 0) & (voltage <= vtop)
        voltage, current = voltage[mask], current[mask]
        v_end = float(voltage[-1])
        top = voltage >= v_end - 0.1 * (v_end - voltage[0])
        if np.count_nonzero(top) >= 2:
            _, offset = np.polyfit(voltage[top] - v_end, current[top] - voltage[top], 1)
        else:
            offset = current[-1] - v_end

        grid = np.arange(0.0, opts["vrange"] + opts["dv"] / 2, opts["dv"])
        resampled = np.interp(grid, voltage, current)
        beyond = grid > v_end
        resampled[beyond] = grid[beyond] + float(offset)

        super().__init__(grid, resampled, **kwargs)

    def __str__(self) -> str:  # pragma: no cover - debugging aid
        return "Response function object: RespFnFromIVData"


class RespFnPolynomial(RespFn):
    """Response function from the Kennedy (1999) polynomial model.

    The polynomial order controls how sharp the transition at the gap is,
    which makes this class useful for studying how gap sharpness affects
    mixer gain.

    Args:
        p_order (int, optional): Polynomial order, usually 30-50.  Default
            is 50.

    Keyword Args:
        **kwargs: See :class:`RespFn`.

    """

    def __init__(self, p_order: int = 50, **kwargs: Any) -> None:
        self.p_order = p_order
        opts = _default_params(kwargs)
        v = np.arange(0.0, opts["vrange"] + opts["dv"] / 2, opts["dv"])
        super().__init__(v, iv.polynomial(v, p_order), **kwargs)

    def __str__(self) -> str:  # pragma: no cover - debugging aid
        return f"Response function object: RespFnPolynomial, p = {self.p_order}"


class RespFnExponential(RespFn):
    """Response function from the Rashid *et al.* (2016) exponential model.

    Like :class:`RespFnPolynomial`, but with a finite subgap resistance.

    Args:
        vgap (float, optional): Gap voltage in V.  Default is 2.8e-3.
        rn (float, optional): Normal resistance in ohms.  Default is 14.
        rsg (float, optional): Subgap resistance in ohms.  Default is 1000.
        agap (float, optional): Gap linearity coefficient.  Default is 4e4.

    Keyword Args:
        **kwargs: See :class:`RespFn`.

    """

    def __init__(
        self,
        vgap: float = 2.8e-3,
        rn: float = 14,
        rsg: float = 1000,
        agap: float = 4e4,
        **kwargs: Any,
    ) -> None:
        self.vgap, self.rn, self.rsg, self.agap = vgap, rn, rsg, agap
        opts = _default_params(kwargs)
        v = np.arange(0.0, opts["vrange"] + opts["dv"] / 2, opts["dv"])
        current = iv.exponential(v, vgap, rn, rsg, agap)
        super().__init__(v, current, **kwargs)

    def __str__(self) -> str:  # pragma: no cover - debugging aid
        return f"Response function object: RespFnExponential, Rsg = {self.rsg}"


class RespFnPerfect(RespFn):
    """Response function from the ideal step I-V curve.

    The DC current is exactly zero below the gap and exactly equal to the
    bias voltage above it.  Both the I-V curve and its KK transform are
    known analytically, so no numerical Kramers-Kronig transform is needed.

    The infinitely sharp transition can be softened with the ``v_smear``
    keyword argument.

    Keyword Args:
        **kwargs: See :class:`RespFn`.

    """

    def __init__(self, **kwargs: Any) -> None:
        opts = _default_params(kwargs)
        if opts["verbose"]:
            print("Generating response function:")

        voltage = _uniform_grid(opts["vrange"], opts["dv"])
        dv = float(voltage[1] - voltage[0])

        if opts["v_smear"] is None:
            # Both curves are known analytically.
            current = iv.perfect(voltage)
            current_kk = _perfect_kk_tabulated(voltage)
            if opts["verbose"]:
                print(" - Using analytic response function")
        else:
            current = (
                gauss_conv(iv.perfect(voltage) - voltage, sigma=opts["v_smear"] / dv)
                + voltage
            )
            current_kk = kk_trans(voltage, current, opts["kk_n"])
            if opts["verbose"]:
                print(f" - Voltage smear: {opts['v_smear']:.4f}")

        self.voltage = voltage
        self.current = current
        self.voltage_kk = voltage
        self.current_kk = current_kk
        self._build_tables(voltage, current, current_kk)

        if opts["verbose"]:
            print("")

    def __str__(self) -> str:  # pragma: no cover - debugging aid
        return "Response function object: RespFnPerfect"
