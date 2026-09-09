"""High-level containers for measured SIS mixer data.

:class:`DCData` holds an unpumped DC I-V curve, everything measured from it
(gap voltage, normal and subgap resistance, critical current density) and
the response functions built from it.  :class:`PumpedData` holds a pumped
I-V curve plus optional hot/cold IF measurements, and runs the two analyses
that turn them into physics: recovering the embedding circuit from the
pumped I-V curve, and the noise temperature and gain from the IF data.

Both take NumPy arrays; two columns for I-V and IF data, three for IF
spectra.

Differences from upstream QMix
------------------------------

* Unknown keyword arguments are rejected instead of silently ignored, so a
  mistyped parameter name is caught rather than leaving the default in
  place.
* Impedance recovery goes through :mod:`qpmix.exp.zemb`, which evaluates the
  error surface as one broadcast expression and then polishes the minimum
  off the search grid.
* The drive level comes from :func:`qpmix.exp.tucker.recover_alpha`, a
  safeguarded Newton iteration rather than a fixed number of bisection
  steps.
* Plotting is optional: :mod:`matplotlib` is imported inside the plotting
  methods, so importing this module does not pull it in.

Examples:

    >>> from qpmix.exp import DCData, PumpedData
    >>> from qpmix.exp.simulate import simulate_dciv, simulate_pumped_iv
    >>> dciv = DCData(simulate_dciv(), verbose=False)
    >>> bool(abs(dciv.vgap - 2.8e-3) < 5e-5)
    True
    >>> pumped = PumpedData(simulate_pumped_iv(alpha=0.8), dciv,
    ...                     freq=230.0, verbose=False)
    >>> bool(abs(pumped.alpha - 0.8) < 0.05)
    True

"""

from __future__ import annotations

from typing import Any

import numpy as np
import scipy.constants as sc

from qpmix.exp.if_data import dcif_data, if_data
from qpmix.exp.iv_data import dciv_curve, iv_curve
from qpmix.exp.parameters import merge_params
from qpmix.exp.tucker import ac_current, recover_alpha
from qpmix.exp.zemb import recover_zemb
from qpmix.mathfn import slope_span_n
from qpmix.respfn import RespFnFromIVData

__all__ = ["DCData", "PumpedData"]


class DCData:
    """An unpumped DC I-V curve and everything measured from it.

    Args:
        dciv (ndarray): Two-column array of bias voltage and tunneling
            current.
        dcif (ndarray, optional): Two-column array of bias voltage and IF
            power measured with no LO, used to calibrate the IF power scale.
            Default is None.

    Keyword Args:
        **kwargs: Any parameter from :data:`qpmix.exp.parameters.PARAMS`.

    Attributes:
        voltage (ndarray): Bias voltage, normalized to the gap voltage.
        current (ndarray): Tunneling current, normalized to the gap current.
        dc (qpmix.exp.iv_data.DCIVData): The full set of measured values.
        vgap (float): Gap voltage, in volts.
        igap (float): Gap current, in amps.
        fgap (float): Gap frequency, in Hz.
        rn (float): Normal-state resistance, in ohms.
        rsg (float): Subgap resistance, in ohms.
        q (float): Quality factor ``rsg / rn``.
        rna (float): Resistance-area product, in ohm m^2.
        jc (float): Critical current density, in A/m^2.
        ileak (float): Leakage current at ``vleak``, in amps.
        offset (tuple): The removed ``(voltage, current)`` offset.
        vint (float): Intercept voltage of the normal-state branch.
        resp (qpmix.respfn.RespFn): Response function from this I-V curve.
        resp_smear (qpmix.respfn.RespFn): The same, slightly smeared, which
            approximates a small amount of heating.
        if_data (ndarray or None): Calibrated DC IF data.
        dcif (qpmix.exp.if_data.DCIFData or None): IF calibration.
        if_noise (float or None): IF noise contribution, in kelvin.

    Raises:
        ValueError: If the input is not a NumPy array.

    """

    def __init__(
        self, dciv: np.ndarray, dcif: np.ndarray | None = None, **kwargs: Any
    ) -> None:
        kw = merge_params(kwargs)
        self.kwargs = kw
        self.comment = kw["comment"]
        self.vleak = kw["vleak"]

        if not isinstance(dciv, np.ndarray):
            raise ValueError("DC I-V data must be a NumPy array.")

        self.voltage, self.current, self.dc = dciv_curve(dciv, **kwargs)

        self.vgap = self.dc.vgap
        self.igap = self.dc.igap
        self.fgap = self.dc.fgap
        self.rn = self.dc.rn
        self.rsg = self.dc.rsg
        self.q = self.rsg / self.rn
        self.rna = self.rn * kw["area"] * 1e-12
        self.jc = self.vgap / self.rna
        self.offset = self.dc.offset
        self.vint = self.dc.vint
        self.ileak = (
            float(np.interp(self.vleak / self.vgap, self.voltage, self.current))
            * self.igap
        )

        self.resp = RespFnFromIVData(
            self.voltage, self.current, verbose=False, v_smear=None
        )
        self.resp_smear = RespFnFromIVData(
            self.voltage, self.current, verbose=False, v_smear=kw["v_smear"]
        )

        if dcif is not None:
            self.if_data, self.dcif = dcif_data(dcif, self.dc, **kwargs)
            self.if_noise = self.dcif.if_noise
            self.corr = self.dcif.corr
            self.shot_slope = self.dcif.shot_slope
            self.if_fit = self.dcif.if_fit
        else:
            self.if_data = None
            self.dcif = None
            self.if_noise = None
            self.corr = None
            self.shot_slope = None
            self.if_fit = None

        if kw["verbose"]:
            print(self.summary())

    def summary(self) -> str:
        """Return a human-readable summary of the measured values.

        Returns:
            str: A multi-line summary.

        """
        lines = [
            f"DC I-V data: {self.comment}",
            f"\tVgap:     \t{self.vgap * 1e3:6.2f}\tmV",
            f"\tfgap:     \t{self.fgap / 1e9:6.2f}\tGHz",
            f"\tRn:       \t{self.rn:6.2f}\tohms",
            f"\tRsg:      \t{self.rsg:6.2f}\tohms",
            f"\tQ:        \t{self.q:6.2f}",
            f"\tJc:       \t{self.jc / 1e7:6.2f}\tkA/cm^2",
            f"\tIleak:    \t{self.ileak * 1e6:6.2f}\tuA",
            f"\tOffset:   \t{self.offset[0] * 1e3:6.2f}\tmV",
            f"\t          \t{self.offset[1] * 1e6:6.2f}\tuA",
            f"\tVint:     \t{self.vint * 1e3:6.2f}\tmV",
        ]
        if self.if_noise is not None:
            lines.append(f"\tIF noise: \t{self.if_noise:6.2f}\tK")
        return "\n".join(lines)

    def print_info(self) -> None:  # pragma: no cover - thin wrapper
        """Print :meth:`summary` to the terminal."""
        print(self.summary())

    def __str__(self) -> str:  # pragma: no cover - debugging aid
        return self.summary()

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"DCData(Vgap={self.vgap / 1e-3:.2f} mV, Rn={self.rn:.2f} ohms)"

    def plot_dciv(self, ax=None, vmax_plot: float = 4.0):  # pragma: no cover
        """Plot the DC I-V curve.

        Args:
            ax (matplotlib.axes.Axes, optional): Axis to draw on.
            vmax_plot (float, optional): Voltage limit in mV.  Default is 4.

        Returns:
            matplotlib.axes.Axes: The axis.

        """
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots()
        ax.plot(self.voltage * self.vgap * 1e3, self.current * self.igap * 1e6)
        ax.set_xlabel("Bias voltage (mV)")
        ax.set_ylabel("Current (uA)")
        ax.set_xlim([0, vmax_plot])
        ax.grid()
        return ax


class PumpedData:
    """A pumped I-V curve, with optional hot/cold IF measurements.

    Args:
        ivdata (ndarray): Two-column array of bias voltage and pumped
            tunneling current.
        dciv (DCData): The matching unpumped measurement.
        if_hot (ndarray, optional): Two-column hot-load IF data.  Default is
            None.
        if_cold (ndarray, optional): Two-column cold-load IF data.  Default
            is None.

    Keyword Args:
        freq (float): LO frequency in GHz.  Required.
        **kwargs: Any other parameter from
            :data:`qpmix.exp.parameters.PARAMS`.

    Attributes:
        voltage (ndarray): Bias voltage, normalized.
        current (ndarray): Pumped tunneling current, normalized.
        freq (float): LO frequency, in GHz.
        vph (float): Photon voltage, normalized to the gap voltage.
        zt (complex or None): Recovered embedding impedance, normalized.
        vt (float or None): Recovered embedding voltage, normalized.
        zw (complex or None): Junction impedance at mid-step, normalized.
        alpha (float or None): Recovered drive level at mid-step.
        fit_good (bool or None): Whether the impedance fit converged well.
        zemb (qpmix.exp.zemb.ZembResult or None): The full fit result.
        tn (ndarray or None): Noise temperature versus bias, in kelvin.
        gain (ndarray or None): Conversion gain versus bias.
        tn_best (float or None): Noise temperature at the best bias point.
        g_db (float or None): Gain at the best bias point, in dB.

    Raises:
        ValueError: If the input is not a NumPy array or ``freq`` is unset.

    """

    def __init__(
        self,
        ivdata: np.ndarray,
        dciv: DCData,
        if_hot: np.ndarray | None = None,
        if_cold: np.ndarray | None = None,
        **kwargs: Any,
    ) -> None:
        kw = merge_params(kwargs)
        self.kwargs = kw
        self.comment = kw["comment"]

        if not isinstance(ivdata, np.ndarray):
            raise ValueError("Pumped I-V data must be a NumPy array.")
        if kw["freq"] is None:
            raise ValueError("The LO frequency ('freq', in GHz) must be given.")

        self.dciv = dciv
        self.dc = dciv.dc
        self.vgap = dciv.vgap
        self.igap = dciv.igap
        self.fgap = dciv.fgap
        self.rn = dciv.rn
        self.offset = dciv.offset
        self.vint = dciv.vint
        self.voltage_dc = dciv.voltage
        self.current_dc = dciv.current

        self.freq = float(kw["freq"])
        self.vph = self.freq * sc.giga / self.fgap

        self.voltage, self.current = iv_curve(ivdata, self.dc, **kwargs)
        self.rdyn = slope_span_n(self.current, self.voltage, 21)

        if kw["analyze_iv"]:
            self._recover_zemb()
        else:
            self.zt = self.vt = self.zw = self.alpha = None
            self.fit_good = None
            self.zemb = None

        self._analyze_if(if_hot, if_cold, kw, kwargs)

        if kw["verbose"]:
            print(self.summary())

    # -- impedance recovery --------------------------------------------

    def _recover_zemb(self) -> None:
        """Recover the Thevenin equivalent circuit from the pumped I-V curve.

        Uses the RF voltage-match method: the drive level is recovered at
        every bias point on the flat part of the first photon step, giving
        the junction's AC voltage and impedance there, and the Thevenin
        source is then fitted to that load line.
        """
        kw = self.kwargs
        fit_low, fit_high = kw["fit_range"]
        vph = self.vph

        if vph >= 1.0:
            raise ValueError(
                f"At {self.freq:.1f} GHz the photon voltage is "
                f"{vph:.2f} * Vgap, so there is no first photon step below "
                f"the gap to fit. Impedance recovery needs a frequency "
                f"below the gap frequency ({self.fgap / 1e9:.1f} GHz)."
            )

        # Use only the middle of the first photon step, where the curve is
        # linear and the inversion for alpha is well conditioned.
        v_low = 1 - vph + vph * fit_low
        v_high = 1 - vph * (1 - fit_high)
        mask = (v_low <= self.voltage) & (self.voltage <= v_high)
        if mask.sum() < 5:
            raise ValueError(
                f"Only {mask.sum()} bias points fall on the first photon "
                f"step at {self.freq} GHz. Check the frequency and "
                f"'fit_range'."
            )
        vb = self.voltage[mask]
        ib = self.current[mask]
        idx_mid = int(np.abs(vb - (1 - vph / 2.0)).argmin())

        resp = self.dciv.resp
        alpha = recover_alpha(
            resp, vb, ib, vph, alpha_max=kw["alpha_max"], num_b=kw["num_b"]
        )
        vj = alpha * vph
        with np.errstate(divide="ignore", invalid="ignore"):
            zj = vj / ac_current(resp, vb, vph, alpha, num_b=kw["num_b"])

        result = recover_zemb(
            vj,
            zj,
            remb_range=kw["remb_range"],
            xemb_range=kw["xemb_range"],
            zemb=kw["zemb"],
            refine=kw["zemb_refine"],
        )

        self.zemb = result
        self.zt = result.zt
        self.vt = result.vt
        self.fit_good = result.fit_good
        self.err_surf = result.err_surf
        self.zw = zj[idx_mid]
        self.alpha = float(alpha[idx_mid])

    # -- IF analysis ---------------------------------------------------

    def _analyze_if(self, if_hot, if_cold, kw, kwargs) -> None:
        """Analyze hot/cold IF data, if it was supplied.

        Args:
            if_hot (ndarray or None): Hot-load IF data.
            if_cold (ndarray or None): Cold-load IF data.
            kw (dict): Merged parameters.
            kwargs (dict): The user's original keyword arguments.

        """
        blank = (
            "if_hot",
            "if_cold",
            "tn",
            "gain",
            "idx_best",
            "if_noise",
            "shot_slope",
            "tn_best",
            "gain_best",
            "g_db",
            "v_best",
            "zj_if",
        )
        if if_hot is None or if_cold is None or not kw["analyze_if"]:
            for name in blank:
                setattr(self, name, None)
            self.good_if_noise_fit = None
            return

        results, idx_best, dcif = if_data(
            if_hot, if_cold, self.dc, dcif=self.dciv.dcif, **kwargs
        )
        self.if_hot = results[:, :2]
        self.if_cold = np.vstack((results[:, 0], results[:, 2])).T
        self.tn = results[:, 3]
        self.gain = results[:, 4]
        self.idx_best = idx_best

        self.if_noise = dcif.if_noise
        self.corr = dcif.corr
        self.shot_slope = dcif.shot_slope
        self.good_if_noise_fit = dcif.if_fit

        self.tn_best = float(self.tn[idx_best])
        self.gain_best = float(self.gain[idx_best])
        with np.errstate(divide="ignore", invalid="ignore"):
            self.g_db = float(10 * np.log10(self.gain_best))
        self.v_best = float(self.if_hot[idx_best, 0])

        i = int(np.abs(self.voltage - self.v_best).argmin())
        p = np.polyfit(self.voltage[i : i + 10], self.current[i : i + 10], 1)
        self.zj_if = self.rn / p[0]

    # -- reporting -----------------------------------------------------

    def summary(self) -> str:
        """Return a human-readable summary of the analysis.

        Returns:
            str: A multi-line summary.

        """
        lines = [f"Pumped data at {self.freq:.1f} GHz: {self.comment}"]
        if self.zt is not None:
            verdict = "good fit" if self.fit_good else "bad fit"
            lines += [
                f"\tImpedance recovery: {verdict}",
                f"\t\tThev. voltage:  \t{self.vt:+6.3f}\t* Vgap",
                f"\t\tThev. impedance:\t{self.zt:+.3f}\t* Rn",
                f"\t\tdrive level:    \t{self.alpha:+6.3f}",
                f"\t\tjunction imp.:  \t{self.zw:+.3f}\t* Rn",
            ]
            with np.errstate(divide="ignore", invalid="ignore"):
                power = (
                    np.abs(self.vt * self.vgap) ** 2 / 8 / np.real(self.zt * self.rn)
                )
            lines.append(f"\t\tavail. power:   \t{power / 1e-9:+7.2f}\tnW")
        if self.tn is not None:
            lines += [
                f"\tNoise temperature:\t{self.tn_best:6.1f}\tK",
                f"\tGain:             \t{self.g_db:+6.2f}\tdB",
                f"\tBest bias:        \t{self.v_best * self.vgap * 1e3:6.3f}\tmV",
            ]
        return "\n".join(lines)

    def print_info(self) -> None:  # pragma: no cover - thin wrapper
        """Print :meth:`summary` to the terminal."""
        print(self.summary())

    def __str__(self) -> str:  # pragma: no cover - debugging aid
        return self.summary()

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"PumpedData(freq={self.freq:.1f} GHz)"

    def plot_iv(self, ax=None, vmax_plot: float = 4.0):  # pragma: no cover
        """Plot the pumped I-V curve against the unpumped one.

        Args:
            ax (matplotlib.axes.Axes, optional): Axis to draw on.
            vmax_plot (float, optional): Voltage limit in mV.  Default is 4.

        Returns:
            matplotlib.axes.Axes: The axis.

        """
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots()
        scale_v, scale_i = self.vgap * 1e3, self.igap * 1e6
        ax.plot(self.voltage_dc * scale_v, self.current_dc * scale_i, label="unpumped")
        ax.plot(
            self.voltage * scale_v,
            self.current * scale_i,
            label=f"pumped, {self.freq:.0f} GHz",
        )
        ax.set_xlabel("Bias voltage (mV)")
        ax.set_ylabel("Current (uA)")
        ax.set_xlim([0, vmax_plot])
        ax.legend()
        ax.grid()
        return ax

    def plot_noise_temp(self, ax=None, vmax_plot: float = 4.0):  # pragma: no cover
        """Plot noise temperature versus bias voltage.

        Args:
            ax (matplotlib.axes.Axes, optional): Axis to draw on.
            vmax_plot (float, optional): Voltage limit in mV.  Default is 4.

        Returns:
            matplotlib.axes.Axes: The axis.

        Raises:
            ValueError: If no IF data was analyzed.

        """
        import matplotlib.pyplot as plt

        if self.tn is None:
            raise ValueError("No IF data was analyzed.")
        if ax is None:
            _, ax = plt.subplots()
        ax.plot(self.if_hot[:, 0] * self.vgap * 1e3, self.tn)
        ax.set_xlabel("Bias voltage (mV)")
        ax.set_ylabel("Noise temperature (K)")
        ax.set_xlim([0, vmax_plot])
        ax.set_ylim([0, np.nanmin(self.tn) * 10])
        ax.grid()
        return ax
