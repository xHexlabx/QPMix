"""The embedding circuit: the Thevenin equivalent seen by the SIS junction.

Every signal applied to the junction is described by a Thevenin voltage
``vt`` and a Thevenin impedance ``zt``, for each tone ``f`` and harmonic
``p``.  :class:`EmbeddingCircuit` holds those, plus the frequencies and the
DC bias sweep, and is the object passed into
:func:`qpmix.qtcurrent.qtcurrent` and
:func:`qpmix.harmonic_balance.harmonic_balance`.

Everything is normalized: voltages to the gap voltage ``Vgap``, currents to
the gap current ``Igap = Vgap / Rn``, impedances to the normal resistance
``Rn``, and frequencies to the gap frequency ``Fgap = Vgap * e / h``.

Indexing convention: ``f`` runs from 1 to ``num_f`` and ``p`` from 1 to
``num_p``; index 0 is reserved for DC and is always zero.  That wastes one
slot per axis but keeps every index in the code matching the corresponding
index in the published equations.

Differences from upstream QMix:

* Argument validation raises :class:`ValueError` / :class:`TypeError`
  instead of using ``assert``, which is stripped out when Python runs with
  ``-O`` and would silently let a malformed circuit through.
* Unit conversion goes through a single lookup table
  (:data:`POWER_UNITS`, :data:`FREQ_UNITS`) rather than a chain of string
  comparisons, so units are applied consistently everywhere.
* Circuits round-trip through JSON (:meth:`EmbeddingCircuit.to_json` /
  :func:`read_circuit`) as well as the legacy text format.

Examples:

    >>> cct = EmbeddingCircuit(num_f=1, num_p=1, vgap=2.8e-3, rn=14.0)
    >>> cct.set_freq(230, f=1, units='GHz')
    >>> round(float(cct.freq[1]), 4)
    0.3397
    >>> cct.zt[1, 1] = 0.3 - 0.3j
    >>> cct.set_available_power(10, units='nW')
    >>> round(cct.available_power(units='nW'), 6)
    10.0

"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import scipy.constants as sc

__all__ = ["FREQ_UNITS", "POWER_UNITS", "EmbeddingCircuit", "read_circuit"]

#: Multipliers that convert a power in the given unit into watts.  ``None``
#: marks the logarithmic units, which are handled separately.
POWER_UNITS: dict[str, float | None] = {
    "w": 1.0,
    "mw": sc.milli,
    "uw": sc.micro,
    "nw": sc.nano,
    "pw": sc.pico,
    "fw": sc.femto,
    "dbm": None,
    "dbw": None,
}

#: Multipliers that convert a frequency in the given unit into hertz, or a
#: voltage into volts.  ``norm`` means the value is already normalized.
FREQ_UNITS: dict[str, tuple[str, float]] = {
    "hz": ("freq", 1.0),
    "mhz": ("freq", sc.mega),
    "ghz": ("freq", sc.giga),
    "thz": ("freq", sc.tera),
    "v": ("volt", 1.0),
    "mv": ("volt", sc.milli),
    "norm": ("norm", 1.0),
}


class EmbeddingCircuit:
    """The Thevenin equivalent circuit seen by the SIS junction.

    Args:
        num_f (int, optional): Number of fundamental tones, 1 to 4.  Default
            is 1.
        num_p (int, optional): Number of harmonics per tone.  Default is 1.
        vb_min (float, optional): Minimum bias voltage, normalized.  Default
            is 0.
        vb_max (float, optional): Maximum bias voltage, normalized.  Default
            is 2.
        vb_npts (int, optional): Number of bias voltage points.  Default is
            201.
        fgap (float, optional): Gap frequency in Hz.  Default is None.
        vgap (float, optional): Gap voltage in V.  Default is None.
        rn (float, optional): Normal-state resistance in ohms.  Default is
            None.
        name (str, optional): A label for this circuit.  Default is ``''``.

    Attributes:
        num_f (int): Number of fundamental tones.
        num_p (int): Number of harmonics.
        num_n (int): ``num_f * num_p``, the number of unknown signals.
        freq (ndarray): Normalized frequency of each tone, shape
            ``(num_f + 1,)``.
        vt (ndarray): Normalized Thevenin voltage, shape
            ``(num_f + 1, num_p + 1)``, complex.
        zt (ndarray): Normalized Thevenin impedance, same shape as ``vt``.
        vb (ndarray): The DC bias voltage sweep, normalized.
        vb_npts (int): Length of ``vb``.
        vgap (float or None): Gap voltage in V.
        fgap (float or None): Gap frequency in Hz.
        rn (float or None): Normal-state resistance in ohms.
        igap (float or None): Gap current in A.
        comment (list): ``comment[f][p]`` labels each signal.
        name (str): A label for this circuit.

    Raises:
        ValueError: If any of the sizes are out of range.

    """

    def __init__(
        self,
        num_f: int = 1,
        num_p: int = 1,
        vb_min: float = 0,
        vb_max: float = 2,
        vb_npts: int = 201,
        fgap: float | None = None,
        vgap: float | None = None,
        rn: float | None = None,
        name: str = "",
    ) -> None:
        if num_f not in (1, 2, 3, 4):
            raise ValueError("Number of tones (num_f) must be 1, 2, 3 or 4.")
        if int(num_p) != num_p or num_p < 1:
            raise ValueError("Number of harmonics (num_p) must be an int >= 1.")
        if int(vb_npts) != vb_npts or vb_npts < 1:
            raise ValueError("Number of bias points (vb_npts) must be an int >= 1.")
        if vb_max < vb_min:
            raise ValueError("vb_max must be greater than or equal to vb_min.")

        self.name = name
        self.num_f = int(num_f)
        self.num_p = int(num_p)
        self.num_n = self.num_f * self.num_p

        # Junction properties (all optional; needed only for SI-unit helpers)
        self.vgap: float | None = None
        self.fgap: float | None = None
        self.rn: float | None = None
        self.igap: float | None = None
        if vgap is not None:
            self.vgap = float(vgap)
        elif fgap is not None:
            self.vgap = float(fgap) * sc.h / sc.e
        if fgap is not None:
            self.fgap = float(fgap)
        elif vgap is not None:
            self.fgap = float(vgap) * sc.e / sc.h
        if rn is not None:
            self.rn = float(rn)
        if self.vgap is not None and self.rn is not None:
            self.igap = self.vgap / self.rn

        self.freq = np.zeros(self.num_f + 1, dtype=float)
        self.vt = np.zeros((self.num_f + 1, self.num_p + 1), dtype=complex)
        self.zt = np.zeros((self.num_f + 1, self.num_p + 1), dtype=complex)

        self.vb_npts = int(vb_npts)
        self.vb = np.linspace(vb_min, vb_max, self.vb_npts)

        self.comment: list[list[str]] = [
            ["" for _ in range(self.num_p + 1)] for _ in range(self.num_f + 1)
        ]

    def __str__(self) -> str:
        suffix = f": {self.name}" if self.name else ""
        return f"Embedding circuit (Tones:{self.num_f}, Harmonics:{self.num_p}){suffix}"

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return self.__str__()

    def initialize_vj(self) -> np.ndarray:
        """Return a correctly shaped, zeroed junction-voltage array.

        Useful when setting the junction voltage directly and skipping
        harmonic balance.

        Returns:
            ndarray: Zeros, shape ``(num_f + 1, num_p + 1, vb_npts)``,
            complex.

        """
        return np.zeros((self.num_f + 1, self.num_p + 1, self.vb_npts), dtype=complex)

    # -- power ----------------------------------------------------------

    def available_power(self, f: int = 1, p: int = 1, units: str = "W") -> float:
        """Available power of tone ``f``, harmonic ``p``.

        Note:
            The gap voltage and normal resistance must be set.

        Args:
            f (int, optional): Tone index.  Default is 1.
            p (int, optional): Harmonic index.  Default is 1.
            units (str, optional): One of ``'W'``, ``'mW'``, ``'uW'``,
                ``'nW'``, ``'pW'``, ``'fW'``, ``'dBm'``, ``'dBW'``.
                Default is ``'W'``.

        Returns:
            float: Available power in the requested units.

        Raises:
            ValueError: If the junction properties are unset or the units
                are not recognised.

        """
        self._require_junction()
        key = units.lower()
        if key not in POWER_UNITS:
            raise ValueError(f"Not a recognized unit for power: {units!r}")

        v_v = self.vt[f, p] * self.vgap
        r_ohms = self.zt[f, p].real * self.rn
        power = float(np.abs(v_v) ** 2 / r_ohms / 8.0) if r_ohms != 0.0 else 0.0

        if key == "dbm":
            with np.errstate(divide="ignore"):
                return float(10 * np.log10(power * 1e3))
        if key == "dbw":
            with np.errstate(divide="ignore"):
                return float(10 * np.log10(power))
        return power / POWER_UNITS[key]

    def set_available_power(
        self, power: float, f: int = 1, p: int = 1, units: str = "W"
    ) -> None:
        """Set the Thevenin voltage to deliver a given available power.

        Note:
            The gap voltage, normal resistance and Thevenin impedance must
            be set first.

        Args:
            power (float): Available power, in the given units.
            f (int, optional): Tone index.  Default is 1.
            p (int, optional): Harmonic index.  Default is 1.
            units (str, optional): See :meth:`available_power`.  Default is
                ``'W'``.

        Raises:
            ValueError: If the junction properties or the embedding
                impedance are unset, or the units are not recognised.

        """
        self._require_junction()
        if self.zt[f, p] == 0:
            raise ValueError("Embedding impedance not set.")
        key = units.lower()
        if key not in POWER_UNITS:
            raise ValueError(f"Not a recognized unit for power: {units!r}")

        if key == "dbm":
            power_w = 10 ** (power / 10) * 1e-3
        elif key == "dbw":
            power_w = 10 ** (power / 10)
        else:
            power_w = power * POWER_UNITS[key]

        r_ohms = self.zt[f, p].real * self.rn
        self.vt[f, p] = np.sqrt(8 * power_w * r_ohms) / self.vgap

    def set_alpha(self, alpha: float, f: int = 1, p: int = 1, zj: float = 0.66) -> None:
        """Set the Thevenin voltage to approximately reach a drive level.

        This is only a guess: the realised drive level is not known until
        the simulation has run, because the junction impedance ``zj``
        depends on the drive level itself.

        Args:
            alpha (float): Target drive level, ``alpha = vj / vph``.
            f (int, optional): Tone index.  Default is 1.
            p (int, optional): Harmonic index.  Default is 1.
            zj (float, optional): Assumed junction impedance, normalized to
                ``Rn``.  Default is 0.66.

        Raises:
            ValueError: If the frequency or embedding impedance is unset.

        """
        if self.zt[f, p] == 0:
            raise ValueError("Embedding impedance must be defined.")
        if self.freq[f] == 0:
            raise ValueError("Frequency must be defined.")
        self.vt[f, p] = alpha * self.freq[f] * (self.zt[f, p] / zj + 1)

    # -- frequency ------------------------------------------------------

    def set_freq(self, value: float, f: int = 1, units: str = "Hz") -> None:
        """Set the frequency of tone ``f``, in physical or normalized units.

        Note:
            The gap frequency or gap voltage must be defined unless
            ``units='norm'``.

        Args:
            value (float): The value, in the given units.
            f (int, optional): Tone index.  Default is 1.
            units (str, optional): ``'Hz'``, ``'MHz'``, ``'GHz'``,
                ``'THz'`` for frequency; ``'V'``, ``'mV'`` for photon
                voltage; ``'norm'`` for an already-normalized value.
                Default is ``'Hz'``.

        Raises:
            ValueError: If the units are not recognised, or the junction
                gap is needed but unset.

        """
        key = units.lower()
        if key not in FREQ_UNITS:
            raise ValueError(f"Units not recognized: {units!r}")
        kind, scale = FREQ_UNITS[key]

        if kind == "norm":
            self.freq[f] = value
        elif kind == "freq":
            if self.fgap is None:
                raise ValueError("Gap frequency must be defined.")
            self.freq[f] = value * scale / self.fgap
        else:
            if self.vgap is None:
                raise ValueError("Gap voltage must be defined.")
            self.freq[f] = value * scale / self.vgap

    def set_name(self, name: str, f: int = 1, p: int = 1) -> None:
        """Label a tone/harmonic combination.

        Purely cosmetic; it has no effect on the simulation.

        Args:
            name (str): The label.
            f (int, optional): Tone index.  Default is 1.
            p (int, optional): Harmonic index.  Default is 1.

        """
        self.comment[f][p] = name

    @property
    def vph(self) -> np.ndarray:
        """Photon voltage, an alias for :attr:`freq`.

        Normalized photon voltage and normalized frequency are the same
        number; kept for backwards compatibility with QMix.
        """
        return self.freq

    # -- reporting and I/O ----------------------------------------------

    def summary(self) -> str:
        """Return a human-readable description of the circuit.

        Returns:
            str: A multi-line summary.

        """
        lines = [str(self)]
        full = None not in (self.fgap, self.vgap, self.rn)
        for f in range(1, self.num_f + 1):
            for p in range(1, self.num_p + 1):
                label = self.comment[f][p]
                if full:
                    fq = self.freq[f] * self.fgap / 1e9
                    lines.append(f"   f={f}, p={p}   {fq:.1f} GHz x {p}   {label}")
                else:
                    lines.append(
                        f"   f={f}, p={p}   freq = {self.freq[f]:.4f} x {p}   {label}"
                    )
                lines.append(f"\tThev. voltage:\t\t{self.vt[f, p].real:.4f} * Vgap")
                if self.freq[f] != 0:
                    vph = self.vt[f, p].real / (self.freq[f] * p)
                    lines.append(f"\t              \t\t{vph:.4f} * Vph")
                lines.append(f"\tThev. impedance:\t{self.zt[f, p]:.2f} * Rn")
                if full:
                    with np.errstate(divide="ignore"):
                        lines.append(
                            f"\tAvail. power:   \t"
                            f"{self.available_power(f, p, 'W'):.2E} W"
                        )
                        lines.append(
                            f"\t                \t"
                            f"{self.available_power(f, p, 'dBm'):.3f} dBm"
                        )
        return "\n".join(lines)

    def print_info(self) -> None:
        """Print :meth:`summary` to the terminal."""
        print(self.summary())
        print("")

    def to_dict(self) -> dict[str, Any]:
        """Serialize the circuit to a plain dictionary.

        Returns:
            dict: JSON-compatible representation.

        """
        return {
            "name": self.name,
            "num_f": self.num_f,
            "num_p": self.num_p,
            "vb_min": float(self.vb[0]),
            "vb_max": float(self.vb[-1]),
            "vb_npts": self.vb_npts,
            "vgap": self.vgap,
            "fgap": self.fgap,
            "rn": self.rn,
            "freq": self.freq.tolist(),
            "vt": [[[z.real, z.imag] for z in row] for row in self.vt],
            "zt": [[[z.real, z.imag] for z in row] for row in self.zt],
            "comment": self.comment,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EmbeddingCircuit:
        """Rebuild a circuit from :meth:`to_dict` output.

        Args:
            data (dict): The dictionary.

        Returns:
            EmbeddingCircuit: The reconstructed circuit.

        """
        cct = cls(
            num_f=data["num_f"],
            num_p=data["num_p"],
            vb_min=data["vb_min"],
            vb_max=data["vb_max"],
            vb_npts=data["vb_npts"],
            vgap=data.get("vgap"),
            rn=data.get("rn"),
            name=data.get("name", ""),
        )
        cct.freq[:] = data["freq"]
        cct.vt[:] = [[complex(re_, im) for re_, im in row] for row in data["vt"]]
        cct.zt[:] = [[complex(re_, im) for re_, im in row] for row in data["zt"]]
        if "comment" in data:
            cct.comment = [list(row) for row in data["comment"]]
        return cct

    def to_json(self, filename: str | Path) -> None:
        """Save the circuit as JSON.

        Args:
            filename (str or Path): Destination path.

        """
        Path(filename).write_text(json.dumps(self.to_dict(), indent=2))

    def save_info(self, filename: str | Path = "embedding-circuit.txt") -> None:
        """Save the circuit in the legacy QMix text format.

        Args:
            filename (str or Path, optional): Destination path.  Default is
                ``'embedding-circuit.txt'``.

        """
        lines = [
            "Embedding Circuit:",
            f"\tNumber of tones:     {self.num_f}",
            f"\tNumber of harmonics: {self.num_p}",
        ]
        for f in range(1, self.num_f + 1):
            for p in range(1, self.num_p + 1):
                lines.append(f"   f={f}, p={p}\t\t\tfreq = {self.freq[f]:.4f} x {p}")
                lines.append(f"\tThev. voltage:\t\t{self.vt[f, p]:.4f} V / V_gap")
                lines.append(f"\tThev. impedance:\t{self.zt[f, p]:.2f} ohms / R_N")
        Path(filename).write_text("\n".join(lines) + "\n")

    def lock(self) -> None:
        """Make the circuit's arrays read-only.  Useful when debugging."""
        for arr in (self.freq, self.vt, self.zt, self.vb):
            arr.flags.writeable = False

    def unlock(self) -> None:
        """Make the circuit's arrays writeable again."""
        for arr in (self.freq, self.vt, self.zt, self.vb):
            arr.flags.writeable = True

    def _require_junction(self) -> None:
        if self.vgap is None:
            raise ValueError("Gap voltage not set.")
        if self.rn is None:
            raise ValueError("Normal resistance not set.")


def read_circuit(filename: str | Path) -> EmbeddingCircuit:
    """Load an embedding circuit from a file.

    Accepts both the JSON format written by
    :meth:`EmbeddingCircuit.to_json` and the legacy text format written by
    :meth:`EmbeddingCircuit.save_info`.

    Args:
        filename (str or Path): Path to the circuit file.

    Returns:
        EmbeddingCircuit: The reconstructed circuit.

    Raises:
        ValueError: If the file cannot be parsed as either format.

    """
    text = Path(filename).read_text()
    stripped = text.lstrip()
    if stripped.startswith("{"):
        return EmbeddingCircuit.from_dict(json.loads(text))

    data = text.splitlines()
    try:
        num_f = int(data[1].split()[-1])
        num_p = int(data[2].split()[-1])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"Could not parse circuit file: {filename}") from exc

    cct = EmbeddingCircuit(num_f, num_p)
    body = data[3:]
    for i, line in enumerate(body):
        parts = line.split()
        if not parts or not parts[0].startswith("f"):
            continue
        f = int(re.search(r"\d+", parts[0]).group())
        p = int(re.search(r"\d+", parts[1]).group())
        cct.freq[f] = float(parts[4])
        cct.vt[f, p] = complex(body[i + 1].split()[2])
        cct.zt[f, p] = complex(body[i + 2].split()[2])
    return cct
