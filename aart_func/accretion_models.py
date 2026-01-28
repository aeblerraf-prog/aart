from dataclasses import dataclass
from typing import Callable, Tuple

import numpy as np

from aart_func.misc import rms
from numpy.lib.scimath import sqrt
from params import accretion_model, betaphi, betar, metric_model, sub_kep


@dataclass(frozen=True)
class AccretionModel:
    name: str
    g_disk: Callable[[np.ndarray, float, np.ndarray, np.ndarray, np.ndarray], np.ndarray]
    g_gas: Callable[[np.ndarray, float, np.ndarray, np.ndarray, np.ndarray], np.ndarray]


def delta_kerr(r: np.ndarray, a: float) -> np.ndarray:
    """Kerr metric function Delta(r)."""
    return r**2 - 2 * r + a**2


def pif_kerr(r: np.ndarray, a: float) -> np.ndarray:
    """Kerr metric function Pi(r)."""
    return (r**2 + a**2) ** 2 - a**2 * delta_kerr(r, a)


def urbar_kerr(r: np.ndarray, a: float) -> np.ndarray:
    """Radial infall four-velocity (contravariant) for Kerr."""
    return -np.sqrt(2 * r * (r**2 + a**2)) / (r**2)


def omegabar_kerr(r: np.ndarray, a: float) -> np.ndarray:
    """Angular velocity of radial infall for Kerr."""
    return (2 * a * r) / pif_kerr(r, a)


def omegahat_kerr(r: np.ndarray, a: float, laux: np.ndarray) -> np.ndarray:
    """Angular velocity of a sub-Keplerian orbit in Kerr."""
    return (a + (1 - 2 / r) * (laux - a)) / (pif_kerr(r, a) / (r**2) - (2 * a * laux) / r)


def uttilde_kerr(r: np.ndarray, a: float, ur_t: np.ndarray, omega_t: np.ndarray) -> np.ndarray:
    """Contravariant time component of the general four-velocity in Kerr."""
    return np.sqrt(
        (1 + ur_t**2 * r**2 / delta_kerr(r, a))
        / (1 - (r**2 + a**2) * omega_t**2 - (2 / r) * (1 - a * omega_t) ** 2)
    )


def ehat_kerr(r: np.ndarray, a: float, laux: np.ndarray) -> np.ndarray:
    """Sub-Keplerian orbital energy in Kerr."""
    return np.sqrt(
        delta_kerr(r, a)
        / (pif_kerr(r, a) / (r**2) - (4 * a * laux) / r - (1 - 2 / r) * laux**2)
    )


def nuhat_kerr(r: np.ndarray, a: float, laux: np.ndarray, ehat_aux: np.ndarray) -> np.ndarray:
    """Sub-Keplerian radial velocity in Kerr."""
    return r / delta_kerr(r, a) * np.sqrt(
        np.abs(
            pif_kerr(r, a) / (r**2)
            - (4 * a * laux) / r
            - (1 - 2 / r) * laux**2
            - delta_kerr(r, a) / (ehat_aux**2)
        )
    )


def lhat_kerr(r: np.ndarray, a: float) -> np.ndarray:
    """Sub-Keplerian specific angular momentum in Kerr."""
    return sub_kep * (r**2 + a**2 - 2 * a * np.sqrt(r)) / (np.sqrt(r) * (r - 2) + a)


def r_potential_kerr(r: np.ndarray, a: float, lamb: np.ndarray, eta: np.ndarray) -> np.ndarray:
    """Radial potential for Kerr redshift calculations."""
    return (r**2 + a**2 - a * lamb) ** 2 - (r**2 - 2 * r + a**2) * (eta + (lamb - a) ** 2)


def g_disk_subkeplerian(
    r: np.ndarray,
    a: float,
    b: np.ndarray,
    lamb: np.ndarray,
    eta: np.ndarray,
) -> np.ndarray:
    """Redshift factor outside ISCO for sub-Keplerian flow."""
    omega_h = omegahat_kerr(r, a, lhat_kerr(r, a))
    omega_t = omega_h + (1 - betaphi) * (omegabar_kerr(r, a) - omega_h)
    ur = (1 - betar) * urbar_kerr(r, a)
    ut = uttilde_kerr(r, a, ur, omega_t)
    uphi = ut * omega_t

    return 1 / (
        ut
        * (
            1
            - b
            * np.sign(ur)
            * sqrt(np.abs(r_potential_kerr(r, a, lamb, eta) * ur**2))
            / delta_kerr(r, a)
            / ut
            - lamb * uphi / ut
        )
    )


def g_gas_subkeplerian(
    r: np.ndarray,
    a: float,
    b: np.ndarray,
    lamb: np.ndarray,
    eta: np.ndarray,
) -> np.ndarray:
    """Redshift factor inside ISCO for sub-Keplerian flow."""
    isco = rms(a)
    lms = lhat_kerr(isco, a)
    omega_h = omegahat_kerr(r, a, lms)
    omega_t = omega_h + (1 - betaphi) * (omegabar_kerr(r, a) - omega_h)

    e_ms = ehat_kerr(isco, a, lms)
    ur_hat = -delta_kerr(r, a) / (r**2) * nuhat_kerr(r, a, lms, e_ms) * e_ms
    ur = ur_hat + (1 - betar) * (urbar_kerr(r, a) - ur_hat)
    ut = uttilde_kerr(r, a, ur, omega_t)
    uphi = omega_t * ut

    return 1 / (
        ut
        * (
            1
            - b
            * np.sign(ur)
            * sqrt(np.abs(r_potential_kerr(r, a, lamb, eta) * ur**2))
            / delta_kerr(r, a)
            / ut
            - lamb * uphi / ut
        )
    )


def _ensure_kerr_metric() -> None:
    if metric_model != "kerr":
        raise NotImplementedError(
            f"Metric model '{metric_model}' is not implemented. "
            "AART's analytic ray-tracing depends on Kerr integrability. "
            "To add Damour-Solodukhin or other wormhole metrics, add new geodesic, "
            "redshift, and conserved-quantity solvers in raytracing_f.py and update "
            "accretion_models.py accordingly."
        )


def get_accretion_model() -> AccretionModel:
    """Return the configured accretion model implementation."""
    _ensure_kerr_metric()
    model_key = accretion_model.lower().replace("_", "-")
    if model_key == "subkeplerian":
        return AccretionModel(
            name="subkeplerian",
            g_disk=g_disk_subkeplerian,
            g_gas=g_gas_subkeplerian,
        )
    if model_key in {"novikov-thorne", "novikovthorne"}:
        raise NotImplementedError(
            "Novikov-Thorne disk support is not implemented yet. "
            "Add the NT specific energy/angular momentum and emissivity profile, "
            "then wire it into get_accretion_model()."
        )
    if model_key == "adaf":
        raise NotImplementedError(
            "ADAF disk support is not implemented yet. "
            "Add the ADAF velocity/emissivity prescriptions and wire them into get_accretion_model()."
        )
    raise ValueError(f"Unknown accretion model '{accretion_model}'.")
