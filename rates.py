"""Decay widths and cross sections, used by kinematics.py --rate.

Formulas follow D. Morrissey, PHYS 528 notes (notes-00, Sec. 2.3):
  Eq. (65)  Gamma = S/(2M) * integral over phase space of |M|^2
  Eq. (59)  sigma = S/(4 E1 E2 v) * integral over phase space of |M|^2,
            v = sqrt((p1.p2)^2 - m1^2 m2^2) / (E1 E2)
  Eq. (63)  tau = 1/Gamma,   Eq. (64)  BR_f = Gamma_f / Gamma
For two-body final states these reduce to
  Gamma        = S k |M|^2 / (8 pi M^2)              (parent at rest)
  dsigma/dOmega = S k |M|^2 / (64 pi^2 s p)          (CM frame)
where p = |p1| = |p2| and k = |p3| = |p4|, S = 1/2 for two identical final
particles, and |M|^2 is already summed/averaged over spins.

For n >= 3 final particles the phase-space integral is done numerically
(Monte Carlo), using the recursion
  dPhi_n(P; p1..pn) = dPhi_2(P; p1, Q) dQ^2/(2 pi) dPhi_{n-1}(Q; p2..pn),
  integral of dPhi_2 = k / (4 pi M),
with dPhi_n = prod d^3p/((2pi)^3 2E) (2pi)^4 delta^4, as in Eqs. (59), (65).

All numbers are in GeV units. Override any value with --set NAME=VALUE.
"""

import numpy as np
import sympy as sp

from dirac_trace_calculator import Pair, to_latex

# PDG values in GeV (Review of Particle Physics 2024).
PDG = {
    "G_F": 1.1663788e-5,
    "V_ud": 0.97367,
    "V_us": 0.22431,
    "f_pi": 0.1302,          # f_pi ~ 130 MeV normalisation
    "f_K": 0.1557,
    "alpha": 1 / 137.035999,
    "m_e": 0.51099895e-3,
    "m_mu": 0.1056583755,
    "m_tau": 1.77693,
    "m_pi": 0.13957039,
    "m_K": 0.493677,
    "m_nu_e": 0,             # neutrino masses default to zero
    "m_nu_mu": 0,
    "m_nu_tau": 0,
}
PDG["e"] = (4 * 3.141592653589793 * PDG["alpha"]) ** 0.5

# Measured lifetimes in seconds, keyed by the parent's mass symbol (for BR).
LIFETIME = {"m_tau": 290.3e-15, "m_mu": 2.1969811e-6, "m_pi": 2.6033e-8, "m_K": 1.2380e-8}

HBAR = 6.582119569e-25       # GeV s
GEV2_TO_PB = 0.3893794e9     # 1 GeV^-2 in picobarn


def momentum(M, m1, m2):
    """|p| of either daughter when M decays at rest to masses m1, m2."""
    kallen = M**4 + m1**4 + m2**4 - 2*M**2*m1**2 - 2*M**2*m2**2 - 2*m1**2*m2**2
    return sp.sqrt(kallen) / (2 * M)


def two_body_width(M2, M, m1, m2, S=1):
    k = momentum(M, m1, m2)
    return S * k * M2 / (8 * sp.pi * M**2)


def cm_cross_section(M2, m1, m2, m3, m4, S=1):
    """Return dsigma/dOmega and sigma (integrated over theta and phi)."""
    E_cm, theta = sp.symbols("E_cm theta")
    s = E_cm**2
    p = momentum(E_cm, m1, m2)
    k = momentum(E_cm, m3, m4)
    dsigma = S * k * M2 / (64 * sp.pi**2 * s * p)
    sigma = 2 * sp.pi * sp.integrate(dsigma * sp.sin(theta), (theta, 0, sp.pi))
    return dsigma, sp.simplify(sigma)


def numeric(expr, overrides):
    """Substitute PDG values (and overrides); None if symbols are left over."""
    values = dict(PDG)
    values.update(overrides)
    expr = expr.subs({sp.Symbol(name): value for name, value in values.items()})
    if expr.free_symbols:
        names = ", ".join(sorted(str(x) for x in expr.free_symbols))
        print(f"  (no number: give values with --set for {names})")
        return None
    return float(sp.re(expr.evalf()))


def report(M2, masses_in, masses_out, frame, S, overrides):
    """Print the rate and return LaTeX lines for the output file."""
    if len(masses_in) == 1:
        M, (m1, m2) = masses_in[0], masses_out
        width = sp.factor(two_body_width(M2, M, m1, m2, S))
        print("\nDecay width  Gamma = S k |M|^2 / (8 pi M^2):")
        print("  Gamma =", width)
        value = numeric(width, overrides)
        if value is not None:
            print(f"  Gamma = {value:.6e} GeV")
            print(f"  lifetime = hbar/Gamma = {HBAR / value:.6e} s  (this channel alone)")
            if str(M) in LIFETIME:
                total = HBAR / LIFETIME[str(M)]
                print(f"  BR = Gamma / Gamma_total = {value / total:.4%}"
                      f"  (Gamma_total from measured lifetime {LIFETIME[str(M)]:.4e} s)")
        return [r"\Gamma = " + to_latex(width)]

    if frame != "cm":
        raise ValueError("Cross sections are computed in the cm frame; use --frame cm")
    dsigma, sigma = cm_cross_section(M2, *masses_in, *masses_out, S)
    print("\nCross section  dsigma/dOmega = S k |M|^2 / (64 pi^2 s p):")
    print("  dsigma/dOmega =", sp.factor(dsigma))
    print("  sigma =", sigma)
    value = numeric(sigma, overrides)
    if value is not None:
        print(f"  sigma = {value:.6e} GeV^-2 = {value * GEV2_TO_PB:.6e} pb")
    return [r"\frac{d\sigma}{d\Omega} = " + to_latex(sp.factor(dsigma)),
            r"\sigma = " + to_latex(sigma)]


# ---------------- Monte Carlo for n-body final states ----------------

def number(expr, overrides):
    """Like numeric(), but raises an error when a value is missing."""
    values = dict(PDG)
    values.update(overrides)
    expr = sp.sympify(expr).subs({sp.Symbol(name): v for name, v in values.items()})
    if expr.free_symbols:
        names = ", ".join(sorted(str(x) for x in expr.free_symbols))
        raise ValueError(f"Give numerical values with --set for: {names}")
    return float(expr)


def boost(p, P):
    """Boost momenta p (N x 4) from the rest frame of P to the frame where P is given."""
    E, P_vec = P[:, 0], P[:, 1:]
    M = np.sqrt(E**2 - np.sum(P_vec**2, axis=1))
    P_dot_p = np.sum(P_vec * p[:, 1:], axis=1)
    energy = (E * p[:, 0] + P_dot_p) / M
    factor = (p[:, 0] + P_dot_p / (E + M)) / M
    return np.column_stack([energy, p[:, 1:] + P_vec * factor[:, None]])


def phase_space(P, masses, rng):
    """Random n-body momenta with total momentum P (N x 4) and their weights.

    The average of weight * f over the events equals the integral of f dPhi_n.
    """
    N = len(P)
    M = np.sqrt(P[:, 0]**2 - np.sum(P[:, 1:]**2, axis=1))
    m1 = masses[0]
    if np.any(M < sum(masses)):
        raise ValueError("Energy is below the threshold for this final state")

    # Split P into particle 1 and a system Q holding the remaining particles.
    if len(masses) == 2:
        mQ = np.full(N, float(masses[1]))
        weight = np.ones(N)
    else:
        low, high = sum(masses[1:]), M - m1
        mQ = low + (high - low) * rng.random(N)
        weight = (high - low) * 2 * mQ / (2 * np.pi)       # dQ^2 / (2 pi)

    kallen = (M**2 - (m1 + mQ)**2) * (M**2 - (m1 - mQ)**2)
    k = np.sqrt(np.maximum(kallen, 0)) / (2 * M)
    weight = weight * k / (4 * np.pi * M)                    # two-body phase space

    cos_t = 2 * rng.random(N) - 1
    sin_t = np.sqrt(1 - cos_t**2)
    phi = 2 * np.pi * rng.random(N)
    direction = np.column_stack([sin_t * np.cos(phi), sin_t * np.sin(phi), cos_t])
    p1 = np.column_stack([np.sqrt(k**2 + m1**2), k[:, None] * direction])
    Q = np.column_stack([np.sqrt(k**2 + mQ**2), -k[:, None] * direction])
    p1, Q = boost(p1, P), boost(Q, P)

    if len(masses) == 2:
        return [p1, Q], weight
    rest, rest_weight = phase_space(Q, masses[1:], rng)
    return [p1] + rest, weight * rest_weight


def dot(a, b):
    return a[:, 0] * b[:, 0] - np.sum(a[:, 1:] * b[:, 1:], axis=1)


def monte_carlo_rate(M2, incoming, outgoing, masses_in, masses_out, S, overrides,
                     events=100000):
    """Gamma (1 -> n) or sigma (2 -> n, CM frame) from Eqs. (65) and (59)."""
    rng = np.random.default_rng(1)
    m_in = [number(m, overrides) for m in masses_in]
    m_out = [number(m, overrides) for m in masses_out]

    if len(incoming) == 1:
        M = m_in[0]
        P = np.tile([M, 0.0, 0.0, 0.0], (events, 1))
        momenta = {incoming[0]: P}
        prefactor = S / (2 * M)                                # Eq. (65)
    else:
        E_cm = number(sp.Symbol("E_cm"), overrides)
        m1, m2 = m_in
        p = np.sqrt((E_cm**2 - (m1 + m2)**2) * (E_cm**2 - (m1 - m2)**2)) / (2 * E_cm)
        p1 = np.tile([np.sqrt(p**2 + m1**2), 0.0, 0.0, p], (events, 1))
        p2 = np.tile([np.sqrt(p**2 + m2**2), 0.0, 0.0, -p], (events, 1))
        P = p1 + p2
        momenta = {incoming[0]: p1, incoming[1]: p2}
        # 4 E1 E2 v = 4 sqrt((p1.p2)^2 - m1^2 m2^2) = 4 p E_cm    Eq. (59)
        prefactor = S / (4 * p * E_cm)

    final, weight = phase_space(P, m_out, rng)
    momenta.update(zip(outgoing, final))

    # Turn |M|^2 into a numpy function of the scalar products.
    values = dict(PDG)
    values.update(overrides)
    M2 = sp.sympify(M2).subs({sp.Symbol(n): v for n, v in values.items()})
    pairs = sorted(M2.atoms(Pair), key=str)
    placeholders = sp.symbols(f"x0:{len(pairs)}")
    M2 = M2.subs(dict(zip(pairs, placeholders)))
    if M2.free_symbols - set(placeholders):
        names = ", ".join(sorted(str(x) for x in M2.free_symbols - set(placeholders)))
        raise ValueError(f"Give numerical values with --set for: {names}")
    f = sp.lambdify(placeholders, M2, "numpy")
    products = [dot(momenta[str(a)], momenta[str(b)]) for a, b in (q.args for q in pairs)]

    samples = prefactor * weight * np.broadcast_to(f(*products), (events,))
    return samples.mean(), samples.std() / np.sqrt(events)


def monte_carlo_report(M2, incoming, outgoing, masses_in, masses_out, S, overrides, events):
    value, error = monte_carlo_rate(M2, incoming, outgoing, masses_in, masses_out,
                                    float(S), overrides, events)
    print(f"\nMonte Carlo over {len(outgoing)}-body phase space ({events} events):")
    if len(incoming) == 1:
        print(f"  Gamma = {value:.6e} +- {error:.1e} GeV")
        print(f"  lifetime = hbar/Gamma = {HBAR / value:.6e} s  (this channel alone)")
        parent = str(masses_in[0])
        if parent in LIFETIME:
            print(f"  BR = {value / (HBAR / LIFETIME[parent]):.4%}")
        return [rf"\Gamma = {value:.4e} \pm {error:.1e}\ \mathrm{{GeV}}"]
    print(f"  sigma = {value:.6e} +- {error:.1e} GeV^-2 = {value * GEV2_TO_PB:.6e} pb")
    return [rf"\sigma = {value * GEV2_TO_PB:.4e} \pm {error * GEV2_TO_PB:.1e}\ \mathrm{{pb}}"]
