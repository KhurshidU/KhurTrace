"""Kinematics companion to dirac_trace_calculator.py.

Replaces every scalar product SP[a,b] in a trace result by its value for a
chosen process and frame, then prints, saves (LaTeX) and displays the result.

Run:  python kinematics.py
Or:   python kinematics.py --input trace_1st_prong.txt --process "p -> k + q" \
          --mass p=m_tau --mass k=m_pi --mass q=m_nu_tau --frame rest

PROCESS is written with the momentum names used in your expression:
  "p -> k + q"            1 -> 2 decay
  "p1 + p2 -> k1 + k2"    2 -> 2 scattering
Masses default to m_<name> (e.g. m_p1) unless given with --mass name=symbol.

FRAMES
  Decay, "rest": parent at rest. Two-body decay kinematics is fixed by the
    masses, so every product is a function of the masses only.
  Scattering, "invariant": in terms of the Mandelstam variables s, t, u
    s = (p1+p2)^2, t = (p1-k1)^2, u = (p1-k2)^2.
  Scattering, "cm": centre-of-mass frame; total energy E_cm and the angle
    theta between p1 and k1.
  Scattering, "lab": p2 at rest (fixed target); beam energy E_beam of p1 and
    lab energy E_3 of the outgoing particle k1.

Scalar products are Lorentz invariant; the frame only decides which variables
(masses, s/t/u, angles, energies) the answer is written in. Momentum
conservation and on-shell conditions are imposed here, not in the calculator.
In the cm and lab frames, symbols s, t, u in your expression are replaced too.
"""

import argparse
from pathlib import Path

import sympy as sp

from dirac_trace_calculator import (evaluate, latex_document, read_expression_file,
                                    show_latex_panel)

# ---- Default settings: edit these, or override them on the command line ----
EXPRESSION = """1/4*G_F^2*(V_ud)^2*f_pi^2*(m_tau^2 * DiracTrace[(GS[q] + m_nu_tau) . (1 + GA[5]) . (GS[p] + m_tau) . (1 - GA[5])]
+ m_nu_tau^2 * DiracTrace[(GS[q] + m_nu_tau) . (1 - GA[5]) . (GS[p] + m_tau) . (1 + GA[5])] - 4*m_nu_tau*m_tau^2*DiracTrace[GS[q] + m_nu_tau])
"""
PROCESS = "p -> k + q"                 # tau(p) -> pi(k) + nu_tau(q)
MASSES = {"p": "m_tau", "k": "m_pi", "q": "m_nu_tau"}
FRAME = "rest"


def parse_process(process):
    """'p1 + p2 -> k1 + k2' gives (['p1', 'p2'], ['k1', 'k2'])."""
    left, right = process.split("->")
    incoming = [name.strip() for name in left.split("+")]
    outgoing = [name.strip() for name in right.split("+")]
    return incoming, outgoing


def sp_key(a, b):
    return f"SP[{a},{b}]"


def two_body_decay(P, k1, k2, M, m1, m2):
    """Parent P (mass M) at rest decays to k1 (mass m1) and k2 (mass m2)."""
    products = {
        sp_key(P, P): M**2,
        sp_key(k1, k1): m1**2,
        sp_key(k2, k2): m2**2,
        sp_key(P, k1): (M**2 + m1**2 - m2**2) / 2,
        sp_key(P, k2): (M**2 + m2**2 - m1**2) / 2,
        sp_key(k1, k2): (M**2 - m1**2 - m2**2) / 2,
    }
    # Rest-frame energies and momentum, printed for reference.
    E1 = (M**2 + m1**2 - m2**2) / (2 * M)
    E2 = (M**2 + m2**2 - m1**2) / (2 * M)
    momentum = sp.sqrt(kallen(M**2, m1**2, m2**2)) / (2 * M)
    print("Rest frame of", P)
    print(f"  E_{k1} =", E1)
    print(f"  E_{k2} =", E2)
    print(f"  |{k1}| = |{k2}| =", momentum)
    return products


def kallen(a, b, c):
    """Kallen triangle function lambda(a, b, c)."""
    return a**2 + b**2 + c**2 - 2*a*b - 2*a*c - 2*b*c


def mandelstam_products(p1, p2, k1, k2, m1, m2, m3, m4, s, t, u):
    """All scalar products for p1 + p2 -> k1 + k2, given s, t, u."""
    return {
        sp_key(p1, p1): m1**2,
        sp_key(p2, p2): m2**2,
        sp_key(k1, k1): m3**2,
        sp_key(k2, k2): m4**2,
        sp_key(p1, p2): (s - m1**2 - m2**2) / 2,
        sp_key(k1, k2): (s - m3**2 - m4**2) / 2,
        sp_key(p1, k1): (m1**2 + m3**2 - t) / 2,
        sp_key(p2, k2): (m2**2 + m4**2 - t) / 2,
        sp_key(p1, k2): (m1**2 + m4**2 - u) / 2,
        sp_key(p2, k1): (m2**2 + m3**2 - u) / 2,
    }


def scattering(p1, p2, k1, k2, m1, m2, m3, m4, frame):
    mass_sum = m1**2 + m2**2 + m3**2 + m4**2

    if frame == "invariant":
        s, t, u = sp.symbols("s t u")

    elif frame == "cm":
        E_cm, theta = sp.symbols("E_cm theta")
        s = E_cm**2
        E1 = (s + m1**2 - m2**2) / (2 * E_cm)
        E3 = (s + m3**2 - m4**2) / (2 * E_cm)
        p_in = sp.sqrt(kallen(s, m1**2, m2**2)) / (2 * E_cm)
        p_out = sp.sqrt(kallen(s, m3**2, m4**2)) / (2 * E_cm)
        t = m1**2 + m3**2 - 2 * (E1 * E3 - p_in * p_out * sp.cos(theta))
        u = mass_sum - s - t
        print("CM frame: E_cm = sqrt(s), theta = angle between", p1, "and", k1)

    elif frame == "lab":
        if m2 == 0:
            raise ValueError(f"A massless {p2} cannot be at rest; use the cm frame")
        E_beam, E_3 = sp.symbols("E_beam E_3")
        s = m1**2 + m2**2 + 2 * m2 * E_beam       # p2 at rest
        u = m2**2 + m3**2 - 2 * m2 * E_3          # (p2 - k1)^2
        t = mass_sum - s - u
        print("Lab frame:", p2, "at rest, E_beam = energy of", p1,
              ", E_3 = energy of", k1)

    else:
        raise ValueError("Scattering frames are: invariant, cm, lab")

    products = mandelstam_products(p1, p2, k1, k2, m1, m2, m3, m4, s, t, u)
    if frame != "invariant":
        # s, t, u written in the expression (e.g. a 1/s^2 propagator) also get replaced.
        products["s"] = s
        products["t"] = t
        products["u"] = u
    return products


def mass_value(text):
    """'0' or '1.5' gives a number; anything else (e.g. 'm_tau') a symbol."""
    if text.replace(".", "", 1).isdigit():
        return sp.Rational(text)
    return sp.Symbol(text)


def kinematic_rules(process, masses, frame):
    incoming, outgoing = parse_process(process)
    names = incoming + outgoing
    m = [mass_value(masses.get(name, "m_" + name)) for name in names]

    if len(incoming) == 1 and len(outgoing) == 2:
        if frame != "rest":
            raise ValueError("For a 1 -> 2 decay use --frame rest")
        return two_body_decay(*names, *m)
    if len(incoming) == 2 and len(outgoing) == 2:
        return scattering(*names, *m, frame)
    raise ValueError("Supported processes: 'a -> b + c' and 'a + b -> c + d'")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--expr", help="Expression (default: EXPRESSION in this file)")
    parser.add_argument("--input", type=Path, help="Text file containing an expression")
    parser.add_argument("--process", default=PROCESS, help='e.g. "p -> k + q"')
    parser.add_argument("--mass", action="append", default=[], metavar="NAME=SYMBOL",
                        help="Mass symbol of a momentum, e.g. p=m_tau; repeat as needed")
    parser.add_argument("--frame", default=FRAME, help="rest, invariant, cm or lab")
    parser.add_argument("--latex", type=Path, default=Path("kinematics_latex.txt"),
                        help="LaTeX output file (default: kinematics_latex.txt)")
    parser.add_argument("--no-panel", action="store_true", help="Do not open the window")
    args = parser.parse_args()

    if args.input:
        expression = read_expression_file(args.input)
    elif args.expr:
        expression = args.expr
    else:
        expression = EXPRESSION

    # Command-line masses replace the defaults only when a process is given there.
    masses = dict(MASSES) if args.process == PROCESS else {}
    for item in args.mass:
        name, symbol = item.split("=", 1)
        masses[name.strip()] = symbol.strip()

    try:
        rules = kinematic_rules(args.process, masses, args.frame)
        print("\nScalar products:")
        for key, value in rules.items():
            print(f"  {key} = {sp.factor(value)}")

        result = evaluate(expression, rules)
        result = sp.factor_terms(sp.expand(result))
        print("\nResult:\n" + str(result))

        args.latex.write_text(latex_document(expression, result))
        print(f"LaTeX written to {args.latex}")
        if not args.no_panel:
            show_latex_panel(result, title="Trace with kinematics")
    except ValueError as error:
        parser.exit(2, f"Input error: {error}\n")


if __name__ == "__main__":
    main()
