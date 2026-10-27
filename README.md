# KhurTrace

Symbolic Dirac traces and QFT kinematics in Python, with LaTeX output.

- `dirac_trace_calculator.py`: evaluates gamma-matrix traces written in FeynCalc-style notation (`DiracTrace`, `GS`, `GA`, `SP`, ...), including γ⁵ and Lorentz index contraction.
- `kinematics.py`: replaces scalar products with their values for a 1 → 2 decay or a 2 → 2 scattering in the rest, centre-of-mass or lab frame.

Both scripts print the result, save it in LaTeX syntax to a `.txt` file, and show it typeset in a separate window.

## Install

```bash
python -m pip install -r requirements.txt
```

## Usage

Evaluate a trace:

```bash
python dirac_trace_calculator.py --expr 'DiracTrace[(GS[p]+m) . (GS[q]-m)]'
```

Add kinematics, for example e⁺e⁻ → μ⁺μ⁻ in the centre-of-mass frame with massless particles:

```bash
python kinematics.py \
  --expr 'e^4/(4*s^2)*DiracTrace[GS[p2].GA[mu].GS[p1].GA[nu]]*DiracTrace[GS[k1].GA[mu].GS[k2].GA[nu]]' \
  --process "p1 + p2 -> k1 + k2" --mass p1=0 --mass p2=0 --mass k1=0 --mass k2=0 --frame cm
```

Result: `e**4*(cos(theta)**2 + 1)`.

Longer expressions can go in a plain-text file passed with `--input file.txt`. Run either script with `--help` for the full input syntax and options.

## Conventions

Four dimensions, metric (+, −, −, −), γ⁵ = iγ⁰γ¹γ²γ³, ε⁰¹²³ = +1, Tr(1) = 4.
No dimensional regularization or loop integrals. Amplitudes, spin sums and averaging factors are supplied by the user.

## Author

Khurshid Usmanov
