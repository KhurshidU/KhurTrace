# KhurTrace

Symbolic Dirac traces, QFT kinematics, decay widths and cross sections in Python, with LaTeX output.

- `dirac_trace_calculator.py`: evaluates gamma-matrix traces written in FeynCalc-style notation (`DiracTrace`, `GS`, `GA`, `SP`, ...), including γ⁵ and Lorentz index contraction.
- `kinematics.py`: replaces scalar products with their values for a 1 → 2 decay or a 2 → 2 scattering in the rest, centre-of-mass or lab frame.
- `rates.py`: turns a spin-averaged |M|² into a decay width, lifetime, branching ratio or cross section, using PDG values for the constants. Used through `kinematics.py --rate`.

Results are printed, saved in LaTeX syntax to a `.txt` file, and shown typeset in a separate window.

## Install

```bash
python -m pip install -r requirements.txt
```

Requires SymPy, NumPy and Matplotlib.

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

### Decay widths and cross sections

Add `--rate` to get the physical rate:

```bash
# Same process as above: dsigma/dOmega and sigma = 4 pi alpha^2 / (3s), here 868.5 pb at sqrt(s) = 10 GeV
python kinematics.py ... --frame cm --rate --set E_cm=10

# Muon decay mu -> e nu nu (3-body, Monte Carlo): Gamma = 3.00e-19 GeV, lifetime 2.19 us
python kinematics.py --expr '64*G_F^2*SP[p,k2]*SP[k1,k3]' --process 'p -> k1 + k2 + k3' \
  --mass p=m_mu --mass k1=m_e --mass k2=0 --mass k3=0 --frame rest --rate
```

- **Two-body final states** (1 → 2, 2 → 2) use closed-form expressions, giving symbolic and numerical results.
- **Three or more final particles** (1 → n, 2 → n) are integrated numerically over phase space with Monte Carlo. The result is quoted with its statistical error; raise the precision with `--events`.
- **Numbers** come from the PDG values in `rates.py` (G_F, V_ud, f_π, α, lepton and meson masses, lifetimes). Override any of them, or set a new one, with `--set NAME=VALUE` in GeV, for example `--set m_nu_tau=0.01`. Branching ratios use the measured lifetime of the parent.
- **Symmetry factor:** use `--symmetry 1/2` for two identical final particles.

The rate formulas follow D. Morrissey's PHYS 528 lecture notes (UBC):

- decay rate Γ = S/(2M) ∫ dΦ |M|²
- cross section σ = S/(4E₁E₂v) ∫ dΦ |M|², where v = √((p₁·p₂)² − m₁²m₂²)/(E₁E₂)
- lifetime τ = 1/Γ and branching ratio BR = Γ_f/Γ

Longer expressions can go in a plain-text file passed with `--input file.txt`. Run either script with `--help` for the full input syntax and options.

## Conventions

Four dimensions, metric (+, −, −, −), γ⁵ = iγ⁰γ¹γ²γ³, ε⁰¹²³ = +1, Tr(1) = 4, natural units with masses and energies in GeV.
Tree level only: no dimensional regularization, loop integrals or radiative corrections. Amplitudes, spin sums and averaging factors are supplied by the user.

## Author

Khurshid Usmanov
