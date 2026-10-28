"""Symbolic Dirac traces in FOUR dimensions, using only SymPy.

Install: python -m pip install sympy
Run:     python dirac_trace_calculator.py
Or:      python dirac_trace_calculator.py --expr 'DiracTrace[GS[p] . GS[q]]'
Or:      python dirac_trace_calculator.py --input my_expression.txt

INPUT LANGUAGE (a small subset of Mathematica/FeynCalc notation):
  DiracTrace[...]  a Dirac trace; write each fermion trace separately
  GS[p]           p_slash, for an arbitrary symbolic four-momentum p
  GA[mu]          gamma with Lorentz index mu; repeated mu implies contraction
  GA[mu,nu,...]   shorthand for GA[mu] . GA[nu] . ...
  GS[p,q,...]     shorthand for GS[p] . GS[q] . ...
  GA[5]           gamma^5; PL = (1-GA[5])/2; PR = (1+GA[5])/2
  SP[p,q]         Minkowski scalar product p.q
  FV[p,mu]        momentum component; MT[mu,nu] is the metric tensor
  EPS[p,q,r,s]    epsilon contracted with four momenta (local helper)
  m, e, s, ... arbitrary scalar symbols; I is the imaginary unit
  .            ordered matrix product; use * for scalar multiplication
  ^ or **      powers; scalars inside a trace multiply the 4x4 identity

Use explicit * between scalar factors: 4*s^2, not 4s^2. This is a Python
calculator accepting the subset above, not a Mathematica interpreter or
FeynCalc installation. Commands/options outside this subset are not supported.
DiracSimplify[...] and Contract[...] are accepted as optional wrappers.
The earlier Tr(...), slash(p), G(mu), G(-mu), and G5 syntax also works.
Keep the two Lorentz-index styles separate within a contraction.

Conventions: g = diag(+1,-1,-1,-1), gamma5 = i gamma0 gamma1 gamma2 gamma3,
epsilon^{0123}=+1, Tr(1)=4. Thus DiracTrace[GA[mu,nu,rho,sigma,5]]
= -4 I epsilon^{mu nu rho sigma}. EPS(p,q,r,s) means
epsilon^{mu nu rho sigma} p_mu q_nu r_rho s_sigma.
In the FeynCalc-style input each dummy index occurs exactly twice, using the
same name: GA[mu] ... GA[mu]. No minus sign on the second occurrence is needed.
An index occurring once is free. The explicit legacy G syntax requires one
upper and one lower occurrence per dummy index.

This evaluates supplied trace expressions. It does NOT derive amplitudes,
conjugate spinor chains, perform spin/polarization/color sums automatically,
or infer spin-average factors, mass-shell conditions, or momentum conservation.
Supply these yourself. No dimensional regularization or loop integration.
Long gamma strings grow combinatorially; use FORM for large calculations.

Examples:
  DiracTrace[(GS[p]+m) . (GS[q]-m)]
  DiracTrace[GS[p] . GA[mu] . GS[q] . GA[mu]]
  DiracTrace[GS[p] . GA[mu] . (1-GA[5]) . GS[q] . GA[nu] . (1-GA[5])]
  e^4/(4*s^2)*DiracTrace[GS[p2].GA[mu].GS[p1].GA[nu]]*DiracTrace[GS[k1].GA[mu].GS[k2].GA[nu]]

Optional substitutions and saving:
  python dirac_trace_calculator.py --expr 'DiracTrace[GS[p,p]]' --set 'SP[p,p]=m^2' --output result.txt

LaTeX output: every run writes the result in LaTeX to dirac_trace_latex.txt
(change with --latex out.txt) and opens a window showing the rendered result,
one term per line (needs matplotlib; skip with --no-panel). Greek names become
Greek letters and extra underscores nest subscripts: m_nu_tau -> m_{\nu_{\tau}}.

Python use:
  from dirac_trace_calculator import evaluate
  result = evaluate('DiracTrace[(GS[p]+m) . (GS[q]-m)]')

Design references (the evaluator below is original code):
  https://github.com/sympy/sympy
  https://docs.sympy.org/latest/modules/physics/hep/index.html
  https://github.com/form-dev/form
  https://github.com/tueda/python-form
  https://feyncalc.github.io/FeynCalcBook/GA.html
  https://feyncalc.github.io/FeynCalcBook/GS.html
  https://feyncalc.github.io/FeynCalcBook/DiracTrace.html
"""

import argparse
import ast
import io
import re
import tokenize
from functools import lru_cache
from itertools import combinations, permutations
from pathlib import Path

import sympy as sp
from sympy.printing.latex import LatexPrinter

# Edit this expression, use --expr, or put a multiline expression in a text file.
'''EXPRESSION = """
e^4/(4*s^2)
* DiracTrace[GS[p2] . GA[mu] . GS[p1] . GA[nu]]
* DiracTrace[GS[k1] . GA[mu] . GS[k2] . GA[nu]]
"""'''
EXPRESSION = """1/4*G_F^2*(V_ud)^2*f_pi^2*(m_tau^2 * DiracTrace[(GS[q] + m_nu_tau) . (1 + GA[5]) . (GS[p] + m_tau) . (1 - GA[5])]
+ m_nu_tau^2 * DiracTrace[(GS[q] + m_nu_tau) . (1 - GA[5]) . (GS[p] + m_tau) . (1 + GA[5])] - 4*m_nu_tau*m_tau^2*DiracTrace[GS[q] + m_nu_tau])
"""
class Index(sp.Function):
    nargs = 2  # name, +1/-1 for explicit variance, 0 for FeynCalc convention

    def _sympystr(self, printer):
        return ("-" if self.args[1] == -1 else "") + str(self.args[0])


def index(label):
    sign, name = sp.sympify(label).as_coeff_Mul()
    if sign not in (1, -1) or not isinstance(name, sp.Symbol):
        raise ValueError("Use a named Lorentz index, e.g. mu or -mu.")
    return Index(name, sign)


def implicit_index(label):
    if not isinstance(label, sp.Symbol) or not label.is_commutative:
        raise ValueError("Use GA[mu], FV[p,mu], or MT[mu,nu] with plain index names; do not use -mu.")
    return Index(label, 0)


def normalize_input(text):
    """Translate supported brackets, Dot and powers, preserving decimal numbers."""
    # Accept named Mathematica Greek characters such as backslash-[Mu].
    text = re.sub(r'\\\[([A-Za-z]+)\]', r'\1', text)
    operators = {'[': '(', ']': ')', '.': '*', '^': '**'}
    tokens = tokenize.generate_tokens(io.StringIO(text).readline)
    return tokenize.untokenize((t.type, operators.get(t.string, t.string)
                               if t.type == tokenize.OP else t.string)
                              for t in tokens)


def tensor_latex(head, slots, printer):
    """LaTeX for head with ordered slots; lower legacy indices go downstairs."""
    out, runs = head, []
    for slot in slots:
        lower = isinstance(slot, Index) and slot.args[1] == -1
        name = printer._print(slot.args[0] if isinstance(slot, Index) else slot)
        if runs and runs[-1][0] == lower:
            runs[-1][1].append(name)
        else:
            runs.append((lower, [name]))
    for k, (lower, names) in enumerate(runs):
        out += ("{}" if k else "") + ("_" if lower else "^") + "{" + " ".join(names) + "}"
    return out


def with_exponent(latex, exp):
    # SymPy's Pow printer passes exp already rendered as a LaTeX string.
    return latex if exp is None else rf"\left({latex}\right)^{{{exp}}}"


def permutation_sign(items):
    inversions = sum(items[i] > items[j] for i in range(len(items))
                     for j in range(i + 1, len(items)))
    return -1 if inversions % 2 else 1


class Pair(sp.Function):
    """Metric, momentum component, or scalar product, depending on its slots."""
    nargs = 2

    @classmethod
    def eval(cls, a, b):
        if sp.default_sort_key(a) > sp.default_sort_key(b):
            return cls(b, a)

    def _sympystr(self, printer):
        a, b = self.args
        if isinstance(a, Index) and isinstance(b, Index):
            return f"metric({printer.doprint(a)}, {printer.doprint(b)})"
        if isinstance(a, Index) or isinstance(b, Index):
            i, p = (a, b) if isinstance(a, Index) else (b, a)
            return f"V({printer.doprint(p)}, {printer.doprint(i)})"
        return f"SP({printer.doprint(a)}, {printer.doprint(b)})"

    def _latex(self, printer, exp=None):
        a, b = self.args
        if isinstance(a, Index) and isinstance(b, Index):
            return with_exponent(tensor_latex("g", (a, b), printer), exp)
        if isinstance(a, Index) or isinstance(b, Index):
            i, p = (a, b) if isinstance(a, Index) else (b, a)
            head = "{" + printer._print(p) + "}"
            return with_exponent(tensor_latex(head, (i,), printer), exp)
        latex = rf"\left({printer._print(a)} \cdot {printer._print(b)}\right)"
        return latex if exp is None else f"{latex}^{{{exp}}}"


class Epsilon(sp.Function):
    nargs = 4

    @classmethod
    def eval(cls, *args):
        if len(set(args)) < 4:
            return sp.S.Zero
        order = sorted(range(4), key=lambda j: sp.default_sort_key(args[j]))
        if order != [0, 1, 2, 3]:
            return permutation_sign(order) * cls(*(args[j] for j in order))

    def _sympystr(self, printer):
        return "EPS(" + ", ".join(printer.doprint(a) for a in self.args) + ")"

    def _latex(self, printer, exp=None):
        # Momenta in a slot mean contraction: epsilon^{p q mu nu} = epsilon^{a b mu nu} p_a q_b.
        return with_exponent(tensor_latex(r"\epsilon", self.args, printer), exp)


# Matrix-valued placeholders keep multiplication noncommutative until tracing.
Gamma = sp.Function("Gamma", nargs=1, commutative=False)
Slash = sp.Function("Slash", nargs=1, commutative=False)
Trace = sp.Function("Trace", nargs=1, commutative=True)
G5 = sp.Symbol("G5", commutative=False)


def vector(p):
    if not isinstance(p, sp.Symbol) or not p.is_commutative:
        raise ValueError("Use a named momentum, e.g. slash(p). Expand slash(p+q) as slash(p)+slash(q).")
    return p


def parse(text):
    """Parse just the documented arithmetic language, without Python eval."""
    def ga(*labels):
        if not labels:
            raise ValueError("GA needs an index or 5, e.g. GA[mu] or GA[5].")
        return sp.Mul(*(G5 if i == 5 else Gamma(implicit_index(i)) for i in labels))

    def gs(*momenta):
        if not momenta:
            raise ValueError("GS needs a momentum, e.g. GS[p].")
        return sp.Mul(*(Slash(vector(p)) for p in momenta))

    functions = {
        "DiracTrace": Trace, "GA": ga, "GS": gs,
        "FV": lambda p, i: Pair(vector(p), implicit_index(i)),
        "MT": lambda i, j: Pair(implicit_index(i), implicit_index(j)),
        "DiracSimplify": lambda expr: expr, "Contract": lambda expr: expr,
        "Tr": Trace, "G": lambda i: Gamma(index(i)),
        "slash": lambda p: Slash(vector(p)),
        "SP": lambda p, q: Pair(vector(p), vector(q)),
        "V": lambda p, i: Pair(vector(p), index(i)),
        "metric": lambda i, j: Pair(index(i), index(j)),
        "EPS": lambda *p: Epsilon(*(vector(x) for x in p)),
    }
    constants = {"G5": G5, "PL": (1-G5)/2, "PR": (1+G5)/2, "I": sp.I}

    def visit(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return sp.Rational(str(node.value))
        if isinstance(node, ast.Name):
            return constants.get(node.id, sp.Symbol(node.id))
        if isinstance(node, ast.UnaryOp):
            if isinstance(node.op, ast.USub):
                return -visit(node.operand)
            if isinstance(node.op, ast.UAdd):
                return visit(node.operand)
        if isinstance(node, ast.BinOp):
            a, b = visit(node.left), visit(node.right)
            if isinstance(node.op, ast.Add): return a + b
            if isinstance(node.op, ast.Sub): return a - b
            if isinstance(node.op, ast.Mult): return a * b
            if isinstance(node.op, ast.Div): return a / b
            if isinstance(node.op, ast.Pow): return a ** b
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id in functions and not node.keywords):
            return functions[node.func.id](*(visit(a) for a in node.args))
        raise ValueError("Use only the documented functions and +, -, *, /, ., ^, **.")

    if re.search(r'_\{', text):
        raise ValueError("Write subscripts without braces: m_tau, m_nu_tau (not m_{tau}).")
    try:
        source = normalize_input("(" + text.strip() + "\n)")
    except tokenize.TokenError:
        raise ValueError("Unbalanced brackets: check that every ( and [ is closed.") from None
    return visit(ast.parse(source, mode="eval").body)


@lru_cache(maxsize=4096)
def pairings(word):
    """Scalar Clifford part: pair the first slot with every possible partner."""
    if not word:
        return sp.S.One
    if len(word) % 2:
        return sp.S.Zero
    return sp.Add(*( (-1)**(j-1) * Pair(word[0], word[j])
                    * pairings(word[1:j] + word[j+1:])
                    for j in range(1, len(word)) ))


@lru_cache(maxsize=4096)
def trace_word(word):
    # Anticommute every gamma5 to the right; pairs of gamma5 then cancel.
    slots, n5, sign = [], 0, 1
    for item in word:
        if item == G5:
            n5 += 1
        else:
            sign *= (-1)**n5
            slots.append(item)
    slots = tuple(slots)
    if n5 % 2 == 0:
        return sign * 4 * pairings(slots)
    if len(slots) < 4 or len(slots) % 2:
        return sp.S.Zero

    # The grade-four Clifford part supplies epsilon. Leave four slots unpaired,
    # contract all others, and account for the permutation of the original word.
    result = 0
    for kept in combinations(range(len(slots)), 4):
        rest = tuple(j for j in range(len(slots)) if j not in kept)
        parity = permutation_sign(rest + kept)
        result += (parity * pairings(tuple(slots[j] for j in rest))
                   * Epsilon(*(slots[j] for j in kept)))
    return -4 * sp.I * sign * result


def trace_expression(expr):
    """Expand sums/masses inside one trace, preserving gamma-matrix order."""
    result = 0
    for term in sp.Add.make_args(sp.expand(expr)):
        scalar, matrices = term.args_cnc()
        if any(x.has(Trace) for x in scalar):
            raise ValueError("Nested traces are not supported; write their product outside Tr.")
        word = []
        for factor in matrices:
            base, power = factor.as_base_exp()
            if not power.is_Integer or power < 0:
                raise ValueError("Matrix powers must be nonnegative integers.")
            if base == G5:
                item = G5
            elif base.func in (Gamma, Slash):
                item = base.args[0]
            else:
                raise ValueError(f"Unsupported matrix factor: {factor}")
            word.extend([item] * int(power))
        result += sp.Mul(*scalar) * trace_word(tuple(word))
    return result


def evaluate_traces(expr):
    if expr.func == Trace:
        return trace_expression(expr.args[0])
    if expr.is_Add:
        return sp.Add(*(evaluate_traces(a) for a in expr.args))
    if expr.is_Mul:
        return sp.Mul(*(evaluate_traces(a) for a in expr.args))
    if expr.is_Pow:
        return evaluate_traces(expr.base) ** expr.exp
    if expr.has(Gamma, Slash, G5):
        raise ValueError("Put all gamma matrices and slashed momenta inside Tr(...).")
    return expr


def tensor_factors(term):
    """Separate scalar prefactors and expand positive powers of Pair/Epsilon."""
    coefficient, factors = sp.S.One, []
    for f in sp.Mul.make_args(term):
        base, power = f.as_base_exp()
        if isinstance(base, (Pair, Epsilon)) and power.is_Integer and power > 0:
            factors.extend([base] * int(power))
        elif f.has(Index):
            raise ValueError("Indexed tensors must occur polynomially; no indexed denominators.")
        else:
            coefficient *= f
    return coefficient, factors


def contract_term(term):
    coefficient, factors = tensor_factors(term)
    eps = [i for i, f in enumerate(factors) if isinstance(f, Epsilon)]
    if len(eps) >= 2:
        i, j = eps[:2]
        a, b = factors[i].args, factors[j].args
        # Minkowski signature: epsilon epsilon = -det(metric).
        determinant = -sp.Add(*(permutation_sign(p) * sp.Mul(*(
            Pair(a[k], b[p[k]]) for k in range(4))) for p in permutations(range(4))))
        other = sp.Mul(*(f for k, f in enumerate(factors) if k not in (i, j)))
        return contract(coefficient * other * determinant, check_free=False)

    while True:
        occurrences = {}
        for i, f in enumerate(factors):
            for j, a in enumerate(f.args):
                if isinstance(a, Index):
                    occurrences.setdefault(a.args[0], []).append((i, j, a.args[1]))
        repeated = [(name, positions) for name, positions in occurrences.items()
                    if len(positions) > 1]
        if not repeated:
            return coefficient * sp.Mul(*factors)
        name, positions = repeated[0]
        if len(positions) != 2 or sum(x[2] for x in positions) != 0:
            raise ValueError(f"Dummy index {name} must occur exactly twice. Use the same name in GA/FV/MT notation, or one upper and one lower occurrence in legacy G/V/metric notation; do not mix them.")
        (i, a, _), (j, b, _) = positions
        if i == j:
            if isinstance(factors[i], Epsilon):
                return sp.S.Zero
            coefficient *= 4  # g^mu_mu = 4
            factors.pop(i)
        else:
            if isinstance(factors[j], Pair):
                i, j, a, b = j, i, b, a
            if not isinstance(factors[i], Pair):
                raise ValueError("Unexpected epsilon contraction.")
            replacement = factors[i].args[1-a]
            args = list(factors[j].args)
            args[b] = replacement
            new_factor = factors[j].func(*args)
            factors = [f for k, f in enumerate(factors) if k not in (i, j)]
            if new_factor == 0:
                return sp.S.Zero
            scalar, extra = tensor_factors(new_factor)
            coefficient *= scalar
            factors.extend(extra)


def contract(expr, check_free=True):
    result = sp.expand(sp.Add(*(contract_term(t) for t in sp.Add.make_args(sp.expand(expr)))))
    if check_free and result != 0:
        free_sets = {frozenset(t.atoms(Index)) for t in sp.Add.make_args(result)}
        if len(free_sets) > 1:
            raise ValueError("Terms in a tensor sum must have the same free Lorentz indices.")
    return result


def evaluate(text, substitutions=None):
    """Return a SymPy result; optional replacements are applied simultaneously."""
    result = contract(evaluate_traces(parse(text)))
    if substitutions:
        rules = {parse(k) if isinstance(k, str) else k:
                 parse(v) if isinstance(v, str) else v
                 for k, v in substitutions.items()}
        result = sp.expand(result.subs(rules, simultaneous=True))
    return result


class NestedSubscriptPrinter(LatexPrinter):
    """Names with several underscores nest: m_nu_tau -> m_{\\nu_{\\tau}}.

    Also prints the unevaluated input: traces, gamma matrices, slashed momenta.
    """
    panel = False  # matplotlib has no \slashed, so the panel overlays a slash

    def _print_Function(self, expr, exp=None):
        # SymPy skips _print_<Name> for sp.Function("Name") objects, so dispatch here.
        method = getattr(self, "_print_" + expr.func.__name__, None)
        if expr.func in (Trace, Gamma, Slash):
            return method(expr, exp)
        return super()._print_Function(expr, exp)

    def _print_Trace(self, expr, exp=None):
        latex = rf"\mathrm{{Tr}}\left[{self._print(expr.args[0])}\right]"
        return latex if exp is None else f"{latex}^{{{exp}}}"

    def _print_Gamma(self, expr, exp=None):
        latex = tensor_latex(r"\gamma", expr.args, self)
        return latex if exp is None else f"\\left({latex}\\right)^{{{exp}}}"

    def _print_Slash(self, expr, exp=None):
        p = self._print(expr.args[0])
        latex = rf"{{{p}}}\!\!\!/\," if self.panel else rf"\slashed{{{p}}}"
        return latex if exp is None else f"\\left({latex}\\right)^{{{exp}}}"

    def _print_Symbol(self, expr, *args, **kwargs):
        if expr == G5:
            return r"\gamma^{5}"
        parts = expr.name.split('_')
        if len(parts) < 3 or not all(parts):
            return super()._print_Symbol(expr, *args, **kwargs)
        out = None
        for part in reversed(parts):
            name = super()._print_Symbol(sp.Symbol(part))
            if out is None:
                out = name
            else:
                out = ("{" + name + "}" if "_" in name else name) + "_{" + out + "}"
        return out


def to_latex(expr, panel=False, **settings):
    printer = NestedSubscriptPrinter(settings)
    printer.panel = panel
    return printer.doprint(expr)


def initial_latex(expression, panel=False):
    """LaTeX for the unevaluated input; long fractions print as 1/4 (...)."""
    return to_latex(parse(expression), panel, long_frac_ratio=2)


def latex_lines(result):
    """One LaTeX string per term, with explicit leading signs after the first."""
    lines = []
    for k, term in enumerate(sp.Add.make_args(result) if result.is_Add else [result]):
        negative = term.could_extract_minus_sign()
        body = to_latex(-term if negative else term)
        lines.append(("- " if negative else ("+ " if k else "")) + body)
    return lines


def latex_document(expression, result):
    initial = initial_latex(expression)
    aligned = " \\\\\n  &\\quad ".join(latex_lines(result))
    return ('% Conventions: four dimensions, metric (+---), epsilon^{0123} = +1\n'
            + '% Needs \\usepackage{amsmath} and \\usepackage{slashed}\n'
            + ''.join(f'% Input: {line}\n' for line in expression.strip().splitlines())
            + '\n% Initial expression:\n' + initial + '\n'
            + '\n% Result (inline):\n' + to_latex(result) + '\n'
            + '\n% Initial expression = result, multi-line:\n\\begin{align*}\n  &'
            + initial + ' \\\\\n  &= ' + aligned + '\n\\end{align*}\n')


def show_latex_panel(result, title='Dirac trace result', expression=None):
    """Render the (initial expression and) result in a separate matplotlib window."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print('Install matplotlib to show the LaTeX panel: python -m pip install matplotlib')
        return
    lines = latex_lines(result)
    lines[0] = '= ' + lines[0]
    if expression is not None:
        lines.insert(0, initial_latex(expression, panel=True))
    fig = plt.figure(figsize=(8, 0.55 * len(lines) + 0.6))
    fig.canvas.manager.set_window_title(title)
    texts = [fig.text(0.02, 1 - (k + 0.8) / (len(lines) + 0.6), f'${line}$', fontsize=15)
             for k, line in enumerate(lines)]
    fig.canvas.draw()  # measure the rendered lines, then widen the window to fit
    width = max(t.get_window_extent().width for t in texts) / fig.dpi
    fig.set_size_inches(max(4, width / 0.96 + 0.2), fig.get_size_inches()[1], forward=True)
    plt.show()


def read_expression_file(path):
    """Read a plain-text expression, tolerating Python-style surrounding quotes."""
    text = path.read_text().strip()
    if text.startswith('{\\rtf'):
        raise ValueError(f"{path} is saved as rich text (RTF). In TextEdit use "
                         "Format > Make Plain Text (Shift-Cmd-T), then save again.")
    quotes = '"\'“”'  # straight and curly quotes
    return text.strip(quotes).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--expr', help='Expression enclosed in shell quotes')
    group.add_argument('--input', type=Path, help='Text file containing an expression')
    parser.add_argument('--set', action='append', default=[], metavar='LHS=RHS',
                        help='Optional scalar-product or scalar substitution; repeat as needed')
    parser.add_argument('--output', type=Path, help='Also save the input and result as text')
    parser.add_argument('--latex', type=Path, default=Path('dirac_trace_latex.txt'),
                        help='Save the result in LaTeX syntax (default: dirac_trace_latex.txt)')
    parser.add_argument('--no-panel', action='store_true',
                        help='Do not open the window showing the rendered result')
    args = parser.parse_args()
    try:
        expression = read_expression_file(args.input) if args.input else (args.expr or EXPRESSION)
        rules = {}
        for rule in args.set:
            lhs, rhs = rule.split('=', 1)
            rules[lhs.strip()] = rhs.strip()
        result = evaluate(expression, rules)
        output = ('Conventions: four dimensions, metric (+---), epsilon^0123 = +1\n\n'
                  + 'Input:\n' + expression.strip() + '\n\nResult:\n' + str(result) + '\n')
        print(output)
        if result.has(Index):
            print('The result has free Lorentz indices; it is a tensor, not a scalar.')
        if args.output:
            args.output.write_text(output)
        args.latex.write_text(latex_document(expression, result))
        print(f'LaTeX written to {args.latex}')
        if not args.no_panel:
            show_latex_panel(result, expression=expression)
    except (ValueError, TypeError, SyntaxError, tokenize.TokenError, OSError) as error:
        parser.exit(2, f'Input error: {error}\n')


if __name__ == '__main__':
    main()
