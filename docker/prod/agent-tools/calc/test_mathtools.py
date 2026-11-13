"""mathtools: answers, and that chat text can't run Python. Run in the image:
docker run --rm -v $PWD/test_mathtools.py:/app/test_mathtools.py:ro agent-tools-calc:local python -m unittest test_mathtools
"""

import unittest

from mathtools import MathError, parse, physics_units, symbolic_math


class SymbolicMath(unittest.TestCase):
    def test_calculus(self):
        self.assertIn("= x^2*cos(x) + 2*x*sin(x)", symbolic_math("derivative", "x^2 sin(x)"))
        self.assertIn("= y*cos(x*y)", symbolic_math("derivative", "sin(x*y)"))
        self.assertIn("d^2/dx^2", symbolic_math("derivative", "x^3", order=2))
        self.assertIn("= x^3/3 + C", symbolic_math("integral", "x^2"))
        self.assertIn("= 2", symbolic_math("integral", "sin(x)", at="0, pi"))
        self.assertIn("= 1", symbolic_math("limit", "sin(x)/x", at="0"))
        self.assertIn("= ∞", symbolic_math("limit", "1/x", at="0+"))
        self.assertIn("= e", symbolic_math("limit", "(1+1/n)^n", at="oo"))
        self.assertIn("no limit", symbolic_math("limit", "sin(1/x)", at="0"))
        self.assertIn("x^3/6", symbolic_math("series", "sin(x)"))
        self.assertIn("= sqrt(pi)", symbolic_math("integral", "e^(-x^2)", at="-oo, oo"))

    def test_algebra(self):
        out = symbolic_math("solve", "x^2-5x+6=0")
        self.assertIn("x = 2", out)
        self.assertIn("x = 3", out)
        out = symbolic_math("solve", "x+y=3; x-y=1")
        self.assertIn("x = 2", out)
        self.assertIn("y = 1", out)
        self.assertIn("(x - 2)*(x + 2)", symbolic_math("factor", "x^2-4"))
        self.assertIn("= 1", symbolic_math("simplify", "sin(x)^2+cos(x)^2"))
        self.assertIn("≈ 4.44288", symbolic_math("evaluate", "sqrt(2)π"))
        self.assertIn("x = -i", symbolic_math("solve", "x^2+1=0"))

    def test_matrices(self):
        self.assertIn("= -2", symbolic_math("determinant", "[[1,2],[3,4]]"))
        self.assertIn("= -2", symbolic_math("determinant", "[1, 2; 3, 4]"))
        self.assertIn("[[-2, 1], [3/2, -1/2]]", symbolic_math("inverse", "[[1,2],[3,4]]"))
        out = symbolic_math("eigenvalues", "[[2,0],[0,3]]")
        self.assertIn("2", out)
        self.assertIn("3", out)
        self.assertIn("λ = 1", symbolic_math("eigenvectors", "[[2,1],[1,2]]"))
        self.assertIn("= 1", symbolic_math("rank", "[[1,2],[2,4]]"))
        self.assertIn("no inverse", symbolic_math("inverse", "[[1,2],[2,4]]"))
        self.assertIn("[[2, 1], [4, 3]]", symbolic_math("evaluate", "[[1,2],[3,4]]*[[0,1],[1,0]]"))

    def test_rejects_python(self):
        for text in ("__import__('os').system('id')", "x.__class__", "().__class__", "open('/etc/passwd')",
                     "lambda: 1", "[x for x in (1,)]", "sin('1')", "Symbol('a b')", "globals()",
                     "Float('nan; 1')", "a if b else c", "x[0]", "getattr(x, 'y')", "eval('1')",
                     "S('x')", "sympify('x')"):
            with self.assertRaises(MathError, msg=text):
                parse(text)

    def test_errors(self):
        with self.assertRaises(MathError):
            symbolic_math("dance", "x")
        with self.assertRaises(MathError):
            symbolic_math("derivative", "a*b")
        with self.assertRaises(MathError):
            symbolic_math("determinant", "[[1,2,3],[4,5,6]]")
        self.assertIn("= x*cos(x*y)", symbolic_math("derivative", "sin(x*y)", variable="y"))


class PhysicsUnits(unittest.TestCase):
    def test_conversions(self):
        self.assertIn("= 20 m/s", physics_units("72 km/h", "m/s"))
        self.assertIn("298.15 K", physics_units("25 degC", "K"))
        self.assertIn("77 °F", physics_units("25 °C", "degF"))
        self.assertIn("2.4796", physics_units("planck_constant*speed_of_light/(500 nm)", "eV"))
        self.assertIn("2.99792×10⁸ m/s", physics_units("speed_of_light"))
        self.assertIn("1.60218×10⁻¹⁹", physics_units("elementary_charge", "C"))
        self.assertIn("9.80665", physics_units("standard_gravity", "m/s^2"))
        self.assertIn("6.674", physics_units("gravitational_constant"))
        self.assertIn("1.38065×10⁻²³", physics_units("boltzmann_constant", "J/K"))
        self.assertIn("6.02214×10²³", physics_units("avogadro_constant", "1/mol"))
        self.assertIn("8.85419×10⁻¹²", physics_units("vacuum_permittivity", "F/m"))
        self.assertIn("1.05457×10⁻³⁴", physics_units("hbar", "J*s"))
        self.assertIn("9.10938×10⁻³¹", physics_units("electron_mass", "kg"))
        self.assertIn("5 μm", physics_units("0.005 mm", "µm"))

    def test_errors(self):
        with self.assertRaises(MathError):
            physics_units("5 kg", "m")
        with self.assertRaises(MathError):
            physics_units("5 blorps")
        with self.assertRaises(MathError):
            physics_units("__import__('os')")


if __name__ == "__main__":
    unittest.main()
