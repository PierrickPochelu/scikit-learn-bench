import unittest
from io import StringIO
import sys

from scikit_learn_bench import CONST
from scikit_learn_bench.display import print_table, smart_round

class TestUtilityFunctions(unittest.TestCase):

    def test_smart_round(self):

        CONST.ROUNDING = 3
        CONST.THRESHOLD_DECIMAL = 100

        self.assertEqual(smart_round(12345.2546), 12345)
        self.assertEqual(smart_round(0.012345), 0.0123)
        self.assertEqual(smart_round(1.2345), 1.23)
        self.assertEqual(smart_round(123.456), 123)
        self.assertEqual(smart_round(0.00056789), 0.000568)
        self.assertEqual(smart_round(999999), 999999)
        self.assertEqual(smart_round(0), 0)


    def test_print_table_output(self):
        CONST.ROUNDING = 3
        CONST.THRESHOLD_DECIMAL = 100

        score = {
            "AlgorithmA": (0.012345, 123.456),
            "AlgorithmB": (10.234, 10.00001),
            "AlgorithmC": (1.2345, 0.00056789),
        }

        expected_lines = [
            "AlgorithmA 0.0123   123        ",
            "AlgorithmC 1.23     0.000568   ",
            "AlgorithmB 10.2     10.0       ",
        ]

        captured_output = StringIO()
        sys.stdout = captured_output
        print_table(score,1)
        sys.stdout = sys.__stdout__

        output_lines = captured_output.getvalue().split('\n')

        #expected_lines = [l.strip() for l in expected_lines]
        for line, expected in zip(output_lines, expected_lines):
            self.assertTrue(line==expected, f"'{line}'!='{expected}'")

if __name__ == "__main__":
    unittest.main()
