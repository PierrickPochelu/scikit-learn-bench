import time
import unittest

from scikit_learn_bench import Timeout
from scikit_learn_bench import CONST

def f(x,y,z):
    time.sleep(1.)
    return x+y+z

class MyTestCase(unittest.TestCase):
    def test_timeout(self):
        s = time.time()
        y = Timeout.timeout_warp(f, timeout=2.)(1, 2, 3)
        enlapsed_time = time.time() - s

        self.assertTrue(enlapsed_time >= 1.)
        self.assertTrue(y==6)

    def test_timeout_ring(self):
        s = time.time()
        y = Timeout.timeout_warp(f, timeout=0.1)(1, 2, 3)
        enlapsed_time = time.time() - s

        self.assertTrue(enlapsed_time < 1.)
        self.assertTrue(y == CONST.NANSTR)
