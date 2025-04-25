import unittest
import os
import shutil
import time


from scikit_learn_bench.core import bench
from scikit_learn_bench import CONST

PROFILER_PATH="profiling"



class MyTestCase(unittest.TestCase):
    def test_time(self):
        """ Nota bene: in this test min>max to trigger artifically the timeout all the time """
        profiler_type="time"
        default=CONST.IS_MAX_PROF_TIME
        CONST.IS_MAX_PROF_TIME=True
        s=time.time()
        scores = bench(num_samples=10, num_features=2, num_output=2, max_prof_time=0.1, min_prof_time=0.2, ml_type="clu",
                       profiler_type=profiler_type, line_profiler_path=PROFILER_PATH)
        enlapsed_time=time.time()-s
        self.assertTrue(enlapsed_time < 100)
        CONST.IS_MAX_PROF_TIME = default

        #TODO: timeout is still an unpexerimental feature in this project.
        # It seems that the algorithm 'ExtraTreesClassifier' is breaking everything if we try to timeout it