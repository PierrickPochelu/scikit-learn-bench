import unittest
import os
import shutil

from tests.test_core import is_scores_type_valid
from scikit_learn_bench.core import bench

PROFILER_PATH="profiling"



class MyTestCase(unittest.TestCase):
    def test_time(self):
        profiler_type="time"
        scores = bench(num_samples=10, num_features=2, num_output=2, fix_comp_time=0.01, ml_type="cla",
                       profiler_type=profiler_type, line_profiler_path=PROFILER_PATH)
        self.assertTrue(is_scores_type_valid(scores), scores)
        first_prof_result = list(scores.values())[0]
        self.assertTrue(len(first_prof_result)==2)

    def test_time_memory(self):
        profiler_type="timememory"
        scores = bench(num_samples=10, num_features=2, num_output=2, fix_comp_time=0.01, ml_type="cla",
                       profiler_type=profiler_type, line_profiler_path=PROFILER_PATH)
        self.assertTrue(is_scores_type_valid(scores), scores)
        first_prof_result = list(scores.values())[0]
        self.assertTrue(len(first_prof_result)==4)

    def test_time_line(self):
        if os.path.exists(PROFILER_PATH):
            shutil.rmtree(PROFILER_PATH)

        profiler_type="timeline"
        scores = bench(num_samples=10, num_features=2, num_output=2, fix_comp_time=0.01, ml_type="cla",
                       profiler_type=profiler_type, line_profiler_path=PROFILER_PATH)
        self.assertTrue(is_scores_type_valid(scores), scores)
        first_prof_result = list(scores.values())[0]
        self.assertTrue(len(first_prof_result)==2)

        self.assertTrue(os.path.exists(PROFILER_PATH))
        for ml_algo_name in scores:
            file_name = ml_algo_name+".prof"
            expected_path=os.path.join(PROFILER_PATH, file_name)
            self.assertTrue(os.path.exists(expected_path))
