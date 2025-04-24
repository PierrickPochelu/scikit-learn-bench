import unittest
from src.scikit_learn_bench.core import bench
from src.scikit_learn_bench import CONST


def is_scores_type_valid(scores)->bool:
    for ml_algo_name, score_vector in scores.items():
        if not isinstance(ml_algo_name,str):
            return False
        for v in score_vector:
            is_valid = isinstance(v,float) or isinstance(v,int) or v == CONST.NANSTR
            if not is_valid:
                return False
    return True

class MyTestCase(unittest.TestCase):
    def test_clu(self):
        scores = bench(num_samples=10, num_features=2, num_output=1, fix_comp_time=0.01, ml_type="clu")
        self.assertGreater(len(scores), 10)

        self.assertTrue("KMeans" in scores)
        self.assertTrue("DBSCAN" in scores)
        self.assertTrue("AgglomerativeClustering" in scores)

        self.assertTrue("MLPRegressor" not in scores)
        self.assertTrue("KNeighborsRegressor" not in scores)
        self.assertTrue("Ridge" not in scores)

        self.assertTrue(scores["KMeans"][0] > 1)
        self.assertTrue(scores["KMeans"][1] > 1)
        self.assertTrue(scores["AgglomerativeClustering"][0] > 1)
        self.assertTrue(scores["AgglomerativeClustering"][1] == CONST.NANSTR)

        self.assertTrue(is_scores_type_valid(scores))

    def test_tra(self):
        scores = bench(num_samples=10, num_features=2, num_output=1, fix_comp_time=0.01, ml_type="tra")
        self.assertGreater(len(scores), 10)

        self.assertTrue("TSNE" in scores)
        self.assertTrue("PCA" in scores)
        self.assertTrue("TfidfTransformer" in scores)

        self.assertTrue("DecisionTreeClassifier" not in scores)
        self.assertTrue("GaussianProcessClassifier" not in scores)
        self.assertTrue("DBSCAN" not in scores)


        self.assertTrue(is_scores_type_valid(scores))

    def test_cla(self):
        scores=bench(num_samples=10, num_features=2, num_output=2, fix_comp_time=0.01, ml_type="cla")
        self.assertGreater(len(scores),10)

        self.assertTrue("GaussianProcessClassifier" in scores)
        self.assertTrue("MLPClassifier" in scores)
        self.assertTrue("DecisionTreeClassifier" in scores)

        self.assertTrue("KNeighborsRegressor" not in scores)
        self.assertTrue("Ridge" not in scores)
        self.assertTrue("MLPRegressor"  not in scores)

        self.assertTrue(scores["DummyClassifier"][0] > 1)
        self.assertTrue(scores["DummyClassifier"][0] > 1)

        self.assertTrue(is_scores_type_valid(scores))

    def test_reg_1D(self):
        scores = bench(num_samples=10, num_features=2, num_output=1, fix_comp_time=0.01, ml_type="reg")
        self.assertTrue(len(scores) >10)

        self.assertTrue("KNeighborsRegressor" in scores)
        self.assertTrue("Ridge" in scores)
        self.assertTrue("MLPRegressor" in scores)

        self.assertTrue("GaussianProcessClassifier" not in scores)
        self.assertTrue("MLPClassifier" not in scores)
        self.assertTrue("DecisionTreeClassifier" not in scores)

        self.assertTrue(scores["DummyRegressor"][0] > 1)
        self.assertTrue(scores["DummyRegressor"][0] > 1)

        self.assertTrue(is_scores_type_valid(scores))

    def test_reg_nD(self):
        scores = bench(num_samples=10, num_features=2, num_output=2, fix_comp_time=0.01, ml_type="reg")
        self.assertTrue(len(scores) >10)

        self.assertTrue("KNeighborsRegressor" in scores)
        self.assertTrue("Ridge" in scores)
        self.assertTrue("MLPRegressor" in scores)

        self.assertTrue("GaussianProcessClassifier" not in scores)
        self.assertTrue("MLPClassifier" not in scores)
        self.assertTrue("DecisionTreeClassifier" not in scores)

        self.assertTrue(scores["DummyRegressor"][0] > 1)
        self.assertTrue(scores["DummyRegressor"][0] > 1)

        self.assertTrue(is_scores_type_valid(scores))