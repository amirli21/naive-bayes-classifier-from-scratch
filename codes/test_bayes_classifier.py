import unittest
from bayes_classifier import *


class TestBayesClassifier(unittest.TestCase):

    def test_separate_by_class(self):

        dataset: Dataset = [
            [3.393533211, 2.331273381, 0],
            [3.110073483, 1.781539638, 0],
            [1.343808831, 3.368360954, 0],
            [3.582294042, 4.67917911, 0],
            [2.280362439, 2.866990263, 0],
            [7.423436942, 4.696522875, 1],
            [5.745051997, 3.533989803, 1],
            [9.172168622, 2.511101045, 1],
            [7.792783481, 3.424088941, 1],
            [7.939820817, 0.791637231, 1],
        ]
        separated = {
            0: [[3.393533211, 2.331273381, 0], [3.110073483, 1.781539638, 0], [1.343808831, 3.368360954, 0],
                [3.582294042, 4.67917911, 0],
                [2.280362439, 2.866990263, 0]],
            1: [[7.423436942, 4.696522875, 1], [5.745051997, 3.533989803, 1], [9.172168622, 2.511101045, 1],
                [7.792783481, 3.424088941, 1], [7.939820817, 0.791637231, 1]]
        }

        self.assertEqual(separated, separate_by_class(dataset=dataset), "Values don't match.")

    def test_mean(self):

        data = [3, 5, 6]
        self.assertEqual(mean(data), 4.666666666666667, "There must be an error during calculation of the mean.")

    def test_sample_stddev(self):
        data = [3, 5, 6]
        self.assertEqual(round(sample_stddev(data), 3), 1.528)


if __name__ == '__main__':
    unittest.main()
