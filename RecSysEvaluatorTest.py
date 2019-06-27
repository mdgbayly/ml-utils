import unittest
import numpy as np
import pandas as pd

from RecSysModel import RecSysModel
from RecSysEvaluator import RecSysEvaluator

class RecSysEvaluatorTestCase(unittest.TestCase):

    def setUp(self):
        self.interactions_test = pd.DataFrame(columns=['personId', 'contentId', 'eventStrength'])
        self.interactions_test.loc[0] = ['1', 'A', 3.0]
        self.interactions_test.loc[1] = ['1', 'C', 3.0]
        self.interactions_test.loc[2] = ['1', 'E', 3.0]
        self.interactions_test.loc[3] = ['1', 'G', 3.0]
        self.interactions_test.loc[4] = ['1', 'I', 3.0]
        self.interactions_test.loc[5] = ['1', 'N', 3.0]
        self.interactions_test.loc[6] = ['1', 'O', 3.0]
        self.interactions_test.loc[7] = ['1', 'P', 3.0]
        self.interactions_test.loc[8] = ['1', 'Q', 3.0]
        self.interactions_test.loc[9] = ['1', 'R', 3.0]
        self.interactions_test.loc[10] = ['1', 'S', 3.0]
        self.interactions_test = self.interactions_test.set_index('personId')

        self.interactions_train = pd.DataFrame(columns=['personId', 'contentId', 'eventStrength'])
        self.interactions_train = self.interactions_train.set_index('personId')

        self.recommendations = pd.DataFrame(columns=['personId', 'contentId', 'eventStrength'])
        self.recommendations.loc[0] = ['1', 'B', 5.0]
        self.recommendations.loc[1] = ['1', 'C', 3.5]
        self.recommendations.loc[2] = ['1', 'D', 3.0]
        self.recommendations.loc[3] = ['1', 'E', 2.5]
        self.recommendations.loc[4] = ['1', 'F', 2.0]
        self.recommendations.loc[5] = ['1', 'G', 1.5]
        self.recommendations.loc[6] = ['1', 'H', 1.0]
        self.recommendations.loc[7] = ['1', 'I', 0.5]
        self.recommendations.loc[8] = ['1', 'Z1', 0.25]
        self.recommendations.loc[9] = ['1', 'Z2', 0.1]
        self.recommendations.loc[10] = ['1', 'Z3', 0.05]
        self.recommendations.loc[12] = ['1', 'Z4', 0.05]
        self.recommendations.loc[13] = ['1', 'Z5', 0.05]
        self.recommendations.loc[14] = ['1', 'Z6', 0.05]
        self.recommendations.loc[15] = ['1', 'S', 0.05]

        self.model = RecSysModel(self.recommendations)
        self.evaluator = RecSysEvaluator(self.interactions_test, self.interactions_test)

    def test_recall(self):
        metrics, results =\
            self.evaluator.evaluate_model(self.model)

        self.assertEqual(2/11, metrics['recall@5'], 'recall@5')
        self.assertEqual(4/11, metrics['recall@10'], 'recall@10')
        self.assertEqual(5/11, metrics['recall@25'], 'recall@25')

    def test_precision(self):
        metrics, results =\
            self.evaluator.evaluate_model(self.model)

        self.assertEqual(2/11, metrics['precision@5'])
        self.assertEqual(4/11, metrics['precision@10'])
        self.assertEqual(4/11, metrics['precision@25'])


if __name__ == '__main__':
    unittest.main()
