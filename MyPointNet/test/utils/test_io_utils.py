from MyPointNet.src.io import TrainingConfiguration

import unittest


class TestTrainingConfiguration(unittest.TestCase):

    def test_flags(self):
        train_config = TrainingConfiguration()
        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
