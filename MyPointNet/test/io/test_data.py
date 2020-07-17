from src.io.data import PointNetTfUtils


import unittest


class TestIO(unittest.TestCase):

    def test_parse_configuration(self):
        pn_utils = PointNetTfUtils(batch_size=2, num_points=5, input_channels=3, output_dim=1)
        input_pl, labels_pl, is_training_pl = pn_utils.get_placeholders()
        self.assertEqual(input_pl.shape, (2,5,3))
        self.assertEqual(labels_pl.shape, (2,))
        self.assertEqual(is_training_pl.shape, ())


if __name__ == '__main__':
    unittest.main()
