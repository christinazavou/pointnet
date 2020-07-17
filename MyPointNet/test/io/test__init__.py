from src.io import parse_configuration

import sys
import unittest
from unittest.mock import patch


class TestIO(unittest.TestCase):

    def test_parse_configuration(self):
        testargs = ['test__init__.py', "-c", "../../config/train.ini", "--gpu", "5"]
        with patch.object(sys, 'argv', testargs):
            res = parse_configuration()
            self.assertTrue(res.gpu == '5')


if __name__ == '__main__':
    unittest.main()
