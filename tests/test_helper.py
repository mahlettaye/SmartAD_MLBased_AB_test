import unittest

import sys
sys.path.insert(0, '/home/mahlet/10ac/Smart_Ad_AB_test/scripts/')

from helper import data_loader
class TestSum(unittest.TestCase):
    
    def test_list_int(self):
        """
        Test that it can sum a list of integers
        """
        
        result = data_loader("data/AdSmartABdata.csv")
        self.assertEqual(result, "data/AdSmartABdata.csv")

if __name__ == '__main__':
    # to run test terminal python3 -m unittest discover -v -s . -p "*Test_*.py"
    unittest.main()