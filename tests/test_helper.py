import unittest
import pandas as pd 

import sys
sys.path.insert(0, '/home/mahlet/10ac/Smart_Ad_AB_test/')
#from helper import duplicate_calculator,data_summary
from scripts.model import data_spliter

class TestSum(unittest.TestCase):
    
    def test_duplicate_calculator(self):
        """
        Test that it can sum a list of integers
        """
        data= pd.read_csv("data/processed_data.csv")
        result,result2 = data_spliter(data)

        self.assertEqual(type(result), "pandas.core.frame.DataFrame", "result is not dataframe") 

if __name__ == '__main__':
    # to run test terminal python3 -m unittest discover -v -s . -p "*Test_*.py"
    unittest.main()