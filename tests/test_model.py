import unittest
import pandas as pd 

import sys
sys.path.insert(0, '/home/mahlet/10ac/Smart_Ad_AB_test/')
#from helper import duplicate_calculator,data_summary
from scripts.model import Modeling

class TestSum(unittest.TestCase):
    """
		A class for unit-testing class for helper.py file

		Args:
        -----
			unittest.TestCase this allows the new class to inherit
			from the unittest module
	"""
    
     
    
    def test_data_spliter(self):
        """
        test for returned dataframe column length 
        """
        data= pd.read_csv("data/processed_data.csv")
        
        result,result2 = Modeling.data_spliter(data)
        actual = len (result.columns)
        expected = 5

        self.assertEqual(actual, expected, "result is not dataframe")
    

if __name__ == '__main__':
    # to run test terminal python3 -m unittest discover -v -s . -p "*Test_*.py"
    unittest.main()