import unittest
import pandas as pd 

import sys
sys.path.insert(0, '/home/mahlet/10ac/Smart_Ad_AB_test/')
#from helper import duplicate_calculator,data_summary
from scripts.model import data_spliter

class TestSum(unittest.TestCase):
    """
	A class for unit-testing function in the helper.py file

	Args:
        -----
	   unittest.TestCase this allows the new class to inherit
	   from the unittest module
	"""
    
    def test_data_path(self):
        """
        Test duplicate calculator function 
        """
        data= pd.read_csv("data/processed_data.csv")
        result,result2 = data_spliter(data)

        self.assertEqual(type(result), "pandas.core.frame.DataFrame", "result is not dataframe") 
    
    

if __name__ == '__main__':
    # to run test terminal python3 -m unittest discover -v -s . -p "*test_*.py"
    unittest.main()
