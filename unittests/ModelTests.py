#!/usr/bin/env python
"""
model tests
"""


import unittest
import sys
sys.path.append('../')
## import model specific functions and variables
from model import *

class ModelTest(unittest.TestCase):
    """
    test the essential functionality
    """
        
    def test_01_train(self):
        """
        test the train functionality
        """
        data_dir=os.path.join(".","data","cs-train")
        ## train the model
        model_train(data_dir,test=True)
        self.assertTrue(os.path.exists(data_dir))

    def test_02_load(self):
        """
        test the train functionality
        """
                        
        ## train the model
        all_data, all_models = model_load()
        data = all_data['all']
        target_date = "{}-{}-{}".format('2018','05','01')
        self.assertTrue('all' in all_models.keys())
        self.assertTrue(target_date in data['dates'])

       
    def test_03_predict(self):
        """
        test the predict function input
        """

    
        ## ensure that a list can be passed
        country='all'
        year='2018'
        month='01'
        day='05'
        result = model_predict(country,year,month,day)
        y_pred = result['y_pred']
        for p in y_pred:
            self.assertTrue(isinstance(p,float))

          
### Run the tests
if __name__ == '__main__':
    unittest.main()
