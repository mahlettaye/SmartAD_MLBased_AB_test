
import pandas as pd
#import seaborn as sns
import matplotlib.pyplot as plt

import re
import os

from logger import log_writer

def data_loader(path):
    
    """Data_loader function will load the data 
    It will catch an exception and writes to the log file
    """
    try:
        data = pd.read_csv(path)
        log_writer("You have entered correct path") 
    except FileNotFoundError as e:
        log_writer("Incorrect file path") 
    return data

#Duplicate calculate
def duplicate_calculator(df):
  dups = df.duplicated()
  # report if there are any duplicates
  
  return print(dups.any())

def data_summary (df):
    #this function prints summary information 
    print ("Information about columns", df.info())
    print ("Your data shape is:", df.shape)
    print ("Your data row count is", df.size)
    print("The number of missing value(s): {}".format(df.isnull().sum().sum()))
    print("Columons having columns value:{}".format(df.columns[df.isnull().any()]))



  


