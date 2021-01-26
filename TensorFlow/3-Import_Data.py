import numpy as np
import pandas as pd

#Load data from csv
root_square_cases = pd.read_csv('Test_root_square.csv')
#Convert to numpy array
root_square_cases = np.array(root_square_cases)