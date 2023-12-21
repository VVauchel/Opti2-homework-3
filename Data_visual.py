import numpy as np
import pandas as pd
import Fun_Jac_Hess_v2 as fun
import SPFM as SPFM
from datetime import datetime

# Consider most updated results
filename = ('results_20231221120610.csv')

# Read the CSV file into a DataFrame
result_df = pd.read_csv(filename)


# Add the Test accuracy column
result_df['Test_accuracy'] = None

# Read Test_Data
B = SPFM.Read_Data_Test(0)

# Assign accuracy values
for index, row in result_df.iterrows():

    h_string = result_df['h'][index]
    # Convert the string to a NumPy array
    h = np.fromstring(h_string.replace('[', '').replace(']', ''), sep=' ')
    c = result_df['c'][index]

    # Example: Setting 'Test_accuracy' to the square of the 'Score' column
    result_df.at[index, 'Test_accuracy'] = SPFM.classifier(h, c, B)[0]

print(result_df)

smallest_time_rows = result_df[result_df.Num_Dig == 27].nsmallest(3,'time')


small_time_theta = smallest_time_rows['theta'].tolist()

print(small_time_theta)
