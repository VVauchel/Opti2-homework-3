import numpy as np
import pandas as pd
import Fun_Jac_Hess_v2 as fun
import SPFM as SPFM
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns



Num_Dig = [5]
theta = [.7]
tau_range = [.1]
lambd_range = [1]

# Consider the file with the data you want to get
filename = 'results_20231221120610.csv'

# Read the CSV file into a DataFrame
result_df = pd.read_csv(filename)

# Initialize an empty DataFrame
#result_df = pd.DataFrame(columns=['Num_Dig', 'lambda', 'tau', 'theta', 'eps', 'h', 'c', 'time'])

for i in Num_Dig:
    # Read data
    A, B = SPFM.Read_Data(i*9, i*9)

    for lambd in lambd_range:
        # Update starting point
        SPFM.update_x0(A, B, lambd)
        for tau in tau_range:
            for t in theta:
                # Evaluate theta from t and number of digits
                print(f'Running Num_digit = {i},lambda = {lambd}, tau = {tau}, theta_fac = {t}')

                eps = .1
                x, time = SPFM.long_path_method(A, B, t, lambd, eps, tau)

                h = x[:len(A[0, :])]
                c = x[len(A[0, :])]

                # Append a new row to the DataFrame
                result_df = result_df._append({'Num_Dig': i*9, 'lambda': lambd, 'tau': tau,
                                              'theta': t, 'eps': eps, 'h': h, 'c': c, 'time': time},
                                             ignore_index=True)

                # Save the DataFrame to a new CSV file with a timestamp
                timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                result_df.to_csv(f'results_{timestamp}.csv', index=False)

# Print or save the final DataFrame if needed
print(result_df)
