import numpy as np
import pandas as pd
import Fun_Jac_Hess_v2 as fun
import SPFM as SPFM
from datetime import datetime

Num_Dig = [1, 2, 3, 5]
theta_factor = [16, 8, 1, 1/8, 1/16]  # theta = fac*(1/v)
tau_range = [.1, .5, .9]
lambd_range = [1, 5, 10]

# Initialize an empty DataFrame
result_df = pd.DataFrame(columns=['Num_Dig', 'lambda', 'tau', 'theta', 'eps', 'h', 'c', 'time'])

for i in Num_Dig:
    # Read data
    A, B = SPFM.Read_Data(i*9, i*9)
    for lambd in lambd_range:
        for tau in tau_range:
            for t in theta_factor:
                # Evaluate theta from t and number of digits
                print(f'Running Num_digit = {i},lambda = {lambd}, tau = {tau}, theta_fac = {t}')
                theta = t*(4*i*9 + 1)**(-1)

                eps = 1
                x, time = SPFM.long_path_method(A, B, theta, lambd, eps, tau)

                h = x[:len(A[0, :])]
                c = x[len(A[0, :])]

                # Append a new row to the DataFrame
                result_df = result_df._append({'Num_Dig': i*9, 'lambda': lambd, 'tau': tau,
                                              'theta': theta, 'eps': eps, 'h': h, 'c': c, 'time': time},
                                             ignore_index=True)

                # Save the DataFrame to a new CSV file with a timestamp
                timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                result_df.to_csv(f'results_{timestamp}.csv', index=False)

# Print or save the final DataFrame if needed
print(result_df)
