import numpy as np
from scipy.optimize import least_squares
import scipy.integrate as it


# Observations
tdata = np.array([0.9, 1.5, 13.8, 19.8, 24.1, 28.2, 35.2, 60.3, 74.6, 81.3])
ydata = np.array([455.2, 428.6, 124.1, 67.3, 43.2, 28.1, 13.1, -0.4, -1.3, -1.5])

# Model is y(t) = x[0] * exp(x[1] * t)
def prediction_error(x):
    def derivs(t, y):
        return x[1] * y

    y0 = x[0:1]

    sol = it.solve_ivp(derivs, (tdata[0], tdata[-1]), y0, t_eval=tdata)

    try:
        return (ydata - sol.y)[0]
    except ValueError:
        import ipdb

        ipdb.set_trace()


# Define the starting point
x0 = np.array([100.0, -1.0])

# We expect exponential decay: set upper bound x[1] <= 0
upper = np.array([1e20, 0.0])

res = least_squares(prediction_error, x0, bounds=(-np.inf, upper), verbose=2)

print(res.x)
import ipdb

ipdb.set_trace()
