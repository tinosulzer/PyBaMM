import numpy as np
from scipy.optimize import least_squares
import scipy.integrate as it
import matplotlib.pyplot as plt


# Observations
tdata = np.array([0.9, 1.5, 13.8, 19.8, 24.1, 28.2, 35.2, 60.3, 74.6, 81.3])
ydata = np.array([455.2, 428.6, 124.1, 67.3, 43.2, 28.1, 13.1, -0.4, -1.3, -1.5])

# Model is y(t) = x[0] * exp(x[1] * t)
def solve(x):
    return x[0] * np.exp(-x[1] * tdata)


def prediction_error(x):
    return solve(x) - ydata


def jac(x):
    return np.vstack([np.exp(-x[1] * tdata), -x[0] * tdata * np.exp(-x[1] * tdata)]).T


# Define the starting point
x0 = np.array([100.0, 1.0])

# We expect exponential decay: set upper bound x[1] <= 0
lower = np.array([-np.inf, 0.0])

# res = least_squares(prediction_error, x0, bounds=(lower, np.inf), verbose=2)
res = least_squares(prediction_error, x0, jac=jac, bounds=(lower, np.inf), verbose=2)

print(res.x)
plt.plot(tdata, solve(res.x), tdata, ydata, "o")
plt.show()
