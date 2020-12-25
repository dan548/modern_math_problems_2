from scipy import integrate
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


theta = 0
delta = 0
scale = 1


def hampel_under_integral(y):
    global theta
    global delta
    global scale
    t = np.exp((theta - y) / scale)
    return (t ** (delta + 1)) * (t - 1) * (t - 1) / ((t + 1) ** (2*delta+4))
    
    
def hampel_integral():
    return integrate.quad(hampel_under_integral, -116, 116)


def hampel_function(y, t, s, d):
    global theta
    global scale
    global delta
    theta = t
    scale = s
    delta = d
    temp = np.exp((theta - y) / scale)
    res = -scale
    res *= ((temp ** delta) * (temp - 1) / ((temp + 1) ** (2*delta+1)))
    integral_result = hampel_integral()
    return res / integral_result


def hampel_function_mean(y, t):
    global theta
    theta = t
    return y - theta


def hampel_function_median(y, t, s):
    global theta
    global scale
    theta = t
    scale = s
    return np.sign(y - theta) * 2 * scale


xs = np.arange(-10., 10., 0.2)
infl_fun_1 = (hampel_function(x, 0, 1, 0)[0] for x in xs)
infl_fun_2 = (hampel_function(x, 0, 1, 0.1)[0] for x in xs)
infl_fun_3 = (hampel_function(x, 0, 1, 0.5)[0] for x in xs)
infl_fun_4 = (hampel_function(x, 0, 1, 1)[0] for x in xs)
plt.plot(xs, list(infl_fun_1), 'r')
plt.plot(xs, list(infl_fun_2), 'b')
plt.plot(xs, list(infl_fun_3), 'g')
plt.plot(xs, list(infl_fun_4), 'y')
plt.show()