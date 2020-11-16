import statistics
import math


def mean(data):
    return statistics.mean(data)


def median(data):
    return statistics.median(data)


def variance(data):
    return statistics.pvariance(data)


def skewness(data):
    m = mean(data)
    s = 0
    for x in data:
        s += pow((x-m), 3)
    sigma_cube = pow(variance(data), 1.5)
    return s / (len(data) * sigma_cube)


def kurtosis(data):
    m = mean(data)
    s = 0
    for x in data:
        s += pow((x-m), 4)
    d_squared = pow(variance(data), 2)
    return s / (len(data) * d_squared)
