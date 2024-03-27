

import numpy as np
import pandas as pd
from localreg import *
import random
import matplotlib.pyplot as plt





def etax(a,s1,s2,beta,x1,x2):
    level = (1 + (2 * a -1) * s1) / 2
    shift = s2 * x1 / (np.abs(x1)) /2 * (np.abs(x1) * (1-np.abs(x2))) ** beta
    return level + shift



def generate_dataset(number,s1,s2,beta):
    number_of_A1 = np.random.binomial(number,0.5)
    number_of_A0 = number - number_of_A1
    x11 = np.random.uniform(low=-1, high=1.0, size=number_of_A1).reshape(-1,1)
    x21 = np.random.uniform(low=-1, high=1.0, size=number_of_A1).reshape(-1,1)
    x1 = np.concatenate([x11,x21], axis = 1)
    x10 = np.random.uniform(low=-1, high=1.0, size=number_of_A0).reshape(-1,1)
    x20 = np.random.uniform(low=-1, high=1.0, size=number_of_A0).reshape(-1,1)
    x0 = np.concatenate([x10,x20], axis = 1)

    eta1 = etax(1,s1,s2,beta,x11,x21).squeeze()
    eta0 = etax(0,s1,s2,beta,x10,x20).squeeze()
    U1 = np.random.uniform(low=0, high=1.0, size=number_of_A1)
    U0 = np.random.uniform(low=0, high=1.0, size=number_of_A0)
    y1 = (U1< eta1) * 1
    y0 = (U0< eta0) * 1
    dataset1 = [x1, y1]
    dataset0 = [x0, y0]
    return [dataset1, dataset0]


