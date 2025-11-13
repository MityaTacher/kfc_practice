from typing import Optional

import numpy as np
import pandas

dataframe = pandas.read_csv("data/data.csv").iloc[:, 1].to_list()
TIME_ARRAY = np.array(dataframe)
TIME_ARRAY = TIME_ARRAY[TIME_ARRAY != 0]
TOTAL_LENGHT = len(TIME_ARRAY)
del dataframe


