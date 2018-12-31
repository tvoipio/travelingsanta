import pandas as pd
import numpy as np

input_filename = 'nicelist.txt'

# Korvatunturi coordinates 68.073611N 29.315278E
EF_COORDS = (68.073611, 29.315278)

def read_input(fname=input_filename):
    input_raw = pd.read_csv(fname, sep=';',
                            header=None,
                            names=['ID', 'lat_deg', 'lon_deg', 'wt'])

    # Add distribution centre (Korvatunturi)

    ef_DF = pd.DataFrame({'ID': [-1], 'lat_deg': [EF_COORDS[0]],
                          'lon_deg': [EF_COORDS[1]], 'wt': [0]})

    return pd.concat([ef_DF, input_raw], ignore_index=True)
