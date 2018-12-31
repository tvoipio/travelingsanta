import numpy as np
import datetime
import logging

lon_sin_col = 'lon_sin'
lat_sin_col = 'lat_sin'
lon_cos_col = 'lon_cos'
lat_cos_col = 'lat_cos'

# Radius of Earth, in kilometres (assuming spherical Earth)
r_earth = 6378.

def calc_trigs(geo_DF, lat_col, lon_col):
    """Calculate sines and cosines from geographical coordinates (in degrees)"""

    geo_DF[lon_sin_col] = np.sin(np.deg2rad(geo_DF[lon_col]))
    geo_DF[lat_sin_col] = np.sin(np.deg2rad(geo_DF[lat_col]))
    geo_DF[lon_cos_col] = np.cos(np.deg2rad(geo_DF[lon_col]))
    geo_DF[lat_cos_col] = np.cos(np.deg2rad(geo_DF[lat_col]))

    return geo_DF


def calc_cartesian(geo_DF):
    """Calculate Cartesian coordinates on unit sphere from sines and cosines"""

    geo_DF['x'] = geo_DF[lat_cos_col]*geo_DF[lon_cos_col]
    geo_DF['y'] = geo_DF[lat_cos_col]*geo_DF[lon_sin_col]
    geo_DF['z'] = geo_DF[lat_sin_col]

    return geo_DF

def calc_gca(point_DF, row1=0, row2=1):
    """Calculate the great circle angle (in radians) between two points

    Assumes precalculated sines and cosines (calc_trigs)"""

    # https://en.wikipedia.org/wiki/Great-circle_distance#From_chord_length

    dx = (point_DF[lat_cos_col].iloc[row2]*point_DF[lon_cos_col].iloc[row2] -
        point_DF[lat_cos_col].iloc[row1]*point_DF[lon_cos_col].iloc[row1])
    dy = (point_DF[lat_cos_col].iloc[row2]*point_DF[lon_sin_col].iloc[row2] -
        point_DF[lat_cos_col].iloc[row1]*point_DF[lon_sin_col].iloc[row1])
    dz = point_DF[lat_sin_col].iloc[row2] - point_DF[lat_sin_col].iloc[row1]

    gca = 2*np.arcsin(np.sqrt(dx**2 + dy**2 + dz**2)/2)

    return gca

def calc_gca_cart(point_DF, row1=0, row2=1):
    """Calculate the great circle angle (in radians) between two points

    Assumes precalculated Cartesian coordinates (calc_cartesian)"""

    dx = point_DF['x'].iloc[row2] - point_DF['x'].iloc[row1]
    dy = point_DF['y'].iloc[row2] - point_DF['y'].iloc[row1]
    dz = point_DF['z'].iloc[row2] - point_DF['z'].iloc[row1]

    return 2*np.arcsin(np.sqrt(dx**2 + dy**2 + dz**2)/2)


def dist_mat(point_DF, output=None, gca_fun=calc_gca_cart, rowinds=None):
    """Calculate (angular) distance matrix for all point combinations

    Returns an upper triangular matrix. If output is not None,
    print rowind every `output` rows."""

    n_points = point_DF.shape[0]

    dist_mat_l = np.zeros((n_points, n_points))

    if rowinds is None:
        rowinds = range(n_points-1)

    for rowind in rowinds:
        if output is not None:
            if rowind % output == 0:
                logging.info('dist_mat on rowind %d', rowind)
        for colind in range(rowind+1, n_points):
            dist_mat_l[rowind, colind] = \
              gca_fun(point_DF, row1=rowind, row2=colind)

    return dist_mat_l[rowinds,:]
