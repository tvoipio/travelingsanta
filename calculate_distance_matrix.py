import geo_process as gp
import preprocess_data as pp
import pickle
import logging
import numpy as np

from multiprocessing import Pool


def calc_dist_helper(data):
    nice_data = data['nice_data']
    minrow = data['minrow']
    maxrow = data['maxrow']

    logging.info('Calculating distance matrix rows %d to %d',
                 minrow, maxrow)

    return gp.dist_mat(nice_data, output=None, gca_fun=gp.calc_gca,
                       rowinds=range(minrow, maxrow+1))


if __name__ == '__main__':

    out_file = 'dist_mat.pkl'
    out_interval = 50

    nice_file = 'nice_processed.pkl'

    logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s',
                        level=logging.INFO,
                        datefmt='%Y-%m-%d %H:%M:%S')

    logging.info('Loading data')
    nice_raw = pp.read_input()

    # For debugging purposes.
    # nice_raw = nice_raw.head(432)

    logging.info('Calculating sines and cosines')
    nice_data = gp.calc_trigs(nice_raw, lat_col='lat_deg', lon_col='lon_deg')

    logging.info('Calculating Cartesian coordinates')
    nice_data = gp.calc_cartesian(nice_data)

    logging.info('Writing processed nice list to %s', nice_file)
    with open(nice_file, 'wb') as f:
        pickle.dump(nice_data, f)

    logging.info('Calculating distance matrix')
    # dist_mat = gp.dist_mat(nice_data, output=out_interval, gca_fun=gp.calc_gca)

    with Pool() as pool:
        breaks = list(range(0, nice_data.shape[0]+1, 50))
        if breaks[-1] < nice_data.shape[0]:
            breaks.append(nice_data.shape[0])

        dist_iter = ({'nice_data': nice_data, 'minrow': breaks[ind],
                      'maxrow': breaks[ind+1]-1}
                     for ind in range(len(breaks)-1))

        dist_mat_slices = pool.map(calc_dist_helper, dist_iter)

        dist_mat = np.vstack(dist_mat_slices)

    logging.info('Writing distance matrix to %s', out_file)
    with open(out_file, 'wb') as f:
        pickle.dump(dist_mat, f)

    logging.info('Done')
