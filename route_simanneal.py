# Optimize Santa's distribution routes using simulated annealing

import pickle
import numpy as np
import pandas as pd
import logging
import itertools
import random

import geo_process as gp

import sklearn.cluster as clust

from single_route_anneal import SingleSantaProblem

from simanneal import Annealer

nice_file = 'nice_processed.pkl'
dist_file = 'dist_mat.pkl'

# Total mass carried with each route limited to 10,000 kg (1e7 grams)
weight_limit = 1e7

# Number of trips - based on data, minimum amount is 531 assuming
# that mass could be distributed arbitrarily
num_trips = 540
min_routes = 531

# Paranoia mode (setting to True enables extra checks)
paranoid = False

# State: a dictionary with 'lengths' = np.array of route lengths
# and 'routes', list of lists: inner list represents a route,
# each element is a row number in the nice_data DF; index of the
# outer list is the route number


def load_data(nice=nice_file, dist=dist_file):
    """Load data for route optimization"""

    with open(nice, 'rb') as f:
        nice_data = pickle.load(f)

    with open(dist, 'rb') as f:
        dist_data = pickle.load(f)

    return {'nice_list': nice_data, 'dist_mat': dist_data}


def validate_state(state, nice_data,
                   check_duplicates=False, check_orphans=False):
    """Validate a state"""

    # Validate weights
    # route_weights = [nice_data['wt'].iloc[child_inds].values.sum()
    #                 for child_inds in state['routes']]

    route_weights = compute_route_weights(state, nice_data)

    route_weights_sorted = route_weights[:]
    route_weights_sorted.sort(reverse=True)

    logging.debug('Largest 10 route weight sums: %s', ';'
                  .join([str(num) for num in route_weights_sorted[:10]]))

    wt_within_limit = all([(route_weight <= weight_limit)
                           for route_weight in route_weights])

    duplicates_found = False
    orphans_found = False

    if check_duplicates or check_orphans:
        inds_in_routes = list(itertools.chain(*state['routes']))

    # Check for duplicates
    if check_duplicates:
        num_inds = len(inds_in_routes)
        num_unique_inds = len(set(inds_in_routes))

        duplicates_found = (num_unique_inds != num_inds)

        if duplicates_found:
            logging.warn('Duplicates found!')

    # Check for orphans (assuming that first row of nice_data is Santa)
    if check_orphans:
        orphan_inds = set(range(1, nice_data.shape[0])) - set(inds_in_routes)

        orphans_found = (len(orphan_inds) > 0)

        if orphans_found:
            logging.warn('Orphans found!')

    return (wt_within_limit and not (duplicates_found or orphans_found))


def optimize_route(state, nice_data, dist_matrix, route_index,
                   quiet=True):
    """Optimize a single route to have the shortest distance

    Returns the new length of the route. Routes are cycles starting
    from row 0 (Santa's base of operations at Korvatunturi) and also
    ending there."""

    route_order = state['routes'][route_index]

    if not quiet:
        logging.info('Optimization for subroute %d, %d gifts',
                     route_index, len(route_order))

    if len(route_order) > 1:
        ssp = SingleSantaProblem(route_order, dist_matrix)

        # Empirically determined parameters
        ssp.Tmax = 50
        ssp.Tmin = 0.1
        ssp.steps = 20000

        if quiet:
            ssp.updates = 0

        route, route_len = ssp.anneal()

        if not quiet:
            logging.info('Optimized initial subroute %d, total distance %f',
                         route_index, route_len)

    elif len(route_order) == 1:
        # Only one stop
        route = route_order
        # Route length is twice the Korvatunturi-child distance
        route_len = 2*dist_matrix[0, route_order[0]]
    else:
        # Empty (sub)route
        route = []
        route_len = 0

    state['routes'][route_index] = route
    state['lengths'][route_index] = route_len

    return state


def optimize_routes(state, nice_data, dist_matrix, route_inds=None,
                    *args, **kwargs):
    """Optimize each (sub)route in state

    For each (sub)route in state, find the ordering which results in the
    shortest route length. If route_inds is specified, only optimize
    the routes with that index."""

    if route_inds is None:
        route_inds = list(range(len(state['routes'])))

    maxprint = 5
    route_str = ', '.join([str(num) for num in route_inds[:maxprint]])

    if len(route_inds) > maxprint:
        route_str += ", and {:d} others".format(len(route_inds) - maxprint)

    logging.debug('Optimizing routes %s', route_str)

    for ind in route_inds:
        state = optimize_route(state, nice_data, dist_matrix, ind,
                               *args, **kwargs)

    return state


def populate_state(state, nice_data, dist_matrix, random_seed=None,
                   *args, **kwargs):
    """Populate the initial state.

    Wrapper function for e.g. populate_state_clusters or
    populate_state_random"""
    return populate_state_clusters(state, nice_data, dist_matrix,
                                   random_seed=random_seed, *args, **kwargs)


def populate_state_clusters(state, nice_data, dist_matrix, random_seed=None,
                            split_strategy='knn'):
    """Populate a previously empty state

    Populates state by clustering. Number of clusters is the number of
    routes in `state`; if the points in a cluster exceed the weight limit,
    the cluster is split using strategy defined by `split_strategy`"""

    num_clust = len(state['routes'])

    clust_alg = clust.KMeans(n_clusters=num_clust, copy_x=True,
                             random_state=random_seed)

    logging.info('Clustering %d points into %d clusters using KNN',
                 nice_data.shape[0], num_clust)

    # Korvatunturi is included in one of the clusters, but this should
    # not matter that much.
    # Clustering is done on Cartesian coordinates with the assumption that
    # each cluster is confined to a relatively small geographic area,
    # and thus the Cartesian distance (i.e. chord) is a reasonable
    # approximation for the great circle distance
    clust_alg.fit(nice_data[['x', 'y', 'z']])
    clusters = clust_alg.predict(nice_data[['x', 'y', 'z']])
    cluster_labels = list(set(clusters))
    cluster_labels.sort()

    cluster_centers = clust_alg.cluster_centers_

    logging.info('Assigning present to routes according to clusters')

    santa_base_cluster = clusters[0]

    # Iterate over clusters and assign all points in each cluster to a route
    for label_num in range(len(cluster_labels)):
        cluster_matches = np.where(clusters == cluster_labels[label_num])[0]
        cluster_matches = list(cluster_matches)

        if cluster_labels[label_num] == santa_base_cluster:
            logging.info('Korvatunturi in cluster %d, removing from route',
                         cluster_labels[label_num])
            cluster_matches.remove(0)
            logging.info('Route %d now visits %d places',
                         cluster_labels[label_num], len(cluster_matches))

        state['routes'][label_num] = cluster_matches

    # Split routes which are over the weight limit
    state, cluster_centers = split_routes(state, nice_data, cluster_centers)

    # All gifts should now be distributed along routes, let's recheck
    if not validate_state(state, nice_data, check_duplicates=True,
                          check_orphans=True):
        raise RuntimeError('State not OK after populating all gifts')

    # Combine routes where possible
    state = combine_routes(state, nice_data, cluster_centers,
                           n_route_cutoff=min_routes)

    # All gifts should still be distributed along routes, let's recheck
    if not validate_state(state, nice_data, check_duplicates=True,
                          check_orphans=True):
        raise RuntimeError('State not OK after combining routes')

    logging.info('Initial populating done, optimizing initial route order')

    # Optimize route lengths
    state = optimize_routes(state, nice_data, dist_matrix, quiet=False)

    # Finally
    return state


def split_routes(state, nice_data, cluster_centers):
    """Split routes over the weight limit"""

    logging.info("Splitting overweight routes")

    route_weights = compute_route_weights(state, nice_data)
    num_iter = 0

    cluster_centers_new = cluster_centers.copy()

    # Iterate while any route is overweight
    while any([route_weight > weight_limit for route_weight in route_weights]):
        # Pick the route with the most weight
        split_index = np.array(route_weights).argmax()

        cluster_nice = nice_data.iloc[state['routes'][split_index]]

        # Pick the two heaviest presents as initial new cluster centers
        two_heaviest = (
            cluster_nice.sort_values(by='wt', ascending=False).head(2)
            )
        centers = two_heaviest[['x', 'y', 'z']].values

        knn_clust = clust.KMeans(n_clusters=2, init=centers, n_init=1,
                                 copy_x=True)
        knn_clust.fit(cluster_nice[['x', 'y', 'z']])
        new_labels = knn_clust.predict(cluster_nice[['x', 'y', 'z']])

        new_route_index = len(state['routes'])
        state['lengths'].append(0)
        state['routes'].append([])

        if len(state['lengths']) != len(state['routes']):
            logging.warn("Length of state['lengths'] (%d) does not match " +
                         "length of state['routes']! (%d)",
                         len(state['lengths']), len(state['routes']))

        # We trust that KMeans returns integer indices 0, ..., n-1
        route1_indices = \
            cluster_nice.iloc[np.where(new_labels == 0)[0]].index.values
        route2_indices = \
            cluster_nice.iloc[np.where(new_labels == 1)[0]].index.values

        state['routes'][split_index] = list(route1_indices)
        state['routes'][new_route_index] = list(route2_indices)

        # Replace old cluster center information with new one
        # (adding the new cluster center)

        cluster_centers_new = \
            np.vstack((cluster_centers_new, knn_clust.cluster_centers_[1, :]))
        cluster_centers_new[split_index, :] = knn_clust.cluster_centers_[0, :]

        route_weights = compute_route_weights(state, nice_data)
        num_iter += 1

        if num_iter % 10 == 0:
            logging.info('Splitting iteration %d done, now at %d routes',
                         num_iter, new_route_index+1)

    return state, cluster_centers_new


def compute_route_weights(state, nice_data):
    """Compute the weight carried on a route"""

    return [nice_data['wt'].iloc[child_inds].values.sum()
            for child_inds in state['routes']]


def combine_routes(state, nice_data, cluster_centers,
                   cutoff_dist=1500, n_iter_max=2000,
                   n_route_cutoff=0):
    """Combine routes where possible"""

    # logging.info("Combining routes not implemented yet, sorry...")

    # Take a (possibly unordered) route at random; check if there
    # are eligible routes (i.e. where all the stuff in this route fits)
    # whose centers are within cutoff distance (in km) from this route's
    # center; pick the closest one; combine these two routes

    n_iter = 0

    cluster_centers_new = cluster_centers.copy()

    while len(state['routes']) > n_route_cutoff and n_iter < n_iter_max:
        n_iter += 1
        candidate1 = random.sample(range(len(state['routes'])), 1)[0]

        logging.debug('Iteration %d, trying to combine route %d with another',
                      n_iter, candidate1)

        # Construct DataFrame of center locations
        center_DF = pd.DataFrame(cluster_centers_new,
                                 columns=['x', 'y', 'z'])

        # Find routes which, based on total weight, could be combined with this
        route_weights = np.array(compute_route_weights(state, nice_data))
        logging.debug('Current route weight %d, eligibility limit %f',
                      route_weights[candidate1],
                      weight_limit - route_weights[candidate1])
        eligible_routes = list(np.where(
            route_weights < (weight_limit - route_weights[candidate1]))[0])

        # Remove self
        if candidate1 in eligible_routes:
            eligible_routes.remove(candidate1)

        # Calculate distances
        center_distances = np.array([
            gp.calc_gca_cart(center_DF, row1=candidate1, row2=rowind)
            for rowind in eligible_routes])*gp.r_earth

        # Determine distances within cutoff
        eligible_center_distances = (center_distances <= cutoff_dist)

        # Set the distance of noneligible routes to "high enough"
        center_distances[np.logical_not(eligible_center_distances)] = \
            np.Inf

        if eligible_center_distances.any():
            candidate2_ind = center_distances.argmin()
            candidate2_dist = center_distances[candidate2_ind]
            candidate2 = eligible_routes[candidate2_ind]
            new_route = \
                state['routes'][candidate1] + state['routes'][candidate2]
            state['routes'][candidate1] = new_route
            state['routes'].pop(candidate2)
            state['lengths'].pop(candidate2)

            logging.debug('Combining route weights %d, %d, total %d',
                          route_weights[candidate1],
                          route_weights[candidate2],
                          route_weights[candidate1]+route_weights[candidate2])

            points_in_route = nice_data[['x', 'y', 'z']].iloc[new_route].values
            new_center = points_in_route.mean(axis=0)

            cluster_centers_new[candidate1, :] = new_center
            new_indices = list(range(cluster_centers_new.shape[0]))
            new_indices.pop(candidate2)
            cluster_centers_new = cluster_centers_new[new_indices, :]

            if cluster_centers_new.shape[0] != len(state['routes']):
                raise RuntimeError(('Cluster center number ({:d}) ' +
                                    'and route number ({:d}) mismatch')
                                   .format(
                                        cluster_centers_new.shape[0],
                                        len(state['routes'])
                                    ))

            logging.info('Merging route %d to %d ' +
                         '(changes route numbering above %d), ' +
                         'center-to-center distance was %.1f, new route now ' +
                         '%d stops, %d routes remaining',
                         candidate2, candidate1, candidate2, candidate2_dist,
                         len(new_route), len(state['routes']))
        else:
            logging.debug('Could not merge route %d, no eligible routes found',
                          candidate1)

    return state


def populate_state_random(state, nice_data, dist_matrix, random_seed=None):
    """Populate a previously empty state

    Distributes gifts randomly to routes. Ensures that the weight constraint
    is fulfilled, all gifts are distributed, and each subroute is optimized"""

    # Get weights in descending order; gifts are distributed at random
    # starting with the heaviest ones
    weights_sorted = nice_data['wt'].iloc[1:].sort_values(ascending=False)

    num_trips = len(state['routes'])

    logging.info('Populating state randomly to %d routes', num_trips)
    random.seed(random_seed)

    num_iterations = [0]*len(weights_sorted)

    for ind in range(len(weights_sorted)):

        addition_ok = False
        num_iter_max = 100

        while not addition_ok:
            candidate_route = random.randrange(num_trips)

            state['routes'][candidate_route].append(
                weights_sorted.index.values[ind])

            logging.debug('Adding gift %d (weight %d) to route %d',
                          weights_sorted.index.values[ind],
                          weights_sorted.values[ind],
                          candidate_route)

            addition_ok = validate_state(state, nice_data)

            num_iterations[ind] += 1

            # If the addition was not OK, remove the gift from the route

            if not addition_ok:
                state['routes'][candidate_route].pop()

                if num_iterations[ind] > num_iter_max:
                    raise RuntimeError(
                        'Maximum number of iterations exceeded' +
                        ' when populating state, ind = ' + str(ind))

        if ind % 25 == 0:
            logging.info('...item %d placed, cumulative mean ' +
                         'number of iterations %.2f',
                         ind, sum(num_iterations[:ind+1])/float(ind+1))

        if paranoid:
            validate_state(state, nice_data, check_duplicates=True)

    # All gifts should now be distributed along routes, let's recheck
    if not validate_state(state, nice_data, check_duplicates=True,
                          check_orphans=True):
        raise RuntimeError('State not OK after populating all gifts')

    logging.info('Initial populating done, optimizing initial route order')

    # Optimize route lengths
    state = optimize_routes(state, nice_data, dist_matrix, quiet=False)

    return state


class TravellingSantaProblem(Annealer):
    """Solve the Travelling Santa problem with simulated annealing"""

    def __init__(self, state, nice_list, distance_matrix,
                 optim_single_fun, validate_fun):
        self.distance_matrix = distance_matrix
        self.nice_list = nice_list
        self.optim_single_fun = optim_single_fun
        self.validate_fun = validate_fun
        super(TravellingSantaProblem, self).__init__(state)

    def move(self):
        """Move a gift from a route to another"""

        # TODO implement:
        # 1. Pick a route and a gift (/recipient) to move
        # 2. Select candidate route
        # 3. Verify that moving the gift would not violate load limit
        # 4. If load limit is violated, go back to 1
        # 5. When load limit is satisfied, find optimal routings within
        #    affected routes
        # 6. Store new routes and their lengths in the state

        move_ok = False

        while not move_ok:
            len_route1 = 0

            while len_route1 < 1:
                route1, route2 = \
                    random.sample(range(len(self.state['routes'])), 2)
                len_route1 = len(self.state['routes'][route1])

            gift_ind = random.randrange(len_route1)

            nice_ind = self.state['routes'][route1][gift_ind]

            # Verify that the gift can be added to the candidate route
            self.state['routes'][route2].append(nice_ind)
            validation_pass = self.validate_fun(self.state, self.nice_list)

            if not validation_pass:
                # Restore route 2 back to previous state
                logging.debug('Validation did not pass when moving ' +
                              '%d from %d to %d',
                              nice_ind, route1, route2)

                self.state['routes'][route2].pop()
            else:
                # Move the gift to route 2 (remove from route 1)
                self.state['routes'][route1].pop(gift_ind)
                move_ok = True

        # Optimize the changed routes
        self.state = self.optim_single_fun(self.state, self.nice_list,
                                           self.distance_matrix, route1)
        self.state = self.optim_single_fun(self.state, self.nice_list,
                                           self.distance_matrix, route2)

    def energy(self):
        """Calculate total length of all routes"""

        return sum(self.state['lengths'])

    def copy_state(self, state):
        """Return an exact copy of the provided state"""

        new_state = {'lengths': state['lengths'].copy(),
                     'routes': [route[:] for route in state['routes']]}

        return new_state


if __name__ == '__main__':

    loglevel = logging.INFO

    logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s',
                        level=loglevel,
                        datefmt='%Y-%m-%d %H:%M:%S')

    initial_state_file = 'tsp_initial_clust.pkl'
    final_state_file = 'tsp_final_clust.pkl'
    data = load_data()

    nice_data = data['nice_list']
    dist_mat = data['dist_mat']

    try:
        with open(initial_state_file, 'rb') as f:
            logging.info('Loading initial state from %s', initial_state_file)
            initial_state = pickle.load(f)
    except OSError:
        logging.info('Starting to populate initial state')

        # One would think that [[]]*n would be a nice way to create
        # n empty lists, but one would be wrong...
        # (n references to the same list)
        empty_state = {'lengths': [0]*num_trips,
                       'routes': [list() for ind in range(num_trips)]}

        initial_state = populate_state(empty_state, nice_data, dist_mat)

        logging.info('Saving initial state to file')
        with open(initial_state_file, 'wb') as f:
            pickle.dump(initial_state, f)

    logging.info('Starting to optimize the problem')
    tsp = TravellingSantaProblem(initial_state, nice_data, dist_mat,
                                 optimize_route, validate_state)

    # The temperature is in the units of energy, and the acceptance
    # probability is related to the *difference* between current and
    # previous state; hence the Tmax, Tmin may be of the same order as in
    # the single-route optimization since each step shortens one or two routes
    tsp.Tmax = 50
    tsp.Tmin = 0.1
    tsp.steps = 1000
    tsp.updates = 100
    # tsp.steps = 2
    # tsp.updates = 0

    final_state, final_dist = tsp.anneal()

    final_dict = {'final_state': final_state, 'final_dist': final_dist,
                  'initial_state': initial_state}

    r_earth = 6378.

    logging.info('Finished annealing, final distance %f km',
                 final_dist*r_earth)

    logging.info('Saving final state to %s', final_state_file)
    with open(final_state_file, 'wb') as f:
        pickle.dump(final_dict, f)
