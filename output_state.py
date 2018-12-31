import pickle


def load_final_state(fname):
    """Load final state from file"""

    with open(fname, 'rb') as f:
        final_dict = pickle.load(f)

    return final_dict


def output_state(final_state, nice_data, output_fname):
    """Output final route information to file"""

    with open(output_fname, 'w') as f:
        for route in final_state['routes']:
            child_IDs_str = [str(nice_data.ID.iloc[place_ind])
                             for place_ind in route]
            f.write(';'.join(child_IDs_str) + '\n')


if __name__ == "__main__":
    final_state_name = 'tsp_final_clust_comb.pkl'
    nice_name = 'nice_processed.pkl'
    output_fname = 'route_output_clust_comb.txt'

    with open(nice_name, 'rb') as f:
        nice_data = pickle.load(f)

    final_dict = load_final_state(final_state_name)

    output_state(final_dict['final_state'], nice_data, output_fname)
