from simanneal import Annealer
import random

class SingleSantaProblem(Annealer):
    """Solve the traveling salesman problem with simulated annealing

    The state is a list of indices corresponding rows & columns of the distance
    matrix. Santa's workshop (index 0) is automatically added as the first and
    and last stop."""

    def __init__(self, state, distance_matrix, random_seed=None, noupdate=False):
        self.distance_matrix = distance_matrix
        self.noupdate = noupdate
        random.seed(random_seed)
        super(SingleSantaProblem, self).__init__(state)

    def move(self):
        """Swap two locations"""

        ind1, ind2 = random.sample(range(len(self.state)), 2)

        self.state[ind1], self.state[ind2] = self.state[ind2], self.state[ind1]

    def energy(self):
        """Calculate route length"""

        route_len = 0

        # Santa's base to first visit
        route_len += self.distance_matrix[0, self.state[0]]

        for ind in range(len(self.state)-1):
            loc1, loc2 = self.state[ind], self.state[ind+1]
            # The min-max trick ensures that we query the upper triangle
            route_len += self.distance_matrix[min(loc1, loc2), max(loc1, loc2)]

        # From last visit back to Santa's base
        route_len += self.distance_matrix[0, self.state[-1]]

        return route_len

    def update(self, *args, **kwargs):
        """Use default update or do nothing"""

        if not self.noupdate:
            self.default_update(*args, **kwargs)
