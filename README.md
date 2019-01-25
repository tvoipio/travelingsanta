# travelingsanta

This is a solution to Reaktor's Traveling Santa challenge 2018, https://traveling-santa.reaktor.com/

The solution is based using Simulated Annealing in two concentric routes:
the inner loop takes list of gifts on a single route (fulfilling the
weight constraint) and finds the optimal solution (a Traveling Salesman problem);
the outer loop tries to optimize the distribution of gifts between routes.

The solution is known to be nonoptimal at least in two senses:
 - simulated annealing may not be suited to this sort of problem (there
     exist much faster, specific tools to the Capacitated Vehicle Routing
     Problem);
 - the initial state and the cooling schedules have not been optimized at all.

The initial state is found using spatial clustering (k-means on the Cartesian
    coordinates) with the assumption that the points within each cluster are
    reasonably close together and thus the Cartesian distance is a reasonable
    approximation of the actual (great circle) distance.

*How to execute:*
1. Ensure that you have the requirements installed:
  - Python 3.x (I used 3.6.6)
  - `pip install -r requirements.txt`
2. `python calculate_distance_matrix.py`
3. `python route_simanneal.py`

The Jupyter notebook `Visualise Nice Children.ipynb` was used for playing
around with the different components.

Original task:

Help Santa reduce his carbon footprint by optimizing his logistics!

At your disposal, you have Santa’s most highly guarded trade secret, the Nice List. The coveted List details which present shall be given to which child. Here’s the kicker: Santa’s sleigh can only carry 10 metric tons (10,000kg) at a time, and for each trip Santa makes, you'll need to tell him which items to pack.

You can download the list here. In the file, you'll find the wish of each child. Each row contains one wish. For privacy purposes, we've left out the names of the children: instead, their files include a numerical ID, their coordinates on Earth, and the weight of their present in grams. Your job is to find the most optimal routes to deliver all the presents. Santa starts at Korvatunturi, Finland (68.073611N 29.315278E) and, based on your list, flies directly from one coordinate to another until all presents are delivered, and then returns to Korvatunturi. The shorter the overall length of the trip, the less emissions there will be. For the purposes of this task, we assume Earth to be a sphere with radius of 6,378km.

To submit your solution, upload a .csv file below. You are welcome to upload multiple solutions to improve your score. Each row should contain a single gift run starting from Korvatunturi, listing all the children who Santa will visit on the run, their IDs separated with a semicolon. See example file here. You cannot exceed the capacity of the sleigh on a single run. Once you've sent over your solution, we will tally up the total distance covered for all rows. Whoever delivers all the presents while covering the least distance, wins!

(as of 2018-12-31, the best solution on the leaderboard is around 7.618e9 metres)
