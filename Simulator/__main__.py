from environment import Environment
from robot import Robot
from optimalPlanner import RRTPlanner
from descartes import PolygonPatch

from matplotlib import pyplot as plt

rect_limits = [-2, 15]
resolution = 0.1

start = (7.5, 2.5)
end = (0, 12.5)

env = Environment(rect_limits, resolution)
rrt = RRTPlanner(start, end, None, rect_limits)
path = rrt.planning()

fig, ax = plt.subplots()

for coord in env.coords:
    ax.plot(coord[0], coord[1])

x, y = zip(*env.sampled_points)
ax.scatter(x, y, c=env.uncertainity_distribution)

ax.plot([x for (x, y) in path], [y for (x, y) in path], '-r')

plt.show()
