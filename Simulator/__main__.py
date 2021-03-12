from environment import Environment
from robot import Robot
from descartes import PolygonPatch

from matplotlib import pyplot as plt

rect_limits = [-2, 15]
resolution = 0.1

env = Environment(rect_limits, resolution)

fig, ax = plt.subplots()

for coord in env.coords:
    ax.plot(coord[0], coord[1])

x, y = zip(*env.sampled_points)
ax.scatter(x, y, c=env.uncertainity_distribution)

plt.show()
