from environment import Environment
from robot import Robot
from optimalPlanner import RRTPlanner
from descartes import PolygonPatch

from shapely.geometry import Polygon
from matplotlib import pyplot as plt

rect_limits = [-2, 15]
resolution = 0.1

start = (7.5, 2.5)
end = (0, 12.5)

obstacles = [Polygon(((6,-2), (6,6), (5,6), (5,-2))), Polygon(((6,11), (6,15), (5,15), (5,11)))]

env = Environment(rect_limits, resolution)
rrt = RRTPlanner(start, end, obstacles, rect_limits)
path = rrt.planning()

fig, ax = plt.subplots()

for coord in env.coords:
    ax.plot(coord[0], coord[1])

for obstacle in obstacles:
    patch = PolygonPatch(obstacle)
    ax.add_patch(patch)

x, y = zip(*env.sampled_points)
ax.scatter(x, y, c=env.uncertainity_distribution, cmap='winter_r')

ax.plot([x for (x, y) in path], [y for (x, y) in path], '.-r')

fig1 = plt.figure()
ax1 = fig1.add_subplot(111, projection='3d')

ax1.scatter(x,y,env.uncertainity_distribution)

plt.show()
