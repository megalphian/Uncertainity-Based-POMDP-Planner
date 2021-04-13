# Motion Planning under uncertainity in POMDP belief space
A local path planner to minimize the state-dependent uncertainity experienced by the robot while traversing a trajectory. The planner uses an Extended Kalman Filter to estimate the covariance along an uncertainity-unaware path, and an iterative LQG is implemented to minimize the uncertainity of this path. 

This is an implementation of the obstacle-free point robot case explored by [van den Berg et. al.](https://journals.sagepub.com/doi/10.1177/0278364912456319).

## Running the Simulator

The simulator is built in **Python 3.8**, but it should run in any version of Python 3. The simulator requires 3 dependencies:
```
numpy
scipy
matplotlib
```

To install these dependencies, you can use the *requirements.txt* file provided.
```
pip install -r requirements.txt
```

The simulator requires a config to configure the start, goal, initial covariance and cost paramemters. This takes a while to configure, especially the estimate of the inital covariance *(See note at the end)*. I have provided a few test paths that work consistently with the simulator as config files.

To run the simulator:
```
python Simulator --config <config-file-name>.json
```

For example:
```
python Simulator --config config1.json
```

## Notes:
1) When using the provided config files, please change the current directory to the project folder. For Linux and Mac
```
cd kalman-based-pomdp-optimizer
```
2) When testing the Simulator, I have noticed that the path sometimes does not optimize to a lower cost path and the original path is retained. I wasn't able to debug this issue, but re-running the simulator one or two times seems to fix the issue. 

## On choosing an initial covariance
A big factor I noticed while testing this simulator is the value of the initial covariance. In some cases, a small covariance leads to the path not optimizing at all, and a large covariance causes the path to move really far from the light.

The covariance choices that worked were chosen to reflect where the starting point was with respect to the light region. If the path starts closer to the light, a smaller covariance value (0.5 or 1) seems to work fine. If the starting point is farther from the light, then a larger value (2 - 3) is used.