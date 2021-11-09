Required libraries and packages:

1. NUMBA 
	sudo apt-get install python-numba
	After installation, 
	Run numba_test.py in scripts/support directory
	(The script should run without any error and should show elapsed time)

2. Scenario-runner (0.9.8)
	https://github.com/carla-simulator/scenario_runner

3. CARLA Simulator (0.9.8)

4. ROS (melodic)

5. Python (3.6.9)

6. grid_map package: sudo apt-get install ros-$ROS_DISTRO-grid-map
	or visit: https://github.com/ANYbotics/grid_map

7. others: argparse, CSV, Matplotlib

Note: 2, 3 are needed only for creating new scenarios. One may use recorded bags.
	  6 is necessary for better visualisation of grid with color map. 
	  (GridMap causes lag in the simulation, so for good performance use OccupancyGrid message)



Steps to use the repository:

1. Clone the repository

2. Download bagfiles: https://drive.google.com/drive/folders/14noRPVw5wNrb6D1mwjnRfd1XXu-LsA7W?usp=sharing

3. Install prerequisites

4. roscore

5. robag play

6. python rosnode.py

7. open rviz and open config file

