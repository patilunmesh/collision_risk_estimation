Procedure to record Carla bag

1. Roscore

2. open 5 terminals each with following two source commands

	1) source ~/carla-ros-bridge/catkin_ws/devel/setup.bash

	2) export CARLA_ROOT=~/opt/carla-simulator && export SCENARIO_RUNNER_ROOT=~/scenario_runner-0.9.8   && export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla/dist/carla-0.9.8-py2.7-linux-x86_64.egg && export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla/agents && export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla && export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI


3. Commands for each of the 5 terminals:

	T1)     cd opt/carla-simulator/

		./CarlaUE4.sh 

	T2)      cd ~/scenario_runner

		python manual_control.py

	T3)	cd ~/scenario_runner

		python scenario_runner.py --scenario NoSignalJunctionCrossing --reloadWorld

	T4)	rosparam set /use_sim_time false

		cd ~/carla-ros-bridge/catkin_ws

		roslaunch carla_ros_bridge carla_ros_bridge.launch 

	T5)	rosparam set /use_sim_time false

		rosbag record -a

\\ the bag will be saved in home with timestamp

Procedure to get collision risk:

1. Open Rviz with carla config
2. rosbag play -----
3. rosparam set /use_sim_time true
4. python carla_objects.py
5. python orsp_carla.py
6. python carla_risk_plotter.py

Note: Please check argparse in every code to understand the required arguments















	









