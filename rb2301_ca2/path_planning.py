import argparse
import configparser
import json
import math
import os
import time

import numpy as np
import heapq
import rclpy
from rclpy.node import Node
from rclpy.logging import set_logger_level, LoggingSeverity

from rclpy.qos import (
    ReliabilityPolicy,
    QoSProfile,
)
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from PIL import Image
from geometry_msgs.msg import Twist

from copy import deepcopy

np.set_printoptions(
    2, suppress=True, threshold=np.inf
)  # Print numpy arrays to specified d.p., suppress scientific notation (e.g. 1e-5), and do not truncate

set_logger_level("waypoint", level=LoggingSeverity.DEBUG) # Configure to either LoggingSeverity.INFO or LoggingSeverity.DEBUG

occupancy_grid_resolution = 0.2 # Sim grid resolution, in metres per cell (the real maze uses the same 0.2m cells, see ca2_irl_layout.json)
max_translate_velocity = 1.4 # Overwritten in main() based on sim vs real-life; 0.4m/s cap for real life, please keep that in place

_PACKAGE_DIR = os.path.dirname(os.path.realpath(__file__))


# --- Coordinate conversion --------------------------------------------------
# A grid index (i, j) represents a CELL, not a point. That cell's world
# coordinate is its CENTER, e.g. cell [0, 0] is centred half a resolution-step
# away from the grid's origin corner, not exactly on it. This matches how the
# Gazebo world and the real maze are physically laid out (goal tape/markers
# sit in the middle of a cell, not on its boundary line).
def grid_to_world(i:int, j:int, origin:tuple, resolution:float=occupancy_grid_resolution) -> tuple:
    '''Convert grid index (i, j) to the world (x, y) coordinate of that cell's centre.'''
    return (origin[0] + (i + 0.5) * resolution, origin[1] + (j + 0.5) * resolution)

def world_to_grid(x:float, y:float, origin:tuple, resolution:float=occupancy_grid_resolution) -> tuple:
    '''Convert a world (x, y) coordinate to the grid index (i, j) of the cell containing it.'''
    return (int(np.floor((x - origin[0]) / resolution)), int(np.floor((y - origin[1]) / resolution)))


# --- Sim / real-life profiles -----------------------------------------------
# Run without arguments for the simulation. For the real maze pass
# --run test1 | test2 | full  (see README). Everything that differs between
# sim and real -- which map to load, the cell resolution, where the grid's
# [0, 0] corner sits in the frame your pose is reported in, the goal points,
# the speed cap -- lives in one of these profiles.
sim_config = {
    "map_file": "ca2_sim_map.npy",
    "origin": (-1.0, -5.0),
    "resolution": occupancy_grid_resolution,
    "goal_list": [(3.5, -3.5), (3.3, 0.3), (2.5, -3.5), (-0.3, -3.7)],
    "max_translate_velocity": 1.4,
}

IRL_RUNS = ("test1", "test2", "full")


# --- Real maze <-> Optitrack frame -------------------------------------------
# The real maze is a fixed layout (ca2_irl_map.npy + ca2_irl_layout.json, in
# "maze-local" metres: (0, 0) is the outer corner of array cell [0, 0], +x runs
# along array axis 0, +y along axis 1 -- exactly like the sim grid). Optitrack
# reports poses in ITS OWN frame, wherever the maze happens to be taped down.
# optitrack_variables.config holds the 3 numbers that relate the two:
#   origin_x, origin_y : Optitrack coordinates of the maze's (0, 0) corner
#   rotation_deg       : angle of the maze's +x axis in the Optitrack frame
#                        (counter-clockwise positive, degrees)
# The pose from Optitrack is converted into the maze frame before your code
# sees it, so self.origin is (0, 0) and everything else (world_to_grid,
# goal_list, ...) works exactly like in simulation.
def wrap_deg(angle:float) -> float:
    '''Wrap an angle in degrees into [-180, 180).'''
    return (angle + 180.0) % 360.0 - 180.0

def optitrack_to_maze(x:float, y:float, heading_deg:float, frame:dict) -> tuple:
    '''Optitrack-frame pose (x, y, heading in degrees) -> maze-frame pose.'''
    dx, dy = x - frame["origin_x"], y - frame["origin_y"]
    r = math.radians(frame["rotation_deg"])
    c, s = math.cos(r), math.sin(r)
    return (c * dx + s * dy, -s * dx + c * dy,
            wrap_deg(heading_deg - frame["rotation_deg"] - frame.get("heading_offset_deg", 0.0))) # offset: the Motive rigid body's "forward" vs the robot's front

def maze_to_optitrack(mx:float, my:float, heading_deg:float, frame:dict) -> tuple:
    '''Inverse of optitrack_to_maze (handy for working out where to place the robot).'''
    r = math.radians(frame["rotation_deg"])
    c, s = math.cos(r), math.sin(r)
    return (frame["origin_x"] + c * mx - s * my, frame["origin_y"] + s * mx + c * my,
            wrap_deg(heading_deg + frame["rotation_deg"] + frame.get("heading_offset_deg", 0.0)))

def robot_frame(config:dict, robot_number:int) -> dict:
    '''The frame for one robot: the shared origin/rotation plus that robot's heading offset (per-robot value from the config if listed).'''
    frame = dict(config["frame"])
    frame["heading_offset_deg"] = config["heading_offsets"].get(robot_number, frame["heading_offset_deg"])
    return frame

def load_irl_config(run:str) -> dict:
    '''Build the real-maze profile for run "test1", "test2" or "full".'''
    if run not in IRL_RUNS:
        raise ValueError(f"--run must be one of {IRL_RUNS}, got {run!r}")
    with open(os.path.join(_PACKAGE_DIR, "ca2_irl_layout.json")) as f:
        layout = json.load(f)
    parser = configparser.ConfigParser()
    config_path = os.path.join(_PACKAGE_DIR, "optitrack_variables.config")
    if not parser.read(config_path):
        raise FileNotFoundError(f"Could not read {config_path}")
    frame = {
        "origin_x": parser.getfloat("frame", "origin_x"),
        "origin_y": parser.getfloat("frame", "origin_y"),
        "rotation_deg": parser.getfloat("frame", "rotation_deg"),
    }
    # Rigid bodies are created in Motive with the robot facing the maze's +y direction, so the rigid body's zero heading IS maze +y
    # (maze heading 90 deg): heading_maze = reported + 90 = reported - rotation - offset  =>  offset = -(rotation + 90). "auto" means that.
    raw_offset = parser.get("frame", "heading_offset_deg", fallback="auto").strip().lower()
    frame["heading_offset_deg"] = wrap_deg(-(frame["rotation_deg"] + 90.0)) if raw_offset == "auto" else float(raw_offset)
    heading_offsets = {int(k): float(v) for k, v in parser.items("heading_offsets")} if parser.has_section("heading_offsets") else {}
    return {
        "map_file": "ca2_irl_map.npy",
        "origin": (0.0, 0.0), # pose is converted into the maze frame, so the grid corner is (0, 0)
        "resolution": float(layout["resolution"]),
        "goal_list": [tuple(g) for g in layout["runs"][run]["goals"]],
        "start": tuple(layout["runs"][run]["start"]),
        "frame": frame,
        "robot_number": parser.getint("robot", "number"),
        "heading_offsets": heading_offsets, # per-robot override of frame["heading_offset_deg"]
        "max_translate_velocity": 0.4, # Please keep this in place
    }


class WaypointNode(Node):
    '''Node to calculate path and move robot towards given goal_coordinates, using pose info from either gazebo odometer or optitrack'''
    def __init__(self, map_array:np.array, goal_list:list, is_simulation:bool=True, origin:tuple=(0.0, 0.0), resolution:float=occupancy_grid_resolution,
                 frame:dict=None, robot_number:int=3, expected_start:tuple=None, calibrate:bool=False):
        super().__init__('waypoint')
        self.get_logger().info("Starting WaypointNode")

        self.is_simulation = is_simulation

        # Subscribe to the dynamic_pose topic from Gazebo that publishes ground-truth pose data
        if self.is_simulation:
            self.subscription = self.create_subscription(Odometry, 'odom', self.odometer_callback, 2)
        else:
            qos_profile = QoSProfile(depth=2, reliability=ReliabilityPolicy.BEST_EFFORT)

            self.map_sub = self.create_subscription(
                PoseStamped,
                f'/vrpn_mocap/bingda_{robot_number:03d}/pose',
                self.optitrack_callback,
                qos_profile
                )

        self.publisher_ = self.create_publisher(Twist, 'cmd_vel', 10) # Publish to cmd_vel node
        self.timer = self.create_timer(0.05, self.timer_callback)  # Runs at 20Hz. Can be changed.

        self.goal_list = goal_list
        self.map_array = map_array
        self.origin = origin # World (x, y) coordinate of the grid's [0, 0] corner. Use with grid_to_world()/world_to_grid()
        self.resolution = resolution # Metres per grid cell (0.2). Use with grid_to_world()/world_to_grid()

        self.frame = frame # Optitrack->maze frame (real robot only), see optitrack_to_maze()
        self.expected_start = expected_start # Where this run says the robot should be placed (maze frame), real robot only
        self.calibrate = calibrate # Real robot: only print poses, never publish cmd_vel
        self.raw_pose = None # Latest pose exactly as Optitrack reported it (real robot only)
        self._start_checked = False
        self._calibrate_ticks = 0

        self.pose = None
        self.path = [] # Set this to your planned route (a list of grid-index tuples, in travel order) once you've computed it -- it'll automatically show up in the terminal map print
        self._last_printed_path = None

        self.init_other() # !!!

    def _print_calibration(self):
        '''--calibrate: twice a second, print the raw Optitrack pose next to the maze-frame pose and grid cell.'''
        self._calibrate_ticks += 1
        if self._calibrate_ticks % 10:
            return
        ox, oy, oh = self.raw_pose
        mx, my, mh = self.pose
        i, j = world_to_grid(mx, my, self.origin, self.resolution)
        self.get_logger().info(f"optitrack ({ox:+.3f}, {oy:+.3f}, {oh:+7.1f} deg)  ->  maze ({mx:+.3f}, {my:+.3f}, {mh:+7.1f} deg)  cell [{i}, {j}]")

    def _check_start_once(self):
        '''Once, on the first pose: warn if the robot isn't near this run's start point (usually a placement or config mistake).'''
        if self._start_checked or self.expected_start is None:
            return
        self._start_checked = True
        dist = math.hypot(self.pose[0] - self.expected_start[0], self.pose[1] - self.expected_start[1])
        msg = f"Robot is at maze ({self.pose[0]:.2f}, {self.pose[1]:.2f}); this run starts at {self.expected_start} -- {dist:.2f} m away"
        if dist > 0.4:
            self.get_logger().warn(msg + ". Check where the robot is placed; if it is on the start point, ask a TA.")
        else:
            self.get_logger().info(msg + " (ok)")

    def print_map(self):
        '''Prints the occupancy grid to the terminal: walls, your current position ('S'), all goal points ('W'/'G'),
        and your planned route (self.path) if you've set one ('*'). Safe to call anytime pose is known; does nothing
        useful before then. Called automatically from timer_callback() whenever self.path changes.'''
        if self.pose is None:
            return
        shape = self.map_array.shape
        clip = lambda cell: (int(np.clip(cell[0], 0, shape[0]-1)), int(np.clip(cell[1], 0, shape[1]-1)))
        current_cell = clip(world_to_grid(self.pose[0], self.pose[1], self.origin, self.resolution))
        goal_cells = [clip(world_to_grid(gx, gy, self.origin, self.resolution)) for gx, gy in self.goal_list]
        grid = Grid(self.map_array, starting_position=current_cell, goal_position=goal_cells[-1])
        grid.print_grid_map(waypoints=goal_cells, path=self.path)

    def yaw_from_quaternion(self, q):
        '''Returns yaw angle (in rad) for orientation based on given quaternion input q'''
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        return np.arctan2(siny_cosp, cosy_cosp)

    def optitrack_callback(self, msg:PoseStamped):
        '''Callback to calculate 2D pose info from Optitrack node. Pose info includes x and y coordinates, as well as heading in degrees.
        This callback will run everytime the rclpy executor spins'''
        x, y = msg.pose.position.x, msg.pose.position.y
        heading = np.rad2deg(self.yaw_from_quaternion(msg.pose.orientation))
        self.raw_pose = np.array((x,y,heading))
        if self.frame is not None:
            x, y, heading = optitrack_to_maze(x, y, heading, self.frame) # Your code works in the maze frame, like in simulation
        self.pose = np.array((x,y,heading))
        return self.pose

    def odometer_callback(self, msg):
        '''Callback to calculate 2D pose info from Gazebo odomoter. Pose info includes x and y coordinates, as well as heading in degrees.
        This callback will run everytime the rclpy executor spins'''
        latest_pose_msg = msg.pose.pose
        heading = np.rad2deg(self.yaw_from_quaternion(latest_pose_msg.orientation))
        self.pose = np.array((latest_pose_msg.position.x, latest_pose_msg.position.y, heading))
        return self.pose

    def move_2D(self, x:float=0.0, y:float=0.0, turn:float=0.0):
        '''Publishes a Twist message to ROS to move a robot. Inputs are x and y linear velocities, as well as turn (z-axis yaw) angular velocity.'''
        twist_msg = Twist()
        x = np.clip(x, -max_translate_velocity, max_translate_velocity)
        y = np.clip(y, -max_translate_velocity, max_translate_velocity)
        turn = np.clip(turn, -max_translate_velocity*2, max_translate_velocity*2)
        twist_msg.linear.x, twist_msg.linear.y, twist_msg.linear.z = float(x), float(y), 0.0
        twist_msg.angular.x, twist_msg.angular.y, twist_msg.angular.z = 0.0, 0.0, float(turn)
        self.publisher_.publish(twist_msg)

    def set_waypoints(self, waypoints:list):
        '''Set new waypoints when a goal has been reached'''
        self.goal_reached = False
        self.waypoints = waypoints
        self.current_waypoint_idx = 0

    def init_other(self):
        self.WIP_WPs = []
        self.desiredPose = None
        self.outputPose = {'x': 0, 'y': 0, 'h': 0}

        # print(self.map_array) # a 2D array of 99s and 0s, when printed is upside down
        self.getGraph() # get graph from obstacle map in my desired format >:)
        # to consider: 
        # print_map when self.path changes   -> draw_grid_map
        
        self.reached_goal = [False for i in range(len(self.goal_list))]
        self.goals = [world_to_grid(*self.goal_list[i], self.origin, self.resolution) for i in range(len(self.goal_list))]
        self.checkPairs = [[0, 1], [-1, 0], [0, -1], [1, 0]]
        self.WP_idx = 0
        self.currentGoal_idx = -1 # do not make 0


        # Values to modify
        self.size = 0.2
        self.turn_value = 2      # please edit margin of error because it might miss it...
        self.min_run_value = 0.4
        self.run_multiplier = 5. 

        # heading PID gains (tune these). Errors are in degrees, outputs in rad/s.
        self.turn_kp, self.turn_ki, self.turn_kd = 0.05, 0.01, 0.005    # turn-on-the-spot at a waypoint
        self.steer_kp, self.steer_ki, self.steer_kd = 0.03, 0.0, 0.002  # heading correction while driving
        self.steer_limit = 1.0     # max steering rate while driving (rad/s)
        self.min_turn_value = 0.5  # minimum turn rate (rad/s); lower than this and friction stops it just short of the target
        self.heading_tolerance = 2. # degrees; must be wide enough that min_turn_value*dt (~1.4 deg/tick) can't jump over it
        self.dt = 0.05             # controller period, matches create_timer(0.05, ...)
        self.h_integral, self.h_prev_error = 0.0, None # heading PID state, reset on every new desiredPose

        self.size_bounds = 0.2#/2.
        self.delay_counter = 40 # give time to run cmd

        # check_grid_validity
        # draw_grid_map
        # animate_path

    def timer_callback(self):
        if self.delay_counter > 0:  # !!! i also hid self.pose print below
            self.delay_counter -= 1
            self.move_2D(0, 0, 0)
            return

        """Controller loop. Insert path planning and PID control logic here"""
        if self.pose is None:
            return # Does not run if no pose received from Odom or Optitrack
        now = time.time()
        if now - getattr(self, "_last_pose_log", 0.0) >= 1.0: # at most once a second, so it does not bury the map / warnings below
            self._last_pose_log = now
            self.get_logger().debug(f"Pose: {self.pose}")

        if self.calibrate:
            self._print_calibration()
            return # Calibration mode: look, don't drive

        self._check_start_once()

        if self.path != self._last_printed_path: # Prints once immediately (map + start + goals), then again each time self.path changes
            self.print_map()
            self._last_printed_path = list(self.path)
        
        
        # cd rb2301_ca2
        # chmod +x ca2.sh gz_ca2.sh
        # colcon build --symlink-install
        # ./gz_ca2.sh
        # ./ca2.sh 

        # need to make heading stabilize towards desired heading
        # return;
        # gz service -s /gui/move_to/pose --reqtype gz.msgs.GUICamera --reptype gz.msgs.Boolean --timeout 2000 --req 'pose: {position: {x: 2, y: -1.8, z: 4.0}, orientation: {x: 0.0, y: 1.0, z: 0.0, w: 1.0}}'
        # gz service -s /world/empty/set_pose --reqtype gz.msgs.Pose --reptype gz.msgs.Boolean --timeout 1000 --req "name: 'nanocar', position: {x: 0.0, y: 0.0, z: 0.0036}, orientation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0}"



        # self.move_2D(0, 0, 1) # it won't turn becuase of ✨ friction ✨
        # print(self.pose[2])  # yaw 1.57 -> 180.

        # self.move_2D(1, 0) # 0.05: barely moving, 0.1 slow
        # print(self.pose[0]) # it goes up
        # return

        ###### INSERT CODE HERE ######
        if (self.WP_idx >= len(self.WIP_WPs)):
            self.reached_goal[self.currentGoal_idx] = True
            self.WIP_WPs = [] # register new

        if (len(self.WIP_WPs) == 0): # register a new path plan
            self.currentGoal = None
            self.WP_idx = 0

            i = 0
            while i < len(self.goals): 
                if (not self.reached_goal[i]): # get first
                    self.currentGoal = self.goals[i]
                    self.currentGoal_idx = i  # to set as true later
                    break; 
                i+=1
            if ((self.currentGoal == None)): 
                print('Done'); return # all goals are complete, it's done. 

            print("PLANNING", self.currentGoal)
            # do once per goal:
            self.calcH() # calculate heuristic
            self.Astar() # run A* algorithm to find path (list of traversable grid cells)
            self.makeWaypoints() # simplify consecutive cells via waypoints

        else: # only if waypoints exist:
        
            # run continuously:

            # print("RUNNING", self.currentGoal)
            # define desiredPose and outputPose
            if self.desiredPose != None: # previous callback has not completed to reach desiredPose
                # in case you overshoot...
                self.dynamicToNextWaypoint() # insert dynamic PID output pose here
            else: # it has completed so define new state...
                self.STATE = self.moveToNextWaypoint() 
            
            # basically a while ( true )
            # mov = {x: 0, y: 0, heading: 0};

            # ideally it should finish moving first then check for each callback... but eh...
            self.move_2D(self.outputPose["x"], self.outputPose["y"], self.outputPose['h'])
            # get Pose
            actual = {'x': self.pose[0], 'y': self.pose[1], 'h': self.pose[2]}
            # check if desired waypoint (heading & location) is reached 

            printStr = ''
            # printStr += f'x: {self.desiredPose['x']:.2f} {actual['x']:.2f} ({self.outputPose['x']:.2f}) '+('     ' if self.getDifference('x', actual) else ' :)  ' ) if ((self.desiredPose['x'] != None)) else ''
            # printStr += f'y: {self.desiredPose['y']:.2f} {actual['y']:.2f} ({self.outputPose['y']:.2f}) '+('     ' if self.getDifference('y', actual) else ' :)  ' ) if ((self.desiredPose['y'] != None)) else ''
            printStr += f'x: {self.desiredPose['x']:.3f} {actual['x']:.3f} '+('     ' if self.getDifference('x', actual) else ' :)  ' ) if ((self.desiredPose['x'] != None)) else ''
            printStr += f'y: {self.desiredPose['y']:.3f} {actual['y']:.3f} '+('     ' if self.getDifference('y', actual) else ' :)  ' ) if ((self.desiredPose['y'] != None)) else ''
            printStr += f'({self.outputPose['x']:.2f})' if ((self.desiredPose['x'] != None) or (self.desiredPose['y'] != None)) else ''   # only x movement considered
            printStr += f'h: {self.desiredPose['h']%360.:.2f} {actual['h']%360.:.2f} ({self.outputPose['h']:.2f}) ' + ('  ' if self.getDifferenceH(actual) else ' :)  ') if ((self.desiredPose['h'] != None)) else ''
            # print to see approach closer
            print(printStr)
            # check if it reached waypoint.
            # consider margin of error:    ⚠️⚠️⚠️ %360.
            # print(abs(self.desiredPose['h']%360. - actual['h']%360.))          may depend on your turning & running speed...     
            
            # the errors can accumulate oh noes...
            
            if self.getDifference('x', actual): return 
            if self.getDifference('y', actual): return
            if self.getDifferenceH(actual): return  # it's stupid 😭
            # might need to % degrees

            self.desiredPose = None
            self.outputPose = None
            print('next', self.STATE)
            if self.STATE == 'TURNING TO NEXT WAYPOINT': # goal is to turn and it finished turning
                # remove waypoint or move index += 1? (<== let's do this)
                self.WP_idx += 1

        ###### INSERT CODE HERE ######
    def getDifference(self, k, actual):
        return ((self.desiredPose[k] != None) and (abs(self.desiredPose[k] - actual[k]) > self.size_bounds))
    def getDifferenceH(self, actual):
        return ((self.desiredPose['h'] != None) and (abs(wrap_deg(self.desiredPose['h'] - actual['h'])) > self.heading_tolerance))


    def getGraph(self):
        self.maze_arr = [] # 30 cols for 35 rows
        # CA2 map: y axis (left, 30x cols), x axis (up, 35x rows)
        # to preserve axis shape when printing,
        # Arr map: y axis (right -> cols), x axis (down -> rows)
        
        # for y in range(len(self.map_array)):
        #     self.maze_arr.append([])
        #     for x in range(len(self.map_array[y])):
        #         self.maze_arr[y].append([])
        for x in range(len(self.map_array)):  # each row
            self.maze_arr.append([])
            for y in range(len(self.map_array[x])):
                self.maze_arr[x].append([])
        # print('size', len(self.maze_arr), len(self.maze_arr[0])) # 35, 30

        for x in range(len(self.maze_arr)): 
            for y in range(len(self.maze_arr[x])):
                # c = self.map_array[-x][-y]
                c = self.map_array[x][y]  
                # print("#" if c == 99 else "." , end="") # so that the output can be rotated 180. to match the image.
                if (c != 99): # unoccupied
                    self.maze_arr[x][y] = [1, 0] 
                else:
                    self.maze_arr[x][y] = [0, 0] # [filled, heuristic]
            # print()
        # exit()

    def manhattan(self, A, B): return abs(B[0]-A[0]) + abs(B[1]-A[1])
    def calcH(self):
        # get goal
        # for each cell
        # find manhattan distance from cell to goal (grid index distance)

        # print(len(self.maze_arr), len(self.maze_arr[0]), self.currentGoal) # 35, 30, 22x7
        for x in range(len(self.maze_arr)): 
            for y in range(len(self.maze_arr[x])):
                self.maze_arr[x][y][1] = self.manhattan([x, y], self.currentGoal) # this should be integers
                # self.maze_arr[y][x][1] = self.manhattan([y, x], [self.currentGoal[0], self.currentGoal[1]]) 
                # manhattan([4, 25], [22, 7]) = 36

                # 😭😭😭 why are you like this
                # manhattan([25, 4], [7, 22]) = 36
        # print(self.maze_arr) # Below "E" where start is should have values near 36 for the first goal
        # self.debugMap()        
        # exit()

    def debugMap(self):
        # sudo apt --fix-broken install
        # sudo apt install python3-wheel python3-pip
        # # pip install pyperclip    (nevermind...)
        # sudo apt update
        # sudo apt install python3-pyperclip
        # pyperclip.copy(output_text)
        # doesn't work OH FORGET IT

        # (1 is free)
        output_text = 'x y, [free? heuristic]\n'
        for x in range(len(self.maze_arr)):  # 35 rows  
            for y in range(len(self.maze_arr[x])):
                output_text += f"{x} {y} [{self.maze_arr[x][y][0]} {self.maze_arr[x][y][1]}]\n"
        with open('/home/km/Downloads/map_data.txt', "w") as file:
            file.write(output_text)
        # suppose x & y are the indexes of a 2D array. create a 2D grid where # if free = 0, . otherwise. append 2 characters to each based on heuristic. remove spacing.

        exit()

    def findH(self, inp):
        [x, y] = inp
        # print(x, y, self.maze_arr[x][y][1]) 
        # 4, 25, 36 (start)
        # 5, 25, 35 (closer; up x +1)
        # 4, 26, 37 (further; left y +1)
        # exit()
        return self.maze_arr[x][y][1]
    
    def expand(self, parent):
        # return top, bottom, left, right cells (if available)
        children = []
        i = 0
        while (i < len(self.checkPairs)):
            [x, y] = [parent.coord[0] + self.checkPairs[i][0], 
                    parent.coord[1]  + self.checkPairs[i][1]]

            if (0 <= x) and (x < len(self.maze_arr)) and \
                (0 <= y) and (y < len(self.maze_arr[x])): 
                if (self.maze_arr[x][y][0] != 0): # not blue            
                    # print(str(y) + ' ' + str(x), self.maze_arr[x][y])
                    children.append(NODE(parent, 
                        [x, y], 
                        parent.cost + 1, 
                        self.findH([x, y])))

            
            i += 1
        return children
    
    def getPriority(self, frontier):
        minVal = float('inf')
        minKey = 0

        for i in range(len(frontier)): 
            node = frontier[i]
            p = node.cost + node.h
            b = p < minVal
            minVal = p if (b) else minVal # ⚠️ wait... what about tiebreakers
            minKey = i if (b) else minKey # a coordinate

        
        # to_return = frontier[minKey]
        # frontier = frontier[:minKey] + frontier[minKey+1:]
        # return to_return
        return frontier.pop(minKey)  # remove element from frontier & return node


    # First output waypoints should be
    # [4, 25], [4, 17], [27, 17], [27, 14], [22, 14], [22, 7]
    def Astar(self):
        # placeholder: start at [4, 25] and try to avoid a u-turn please
        # Example: self.path = [[4, 25], [2, 25], [0, 25], [0, 5], [1, 5], [2, 5], [3, 5], [3, 6], [3, 7], [3, 8]]
        self.path = []
        [rx, ry] = [*world_to_grid(*self.pose[:2], self.origin, self.resolution) ]        
        # print('Start', rx, ry) # Start: 4, 25   wait why is it 5, 25

        frontier = [] # [ NODE, NODE2, ... ]
        reached  = {} # { NODE: ?, NODE2: ?, ... } "set"
        children = []
        fixedGoal = self.currentGoal[:] # temp copy

        # start location
        start = NODE(None, [rx, ry], 0, 0)
        frontier.append( start ) # keeping track of the axes is a nightmare so just print to see results, okay?
        # print('s', self.path) # default -750, 120

        found_goal = False
        # print('set', start, fixedGoal);
        # print(self.maze_arr[fixedGoal[0]][fixedGoal[1]])

        # printStr = ''
        node = None
        while (len(frontier) > 0):                
            node = self.getPriority(frontier)  # Priority queue: order by g(n) + f(n)
            # REMOVES NODE FROM FRONTIER

            # printStr += str(node["coord"]) + '\t'
            # ok some reason, to get the correct output, [22, 7] has to index [y=22][x=7]
            # print(node.coord, fixedGoal); exit() # start: [4, 25] [22, 7]

            if ((node.coord[0] == fixedGoal[0]) and (node.coord[1] == fixedGoal[1])):
                found_goal = True
                # print("found goal!");
                break
            
            children = self.expand( node )
            # print(node.coord, end = '\t')
            for i in range(len(children)):
                child = children[i]
                # if (child.coord[0] == 5 and child.coord[1] == 27) {print('5', '27', rb.maze_arr[5][27] , self.findH([27, 5]))}  # rb.maze_arr[27][5]   is allowed
                # if (child.coord[1] == 5 and child.coord[0] == 27) {print('27', '5', rb.maze_arr[27][5] , self.findH([5, 27]))} # rb.maze_arr[5][27] 0 so it's illegal!
                if (not (child.ID in reached)):  # filter which nodes to go next here 
                    reached[child.ID] = child
                    frontier.append(child)
                else: # update if node & cost cheaper found
                    prev_p = reached[child.ID].cost + reached[child.ID].h
                    p = child.cost + child.h
                    if (p < prev_p):
                        for t in range(len(frontier)): # ⚠️😋 untested
                            if (frontier[t]==reached[child.ID]): # where frontier node == child to replace                          
                                frontier[t] = child
                                print("check that this works later because right now, i don't know")
                                # can break?                        
                        reached[child.ID] = child
                        
        
        if (found_goal):
            while (node != None):
                self.path.append(node.coord)

                # none should be blocked (self.maze_arr[y][x][0] always 1)
                # print(node.coord, self.findH(node.coord), self.maze_arr[node.coord[1]][node.coord[0]])

                node = node.parent
            self.path.reverse()
            # print(self.path)
        else:
            print("Error: path could not be found")
        # First path: [[4, 25], [4, 24], [4, 23], [4, 22], [4, 21], [4, 20], [4, 19], [4, 18], [4, 17], [5, 17], [6, 17], [7, 17], [8, 17], [9, 17], [10, 17], [11, 17], [12, 17], [13, 17], [14, 17], [15, 17], [16, 17], [17, 17], [18, 17], [19, 17], [20, 17], [21, 17], [22, 17], [23, 17], [24, 17], [25, 17], [26, 17], [27, 17], [27, 16], [27, 15], [27, 14], [26, 14], [25, 14], [24, 14], [23, 14], [22, 14], [22, 13], [22, 12], [22, 11], [22, 10], [22, 9], [22, 8], [22, 7]]
        # exit()

    def makeWaypoints(self):
        self.WPs = [self.path[0]]
        self.WIP_WPs = []
        
        if (len(self.path) > 1):
            cell = 1 # start at 2nd
            latest = []
            while (cell < len(self.path) - 1):
                # if (cell and latest WP cell) share x or share y { do nothing }
                latest = self.WPs[len(self.WPs)-1]
                if ((self.path[cell][0] == latest[0]) or (self.path[cell][1] == latest[1])): ...
                else: self.WPs.append( self.path[cell-1] ) # add previous cell
                cell += 1
            
            self.WPs.append(self.path[len(self.path)-1]); # last cell    
        
        # EXCLUDE FIRST WAYPOINT ⚠️⚠️⚠️
        # self.WPs.pop(0)
        # wait it breaks the thing oops

        self.WIP_WPs = deepcopy(self.WPs) # deep copy the entire thing
        
        # testprint = [ [*self.path[i], *self.maze_arr[self.path[i][0]][self.path[i][1]] ] for i in range(len(self.path)) ] # verify all 3rd values are 1
        # print('pth', testprint)

        # First wps: [[4, 25], [4, 17], [27, 17], [27, 14], [22, 14], [22, 7]]
        #  static pov           right,    up,      right,     down,     right
        print('wps', self.WIP_WPs)
    def moveToNextWaypoint(self):
        # read next queued waypoint
        # check [point A to point B] progress
        self.desiredPose = {'x': None, 'y': None, 'h': None}
        self.outputPose = {'x': 0, 'y': 0, 'h': 0}
        self.h_integral, self.h_prev_error = 0.0, None # new target, clear old integral/derivative

        # replace w/ waypoint: (in terms of world coordinates)
        [x, y] = grid_to_world(*self.WIP_WPs[self.WP_idx], self.origin, self.resolution)
        [rx, ry] = self.pose[:2]
        # print(f"to: {self.WIP_WPs[self.WP_idx]} ({x:.2f}, {y:.2f}), at ({rx:.2f}, {ry:.2f})")        

        # if haven't reached waypoint: move forward (similar to check goal)
            # ⚠️⚠️⚠️ also allow heading adjustment later!!!
        # if have, turn for the next waypoint. once finished, waypoint idx += 1
    
        STATE = ''

        # ⚠️⚠️⚠️ based on 0.2m square...     x +/- 0.2/2 ?
        if ((abs(x - rx) <= self.size_bounds) and (abs(y - ry) <= self.size_bounds )): # has reached waypoint (circular :P)
            STATE = 'TURNING TO NEXT WAYPOINT'

            if self.WP_idx+1 >= len(self.WIP_WPs): return STATE # no more waypoints to turn to, so move on already to next goal...
            # 90 (left)      0          270 (right)
            #            180 (down)
            # increase: CCW

            headingDeg = self.pose[2]%360.
            heading = np.deg2rad(headingDeg)  # ⚠️⚠️⚠️ i forgot radians lololol :|
            # register to turn: (+x or +y, i aint doing diagonals now)
            # read heading and next pos (a u-turn should never happen)
            # Ex: from (0, 2) (upward 0 heading) to (5, 2)   compare 0 & 5
            # turn right (0 < 5)
            # otherwise (0 < -3) turn left
            # Ex: from (3, 5) (left heading) to (3, 2),  compare 5 & 2
            # turn left (5 > 2)
            # otherwise (5 < 8) turn right
            # Ex: other directions
            # (downward heading): reverse Ex1
            # (right heading): reverse Ex2
            # OR
            # rotate coords to match upward heading
            # OR
            # create a line from location + ahead (arbitrary length), check if point is on left or right side of line
            # OR
            # have current coords (rx, ry), next target coords (x, y)
            # find rotation such that line formed from current to infinity will contain target coords
            # ⚠️⚠️⚠️ i searched so i can't explain rn

            # https://www.desmos.com/calculator/cwy8ok9tje
            [x, y] = grid_to_world(*self.WIP_WPs[self.WP_idx+1], self.origin, self.resolution)
            value = np.rad2deg(np.arctan2(y - ry, x - rx))
            # ideally, desired %270
            self.desiredPose['h'] = value%360.
            # print(f"{rx:.3f}, {ry:.3f} to {x:.3f}, {y:.3f}   {value:.3f}")
            self.outputPose['h'] = self.turn_value * (-1 if value < 0 else 1)
            # ⚠️⚠️⚠️ optionally, do PID

        else: # hasn't reached waypoint
            STATE = 'MOVING TO WAYPOINT'
            # ⚠️⚠️⚠️ replace w/ a more reliable method
            self.desiredPose['x'] = x   # should i add 0.1   hmmm
            self.desiredPose['y'] = y

            # consider PID control
            # (copy from below later)



        return STATE       

    def dynamicToNextWaypoint(self): 
        # ⚠️⚠️⚠️ do PID control here and match with previous :)
        # desiredPose is set by moveToNextWaypoint. it does not change
        self.outputPose = {'x': 0, 'y': 0, 'h': 0}

        # [x, y] = grid_to_world(*self.WIP_WPs[self.WP_idx], self.origin, self.resolution)
        [rx, ry] = self.pose[:2]
        headingDeg = self.pose[2]%360.
        heading = np.deg2rad(headingDeg)

        if self.STATE == 'TURNING TO NEXT WAYPOINT':
            if self.WP_idx+1 >= len(self.WIP_WPs): return # no more waypoints to turn to, so move on already to next goal...
            # the line thing is severely unreliable here (idk why) so let's directly use pose :\
            
            # given angle and target angle, how to find which direction to turn?
            # ⚠️⚠️⚠️ i searched so i can't explain rn
            value = wrap_deg(self.desiredPose['h'] - headingDeg) # heading error in degrees, +ve = turn left (CCW)

            # PID on heading error
            self.h_integral = np.clip(self.h_integral + value*self.dt, -20., 20.) # clamp so it can't wind up
            derivative = 0. if self.h_prev_error is None else (value - self.h_prev_error)/self.dt
            self.h_prev_error = value
            turn = self.turn_kp*value + self.turn_ki*self.h_integral + self.turn_kd*derivative
            if abs(turn) < self.min_turn_value: turn = self.min_turn_value * (1 if value >= 0 else -1)
            self.outputPose['h'] = turn
            print(f"[TURN]  target {self.desiredPose['h']%360.:7.2f} deg | current {headingDeg:7.2f} deg | error {value:+7.2f} deg | cmd {turn:+.2f} rad/s")
        # ⚠️⚠️⚠️ turning changes the pose as well. include heading later.
        elif self.STATE == 'MOVING TO WAYPOINT':
            # have: desired pose (x, y), robot pose (rx, ry), robot heading
            [x, y] = [self.desiredPose["x"], self.desiredPose["y"]]
            heading = -heading # hmmm?
            nx = (x - rx) * np.cos(heading) - (y - ry) * np.sin(heading) # target x, y
            ny = (x - rx) * np.sin(heading) + (y - ry) * np.cos(heading) # rotate desired to the perspective of robot, around (rx, ry), ignore +(rx, ry) offset
            print(f"{nx:.3f}, {ny:.3f} around {rx:.3f}, {ry:.3f} to {x:.3f}, {y:.3f}")
            #       1.502, 0.051       around -0.035, 0.001      to -0.100, -1.500

            

            error = nx
            # x should be +ve when moving to goal, then flipping signs if it overshoots...
            self.outputPose['x'] = max(self.run_multiplier*error, self.min_run_value) if (error > 0) else min(self.run_multiplier*error, -self.min_run_value)

            # heading correction: PID on the bearing to the waypoint in the robot frame (+ve = waypoint is to the left)
            if np.hypot(nx, ny) > 0.05: # bearing is noisy when right on top of the waypoint
                bearing = np.rad2deg(np.arctan2(ny, nx))
                if nx < 0: bearing = wrap_deg(bearing - 180.) # overshot and reversing: line the back up instead of spinning round
                self.h_integral = np.clip(self.h_integral + bearing*self.dt, -20., 20.)
                derivative = 0. if self.h_prev_error is None else (bearing - self.h_prev_error)/self.dt
                self.h_prev_error = bearing
                steer = self.steer_kp*bearing + self.steer_ki*self.h_integral + self.steer_kd*derivative
                self.outputPose['h'] = np.clip(steer, -self.steer_limit, self.steer_limit)
                print(f"[DRIVE] target {(headingDeg + bearing)%360.:7.2f} deg | current {headingDeg:7.2f} deg | error {bearing:+7.2f} deg | cmd {self.outputPose['h']:+.2f} rad/s")



class NODE:
    def __init__(self, parent, coord, accumulated_cost, h=0):
        self.coord = coord
        self.cost = accumulated_cost
        self.h = h
        self.parent = parent # also NODE class

        self.ID = str(self.coord[0]) + ', ' + str(self.coord[1]) 

class Grid():
    '''
    Grid class to use with occupancy grid. Contains the following functions:
        check_grid_validity : Uses flood fill to check if there's a valid path from start to goal position
        draw_grid_map : Creates a colour image of the grid, as well as waypoints and full solution path if given. Can show it and/or save it to a .png file
        print_grid_map : Prints an ASCII-art version of the same map (grid, waypoints, path) to the terminal
        animate_path : Prints the ASCII-art map frame by frame in the terminal, animating the robot moving along a given path

    __init__ input Args:
        grid_array : 2D numpy array representing the occupancy grid
        starting_position : tuple of starting indices within the numpy array
        goal_position : tuple of goal indices within the numpy array. Works with negative indices as well
    '''
    def __init__(self, grid_array:np.array=np.array([]), starting_position:tuple=(0,0), goal_position:tuple=(-1,-1)):
        self.grid = grid_array
        self.shape = self.grid.shape
        self.starting_position = starting_position

        # If goal position given with negative index, need to convert to +ve
        if goal_position[0] < 0:
            goal_x = self.shape[0] + goal_position[0]
        else:
            goal_x = goal_position[0]
        if goal_position[1] < 0:
            goal_y = self.shape[1] + goal_position[1]
        else:
            goal_y = goal_position[1]

        self.goal_position = (goal_x, goal_y)

    def check_grid_validity(self):
        '''Use flood fill to check if there's a viable path between start and goal positions'''
        grid = self.grid.copy()
        flood_stack = [(self.goal_position[0], self.goal_position[1])]
        while flood_stack:
            tile = flood_stack[0]
            del flood_stack[0]
            try:
                next_tile = (tile[0]+1, tile[1])
                if grid[next_tile] == 0:
                    grid[next_tile] = 1
                    flood_stack.append(next_tile)
            except:pass
            try:
                next_tile = (tile[0]-1, tile[1])
                if grid[next_tile] == 0:
                    grid[next_tile] = 1
                    flood_stack.append(next_tile)
            except:pass
            try:
                next_tile = (tile[0], tile[1]+1)
                if grid[next_tile] == 0:
                    grid[next_tile] = 1
                    flood_stack.append(next_tile)
            except:pass
            try:
                next_tile = (tile[0], tile[1]-1)
                if grid[next_tile] == 0:
                    grid[next_tile] = 1
                    flood_stack.append(next_tile)
            except:pass
        if grid[self.starting_position] == 1: # Means the flood is able to reach starting position from the ending position
            return True
        else:
            return False

    def _colour_grid(self, waypoints:list=(), path:list=(), obstacle_threshold:float=50) -> np.array:
        '''Builds the (H, W, 3) colour image array shared by draw_grid_map. Maze walls in blue, empty space in white, path taken in green and waypoints in red'''
        image_grid = np.ones((self.grid.shape[0],self.grid.shape[1],3), dtype=np.uint8)
        image_grid[self.grid <= obstacle_threshold] = (255,255,255)
        image_grid[self.grid > obstacle_threshold] = (0,0,255)

        for x, y in path:
            image_grid[x][y] = (0,255,0)

        for point in waypoints:
            image_grid[point] = (255,0,0)

        return image_grid

    def draw_grid_map(self, waypoints:list=(), path:list=(), obstacle_threshold:float=50, save_path:str=None, show:bool=True):
        '''Creates an image of the maze and path taken. Maze walls in blue, empty space in white, path taken in green and waypoints in red

        Args:
            waypoints : list (or other iterable) of tuple coordinates representing all the grid indices for the waypoints. Will be represented in red, takes precedence over path
            path : list (or other iterable) of tuple coordinates representing all the grid indices forming the solution path. Will be represented in green
            obstacle_threshold : Optional float to indicate threshhold for whether a grid is considered occupied. Not important for ca2
            save_path : Optional file path (e.g. "path_result.png") to save the image to, in addition to/instead of showing it
            show : Whether to pop up the image in a viewer (default True). Set to False if you only want to save it
        '''
        image_grid = self._colour_grid(waypoints, path, obstacle_threshold)

        image_grid = np.flip(image_grid, axis=1)[::-1]
        img = Image.fromarray(image_grid, 'RGB')

        # Resize image
        base_width = 500
        wpercent = (base_width / float(img.size[0]))
        hsize = int((float(img.size[1]) * float(wpercent)))
        img = img.resize((base_width, hsize), getattr(Image, "Resampling", Image).NEAREST)

        if save_path:
            img.save(save_path)
            print(f"Saved map image to {save_path}")

        if show:
            img.show()

    def print_grid_map(self, waypoints:list=(), path:list=(), obstacle_threshold:float=50):
        '''Prints an ASCII-art version of the maze to the terminal: '#' wall, '.' free, '*' solution path, 'W' waypoint, 'S' start, 'G' goal

        Args:
            waypoints : list (or other iterable) of tuple coordinates representing all the grid indices for the waypoints
            path : list (or other iterable) of tuple coordinates representing all the grid indices forming the solution path
            obstacle_threshold : Optional float to indicate threshhold for whether a grid is considered occupied. Not important for ca2
        '''
        chars = np.where(self.grid > obstacle_threshold, '#', '.').astype('<U1')

        for x, y in path:
            chars[x, y] = '*'
        for point in waypoints:
            chars[point] = 'W'
        chars[self.starting_position] = 'S'
        chars[self.goal_position] = 'G'

        display = np.flip(chars, axis=1)[::-1]
        print('\n'.join(''.join(row) for row in display))

    def animate_path(self, path:list, waypoints:list=(), obstacle_threshold:float=50, delay:float=0.2):
        '''Animates the robot moving along path, one cell at a time, by reprinting the ASCII-art map to the terminal.
        'R' marks the robot's current cell, '*' cells it has already passed through.

        Args:
            path : list of tuple grid indices forming the solution path, in travel order
            waypoints : list (or other iterable) of tuple coordinates representing all the grid indices for the waypoints
            obstacle_threshold : Optional float to indicate threshhold for whether a grid is considered occupied. Not important for ca2
            delay : Seconds to pause between animation frames
        '''
        path = list(path)
        base_chars = np.where(self.grid > obstacle_threshold, '#', '.').astype('<U1')
        for point in waypoints:
            base_chars[point] = 'W'
        base_chars[self.goal_position] = 'G'

        for step in range(len(path)):
            frame = base_chars.copy()
            for x, y in path[:step]:
                frame[x, y] = '*'
            frame[path[step]] = 'R'

            display = np.flip(frame, axis=1)[::-1]
            print("\033c", end="")  # Clear terminal between frames
            print('\n'.join(''.join(row) for row in display))
            time.sleep(delay)


def main(args=None):
    global max_translate_velocity

    arg_parser = argparse.ArgumentParser(description="RB2301 CA2 path planning")
    arg_parser.add_argument(
        '--run', choices=IRL_RUNS, default=None,
        help="Run on the real maze: test1, test2 or full (starts/goals in README). Omit to run in Gazebo simulation."
    )
    arg_parser.add_argument(
        '--robot', type=int, default=None,
        help="Number of your Bingda robot (3 -> /vrpn_mocap/bingda_003/pose). Default: your ROS_DOMAIN_ID (set to the robot number on the robots), else [robot] number in optitrack_variables.config."
    )
    arg_parser.add_argument(
        '--calibrate', action='store_true',
        help="Real robot: only print Optitrack and maze-frame poses (twice a second) and never publish cmd_vel. Use it to set origin/rotation."
    )
    cli_args, ros_args = arg_parser.parse_known_args(args=args)
    run = cli_args.run or ("test1" if cli_args.calibrate else None)
    is_simulation = run is None # Remember: pass --run test1|test2|full to ca2.sh when testing on the real lab setup

    print("Starting path planning")
    rclpy.init(args=ros_args)

    robot_number = 3
    frame = expected_start = None
    if is_simulation:
        config = sim_config
    else:
        config = load_irl_config(run)
        frame, expected_start = config["frame"], config["start"]
        domain = os.environ.get("ROS_DOMAIN_ID", "")
        if cli_args.robot is not None:
            robot_number = cli_args.robot
        elif domain.isdigit() and 1 <= int(domain) <= 99: # on the Bingda robots ROS_DOMAIN_ID is the robot number
            robot_number = int(domain)
        else:
            robot_number = config["robot_number"]
        frame = robot_frame(config, robot_number)
        expected_start = config["start"]
        print(f"Running on the real maze, run={run}, robot=bingda_{robot_number:03d}, resolution={config['resolution']}")
        print(f"  maze origin in Optitrack frame = ({frame['origin_x']}, {frame['origin_y']}), rotation = {frame['rotation_deg']} deg, robot heading offset = {frame['heading_offset_deg']} deg")
        print(f"  start={expected_start}, goals={config['goal_list']}")
        if cli_args.calibrate:
            print("  CALIBRATE MODE: printing poses only, the robot will not be commanded")

    max_translate_velocity = config["max_translate_velocity"]

    map_array = np.load(os.path.join(_PACKAGE_DIR, config["map_file"]), allow_pickle=True)
    waypoint = WaypointNode(map_array, config["goal_list"], is_simulation, config["origin"], config["resolution"],
                            frame=frame, robot_number=robot_number, expected_start=expected_start, calibrate=cli_args.calibrate)

    # Start spinning the waypoint node and only stop once SystemExit error is raised within the node callback
    try:
        rclpy.spin(waypoint)
    except SystemExit:
        print("Shutting down")

    waypoint.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
