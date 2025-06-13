#!/usr/bin/env python
import gym
import numpy as np
import rospy
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import Twist
from tf.transformations import euler_from_quaternion
# from scripts.pure_pursuit import pure_pursuit
# from scripts.utils import compute_errors
import numpy as np

def pure_pursuit(x, y, psi, path, d=0.2, v=0.5):
    """
    Compute angular velocity using Pure Pursuit.
    Args:
        x, y, psi: Current position and orientation of the robot
        path: nav_msgs/Path, reference path
        d: Look-ahead distance (m)
        v: Current linear velocity (m/s)
    Returns:
        omega: Angular velocity (rad/s)
    """
    min_dist = float('inf')
    closest_idx = 0
    for i, pose in enumerate(path.poses):
        px, py = pose.pose.position.x, pose.pose.position.y
        dist = np.sqrt((x - px)**2 + (y - py)**2)
        if dist < min_dist:
            min_dist = dist
            closest_idx = i
    
    arc_length = 0
    lookahead_idx = closest_idx
    for i in range(closest_idx, len(path.poses)-1):
        p1 = path.poses[i].pose.position
        p2 = path.poses[i+1].pose.position
        arc_length += np.sqrt((p2.x - p1.x)**2 + (p2.y - p1.y)**2)
        if arc_length >= d:
            lookahead_idx = i + 1
            break
    
    px, py = path.poses[lookahead_idx].pose.position.x, path.poses[lookahead_idx].pose.position.y
    alpha = np.arctan2(py - y, px - x) - psi
    omega = 2 * v * np.sin(alpha) / d
    return np.clip(omega, -0.96, 0.96)  # omega_max = 0.96 rad/s

def compute_errors(x, y, psi, path):
    """
    Compute cross-track error (e_p), orientation error (psi_e), and psi_e2.
    Args:
        x, y, psi: Current position and orientation of the robot
        path: nav_msgs/Path
    Returns:
        e_p, psi_e, psi_e2
    """
    min_dist = float('inf')
    closest_idx = 0
    closest_point = None
    for i, pose in enumerate(path.poses):
        px, py = pose.pose.position.x, pose.pose.position.y
        dist = np.sqrt((x - px)**2 + (y - py)**2)
        if dist < min_dist:
            min_dist = dist
            closest_idx = i
            closest_point = (px, py)
    
    if closest_idx < len(path.poses)-1:
        p1 = path.poses[closest_idx].pose.position
        p2 = path.poses[closest_idx+1].pose.position
        tangent = np.array([p2.x - p1.x, p2.y - p1.y])
        tangent = tangent / np.linalg.norm(tangent)
    else:
        tangent = np.array([1, 0])
    
    d = np.array([x - closest_point[0], y - closest_point[1]])
    e_p = d[0] * tangent[1] - d[1] * tangent[0]
    
    psi_ref = np.arctan2(tangent[1], tangent[0])
    psi_e = psi - psi_ref
    psi_e = np.arctan2(np.sin(psi_e), np.cos(psi_e))
    
    arc_length = 0
    lookahead_idx = closest_idx
    for i in range(closest_idx, len(path.poses)-1):
        p1 = path.poses[i].pose.position
        p2 = path.poses[i+1].pose.position
        arc_length += np.sqrt((p2.x - p1.x)**2 + (p2.y - p1.y)**2)
        if arc_length >= 0.2:
            lookahead_idx = i + 1
            break
    if lookahead_idx < len(path.poses)-1:
        p1 = path.poses[lookahead_idx].pose.position
        p2 = path.poses[lookahead_idx+1].pose.position
        tangent = np.array([p2.x - p1.x, p2.y - p1.y])
        tangent = tangent / np.linalg.norm(tangent)
        psi_e2 = np.arctan2(tangent[1], tangent[0])
    else:
        psi_e2 = psi_ref
    
    return e_p, psi_e, psi_e2
#!/usr/bin/env python
import gymnasium as gym
import numpy as np
import rospy
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import Twist
from tf.transformations import euler_from_quaternion
from pure_pursuit import pure_pursuit
from utils import compute_errors

class MiR100Env(gym.Env):
    def __init__(self):
        super(MiR100Env, self).__init__()
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-0.5, high=0.3, shape=(1,), dtype=np.float32)
        self.v_max = 0.5
        self.wheelbase = 0.5
        self.dt = 0.02  # 50 Hz
        self.omega_max = 0.96
        self.state = None
        self.path = None
        self.odom = None
        self.prev_path = None
        rospy.init_node('mir100_env', anonymous=True)
        self.pub_cmd_vel = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.sub_odom = rospy.Subscriber('/odom', Odometry, self.odom_callback)
        self.sub_path = rospy.Subscriber('/path', Path, self.path_callback)
        self.step_count = 0
        self.max_steps = 1000  # 50 giây
        self.goal_threshold = 1.0

    def odom_callback(self, msg):
        self.odom = msg

    def path_callback(self, msg):
        if not msg.poses:
            rospy.logwarn("Nhận path rỗng")
            return
        if self.path is None or len(msg.poses) != len(self.path.poses):
            self.prev_path = self.path
            self.path = msg

    def is_goal_reached(self):
        if self.odom and self.path and self.path.poses:
            x_robot = self.odom.pose.pose.position.x
            y_robot = self.odom.pose.pose.position.y
            x_goal = self.path.poses[-1].pose.position.x
            y_goal = self.path.poses[-1].pose.position.y
            distance = np.sqrt((x_robot - x_goal)**2 + (y_robot - y_goal)**2)
            return distance < self.goal_threshold
        return False

    def reset(self, seed=None, options=None):
        self.state = np.array([np.random.uniform(-0.1, 0.1), np.random.uniform(-0.0873, 0.0873), 0, 0, 0])
        self.step_count = 0
        self.prev_path = None
        return self.state, {}

    def step(self, action):
        self.step_count += 1
        v_dot = np.clip(action[0], -0.5, 0.3)
        v = np.clip(self.state[2] + v_dot * self.dt, 0, self.v_max)
        if self.odom and self.path and self.path.poses:
            x = self.odom.pose.pose.position.x
            y = self.odom.pose.pose.position.y
            orientation = self.odom.pose.pose.orientation
            _, _, psi = euler_from_quaternion([orientation.x, orientation.y, orientation.z, orientation.w])
            omega = pure_pursuit(x, y, psi, self.path, d=0.3, v=v)
            e_p, psi_e, psi_e2 = compute_errors(x, y, psi, self.path)
            self.state = np.array([e_p, psi_e, v, omega, psi_e2])
            reward = -2.0 * abs(e_p) + 4.0 * v * (1 - abs(e_p) / 0.3) - 0.1 * (v < 1e-6)
            if self.is_goal_reached():
                reward += 200.0
            cmd = Twist()
            cmd.linear.x = v
            cmd.angular.z = omega
            self.pub_cmd_vel.publish(cmd)
        else:
            reward = -0.1
            self.state = np.zeros(5)
            rospy.logwarn("Thiếu odom hoặc path hoặc path rỗng")
        done = self.step_count >= self.max_steps or self.is_goal_reached()
        return self.state, reward, done, False, {}

if __name__ == '__main__':
    try:
        env = MiR100Env()
        env.reset()
    except rospy.ROSInterruptException:
        pass