#!/usr/bin/env python
import gym
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from mir100_env import MiR100Env
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

class MiR100Env(gym.Env):
    def __init__(self):
        super(MiR100Env, self).__init__()
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-0.5, high=0.3, shape=(1,), dtype=np.float32)
        self.v_max = 0.5  # Maximum linear velocity
        self.wheelbase = 0.5  # Wheelbase
        self.dt = 0.05  # Sampling period
        self.omega_max = 0.96  # Maximum angular velocity
        self.state = None
        self.path = None
        self.odom = None
        rospy.init_node('mir100_env', anonymous=True)
        self.pub_cmd_vel = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.sub_odom = rospy.Subscriber('/odom', Odometry, self.odom_callback)
        self.sub_path = rospy.Subscriber('/path', Path, self.path_callback)
        self.step_count = 0
        self.max_steps = 400

    def odom_callback(self, msg):
        self.odom = msg

    def path_callback(self, msg):
        self.path = msg

    def reset(self):
        self.state = np.array([np.random.uniform(-0.1, 0.1), np.random.uniform(-0.0873, 0.0873), 0, 0, 0])
        self.step_count = 0
        return self.state

    def step(self, action):
        self.step_count += 1
        v_dot = np.clip(action[0], -0.5, 0.3)
        v = np.clip(self.state[2] + v_dot * self.dt, 0, self.v_max)
        if self.odom and self.path:
            x = self.odom.pose.pose.position.x
            y = self.odom.pose.pose.position.y
            orientation = self.odom.pose.pose.orientation
            _, _, psi = euler_from_quaternion([orientation.x, orientation.y, orientation.z, orientation.w])
            omega = pure_pursuit(x, y, psi, self.path, d=0.2, v=v)
            e_p, psi_e, psi_e2 = compute_errors(x, y, psi, self.path)
            self.state = np.array([e_p, psi_e, v, omega, psi_e2])
            reward = -5.0 * abs(e_p) + 2.5 * v * (1 - abs(e_p) / 0.2) - 0.2 * (v < 1e-6)
            cmd = Twist()
            cmd.linear.x = v
            cmd.angular.z = omega
            self.pub_cmd_vel.publish(cmd)
        else:
            reward = 0
            self.state = np.zeros(5)
        done = self.step_count >= self.max_steps
        return self.state, reward, done, {}
#!/usr/bin/env python
import torch
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from mir100_env import MiR100Env

# Kiểm tra GPU
print(f"GPU available: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

# Khởi tạo môi trường
env = MiR100Env()

# Tạo mô hình SAC
model = SAC(
    "MlpPolicy",
    env,
    learning_rate=3e-4,
    buffer_size=500000,
    learning_starts=5000,
    batch_size=256,
    tau=0.005,
    gamma=0.99,
    ent_coef="auto",
    verbose=1,
    tensorboard_log="/home/duc/catkin_ws/src/mir100_sac/tensorboard/",
    device="cuda" if torch.cuda.is_available() else "cpu"
)

# Callback lưu checkpoint
checkpoint_callback = CheckpointCallback(save_freq=10000, save_path='/home/duc/catkin_ws/src/mir100_sac/checkpoints/', name_prefix='sac_mir100')

# Huấn luyện
model.learn(total_timesteps=5000, callback=checkpoint_callback)

# Lưu mô hình
model.save("/home/duc/catkin_ws/src/mir100_sac/models/sac_mir100")

# Xuất ONNX
# Xuất ONNX
policy = model.policy
policy.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
policy.actor.to(device)  # Ensure actor is on the correct device
dummy_input = torch.randn(1, 5, dtype=torch.float32).to(device)  # Move dummy_input to the same device
torch.onnx.export(
    policy.actor,
    dummy_input,
    "/home/duc/catkin_ws/src/mir100_sac/models/sac_mir100.onnx",
    opset_version=11,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
)
print("Đã xuất mô hình ONNX: /home/duc/catkin_ws/src/mir100_sac/models/sac_mir100.onnx")