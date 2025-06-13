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