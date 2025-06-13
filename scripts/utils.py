import numpy as np
from tf.transformations import euler_from_quaternion

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