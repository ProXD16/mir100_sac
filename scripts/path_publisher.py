#!/usr/bin/env python
import rospy
import numpy as np
from nav_msgs.msg import Path, Odometry
from geometry_msgs.msg import PoseStamped
import yaml
import cv2

class PathPublisher:
    def __init__(self):
        rospy.init_node('path_publisher', anonymous=True)
        self.pub = rospy.Publisher('/path', Path, queue_size=10)
        self.rate = rospy.Rate(1)  # 1 Hz
        self.map_data = None
        self.map_resolution = None
        self.map_origin = None
        self.map_width = None
        self.map_height = None
        self.load_map()
        self.odom = None
        self.current_path = None
        self.sub_odom = rospy.Subscriber('/odom', Odometry, self.odom_callback)
        self.goal_threshold = 1.0
        self.path_published = False

    def load_map(self):
        map_file = rospy.get_param('~map_file', '/home/duc/catkin_ws/src/mir_robot/mir_gazebo/maps/maze.yaml')
        try:
            with open(map_file, 'r') as f:
                map_yaml = yaml.safe_load(f)
            self.map_resolution = map_yaml['resolution']
            self.map_origin = map_yaml['origin'][:2]
            map_img_path = map_yaml['image']
            if not map_img_path.startswith('/'):
                import os
                map_img_path = os.path.join(os.path.dirname(map_file), map_img_path)
            self.map_data = cv2.imread(map_img_path, cv2.IMREAD_GRAYSCALE)
            if self.map_data is None:
                raise ValueError(f"Không thể đọc file bản đồ: {map_img_path}")
            self.map_height, self.map_width = self.map_data.shape
        except Exception as e:
            rospy.logerr(f"Lỗi khi tải bản đồ: {str(e)}")
            raise

    def odom_callback(self, msg):
        self.odom = msg

    def is_free(self, x, y):
        px = int((x - self.map_origin[0]) / self.map_resolution)
        py = int((y - self.map_origin[1]) / self.map_resolution)
        if 0 <= px < self.map_width and 0 <= py < self.map_height:
            pixel_value = self.map_data[self.map_height - 1 - py, px]
            is_free = pixel_value > 150
            return is_free
        return False

    def is_goal_reached(self):
        if self.odom and self.current_path and self.current_path.poses:
            x_robot = self.odom.pose.pose.position.x
            y_robot = self.odom.pose.pose.position.y
            x_goal = self.current_path.poses[-1].pose.position.x
            y_goal = self.current_path.poses[-1].pose.position.y
            distance = np.sqrt((x_robot - x_goal)**2 + (y_robot - y_goal)**2)
            return distance < self.goal_threshold
        return False

    def generate_random_path(self, N_w=20, x_range=(0, 20), y_range=(0, 20)):
        waypoints = []
        if self.odom:
            x_start = self.odom.pose.pose.position.x
            y_start = self.odom.pose.pose.position.y
        else:
            x_start, y_start = 10, 10
        x_start = max(0, min(x_start, 20))
        y_start = max(0, min(y_start, 20))
        if self.is_free(x_start, y_start):
            waypoints.append((x_start, y_start))
        else:
            x_start, y_start = 10, 10
            if self.is_free(x_start, y_start):
                waypoints.append((x_start, y_start))
        
        last_x, last_y = x_start, y_start
        for _ in range(N_w - 1):
            attempts = 0
            max_attempts = 100
            while attempts < max_attempts:
                x = np.random.uniform(max(0, last_x - 1), min(20, last_x + 1))
                y = np.random.uniform(max(0, last_y - 1), min(20, last_y + 1))
                x = max(0, min(x, 20))  # Giới hạn trong bản đồ
                y = max(0, min(y, 20))
                if self.is_free(x, y):
                    waypoints.append((x, y))
                    last_x, last_y = x, y
                    break
                attempts += 1
            if attempts >= max_attempts:
                rospy.logwarn("Không tìm thấy điểm tự do")
                break
        return waypoints if waypoints else [(x_start, y_start)]

    def generate_eight_shaped_path(self, a=7.0, num_points=100):
        waypoints = []
        if self.odom:
            x_center = self.odom.pose.pose.position.x
            y_center = self.odom.pose.pose.position.y
        else:
            x_center, y_center = 10, 10
        x_center = max(0, min(x_center, 20))
        y_center = max(0, min(y_center, 20))
        for lambda_val in np.linspace(0, 2*np.pi, num_points):
            x = x_center + a * np.sin(lambda_val)
            y = y_center + a * np.sin(lambda_val) * np.cos(lambda_val)
            x = max(0, min(x, 20))  # Giới hạn trong bản đồ
            y = max(0, min(y, 20))
            if self.is_free(x, y):
                waypoints.append((x, y))
        return waypoints if waypoints else [(x_center, y_center)]

    def publish_path(self):
        while not rospy.is_shutdown():
            try:
                if self.current_path is None or self.is_goal_reached():
                    path_msg = Path()
                    path_msg.header.frame_id = "map"
                    path_msg.header.stamp = rospy.Time.now()
                    if np.random.random() < 0.5:
                        waypoints = self.generate_random_path()
                    else:
                        waypoints = self.generate_eight_shaped_path()
                    if waypoints:
                        for x, y in waypoints:
                            pose = PoseStamped()
                            pose.header.frame_id = "map"
                            pose.header.stamp = rospy.Time.now()
                            pose.pose.position.x = x
                            pose.pose.position.y = y
                            pose.pose.orientation.w = 1.0
                            path_msg.poses.append(pose)
                        self.current_path = path_msg
                        self.path_published = True
                    else:
                        rospy.logwarn("Không tạo được path")
                        self.current_path = None
                if self.current_path:
                    self.current_path.header.stamp = rospy.Time.now()
                    self.pub.publish(self.current_path)
                else:
                    rospy.logwarn("Không có path để xuất bản")
            except Exception as e:
                rospy.logerr(f"Lỗi publish_path: {str(e)}")
            self.rate.sleep()

if __name__ == '__main__':
    try:
        publisher = PathPublisher()
        publisher.publish_path()
    except rospy.ROSInterruptException:
        pass