import rclpy
from geometry_msgs.msg import Vector3
import numpy as np
from rclpy.node import Node
from serial import Serial
from sensor_msgs.msg import Imu, LaserScan


class Coordinate(Node):
    def __init__(self):
        super().__init__('pose_node')

        self.serial = Serial(
            "/dev/ttyAMA2",
            baudrate=38400,
            bytesize=8,
            parity='N',
            stopbits=1,
        )

        self.laser_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.laser_callback,
            10
        )

        self.imu_angle_sub = self.create_subscription(
            Vector3,
            '/imu/data',
            self.angle_callback,
            10
        )

        self.imu_sub = self.create_subscription(
            Imu,
            '/imu/data',
            self.imu_callback,
            10
        )

        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0

        self.accel_x = 0.0
        self.accel_y = 0.0
        self.accel_z = 0.0

        self.distance_0 = 0.0  # 0 degrees
        self.distance_90 = 0.0  # 90 degrees
        self.distance_180 = 0.0  # 180 degrees (left side)
        self.distance_270 = 0.0  # 270 degrees (back side)

        self.pub_timer = self.create_timer(0.1, self.timer_callback)

        self.last_position = [0.0, 0.0]
        self.last_velocity_x = 0.0
        self.last_velocity_y = 0.0
        self.last_time = self.get_clock().now()
        self.lidar_occluded = False
        self.lidar_position = [0.0, 0.0]

    def angle_callback(self, msg):
        self.roll = msg.x
        self.pitch = msg.y
        self.yaw = msg.z

    def imu_callback(self, msg):
        self.accel_x = msg.linear_acceleration.x
        self.accel_y = msg.linear_acceleration.y
        self.accel_z = msg.linear_acceleration.z

    def laser_callback(self, msg):
        ranges = msg.ranges
        angle_min = msg.angle_min
        angle_increment = msg.angle_increment

        # 定义需要提取的角度范围
        angles_to_extract = [0, 90, 180, 270]
        angle_tolerance = np.radians(2.5)  # 转换为弧度

        # 提取指定角度范围内的距离数据
        extracted_ranges = []
        extracted_angles = []

        for angle in angles_to_extract:
            target_angle_rad = np.radians(angle)

            # 找到最接近目标角度的索引
            index = int((target_angle_rad - angle_min) / angle_increment)

            # 检查索引是否在有效范围内
            if 0 <= index < len(ranges):
                # 提取目标角度附近的几个数据点
                for i in range(max(0, index - 1), min(len(ranges), index + 2)):
                    current_angle = angle_min + i * angle_increment
                    if abs(current_angle - target_angle_rad) <= angle_tolerance:
                        extracted_ranges.append(ranges[i])
                        extracted_angles.append(current_angle)

        # 将提取的激光雷达数据转换为笛卡尔坐标
        points = []
        for i, r in enumerate(extracted_ranges):
            x = r * np.cos(extracted_angles[i])
            y = r * np.sin(extracted_angles[i])
            points.append([x, y])

        # 使用IMU数据进行坐标变换
        rotation_matrix = self.euler_to_rotation_matrix(self.roll, self.pitch, self.yaw)
        transformed_points = []
        for point in points:
            transformed_point = np.dot(rotation_matrix, np.array(point))
            transformed_points.append(transformed_point)

        # 将转换后的激光雷达数据存储在 self.lidar_position 中
        self.lidar_position = transformed_points

        # 将提取的距离数据存储在对应的变量中
        self.distance_0 = extracted_ranges[0] if 0 in angles_to_extract and len(extracted_ranges) > 0 else self.distance_0
        self.distance_90 = extracted_ranges[1] if 90 in angles_to_extract and len(extracted_ranges) > 1 else self.distance_90
        self.distance_180 = extracted_ranges[2] if 180 in angles_to_extract and len(extracted_ranges) > 2 else self.distance_180
        self.distance_270 = extracted_ranges[3] if 270 in angles_to_extract and len(extracted_ranges) > 3 else self.distance_270

    @staticmethod
    def euler_to_rotation_matrix(roll, pitch, yaw):
        # 将欧拉角转换为旋转矩阵
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(roll), -np.sin(roll)],
                       [0, np.sin(roll), np.cos(roll)]])

        Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                       [0, 1, 0],
                       [-np.sin(pitch), 0, np.cos(pitch)]])

        Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                       [np.sin(yaw), np.cos(yaw), 0],
                       [0, 0, 1]])

        R = np.dot(Rz, np.dot(Ry, Rx))

        return R

    def timer_callback(self):
        self.get_logger().info(f'x={self.distance_270:.2f}, y={self.distance_180:.2f}')


def main(args=None):
    rclpy.init(args=args)
    coordinate = Coordinate()
    rclpy.spin(coordinate)
    coordinate.destroy_node()
    rclpy.shutdown()

