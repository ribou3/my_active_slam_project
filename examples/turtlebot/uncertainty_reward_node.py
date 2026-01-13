import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseWithCovarianceStamped
import numpy as np

class UncertaintyRewardNode(Node):
    def __init__(self):
        super().__init__('uncertainty_reward_node')
        # SLAM Toolboxが出力する共分散付きポーズを購読
        self.subscription = self.create_subscription(
            PoseWithCovarianceStamped,
            '/pose', # slam_toolboxの設定により /slam_toolbox/pose 等になる場合あり
            self.pose_callback,
            10)
        self.latest_covariance = None

    def pose_callback(self, msg):
        # 6x6の共分散行列 (flattened 36 elements)
        cov = msg.pose.covariance
        # x, y, yaw(z軸回転) の 3x3 行列を抽出
        # 指標: [0]=x, [1]=xy, [5]=x_yaw, [6]=yx, [7]=y, [11]=y_yaw, [30]=yaw_x, [31]=yaw_y, [35]=yaw
        indices = [0, 1, 5, 6, 7, 11, 30, 31, 35]
        self.latest_covariance = np.array([cov[i] for i in indices]).reshape(3, 3)

    def calculate_d_opt(self):
        if self.latest_covariance is None:
            return 0.0
        # D-最適性基準の計算 (行列式)
        det = np.linalg.det(self.latest_covariance)
        # 非常に小さい値になるため、対数等でスケーリングするのが論文流
        return np.log(1.0 + det)