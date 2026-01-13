#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from std_srvs.srv import Empty
import numpy as np
import time

class TurtlebotEnv(Node):
    """
    ROS 2 と Gazebo のインターフェースを担当するクラス。
    OpenAI Gym のような step(), reset() メソッドを提供します。
    論文の仕様に基づき、0.1秒ごとに一時停止して思考する「離散時間ステップ」を実装しています。
    """
    def __init__(self):
        super().__init__('turtlebot_env_node')

        # --- 通信のセットアップ ---
        self.vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        
        self.unpause = self.create_client(Empty, '/unpause_physics')
        self.pause = self.create_client(Empty, '/pause_physics')
        self.reset_sim = self.create_client(Empty, '/reset_simulation')
        
        self.last_scan = None
        self.input_dim = 100 

        self.get_logger().info('TurtlebotEnv Node has been initialized.')

    def scan_callback(self, msg):
        self.last_scan = msg

    def call_service(self, client):
        if not client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn(f'Service {client.srv_name} not available')
            return
        req = Empty.Request()
        future = client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=1.0)

    # --- 以前の360度ダウンサンプリング（参考用としてコメントアウト） ---
    # def process_scan(self, scan_msg):
    #     if scan_msg is None: return np.zeros(self.input_dim)
    #     state = np.array(scan_msg.ranges)
    #     state[np.isinf(state)] = 3.5
    #     state[np.isnan(state)] = 3.5
    #     indices = np.linspace(0, len(state)-1, self.input_dim).astype(int)
    #     state_reduced = state[indices]
    #     return state_reduced

    def process_scan(self, scan_msg):
        """
        論文準拠：前方180度のみを抽出し、左右の並びを正正。
        論文では前方の180度視野（FOV）に等間隔に配置されたn個のレーザー（n=100）を使用します。
        """
        if scan_msg is None: return np.zeros(self.input_dim)
        
        raw_ranges = np.array(scan_msg.ranges)
        
        # TB3のインデックス: 0(前), 90(左), 180(後), 270(右)
        # 配列を [左90度 ... 正面(0) ... 右90度] の順に並べることでNNに左右を教えます。
        indices_180 = np.concatenate([
            np.arange(90, -1, -1),   # 左90度から正面(0)へ
            np.arange(359, 269, -1)  # 正面から右90度(270)へ
        ])
        state_180 = raw_ranges[indices_180]
        
        # 無限遠などを 3.5m に置換
        state_180[np.isinf(state_180)] = 3.5
        state_180[np.isnan(state_180)] = 3.5
        
        # 100個にダウンサンプリング
        resampled_indices = np.linspace(0, len(state_180)-1, self.input_dim).astype(int)
        return state_180[resampled_indices]

    def check_collision(self, state_reduced):
        valid_distances = state_reduced[state_reduced > 0.13]
        if len(valid_distances) > 0:
            min_dist = np.min(valid_distances)
            # 論文準拠の衝突閾値 0.2m
            return bool(min_dist < 0.20), min_dist
        return False, 3.5

    def step(self, action):
        """
        1ステップ実行: 物理演算再開 -> 0.1秒進行 -> 停止 -> 観測
        """
        # --- PHASE 1: 物理演算の再開と行動の送信 ---
        self.call_service(self.unpause)
        
        vel = Twist()
        # 論文の設定値に近い速度 a1=0.3, a2=a3=0.05
        if action == 0:   # 前進
            vel.linear.x = 0.3 
        elif action == 1: # 左回転
            vel.linear.x = 0.05
            vel.angular.z = 0.3
        elif action == 2: # 右回転
            vel.linear.x = 0.05
            vel.angular.z = -0.3
            
        self.vel_pub.publish(vel)

        # --- PHASE 2: 固定時間の進行 (シミュレーション時刻ベース) ---
        # Gazeboの倍速（RTF）に関わらず、正確にシミュレーション上の0.1秒分だけ移動させます。
        # 論文のアクション適用時間は0.1秒です。
        start_sim_time = self.get_clock().now()
        while True:
            rclpy.spin_once(self, timeout_sec=0.001)
            current_sim_time = self.get_clock().now()
            elapsed_sim_time = (current_sim_time - start_sim_time).nanoseconds / 1e9
            
            # 0.1秒に戻しました（0.05秒だとRTF6倍速環境では制御が間に合わないため）
            if elapsed_sim_time >= 0.1:
                break
        
        # --- PHASE 3: 物理演算の停止 ---
        self.call_service(self.pause)

        # --- PHASE 4: 状態の処理と報酬計算 ---
        state_reduced = self.process_scan(self.last_scan)
        done, min_dist = self.check_collision(state_reduced)
        
        # 報酬関数の定義（論文準拠）
        if not done:
            if action == 0:
                reward = 1.0     # 前進は+1.0
            else:
                reward = -0.05   # 回転は-0.05
        else:
            reward = -100.0      # 衝突は-100.0
            
        return state_reduced, reward, done

    def reset(self):
        """
        エピソード開始時のリセット処理。
        """
        self.call_service(self.unpause)
        self.call_service(self.reset_sim)
        
        stop_vel = Twist()
        self.vel_pub.publish(stop_vel)
        
        self.last_scan = None
        while self.last_scan is None and rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.05)
        
        self.call_service(self.pause)
        return self.process_scan(self.last_scan)