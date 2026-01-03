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
    """
    def __init__(self):
        super().__init__('turtlebot_env_node')

        # --- 通信のセットアップ ---
        # 速度指令パブリッシャー
#        self.vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.vel_pub = self.create_publisher(Twist, '/cmd_vel', 50)
        # Lidarサブスクライバー
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        
        # Gazebo制御用クライアント (物理演算の一時停止・再開・リセット)
        self.unpause = self.create_client(Empty, '/unpause_physics')
        self.pause = self.create_client(Empty, '/pause_physics')
        self.reset_sim = self.create_client(Empty, '/reset_simulation')
        
        # 受信した最新のスキャンデータを保持する変数
        self.last_scan = None
        
        # 状態空間の次元数（DeepQのnetwork_inputsと一致させる必要があります）
        self.input_dim = 100 

    def scan_callback(self, msg):
        """Lidarデータを受信したときに呼ばれるコールバック関数"""
        self.last_scan = msg

    def call_service(self, client):
        """
        サービス呼び出し用ヘルパー関数。
        通信待ちでフリーズしないよう、タイムアウトとFutureを使用します。
        """
        if not client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn(f'Service {client.srv_name} not available')
            return
        
        req = Empty.Request()
        future = client.call_async(req)
        # 完了するまでスピン（最大1秒）
        rclpy.spin_until_future_complete(self, future, timeout_sec=1.0)

    def process_scan(self, scan_msg):
        """
        生のLidarデータをニューラルネットワークの入力形式に変換します。
        """
        if scan_msg is None:
            return np.zeros(self.input_dim)
            
        # 無限遠（検知なし）を3.5mに置換
        state = np.array(scan_msg.ranges)
        state[np.isinf(state)] = 3.5
        state[np.isnan(state)] = 3.5
        
        # データのダウンサンプリング（全データを100個に間引く）
        # これにより計算量を減らし、学習を安定させます
        indices = np.linspace(0, len(state)-1, self.input_dim).astype(int)
        state_reduced = state[indices]
        
        return state_reduced

    def step(self, action):
        """
        1ステップ実行します: 行動 -> 物理進行 -> 観測 -> 停止
        
        Returns:
            state (np.array): 次の状態
            reward (float): 報酬
            done (bool): 終了判定（衝突など）
        """
        # --- PHASE 1: 物理演算の再開と行動の送信 ---
        self.last_scan = None # 古いデータを捨てる
        self.call_service(self.unpause)
        
        # 行動の決定（離散アクション）
        vel = Twist()
        if action == 0:   # 前進
            vel.linear.x = 0.2
        elif action == 1: # 左回転
            vel.linear.x = 0.05
            vel.angular.z = 0.3
        elif action == 2: # 右回転
            vel.linear.x = 0.05
            vel.angular.z = -0.3
            
        self.vel_pub.publish(vel)

        # --- PHASE 2: 新しいセンサーデータの待機 ---
        # 物理演算が動いている間に、次のLidarデータが来るのを待ちます
        start_wait = time.time()
        while self.last_scan is None and rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.01)
            # 万が一データが来ない場合のタイムアウト処理（オプション）
            if time.time() - start_wait > 0.5:
                break
        
        # --- PHASE 3: 物理演算の停止 ---
        # データを受け取ったら即座にシミュレーションを止め、計算時間を確保します
        self.call_service(self.pause)

        # --- PHASE 4: 状態の処理と報酬計算 ---
        state_reduced = self.process_scan(self.last_scan)
        
        # 衝突判定: 最も近い障害物が0.25m未満なら衝突とみなす
        min_distance = np.min(state_reduced)
        done = bool(min_distance < 0.25)
        
        # 報酬関数の定義
        if not done:
            if action == 0:
                reward = 1.0  # 前進できているならプラス
            else:
                reward = -0.05 # 回転は少しペナルティ（直進を推奨するため）
        else:
            reward = -100.0 # 衝突時は大きなペナルティ
            
        return state_reduced, reward, done

    def reset(self):
        """
        エピソード開始時のリセット処理
        """
        # 1. 物理を動かしてからリセット（ROS 2 Gazeboのバグ回避のおまじない）
        self.call_service(self.unpause)
        self.call_service(self.reset_sim)
        
        # 2. リセット直後のデータが来るのを待つ
        self.last_scan = None
        while self.last_scan is None and rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.05)
            
        # 3. データ取得後に停止
        self.call_service(self.pause)
        
        return self.process_scan(self.last_scan)