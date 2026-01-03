#!/usr/bin/env python3
import os
import sys
import rclpy
from rclpy.node import Node
import numpy as np
import time
import csv

from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from std_srvs.srv import Empty

# CPU学習モード（同期の不確実性を減らすため）
os.environ['CUDA_VISIBLE_DEVICES'] = "-1"
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
import deepq

class TurtlebotDqnNode(Node):
    def __init__(self):
        super().__init__('turtlebot_dqn_node')
        self.vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.unpause = self.create_client(Empty, '/unpause_physics')
        self.pause = self.create_client(Empty, '/pause_physics')
        self.reset_sim = self.create_client(Empty, '/reset_simulation')
        self.last_scan = None

    def scan_callback(self, msg):
        self.last_scan = msg

    def call_service(self, client):
        """タイムアウト付きでサービスを呼び出し、デッドロックを防ぐ"""
        if not client.wait_for_service(timeout_sec=1.0):
            return
        req = Empty.Request()
        future = client.call_async(req)
        # 最大1秒待つ（これで衝突時のフリーズを回避）
        rclpy.spin_until_future_complete(self, future, timeout_sec=1.0)

    def step(self, action):
        # --- PHASE 1: ポーズ解除と物理進行 ---
        print(f"\n[STEP START] Action: {action}")
        self.last_scan = None
        self.call_service(self.unpause)
        print(">> Gazebo Unpaused. Moving...")
        
        vel = Twist()
        if action == 0: vel.linear.x = 0.2
        elif action == 1: vel.linear.x = 0.05; vel.angular.z = 0.3
        elif action == 2: vel.linear.x = 0.05; vel.angular.z = -0.3
        
        # 速度命令を送信
        self.vel_pub.publish(vel)

        # 新しいスキャンが届くまで、物理を進めながら待機
        start_wait = time.time()
        while self.last_scan is None and rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.01)
        
        # --- PHASE 2: 即座にポーズ（計算時間を確保） ---
        self.call_service(self.pause)
        wait_duration = time.time() - start_wait
        print(f">> New Scan Received (after {wait_duration:.3f}s). Gazebo Paused.")

        # --- PHASE 3: 状態確定と報酬計算（物理は止まっている） ---
        data = self.last_scan
        state = np.array(data.ranges)
        state[np.isinf(state)] = 3.5
        indices = np.linspace(0, len(state)-1, 100).astype(int)
        state_reduced = state[indices]
        
        done = bool(np.min(state_reduced) < 0.25)
        reward = 1.0 if not done and action == 0 else (0.05 if not done else -100.0)
        
        print(f"[STEP END] Min Lidar: {np.min(state_reduced):.3f} | Done: {done}")
        return state_reduced, reward, done

    def reset(self):
        # 1. 物理を unpause してから reset するのが ROS2 でフリーズしないコツ
        self.call_service(self.unpause)
        self.call_service(self.reset_sim)
        
        # 2. unpauseせず、配置された瞬間のスキャンだけを待つ
        self.last_scan = None
        while self.last_scan is None and rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.05)
            
        # 3. 拾ったら即座に止める
        self.call_service(self.pause)
        
        state = np.array(self.last_scan.ranges)
        state[np.isinf(state)] = 3.5
        indices = np.linspace(0, len(state)-1, 100).astype(int)
        return state[indices]

def main():
    rclpy.init()
    node = TurtlebotDqnNode()
    
    # ログ初期化
    step_log = "step_detailed_log.csv"
    with open(step_log, "w", newline='') as f:
        csv.writer(f).writerow(["epoch", "step", "min_lidar", "action", "q_fwd", "q_left", "q_right", "reward"])

    # 元のROS1設定: minibatch_size=64, learnStart=64, learningRate=0.00025
    minibatch_size = 64
    learnStart = 64
    updateTargetNetwork = 10000
    
    dqn = deepq.DeepQ(100, 3, 1000000, 0.99, 0.00025, learnStart)
    dqn.initNetworks([24, 24])
    
    explorationRate = 1.0 
    stepCounter = 0
    
    try:
        for epoch in range(1, 1001):
            state = node.reset()
            print(f"--- Epoch {epoch} Start | Initial Min Lidar: {np.min(state):.3f} ---")
            cum_reward = 0
            
            for t in range(500):
                q_values = dqn.getQValues(state)
                q_list = np.array(q_values).flatten().tolist()
                action = dqn.selectAction(q_values, explorationRate)
                
                new_state, reward, done = node.step(action)
                dqn.addMemory(state, action, reward, new_state, done)
                
                # 学習開始 (ミニバッチサイズを64に修正)
                if stepCounter >= learnStart:
                    use_target = (stepCounter > updateTargetNetwork)
                    dqn.learnOnMiniBatch(minibatch_size, use_target)
                
                with open(step_log, "a", newline='') as f:
                    csv.writer(f).writerow([epoch, t, f"{np.min(new_state):.3f}", action, f"{q_list[0]:.4f}", f"{q_list[1]:.4f}", f"{q_list[2]:.4f}", reward])
                
                state = new_state
                cum_reward += reward
                stepCounter += 1
                
                if stepCounter % updateTargetNetwork == 0:
                    dqn.updateTargetNetwork()
                    print("Updating target network...")
                
                if done: break
            
            # Epsilon減衰 (ROS1再現)
            explorationRate = max(0.05, explorationRate * 0.995)
            print(f"Epoch {epoch:4d} Result | Reward: {cum_reward:7.1f} | Eps: {explorationRate:.3f} | Steps: {t+1}")

    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()