#!/usr/bin/env python3
import os
import sys
import rclpy
import numpy as np
import csv
import time
import gc

# 分割した環境クラスをインポート
# 同じディレクトリに turtlebot_env.py がある前提です
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from turtlebot_env import TurtlebotEnv
import deepq # 既存のDeepQライブラリ

#@profile
def main():
    # --- ROS 2 の初期化 ---
    rclpy.init()
    
    # 環境インスタンスの作成
    env = TurtlebotEnv()
    
    # --- ハイパーパラメータの設定 (ROS1の設定を踏襲) ---
    epochs = 1000               # 総エピソード数
#    steps_per_epoch = 500       # 1エピソードあたりの最大ステップ数
    steps_per_epoch = 2000       # 1エピソードあたりの最大ステップ数
    minibatch_size = 64         # 学習時のバッチサイズ
    learnStart = 64             # 学習開始までに貯めるメモリ数
#    updateTargetNetwork = 10000 # ターゲットネットワークを更新する頻度
    updateTargetNetwork = 2000 # ターゲットネットワークを更新する頻度
    learningRate = 0.00025      # 学習率
    discountFactor = 0.99       # 割引率
    # memorySize = 1000000        # リプレイバッファサイズ
    memorySize = 30000
    network_inputs = 100        # Lidarの入力次元数
    network_outputs = 3         # アクション数 (前進、左、右)
    hidden_layers = [24, 24]    # 隠れ層の構造
    
    explorationRate = 1.0       # イプシロン（探索率）の初期値
    explorationDecay = 0.995    # イプシロンの減衰率
    min_exploration = 0.05      # イプシロンの最小値

    # --- DQNエージェントの初期化 ---
    dqn = deepq.DeepQ(network_inputs, network_outputs, memorySize, discountFactor, learningRate, learnStart)
    dqn.initNetworks(hidden_layers)
    
    # ログファイルの準備
    step_log = "ros2_training_log.csv"
    with open(step_log, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "step", "min_lidar", "action", "reward", "q_max", "loss_placeholder"])

    stepCounter = 0 # 全エピソード通算のステップ数

    try:
        print("--- Training Start ---")
        
        for epoch in range(1, epochs + 1):
            # エピソードの初期化
            state = env.reset()
            cumulated_reward = 0
            print(f"Current memory size: {dqn.memory.getCurrentSize()} currentPosition:{dqn.memory.currentPosition}") #現在のサイズ
            
            # ステップループ
            for t in range(steps_per_epoch):
                print(f"Step {t}: Getting Q values...") # 追加
                # 1. 行動の選択 (ε-greedy)
                qValues = dqn.getQValues(state)
                action = dqn.selectAction(qValues, explorationRate)
                
                print(f"Step {t}: Acting...") # 追加
                # 2. 環境の実行 (Envクラスに任せる)
                new_state, reward, done = env.step(action)
                
                # 3. メモリへの保存
                dqn.addMemory(state, action, reward, new_state, done)
                
                # 4. 学習 (Experience Replay)
                if stepCounter >= learnStart:
                    # 一定間隔でターゲットネットワークを使用するか切り替え
                    useTarget = (stepCounter > updateTargetNetwork)
                    dqn.learnOnMiniBatch(minibatch_size, useTarget)
                
                # 5. ターゲットネットワークの同期
                if stepCounter % updateTargetNetwork == 0:
                    dqn.updateTargetNetwork()
                    print(f"Update Target Network at step {stepCounter}")

                # ログ保存 (CSV)
                with open(step_log, "a", newline='') as f:
                    csv.writer(f).writerow([epoch, t, f"{np.min(new_state):.3f}", action, reward, np.max(qValues)])
                
                # 次の状態へ更新
                state = new_state
                cumulated_reward += reward
                stepCounter += 1
                
                if done:
                    break # 衝突したらエピソード終了

            # エピソード終了後の処理
            # 探索率(epsilon)を減衰させる
            explorationRate *= explorationDecay
            explorationRate = max(min_exploration, explorationRate)
            
            print(f"Epoch: {epoch:04d} | Steps: {t+1:03d} | Reward: {cumulated_reward:.2f} | Eps: {explorationRate:.4f}")
            
            # 定期的にモデルを保存 (100エピソードごと)
            if epoch % 100 == 0:
                save_path = f"models/ros2_dqn_epoch_{epoch}.h5"
                # ディレクトリがなければ作成
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                dqn.saveModel(save_path)
                print(f"Model saved: {save_path}")
        gc.collect()

    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    finally:
        # 終了処理
        env.destroy_node()
        rclpy.shutdown()
        print("ROS 2 Shutdown complete.")

if __name__ == '__main__':
    main()