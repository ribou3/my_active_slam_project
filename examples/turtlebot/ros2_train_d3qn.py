#!/usr/bin/env python3
import os
import sys
import rclpy
import numpy as np
import csv
import time
import gc
from datetime import datetime
import matplotlib.pyplot as plt

# 環境設定
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from turtlebot_env import TurtlebotEnv
from d3qn_agent import D3QNAgent # さっき作成したクラス

def plot_progress(csv_path, save_path):
    """学習ログからグラフを作成して保存する"""
    try:
        data = np.genfromtxt(csv_path, delimiter=',', skip_header=1)
        if data.ndim == 1: return # データが足りない場合
        
        epochs = np.unique(data[:, 0])
        rewards = []
        q_maxs = []
        
        for e in epochs:
            epoch_data = data[data[:, 0] == e]
            rewards.append(np.sum(epoch_data[:, 4])) # Reward sum
            q_maxs.append(np.mean(epoch_data[:, 5])) # Q_max mean
            
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(epochs, rewards, label='Total Reward')
        plt.title('Reward per Episode')
        plt.xlabel('Epoch')
        plt.ylabel('Reward')
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(epochs, q_maxs, label='Avg Q-Max', color='orange')
        plt.title('Average Q-Value')
        plt.xlabel('Epoch')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
    except Exception as e:
        print(f"Plotting error: {e}")

def main():
    rclpy.init()
    env = TurtlebotEnv()
    
    # --- 実行IDの生成（重複防止） ---
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"training_output/{run_id}"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/models", exist_ok=True)
    
    log_file = f"{output_dir}/d3qn_log.csv"
    print(f"Training Run ID: {run_id}")
    print(f"Logs will be saved to: {output_dir}")

    # --- ハイパーパラメータ (論文準拠) ---
    # Applied Sciences 2020: "A Deep Reinforcement Learning Approach for Active SLAM"
    
    epochs = 500               # 十分な収束のため
    steps_per_epoch = 500      # 1エピソードの最大ステップ
    minibatch_size = 64         # Batch size
    
    # Paper parameters
    learningRate = 0.00025      # 5e-5 (論文値)
    updateTargetNetwork = 3000 # 10,000 steps (論文値)
    memorySize = 20000         # 100,000 steps (論文値)
    discountFactor = 0.99       
    
    # その他
    learnStart = 1000           # メモリを少し溜めてから開始
    network_inputs = 100        # Lidar input size
    network_outputs = 3         # Action size
    
    # Exploration parameters
    explorationRate = 1.0
    explorationDecay = 0.995
    min_exploration = 0.05 # 論文では0.1程度の場合もあるが0.05でOK

    # エージェント初期化
    agent = D3QNAgent(
        inputs=network_inputs,
        outputs=network_outputs,
        memorySize=memorySize,
        discountFactor=discountFactor,
        learningRate=learningRate,
        learnStart=learnStart
    )

    # ログヘッダー
    with open(log_file, 'w') as f:
        f.write('epoch,step,min_lidar,action,reward,q_max,loss\n')

    step_counter = 0

    try:
        for epoch in range(1, epochs + 1):
            # --- 修正ポイント1: reset() の戻り値を state として受け取る ---
            # これにより、np.zeros や t==0 の特殊処理が不要になります
            state = env.reset() 
            
            cumulated_reward = 0
            
            for t in range(steps_per_epoch):
                # 行動選択
                qValues = agent.getQValues(state)
                action = agent.selectAction(qValues, explorationRate)

                # 環境を進める
                new_state, reward, done = env.step(action)
                
                # 次の状態の次元チェック
                if len(new_state) != network_inputs:
                    print(f"Input size mismatch! Expected {network_inputs}, got {len(new_state)}")
                    break

                # Q値（記録用）
                q_max = np.max(qValues)

                # 経験をメモリに保存
                agent.addMemory(state, action, reward, new_state, done)

                # --- 修正ポイント：学習開始タイミングの制御 ---
                # learnStart(1000)溜まるまではメモリを貯めるだけにする
                if step_counter > learnStart:
                    agent.learnOnMiniBatch(minibatch_size)

                # --- 修正ポイント：ターゲットネットワーク更新の厳密化 ---
                # 0回目を除外し、かつ指定ステップ(10000)ごとに更新
                if step_counter > 0 and step_counter % updateTargetNetwork == 0:
                    agent.updateTargetNetwork()
                    print(f"--- Target Network Updated at Step {step_counter} ---")

                # ログ保存 (CSV)
                with open(log_file, "a", newline='') as f:
                    csv.writer(f).writerow([epoch, t, f"{np.min(new_state):.3f}", action, reward, q_max, 0])

                # 状態を更新して次のステップへ
                state = new_state
                cumulated_reward += reward
                step_counter += 1

                # 衝突したらエピソード終了
                if done:
                    break

            # エポック終了後の処理（探索率の減衰）
            explorationRate *= explorationDecay
            explorationRate = max(min_exploration, explorationRate)
            
            print(f"[{run_id}] Epoch: {epoch:04d} | Steps: {t+1:03d} | Reward: {cumulated_reward:.2f} | Eps: {explorationRate:.4f} | Total Steps: {step_counter}")
            # モデル保存 (100エピソードごと)
            if epoch % 100 == 0:
                save_path = f"{output_dir}/models/d3qn_epoch_{epoch}.h5"
                agent.saveModel(save_path)
                print(f"Model saved: {save_path}")
                
                # グラフ出力 (100エピソードごと)
                plot_path = f"{output_dir}/progress_epoch_{epoch}.png"
                plot_progress(log_file, plot_path)
                print(f"Graph updated: {plot_path}")

            gc.collect()

    except KeyboardInterrupt:
        print("\nTraining interrupted.")
    finally:
        agent.saveModel(f"{output_dir}/models/d3qn_interrupted.h5")
        env.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()