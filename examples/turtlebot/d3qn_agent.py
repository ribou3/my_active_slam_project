import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense, Lambda, Add
from tensorflow.keras.optimizers import RMSprop
import os
import memory

class D3QNAgent:
    """
    Dueling Double DQN Agent
    - Dueling Network: V(s) + A(s,a) architecture
    - Double DQN: Decouples action selection from value estimation
    """
    def __init__(self, inputs, outputs, memorySize, discountFactor, learningRate, learnStart):
        self.input_size = inputs
        self.output_size = outputs
        self.memory = memory.Memory(memorySize)
        self.discountFactor = discountFactor
        self.learningRate = learningRate
        self.learnStart = learnStart
        
        # モデルの構築
        self.model = self.build_dueling_model()
        self.target_model = self.build_dueling_model()
        self.updateTargetNetwork()

    def build_dueling_model(self):
        """Dueling Network Architecture"""
        input_layer = Input(shape=(self.input_size,))
        
        # 共通の隠れ層 (Feature Extractor)
        x = Dense(64, activation='relu', kernel_initializer='he_uniform')(input_layer)
        x = Dense(64, activation='relu', kernel_initializer='he_uniform')(x)
        
        # Branch 1: State Value V(s)
        # 状態そのものの価値（スカラー）
        v_stream = Dense(32, activation='relu', kernel_initializer='he_uniform')(x)
        V = Dense(1, kernel_initializer='he_uniform')(v_stream)
        
        # Branch 2: Advantage A(s, a)
        # 各行動の相対的な価値
        a_stream = Dense(32, activation='relu', kernel_initializer='he_uniform')(x)
        A = Dense(self.output_size, kernel_initializer='he_uniform')(a_stream)
        
        # Combine: Q(s, a) = V(s) + (A(s, a) - mean(A(s, a)))
        # Lambda層を使って計算グラフ内で結合
        def combine_v_a(tensors):
            v, a = tensors
            return v + (a - tf.math.reduce_mean(a, axis=1, keepdims=True))
            
        Q = Lambda(combine_v_a)([V, A])
        
        model = Model(inputs=input_layer, outputs=Q)
        model.compile(loss='mse', optimizer=RMSprop(learning_rate=self.learningRate))
        return model

    def updateTargetNetwork(self):
        """Hard update: Copy weights from model to target_model"""
        self.target_model.set_weights(self.model.get_weights())

    def getQValues(self, state):
        # バッチ次元を追加して予測
        return self.model.predict(state.reshape(1, self.input_size), verbose=0)[0]

    def selectAction(self, qValues, explorationRate):
        if np.random.rand() < explorationRate:
            return np.random.randint(0, self.output_size)
        else:
            return np.argmax(qValues)

    def addMemory(self, state, action, reward, newState, isFinal):
        self.memory.addMemory(state, action, reward, newState, isFinal)

    def learnOnMiniBatch(self, batch_size):
        if self.memory.getCurrentSize() < self.learnStart:
            return

        # メモリからサンプリング
        miniBatch = self.memory.getMiniBatch(batch_size)
        
        states = np.array([sample['state'] for sample in miniBatch])
        next_states = np.array([sample['newState'] for sample in miniBatch])
        
        # 1. Double DQN Logic
        # 次の状態での行動選択 -> メインネットワーク (theta)
        # 次の状態の価値評価 -> ターゲットネットワーク (theta-)
        
        # メインモデルで次の状態のQ値を予測し、最大の行動(argmax)を取得
        q_next_main = self.model.predict(next_states, verbose=0)
        best_actions = np.argmax(q_next_main, axis=1)
        
        # ターゲットモデルで次の状態のQ値を予測
        q_next_target = self.target_model.predict(next_states, verbose=0)
        
        # 学習用データ作成
        x_batch = states
        y_batch = self.model.predict(states, verbose=0) # 現在の予測値をベースにする
        
        for i, sample in enumerate(miniBatch):
            action = sample['action']
            reward = sample['reward']
            isFinal = sample['isFinal']
            
            # Double DQN: Target = r + gamma * Q_target(s', argmax Q_main(s', a))
            if isFinal:
                target_val = reward
            else:
                # ターゲットモデルのQ値から、メインモデルが選んだ行動の値を採用する
                target_val = reward + self.discountFactor * q_next_target[i][best_actions[i]]
            
            # 実際に取った行動のQ値だけを更新
            y_batch[i][action] = target_val
            
        # 学習実行
        self.model.fit(x_batch, y_batch, batch_size=batch_size, epochs=1, verbose=0)

    def saveModel(self, path):
        self.model.save(path)