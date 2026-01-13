import random
import numpy as np
from keras import Sequential, optimizers
from keras.layers import Dense, Activation, LeakyReLU, Dropout
from keras.models import load_model
from keras.regularizers import l2
import os
import memory

class DeepQ:
    """
    DQN (Deep Q-Network) の抽象化クラスです。

    理論的な背景:
        従来の Q-learning:
            Q(s, a) += alpha * (reward(s,a) + gamma * max(Q(s') - Q(s,a))
        DQN (この実装):
            target = reward(s,a) + gamma * max(Q(s')
            ニューラルネットワークを使用して、このターゲット値との誤差(MSE)を最小化します。
    """
    def __init__(self, inputs, outputs, memorySize, discountFactor, learningRate, learnStart):
        """
        初期化メソッド:
            - inputs: 入力層のサイズ（状態空間の次元数）
            - outputs: 出力層のサイズ（アクション数）
            - memorySize: リプレイバッファ（メモリ）の最大サイズ
            - discountFactor: 割引率 (gamma) - 将来の報酬をどれくらい重視するか
            - learningRate: 学習率 - 重みの更新幅
            - learnStart: 学習を開始するまでに貯めるメモリの最小ステップ数（ここでは128に設定されることが多い）
        """
        self.input_size = inputs
        self.output_size = outputs
        self.memory = memory.Memory(memorySize) # 経験再生用のメモリインスタンス
        self.discountFactor = discountFactor
        self.learnStart = learnStart
        self.learningRate = learningRate

    #@profile
    def initNetworks(self, hiddenLayers):
        """
        メインネットワークとターゲットネットワークの2つを初期化します。
        DQNでは学習の安定化のために、ターゲット計算用のネットワークを別途用意します。
        """
        # 行動決定および学習を行うメインモデル
        model = self.createModel(self.input_size, self.output_size, hiddenLayers, "LeakyReLU", self.learningRate)
        self.model = model

        # 教師信号（ターゲットQ値）を作成するためのターゲットモデル
        targetModel = self.createModel(self.input_size, self.output_size, hiddenLayers, "LeakyReLU", self.learningRate)
        self.targetModel = targetModel

    #@profile
    def createRegularizedModel(self, inputs, outputs, hiddenLayers, activationType, learningRate):
        """
        L2正則化を含んだニューラルネットワークモデルを作成します。
        過学習を抑制したい場合に使用します。
        """
        activationType = "LeakyReLU"
        bias = True
        dropout = 0
        regularizationFactor = 0.01 # 正則化の強さ
        model = Sequential()

        # 隠れ層がない場合（線形モデル）
        if len(hiddenLayers) == 0:
            model.add(Dense(self.output_size, input_shape=(self.input_size,), kernel_initializer='lecun_uniform', bias=bias))
            model.add(Activation("linear"))
        else :
            # 最初の隠れ層
            if regularizationFactor > 0:
                model.add(Dense(hiddenLayers[0], input_shape=(self.input_size,), kernel_initializer='lecun_uniform', kernel_regularizer=l2(regularizationFactor),  bias=bias))
            else:
                model.add(Dense(hiddenLayers[0], input_shape=(self.input_size,), kernel_initializer='lecun_uniform', bias=bias))

            # 活性化関数の設定
            if (activationType == "LeakyReLU") :
                model.add(LeakyReLU(alpha=0.01))
            else :
                model.add(Activation(activationType))

            # 以降の隠れ層を追加
            for index in range(1, len(hiddenLayers)):
                layerSize = hiddenLayers[index]
                if regularizationFactor > 0:
                    model.add(Dense(layerSize, kernel_initializer='lecun_uniform', kernel_regularizer=l2(regularizationFactor), bias=bias))
                else:
                    model.add(Dense(layerSize, kernel_initializer='lecun_uniform', bias=bias))
                if (activationType == "LeakyReLU") :
                    model.add(LeakyReLU(alpha=0.01))
                else :
                    model.add(Activation(activationType))
                if dropout > 0:
                    model.add(Dropout(dropout))
            
            # 出力層
            model.add(Dense(self.output_size, kernel_initializer='lecun_uniform', bias=bias))
            model.add(Activation("linear"))
        
        # コンパイル: 最適化手法はRMSprop、損失関数は平均二乗誤差(MSE)
        optimizer = optimizers.RMSprop(learning_rate=learningRate, rho=0.9, epsilon=1e-06)
        model.compile(loss="mse", optimizer=optimizer)
        model.summary() # モデル構造の表示
        return model

    #@profile
    def createModel(self, inputs, outputs, hiddenLayers, activationType, learningRate):
        """
        標準的なニューラルネットワークモデルを作成します（正則化なし）。
        """
        model = Sequential()
        # 隠れ層がない場合
        if len(hiddenLayers) == 0:
            model.add(Dense(self.output_size, input_shape=(self.input_size,), kernel_initializer='lecun_uniform'))
            model.add(Activation("linear"))
        else :
            # 最初の隠れ層
            model.add(Dense(hiddenLayers[0], input_shape=(self.input_size,), kernel_initializer='lecun_uniform'))
            if (activationType == "LeakyReLU") :
                model.add(LeakyReLU(alpha=0.01))
            else :
                model.add(Activation(activationType))

            # 追加の隠れ層
            for index in range(1, len(hiddenLayers)):
                # print("adding layer "+str(index))
                layerSize = hiddenLayers[index]
                model.add(Dense(layerSize, kernel_initializer='lecun_uniform'))
                if (activationType == "LeakyReLU") :
                    model.add(LeakyReLU(alpha=0.01))
                else :
                    model.add(Activation(activationType))
            
            # 出力層（Q値は実数なのでActivationはLinear）
            model.add(Dense(self.output_size, kernel_initializer='lecun_uniform'))
            model.add(Activation("linear"))
        
        # モデルのコンパイル
        optimizer = optimizers.RMSprop(learning_rate=learningRate, rho=0.9, epsilon=1e-06)
        model.compile(loss="mse", optimizer=optimizer)
        model.summary()
        return model

    def printNetwork(self):
        """
        現在のモデルの重みをコンソールに出力します（デバッグ用）。
        """
        i = 0
        for layer in self.model.layers:
            weights = layer.get_weights()
            print(("layer ",i,": ",weights))
            i += 1
    #@profile
    def backupNetwork(self, model, backup):
        """
        あるモデル（model）の重みを、別のモデル（backup）にコピーします。
        主にターゲットネットワークの更新に使用されます。
        """
        weightMatrix = []
        for layer in model.layers:
            weights = layer.get_weights()
            weightMatrix.append(weights)
        i = 0
        for layer in backup.layers:
            weights = weightMatrix[i]
            layer.set_weights(weights)
            i += 1

    #@profile
    def updateTargetNetwork(self):
        """
        メインモデルの重みをターゲットモデルにコピー（同期）します。
        これを定期的に行うことで、学習ターゲットが固定され、学習が安定します。
        """
        self.backupNetwork(self.model, self.targetModel)

    # アクションごとのQ値を予測する
    #@profile
    def getQValues(self, state):
        """
        指定された状態（state）に対する、全アクションのQ値を予測して返します。
        """
        predicted = self.model.predict(state.reshape(1,state.size),verbose=0)
        return predicted[0]

    #@profile
    def getTargetQValues(self, state):
        """
        ターゲットネットワークを使用してQ値を予測します。
        学習時の教師データ作成に使用されます。
        """
        #predicted = self.targetModel.predict(state.reshape(1,len(state)))
        predicted = self.targetModel.predict(state.reshape(1,len(state)),verbose=0)

        return predicted[0]

    #@profile
    def getMaxQ(self, qValues):
        """
        Q値の配列の中から最大値を返します。
        """
        return np.max(qValues)

    #@profile
    def getMaxIndex(self, qValues):
        """
        Q値の配列の中で最大値を持つインデックス（アクションID）を返します。
        """
        return np.argmax(qValues)

    # ターゲット関数の計算
    #@profile
    def calculateTarget(self, qValuesNewState, reward, isFinal):
        """
        Bellman方程式に基づいてターゲットQ値を計算します。
        target = reward(s,a) + gamma * max(Q(s')
        
        isFinalがTrue（エピソード終了）の場合、将来の報酬はないため、rewardのみを返します。
        """
        if isFinal:
            return reward
        else :
            return reward + self.discountFactor * self.getMaxQ(qValuesNewState)

    # 最も高いQ値を持つアクションを選択する
    #@profile
    def selectAction(self, qValues, explorationRate):
        """
        ε-greedy法（イプシロン・グリーディ法）によるアクション選択。
        explorationRate (epsilon) の確率でランダムに行動し（探索）、
        それ以外はQ値が最大のアクションを選択（活用）します。
        """
        rand = random.random()
        if rand < explorationRate :
            action = np.random.randint(0, self.output_size)
        else :
            action = self.getMaxIndex(qValues)
        return action

    #@profile
    def selectActionByProbability(self, qValues, bias):
        """
        Q値に基づいた確率分布（ボルツマン分布に近い形）に従ってアクションを選択します。
        Softmax行動選択の一種と考えられます。
        """
        qValueSum = 0
        shiftBy = 0
        # 負の値を処理するためにシフト量を計算
        for value in qValues:
            if value + shiftBy < 0:
                shiftBy = - (value + shiftBy)
        shiftBy += 1e-06

        for value in qValues:
            qValueSum += (value + shiftBy) ** bias

        probabilitySum = 0
        qValueProbabilities = []
        # 各アクションの選択確率を計算
        for value in qValues:
            probability = ((value + shiftBy) ** bias) / float(qValueSum)
            qValueProbabilities.append(probability + probabilitySum)
            probabilitySum += probability
        qValueProbabilities[len(qValueProbabilities) - 1] = 1

        # 確率に基づいて選択
        rand = random.random()
        i = 0
        for value in qValueProbabilities:
            if (rand <= value):
                return i
            i += 1
    #@profile
    def addMemory(self, state, action, reward, newState, isFinal):
        """
        1ステップの遷移（現在の状態、行動、報酬、次の状態、終了判定）を
        リプレイメモリ（経験再生バッファ）に保存します。
        """
        self.memory.addMemory(state, action, reward, newState, isFinal)
        # デバッグ用: 追加された最新のメモリを表示
        #print((self.memory.getMemory(self.memory.getCurrentSize() - 1)))

    #@profile
    def learnOnLastState(self):
        """
        最後に保存されたメモリのみを使って学習を行います（オンライン学習的な挙動）。
        通常、DQNではあまり使用されません。
        """
        if self.memory.getCurrentSize() >= 1:
            return self.memory.getMemory(self.memory.getCurrentSize() - 1)
    #@profile
    def learnOnMiniBatch(self, miniBatchSize, useTargetNetwork=True):
        """
        ミニバッチ学習（高速化修正版）
        np.append の代わりにリストを使用し、推論を一括で行うことで高速化しています。
        """
        # まだメモリが十分に溜まっていない場合は何もしない
        if self.memory.getCurrentSize() < self.learnStart:
            return

        # メモリからバッチを取得
        miniBatch = self.memory.getMiniBatch(miniBatchSize)

        # 1. データを一括で配列に変換 (ループ内での処理を減らす)
        states = np.array([sample['state'] for sample in miniBatch])
        newStates = np.array([sample['newState'] for sample in miniBatch])

        # 2. 現在のQ値と次の状態のQ値を一括推論 (GPU/CPUの並列処理を活用)
        # verbose=0 でプログレスバーを消すことでログ出力の遅延も防ぐ
        qValues = self.model.predict(states, verbose=0)
        
        if useTargetNetwork:
            qValuesNewState = self.targetModel.predict(newStates, verbose=0)
        else:
            qValuesNewState = self.model.predict(newStates, verbose=0)

        # 3. 学習用データの作成
        x_list = []
        y_list = []

        for i, sample in enumerate(miniBatch):
            isFinal = sample['isFinal']
            action = sample['action']
            reward = sample['reward']
            state = sample['state']
            newState = sample['newState']

            # ターゲット（教師信号）を計算
            targetValue = self.calculateTarget(qValuesNewState[i], reward, isFinal)

            # このサンプルのYデータを作成
            y_sample = qValues[i].copy()
            y_sample[action] = targetValue

            # リストに追加 (np.appendより圧倒的に速い)
            x_list.append(state)
            y_list.append(y_sample)

            # 元のコードにあった「終了時は次の状態も学習させる」ロジックを維持
            if isFinal:
                x_list.append(newState)
                y_list.append(np.array([reward] * self.output_size))

        # 4. リストをnumpy配列に変換して学習実行
        X_batch = np.array(x_list)
        Y_batch = np.array(y_list)
        
        self.model.fit(X_batch, Y_batch, batch_size=len(X_batch), epochs=1, verbose=0)

    #@profile
    def saveModel(self, path):
        """
        現在のモデルをファイルに保存します。
        """
        self.model.save(path)

    def loadWeights(self, path):
        """
        保存されたモデルの重みを読み込みます。
        """
        self.model.set_weights(load_model(path).get_weights())