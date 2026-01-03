import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_srvs.srv import Empty
import time
import os

# --- GPU設定 ---
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
import tensorflow as tf

class GazeboGpuTestNode(Node):
    def __init__(self):
        super().__init__('gazebo_gpu_test_node')
        self.vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.reset_client = self.create_client(Empty, '/reset_simulation')
        
        while not self.reset_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Gazebo reset service not available, waiting...')
        
        # GPU初期化
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            self.get_logger().info(f'GPU Detected: {gpus}')
        else:
            self.get_logger().error('GPU NOT FOUND!')

    def run_gpu_load(self):
        # 重い行列演算をGPUで実行
        with tf.device('/GPU:0'):
            a = tf.random.normal([5000, 5000])
            b = tf.random.normal([5000, 5000])
            start = time.time()
            c = tf.matmul(a, b)
            end = time.time()
            self.get_logger().info(f'GPU Matrix Mult Success: {end - start:.4f}s')

    def move_and_reset(self):
        # 1. GPUに負荷をかける
        self.run_gpu_load()

        # 2. Gazeboでロボットを動かす
        vel_msg = Twist()
        vel_msg.linear.x = 0.5
        self.get_logger().info('Moving robot...')
        for _ in range(20): # 約2秒間
            self.vel_pub.publish(vel_msg)
            time.sleep(0.1)
        
        # 3. リセット
        self.get_logger().info('Resetting simulation...')
        self.vel_pub.publish(Twist()) 
        self.reset_client.call_async(Empty.Request())

def main():
    rclpy.init()
    node = GazeboGpuTestNode()
    try:
        while rclpy.ok():
            node.move_and_reset()
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()