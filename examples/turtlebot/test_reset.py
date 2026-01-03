import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_srvs.srv import Empty
import time

class GazeboTestNode(Node):
    def __init__(self):
        super().__init__('gazebo_test_node')
        
        # 1. 速度指令用のパブリッシャー
        self.vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # 2. Gazeboリセット用のサービスクライアント
        self.reset_client = self.create_client(Empty, '/reset_simulation')
        
        # サービスが利用可能になるまで待機
        while not self.reset_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Gazebo reset service not available, waiting...')

    def move_and_reset(self):
        # 速度の設定 (前進 0.2 m/s)
        vel_msg = Twist()
        vel_msg.linear.x = 0.2
        
        self.get_logger().info('Starting 5 seconds run...')
        start_time = time.time()
        
        # 5秒間、速度を送り続ける
        while (time.time() - start_time) < 5.0:
            self.vel_pub.publish(vel_msg)
            time.sleep(0.1)
        
        # 停止
        self.get_logger().info('Time up! Stopping and Resetting...')
        self.vel_pub.publish(Twist()) 
        
        # リセットサービスの呼び出し
        req = Empty.Request()
        self.reset_client.call_async(req)

def main():
    rclpy.init()
    node = GazeboTestNode()
    
    try:
        while rclpy.ok():
            node.move_and_reset()
            time.sleep(1.0) # リセット後のインターバル
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
