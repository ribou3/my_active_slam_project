import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan

class ScanRelay(Node):
    def __init__(self):
        super().__init__('scan_relay')
        # オリジナルの /scan を購読
        self.subscription = self.create_subscription(LaserScan, '/scan', self.listener_callback, 10)
        # 時刻を修正した /scan_fixed を発行
        self.publisher = self.create_publisher(LaserScan, '/scan_fixed', 10)

    def listener_callback(self, msg):
        # 時刻を「今」に書き換える
        msg.header.stamp = self.get_clock().now().to_msg()
        # フレーム名を明示的に指定
        msg.header.frame_id = 'base_scan'
        self.publisher.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = ScanRelay()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()