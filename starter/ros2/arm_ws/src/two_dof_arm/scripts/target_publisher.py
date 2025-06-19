#!/usr/bin/env  python3

import rclpy
from rclpy.node import Node 
from geometry_msgs.msg import PointStamped
import math
import sys

class TargetPublisher(Node):
    def __init__(self):
        super().__init__('target_publisher')

        self.publisher_ = self.create_publisher(PointStamped, 'target_position', 10)
        self.declare_parameter('mode', 'circle')
        self.mode = self.get_parameter('mode').get_parameter_value().string_value

        if self.mode == 'circle':
            self.radius = 1.2
            self.angle = 0.0
            self.angular_speed = 0.5

            self.timer = self.create_timer(0.1, self.circle_callback)
            self.get_logger().info('activate circular trajectory mode')
        
        elif self.mode == 'manual':
            self.timer = self.create_timer(1.0, self.manual_callback)
            self.get_logger().info('activate manual input mode')
        
        elif self.mode == 'interactive':
            self.timer = self.create_timer(0.1, self.interactive_callback)
            self.setup_keyboard()
            self.target_x = 1.0
            self.target_y = 0.0
            self.get_logger().info('activate interactive mode - use wasd controls')
    
    def circle_callback(self):
        msg = PointStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"

        msg.point.x = self.radius * math.cos(self.angle)
        msg.point.y = self.radius * math.sin(self.angle)
        msg.point.z = 0.0

        self.publisher_.publish(msg)

        self.angle += self.angular_speed * 0.1
        if self.angle > 2 * math.pi:
            self.angle -= 2*math.pi
    
    def manual_callback(self):
        try:
            x = float(input("Enter the x coordinate:"))
            y = float(input("Enter the y coordinate:"))

            msg = PointStamped()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = "base_link"
            msg.point.x = x
            msg.point.y = y
            msg.point.z = 0.0

            self.publisher_.publish(msg)
            self.get_logger().info(f'Send target location: ({x: .2f}, {y: .2f})')
        except ValueError:
            self.get_logger().error('Invalid input, please enter a number')
        except KeyboardInterrupt:
            sys.exit()
    
    def interactive_callback(self):
        if hasattr(self, 'key_pressed'):
            step = 0.05
            if self.key_pressed == 'w':
                self.target_y += step
            elif self.key_pressed == 's':
                self.target_y -= step 
            elif self.key_pressed == 'a':
                self.target_x -= step 
            elif self.key_pressed == 'd':
                self.target_x += step
            
            distance = math.sqrt(self.target_x**2 + self.target_y**2)
            max_reach = 1.8
            
            if distance > max_reach:
                self.target_x *= max_reach / distance
                self.target_y *= max_reach / distance
            
            msg = PointStamped()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = "base_link"
            msg.point.x = self.target_x
            msg.point.y = self.target_y
            msg.point.z = 0.0

            self.publisher_.publish(msg)
            self.key_pressed = None 
    
    def setup_keyboard(self):
        import termios
        import tty 
        import threading

        def get_key():
            import sys, select
            old_settings = termios.tcgetattr(sys.stdin)
            try:
                tty.setraw(sys.stdin.fileno())
                while rclpy.ok():
                    if select.select([sys.stdin], [], [], 0.1)[0]:
                        key = sys.stdin.read(1)
                        if key in ['w', 'a', 's', 'd', 'q']:
                            if key == 'q':
                                break
                            self.key_pressed = key
            finally:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        keyboard_thread = threading.Thread(target=get_key)
        keyboard_thread.daemon = True
        keyboard_thread.start()

def main(args=None):
    rclpy.init(args=args)
    target_publisher = TargetPublisher()

    try:
        rclpy.spin(target_publisher)
    except KeyboardInterrupt:
        pass

    target_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()