#ifndef PRINTER_CONTROLLER_HPP
#define PRINTER_CONTROLLER_HPP

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <vector>
#include <queue>

namespace additive_sim {

class PrinterController : public rclcpp::Node {
public:
    PrinterController();
    
    // 公共方法
    void moveToPosition(double x, double y, double z);
    
    // Setter方法
    void setMoveSpeed(double speed) { move_speed_ = speed; }
    void setLayerHeight(double height) { layer_height_ = height; }
    
    // Getter方法
    double getMoveSpeed() const { return move_speed_; }
    double getLayerHeight() const { return layer_height_; }
    double getCurrentX() const { return current_x_; }
    double getCurrentY() const { return current_y_; }
    double getCurrentZ() const { return current_z_; }
    
    // 公共成员变量（用于控制）
    bool is_printing_;
    bool is_extruding_;
    std::vector<geometry_msgs::msg::Point> printed_points_;
    std::queue<geometry_msgs::msg::Point> path_queue_;
    
private:
    void timerCallback();
    void updateJointStates();
    void publishPrintedMaterial();
    void publishToolPath();
    
    // Publishers
    rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr joint_pub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr material_pub_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr path_pub_;
    
    // Timer
    rclcpp::TimerBase::SharedPtr timer_;
    
    // TF broadcaster
    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
    
    // Joint states
    sensor_msgs::msg::JointState joint_states_;
    
    // Current position
    double current_x_, current_y_, current_z_;
    double target_x_, target_y_, target_z_;
    
    // Movement parameters
    double move_speed_;
    double extrusion_rate_;
    
    // Layer info
    int current_layer_;
    double layer_height_;
};

} // namespace additive_sim

#endif // PRINTER_CONTROLLER_HPP