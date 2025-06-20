#include "additive_manufacturing_sim/printer_controller.hpp"
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <cmath>

namespace additive_sim {

PrinterController::PrinterController() 
    : Node("printer_controller"),
      current_x_(0.0), current_y_(0.0), current_z_(0.1),
      target_x_(0.0), target_y_(0.0), target_z_(0.1),
      is_printing_(false), is_extruding_(false),
      move_speed_(0.05), extrusion_rate_(0.001),
      current_layer_(0), layer_height_(0.002) {
    
    // Initialize publishers
    joint_pub_ = this->create_publisher<sensor_msgs::msg::JointState>(
        "joint_states", 10);
    material_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
        "printed_material", 10);
    path_pub_ = this->create_publisher<visualization_msgs::msg::Marker>(
        "tool_path", 10);
    
    // Initialize TF broadcaster
    tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
    
    // Initialize joint states
    joint_states_.name = {"base_to_x_rail", "x_rail_to_y_carriage", "y_carriage_to_extruder"};
    joint_states_.position = {0.0, 0.0, 0.1};
    joint_states_.velocity = {0.0, 0.0, 0.0};
    joint_states_.effort = {0.0, 0.0, 0.0};
    
    // Create timer for control loop
    timer_ = this->create_wall_timer(
        std::chrono::milliseconds(50),
        std::bind(&PrinterController::timerCallback, this));
    
    RCLCPP_INFO(this->get_logger(), "Printer controller initialized");
}

void PrinterController::timerCallback() {
    updateJointStates();
    
    // 只在打印和挤出时记录点
    if (is_printing_ && is_extruding_) {
        // 记录打印点
        geometry_msgs::msg::Point p;
        p.x = current_x_;
        p.y = current_y_;
        p.z = current_z_;
        printed_points_.push_back(p);
        
        // 每添加一定数量的点就发布一次，避免过于频繁
        if (printed_points_.size() % 10 == 0) {
            RCLCPP_DEBUG(this->get_logger(), "已打印 %zu 个点", printed_points_.size());
        }
    }

    if (is_printing_ && is_extruding_ && printed_points_.size() % 50 == 0) {
        RCLCPP_INFO(this->get_logger(), "正在挤出: 位置(%.3f, %.3f, %.3f), 已记录 %zu 个点", 
                current_x_, current_y_, current_z_, printed_points_.size());
    }
    
    // 始终发布可视化
    publishPrintedMaterial();
    publishToolPath();
}

void PrinterController::updateJointStates() {
    // 更新关节状态
    joint_states_.header.stamp = this->now();
    
    // 平滑移动到目标位置
    double dx = target_x_ - current_x_;
    double dy = target_y_ - current_y_;
    double dz = target_z_ - current_z_;
    
    double distance = std::sqrt(dx*dx + dy*dy + dz*dz);
    
    if (distance > 0.001) {
        double factor = std::min(move_speed_ / distance, 1.0);
        current_x_ += dx * factor;
        current_y_ += dy * factor;
        current_z_ += dz * factor;
    }
    
    // 更新关节位置
    joint_states_.position[0] = current_x_;
    joint_states_.position[1] = current_y_;
    joint_states_.position[2] = 0.45 - current_z_; // Z轴反向
    
    joint_pub_->publish(joint_states_);
}

void PrinterController::moveToPosition(double x, double y, double z) {
    target_x_ = x;
    target_y_ = y;
    target_z_ = z;
}

void PrinterController::publishPrintedMaterial() {
    if (printed_points_.empty()) {
        return;
    }
    
    visualization_msgs::msg::MarkerArray marker_array;
    
    // 使用LINE_STRIP来连接打印的点，效率更高
    visualization_msgs::msg::Marker line_marker;
    line_marker.header.frame_id = "world";
    line_marker.header.stamp = this->now();
    line_marker.ns = "printed_lines";
    line_marker.id = 0;
    line_marker.type = visualization_msgs::msg::Marker::LINE_STRIP;
    line_marker.action = visualization_msgs::msg::Marker::ADD;
    
    line_marker.scale.x = 0.004; // 线宽
    line_marker.pose.orientation.w = 1.0;
    
    // 将所有点添加到线条中
    line_marker.points = printed_points_;
    
    // 根据高度设置颜色渐变
    for (size_t i = 0; i < printed_points_.size(); ++i) {
        std_msgs::msg::ColorRGBA color;
        int layer = static_cast<int>(printed_points_[i].z / layer_height_);
        
        // 彩虹色渐变
        float hue = (layer * 30) % 360; // 每层改变30度
        float r, g, b;
        
        // HSV to RGB conversion
        float h = hue / 60.0;
        float c = 1.0;
        float x = c * (1 - std::abs(std::fmod(h, 2) - 1));
        
        if (h < 1) { r = c; g = x; b = 0; }
        else if (h < 2) { r = x; g = c; b = 0; }
        else if (h < 3) { r = 0; g = c; b = x; }
        else if (h < 4) { r = 0; g = x; b = c; }
        else if (h < 5) { r = x; g = 0; b = c; }
        else { r = c; g = 0; b = x; }
        
        color.r = r;
        color.g = g;
        color.b = b;
        color.a = 1.0;
        
        line_marker.colors.push_back(color);
    }
    
    marker_array.markers.push_back(line_marker);
    
    // 同时添加一些球体标记关键点（每N个点一个）
    int step = std::max(1, static_cast<int>(printed_points_.size() / 1000)); // 最多1000个球
    for (size_t i = 0; i < printed_points_.size(); i += step) {
        visualization_msgs::msg::Marker sphere_marker;
        sphere_marker.header.frame_id = "world";
        sphere_marker.header.stamp = this->now();
        sphere_marker.ns = "printed_spheres";
        sphere_marker.id = i;
        sphere_marker.type = visualization_msgs::msg::Marker::SPHERE;
        sphere_marker.action = visualization_msgs::msg::Marker::ADD;
        
        sphere_marker.pose.position = printed_points_[i];
        sphere_marker.pose.orientation.w = 1.0;
        
        // 球体大小
        sphere_marker.scale.x = 0.005;
        sphere_marker.scale.y = 0.005;
        sphere_marker.scale.z = 0.003;
        
        // 颜色
        int layer = static_cast<int>(printed_points_[i].z / layer_height_);
        sphere_marker.color.r = 0.2 + 0.8 * (layer % 3) / 3.0;
        sphere_marker.color.g = 0.2 + 0.8 * ((layer + 1) % 3) / 3.0;
        sphere_marker.color.b = 0.2 + 0.8 * ((layer + 2) % 3) / 3.0;
        sphere_marker.color.a = 1.0;
        
        sphere_marker.lifetime = rclcpp::Duration::from_seconds(0);
        
        marker_array.markers.push_back(sphere_marker);
    }
    
    // 发布状态信息
    if (printed_points_.size() % 100 == 0) {
        RCLCPP_INFO(this->get_logger(), "已可视化 %zu 个打印点", printed_points_.size());
    }
    
    material_pub_->publish(marker_array);
}

void PrinterController::publishToolPath() {
    visualization_msgs::msg::Marker path_marker;
    path_marker.header.frame_id = "world";
    path_marker.header.stamp = this->now();
    path_marker.ns = "tool_path";
    path_marker.id = 0;
    path_marker.type = visualization_msgs::msg::Marker::LINE_STRIP;
    path_marker.action = visualization_msgs::msg::Marker::ADD;
    
    path_marker.scale.x = 0.002;
    path_marker.color.r = 0.0;
    path_marker.color.g = 1.0;
    path_marker.color.b = 0.0;
    path_marker.color.a = 0.5;
    
    // 添加当前位置
    geometry_msgs::msg::Point current;
    current.x = current_x_;
    current.y = current_y_;
    current.z = current_z_;
    path_marker.points.push_back(current);
    
    // 添加未来路径点
    if (!path_queue_.empty()) {
        // 显示接下来的路径点
        std::queue<geometry_msgs::msg::Point> temp_queue = path_queue_;
        int count = 0;
        while (!temp_queue.empty() && count < 50) {
            path_marker.points.push_back(temp_queue.front());
            temp_queue.pop();
            count++;
        }
    }
    
    path_marker.lifetime = rclcpp::Duration::from_seconds(0);
    path_pub_->publish(path_marker);
}

} // namespace additive_sim