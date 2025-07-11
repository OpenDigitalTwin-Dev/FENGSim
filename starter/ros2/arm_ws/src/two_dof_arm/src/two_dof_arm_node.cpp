#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/point_stamped.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2/LinearMath/Quaternion.h>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <Eigen/Dense>
#include <cmath>

class TwoDOFArm : public rclcpp::Node
{
public:
    TwoDOFArm() : Node("two_dof_arm_node")
    {
        L1_ = 1.0;
        L2_ = 0.8;

        theta1_ = 0.0;
        theta2_ = 0.0;

        joint_pub_ = this->create_publisher<sensor_msgs::msg::JointState>("joint_states", 10);
        marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("arm_markers", 10);

        target_sub_ = this->create_subscription<geometry_msgs::msg::PointStamped>(
            "target_position", 10, 
            std::bind(&TwoDOFArm::targetCallback, this, std::placeholders::_1));

        tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(50),
            std::bind(&TwoDOFArm::timerCallback, this));

        RCLCPP_INFO(this->get_logger(), "The 2DOF robot node has been started");
    }
private:
    // Positive kinematics
    Eigen::Vector2d forwardKinematics(double theta1, double theta2)
    {
        Eigen::Vector2d position;
        position.x() = L1_ * cos(theta1) + L2_ * cos(theta1 + theta2);
        position.y() = L1_ * sin(theta1) + L2_ * sin(theta1 + theta2);
        return position;
    }

    // Inverse Kinematics
    bool inverseKinematics(const Eigen::Vector2d& target, double& theta1, double& theta2)
    {
        double x = target.x();
        double y = target.y();
        
        // Check if the target is reachable
        double r_squared = x * x + y * y;
        double r = sqrt(r_squared);
        
        if (r > (L1_ + L2_) || r < fabs(L1_ - L2_))
        {
            RCLCPP_WARN(this->get_logger(), "Target location unreachable: (%.2f, %.2f)", x, y);
            return false;
        }
        
        // compute theta2
        double cos_theta2 = (r_squared - L1_ * L1_ - L2_ * L2_) / (2 * L1_ * L2_);
        cos_theta2 = std::max(-1.0, std::min(1.0, cos_theta2)); // Limited to the range [-1, 1]
        
        // Choose the elbow-up solution
        theta2 = acos(cos_theta2);
        
        // compute theta1
        double beta = atan2(L2_ * sin(theta2), L1_ + L2_ * cos(theta2));
        theta1 = atan2(y, x) - beta;
        
        // Normalized angle to [-π, π]
        theta1 = atan2(sin(theta1), cos(theta1));
        theta2 = atan2(sin(theta2), cos(theta2));
        
        return true;
    }
    // Target position callback
    void targetCallback(const geometry_msgs::msg::PointStamped::SharedPtr msg)
    {
        Eigen::Vector2d target(msg->point.x, msg->point.y);
        
        double new_theta1, new_theta2;
        if (inverseKinematics(target, new_theta1, new_theta2))
        {
            // Smooth interpolation
            target_theta1_ = new_theta1;
            target_theta2_ = new_theta2;
            interpolating_ = true;
            interpolation_progress_ = 0.0;
            
            RCLCPP_INFO(this->get_logger(), 
                "Target location:(%.2f, %.2f),Target Angle: (%.2f°, %.2f°)", 
                target.x(), target.y(), 
                new_theta1 * 180 / M_PI, new_theta2 * 180 / M_PI);
        }
    }
    
    // Timer callback
    void timerCallback()
    {
        // Smooth interpolated motion
        if (interpolating_)
        {
            interpolation_progress_ += 0.05;
            if (interpolation_progress_ >= 1.0)
            {
                interpolation_progress_ = 1.0;
                interpolating_ = false;
            }
            
            double t = (1 - cos(interpolation_progress_ * M_PI)) / 2; 
            theta1_ = theta1_ + t * (target_theta1_ - theta1_);
            theta2_ = theta2_ + t * (target_theta2_ - theta2_);
        }
        
        // Publish joint status
        publishJointStates();
        
        // Release TF Transform
        publishTransforms();
        
        // Publish Visual Markup
        publishMarkers();
    }
    
    // Publish joint status
    void publishJointStates()
    {
        auto joint_state = sensor_msgs::msg::JointState();
        joint_state.header.stamp = this->now();
        joint_state.header.frame_id = "base_link";
        
        joint_state.name = {"joint1", "joint2"};
        joint_state.position = {theta1_, theta2_};
        joint_state.velocity = {0.0, 0.0};
        joint_state.effort = {0.0, 0.0};
        
        joint_pub_->publish(joint_state);
    }
    
    // Release TF Transform
    void publishTransforms()
    {
        std::vector<geometry_msgs::msg::TransformStamped> transforms;
        
        // base_link -> link1
        geometry_msgs::msg::TransformStamped transform1;
        transform1.header.stamp = this->now();
        transform1.header.frame_id = "base_link";
        transform1.child_frame_id = "link1";
        transform1.transform.translation.x = L1_ * cos(theta1_) / 2;
        transform1.transform.translation.y = L1_ * sin(theta1_) / 2;
        transform1.transform.translation.z = 0.0;
        
        tf2::Quaternion q1;
        q1.setRPY(0, 0, theta1_);
        transform1.transform.rotation.x = q1.x();
        transform1.transform.rotation.y = q1.y();
        transform1.transform.rotation.z = q1.z();
        transform1.transform.rotation.w = q1.w();
        transforms.push_back(transform1);
        
        // link1 -> link2
        geometry_msgs::msg::TransformStamped transform2;
        transform2.header.stamp = this->now();
        transform2.header.frame_id = "link1";
        transform2.child_frame_id = "link2";
        transform2.transform.translation.x = L1_ * cos(theta1_) / 2 + L2_ * cos(theta1_ + theta2_) / 2;
        transform2.transform.translation.y = L1_ * sin(theta1_) / 2 + L2_ * sin(theta1_ + theta2_) / 2;
        transform2.transform.translation.z = 0.0;
        
        tf2::Quaternion q2;
        q2.setRPY(0, 0, theta1_ + theta2_);
        transform2.transform.rotation.x = q2.x();
        transform2.transform.rotation.y = q2.y();
        transform2.transform.rotation.z = q2.z();
        transform2.transform.rotation.w = q2.w();
        transforms.push_back(transform2);
        
        tf_broadcaster_->sendTransform(transforms);
    }
    
    // Publish Visual Markup
    void publishMarkers()
    {
        visualization_msgs::msg::MarkerArray marker_array;
        
        // Workspace Boundaries
        auto workspace_marker = createWorkspaceMarker(0, "base_link");
        marker_array.markers.push_back(workspace_marker);
        
        // Target location marker (if moving)
        if (interpolating_)
        {
            Eigen::Vector2d target_pos = forwardKinematics(target_theta1_, target_theta2_);
            auto target_marker = createSphereMarker(1, "base_link", target_pos.x(), target_pos.y(), 0, 0.08, 1.0, 0.0, 1.0);
            target_marker.color.a = 0.5; // translucent
            marker_array.markers.push_back(target_marker);
        }
        
        marker_pub_->publish(marker_array);
    }
    
    // Create a cylinder mark
    visualization_msgs::msg::Marker createCylinderMarker(int id, const std::string& frame_id,
        double x, double y, double z, double length, double radius, double r, double g, double b)
    {
        visualization_msgs::msg::Marker marker;
        marker.header.frame_id = frame_id;
        marker.header.stamp = this->now();
        marker.ns = "arm";
        marker.id = id;
        marker.type = visualization_msgs::msg::Marker::CYLINDER;
        marker.action = visualization_msgs::msg::Marker::ADD;
        
        marker.pose.position.x = x;
        marker.pose.position.y = y;
        marker.pose.position.z = z;
        marker.pose.orientation.x = 0.0;
        marker.pose.orientation.y = 0.0;
        marker.pose.orientation.z = 0.0;
        marker.pose.orientation.w = 1.0;
        
        marker.scale.x = radius * 2;
        marker.scale.y = radius * 2;
        marker.scale.z = length;
        
        marker.color.a = 1.0;
        marker.color.r = r;
        marker.color.g = g;
        marker.color.b = b;
        
        return marker;
    }
    
    // Create sphere marker
    visualization_msgs::msg::Marker createSphereMarker(int id, const std::string& frame_id,
        double x, double y, double z, double radius, double r, double g, double b)
    {
        visualization_msgs::msg::Marker marker;
        marker.header.frame_id = frame_id;
        marker.header.stamp = this->now();
        marker.ns = "arm";
        marker.id = id;
        marker.type = visualization_msgs::msg::Marker::SPHERE;
        marker.action = visualization_msgs::msg::Marker::ADD;
        
        marker.pose.position.x = x;
        marker.pose.position.y = y;
        marker.pose.position.z = z;
        marker.pose.orientation.w = 1.0;
        
        marker.scale.x = radius;
        marker.scale.y = radius;
        marker.scale.z = radius;
        
        marker.color.a = 1.0;
        marker.color.r = r;
        marker.color.g = g;
        marker.color.b = b;
        
        return marker;
    }
    
    // Creating workspace boundary markers
    visualization_msgs::msg::Marker createWorkspaceMarker(int id, const std::string& frame_id)
    {
        visualization_msgs::msg::Marker marker;
        marker.header.frame_id = frame_id;
        marker.header.stamp = this->now();
        marker.ns = "workspace";
        marker.id = id;
        marker.type = visualization_msgs::msg::Marker::LINE_STRIP;
        marker.action = visualization_msgs::msg::Marker::ADD;
        
        marker.scale.x = 0.01;
        marker.color.a = 0.3;
        marker.color.r = 0.5;
        marker.color.g = 0.5;
        marker.color.b = 0.5;
        
        // Generate workspace boundary points
        int num_points = 100;
        for (int i = 0; i <= num_points; ++i)
        {
            double angle = 2 * M_PI * i / num_points;
            geometry_msgs::msg::Point p;
            p.x = (L1_ + L2_) * cos(angle);
            p.y = (L1_ + L2_) * sin(angle);
            p.z = 0;
            marker.points.push_back(p);
        }
        
        return marker;
    }
    
    
    double L1_, L2_;  // link length
    double theta1_, theta2_;  
    double target_theta1_, target_theta2_; 
    bool interpolating_ = false;  
    double interpolation_progress_ = 0.0; 
    

    rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr joint_pub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;
    rclcpp::Subscription<geometry_msgs::msg::PointStamped>::SharedPtr target_sub_;
    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
    rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<TwoDOFArm>());
    rclcpp::shutdown();
    return 0;
}