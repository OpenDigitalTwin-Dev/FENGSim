#ifndef PATH_PLANNER_HPP
#define PATH_PLANNER_HPP

#include <vector>
#include <geometry_msgs/msg/point.hpp>

namespace additive_sim {

class PathPlanner {
public:
    PathPlanner();
    
    // 生成简单的方形螺旋路径
    std::vector<geometry_msgs::msg::Point> generateSquareSpiralPath(
        double width, double height, int layers, double layer_height);
    
    // 生成圆形路径
    std::vector<geometry_msgs::msg::Point> generateCircularPath(
        double radius, int layers, double layer_height, int points_per_layer);
    
    // 生成自定义模型路径（简化版）
    std::vector<geometry_msgs::msg::Point> generateCustomPath();
    
private:
    double infill_density_;
    double nozzle_diameter_;
};

} // namespace additive_sim

#endif // PATH_PLANNER_HPP