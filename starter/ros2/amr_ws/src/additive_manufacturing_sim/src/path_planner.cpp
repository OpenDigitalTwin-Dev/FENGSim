#include "additive_manufacturing_sim/path_planner.hpp"
#include <cmath>

namespace additive_sim {

PathPlanner::PathPlanner() 
    : infill_density_(0.8), nozzle_diameter_(0.004) {
}

std::vector<geometry_msgs::msg::Point> PathPlanner::generateSquareSpiralPath(
    double width, double height, int layers, double layer_height) {
    
    std::vector<geometry_msgs::msg::Point> path;
    
    for (int layer = 0; layer < layers; ++layer) {
        double z = 0.04 + layer * layer_height;
        double current_width = width;
        double current_height = height;
        double x = -width / 2;
        double y = -height / 2;
        
        // 外围轮廓
        while (current_width > nozzle_diameter_ * 2 && current_height > nozzle_diameter_ * 2) {
            // 底边
            for (double px = x; px <= x + current_width; px += nozzle_diameter_) {
                geometry_msgs::msg::Point p;
                p.x = px;
                p.y = y;
                p.z = z;
                path.push_back(p);
            }
            
            // 右边
            for (double py = y; py <= y + current_height; py += nozzle_diameter_) {
                geometry_msgs::msg::Point p;
                p.x = x + current_width;
                p.y = py;
                p.z = z;
                path.push_back(p);
            }
            
            // 顶边
            for (double px = x + current_width; px >= x; px -= nozzle_diameter_) {
                geometry_msgs::msg::Point p;
                p.x = px;
                p.y = y + current_height;
                p.z = z;
                path.push_back(p);
            }
            
            // 左边
            for (double py = y + current_height; py >= y + nozzle_diameter_ * 2; py -= nozzle_diameter_) {
                geometry_msgs::msg::Point p;
                p.x = x;
                p.y = py;
                p.z = z;
                path.push_back(p);
            }
            
            // 向内收缩
            x += nozzle_diameter_ * 2;
            y += nozzle_diameter_ * 2;
            current_width -= nozzle_diameter_ * 4;
            current_height -= nozzle_diameter_ * 4;
        }
        
        // 填充内部（简单的来回扫描）
        if (infill_density_ > 0) {
            double infill_spacing = nozzle_diameter_ / infill_density_;
            bool direction = (layer % 2 == 0);
            
            if (direction) {
                // 横向填充
                for (double py = -height/2 + nozzle_diameter_ * 3; 
                     py < height/2 - nozzle_diameter_ * 3; 
                     py += infill_spacing) {
                    
                    for (double px = -width/2 + nozzle_diameter_ * 3; 
                         px < width/2 - nozzle_diameter_ * 3; 
                         px += nozzle_diameter_) {
                        geometry_msgs::msg::Point p;
                        p.x = (static_cast<int>(py / infill_spacing) % 2 == 0) ? px : (width/2 - nozzle_diameter_ * 3 - (px + width/2));
                        p.y = py;
                        p.z = z;
                        path.push_back(p);
                    }
                }
            } else {
                // 纵向填充
                for (double px = -width/2 + nozzle_diameter_ * 3; 
                     px < width/2 - nozzle_diameter_ * 3; 
                     px += infill_spacing) {
                    
                    for (double py = -height/2 + nozzle_diameter_ * 3; 
                         py < height/2 - nozzle_diameter_ * 3; 
                         py += nozzle_diameter_) {
                        geometry_msgs::msg::Point p;
                        p.x = px;
                        p.y = (static_cast<int>(px / infill_spacing) % 2 == 0) ? py : (height/2 - nozzle_diameter_ * 3 - (py + height/2));
                        p.z = z;
                        path.push_back(p);
                    }
                }
            }
        }
    }
    
    return path;
}

std::vector<geometry_msgs::msg::Point> PathPlanner::generateCircularPath(
    double radius, int layers, double layer_height, int points_per_layer) {
    
    std::vector<geometry_msgs::msg::Point> path;
    
    for (int layer = 0; layer < layers; ++layer) {
        double z = 0.04 + layer * layer_height;
        double current_radius = radius;
        
        // 外围轮廓
        while (current_radius > nozzle_diameter_) {
            for (int i = 0; i <= points_per_layer; ++i) {
                double angle = 2.0 * M_PI * i / points_per_layer;
                geometry_msgs::msg::Point p;
                p.x = current_radius * std::cos(angle);
                p.y = current_radius * std::sin(angle);
                p.z = z;
                path.push_back(p);
            }
            current_radius -= nozzle_diameter_ * 2;
        }
        
        // 填充内部（螺旋填充）
        if (infill_density_ > 0 && radius > nozzle_diameter_ * 4) {
            double infill_spacing = nozzle_diameter_ / infill_density_;
            double r = infill_spacing;
            
            while (r < radius - nozzle_diameter_ * 2) {
                int points = static_cast<int>(2 * M_PI * r / nozzle_diameter_);
                for (int i = 0; i < points; ++i) {
                    double angle = 2.0 * M_PI * i / points + layer * M_PI / 4; // 每层旋转45度
                    geometry_msgs::msg::Point p;
                    p.x = r * std::cos(angle);
                    p.y = r * std::sin(angle);
                    p.z = z;
                    path.push_back(p);
                }
                r += infill_spacing;
            }
        }
    }
    
    return path;
}

std::vector<geometry_msgs::msg::Point> PathPlanner::generateCustomPath() {
    std::vector<geometry_msgs::msg::Point> path;
    
    // 生成一个简单的金字塔形状
    int total_layers = 50;
    double base_size = 0.2;
    double top_size = 0.02;
    double layer_height = 0.002;  // 使用局部变量
    
    for (int layer = 0; layer < total_layers; ++layer) {
        double z = 0.04 + layer * layer_height;  // 使用局部变量
        double layer_ratio = 1.0 - static_cast<double>(layer) / total_layers;
        double current_size = top_size + (base_size - top_size) * layer_ratio;
        
        // 生成正方形轮廓
        int points_per_side = static_cast<int>(current_size / nozzle_diameter_);
        
        // 外轮廓
        for (int i = 0; i <= points_per_side; ++i) {
            geometry_msgs::msg::Point p1, p2, p3, p4;
            double t = static_cast<double>(i) / points_per_side;
            
            // 底边
            p1.x = -current_size/2 + t * current_size;
            p1.y = -current_size/2;
            p1.z = z;
            path.push_back(p1);
            
            // 右边
            p2.x = current_size/2;
            p2.y = -current_size/2 + t * current_size;
            p2.z = z;
            path.push_back(p2);
            
            // 顶边
            p3.x = current_size/2 - t * current_size;
            p3.y = current_size/2;
            p3.z = z;
            path.push_back(p3);
            
            // 左边
            p4.x = -current_size/2;
            p4.y = current_size/2 - t * current_size;
            p4.z = z;
            path.push_back(p4);
        }
        
        // 简单的内部填充
        if (current_size > nozzle_diameter_ * 4) {
            double fill_spacing = nozzle_diameter_ * 2;
            for (double y = -current_size/2 + fill_spacing; y < current_size/2 - fill_spacing; y += fill_spacing) {
                bool forward = (static_cast<int>((y + current_size/2) / fill_spacing) % 2 == 0);
                
                if (forward) {
                    for (double x = -current_size/2 + fill_spacing; x < current_size/2 - fill_spacing; x += nozzle_diameter_) {
                        geometry_msgs::msg::Point p;
                        p.x = x;
                        p.y = y;
                        p.z = z;
                        path.push_back(p);
                    }
                } else {
                    for (double x = current_size/2 - fill_spacing; x > -current_size/2 + fill_spacing; x -= nozzle_diameter_) {
                        geometry_msgs::msg::Point p;
                        p.x = x;
                        p.y = y;
                        p.z = z;
                        path.push_back(p);
                    }
                }
            }
        }
    }
    
    return path;
}
  

} // namespace additive_sim