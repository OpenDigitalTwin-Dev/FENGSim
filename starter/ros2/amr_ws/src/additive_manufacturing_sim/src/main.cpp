#include <rclcpp/rclcpp.hpp>
#include "additive_manufacturing_sim/printer_controller.hpp"
#include "additive_manufacturing_sim/path_planner.hpp"
#include <std_srvs/srv/trigger.hpp>
#include <thread>
#include <chrono>
#include <memory>
#include <vector>
#include <cmath>
#include <queue>
#include <string>

namespace additive_sim {

class AdditiveManufacturingNode : public rclcpp::Node {
public:
    AdditiveManufacturingNode() : Node("additive_manufacturing_node"),
                                   current_point_index_(0),
                                   is_printing_(false),
                                   print_mode_(PrintMode::CUSTOM) {
        
        // 创建打印机控制器节点
        printer_controller_ = std::make_shared<PrinterController>();
        
        // 创建路径规划器
        path_planner_ = std::make_unique<PathPlanner>();
        
        // 声明和获取参数
        this->declare_parameter<std::string>("print_mode", "custom");
        this->declare_parameter<double>("print_speed", 0.05);
        this->declare_parameter<double>("layer_height", 0.002);
        this->declare_parameter<int>("num_layers", 50);
        
        std::string mode_str = this->get_parameter("print_mode").as_string();
        printer_controller_->setMoveSpeed(this->get_parameter("print_speed").as_double());
        printer_controller_->setLayerHeight(this->get_parameter("layer_height").as_double());
        int num_layers = this->get_parameter("num_layers").as_int();
        
        // 根据参数选择打印模式
        if (mode_str == "square") {
            print_mode_ = PrintMode::SQUARE;
            print_path_ = path_planner_->generateSquareSpiralPath(0.15, 0.15, num_layers, printer_controller_->getLayerHeight());
        } else if (mode_str == "circle") {
            print_mode_ = PrintMode::CIRCLE;
            print_path_ = path_planner_->generateCircularPath(0.1, num_layers, printer_controller_->getLayerHeight(), 100);
        } else {
            print_mode_ = PrintMode::CUSTOM;
            print_path_ = path_planner_->generateCustomPath();
        }
        
        // 初始化路径队列
        initializePathQueue();
        
        // 创建定时器来执行打印过程
        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(20),
            std::bind(&AdditiveManufacturingNode::printingProcess, this));
        
        // 创建服务（可选：用于控制打印）
        start_service_ = this->create_service<std_srvs::srv::Trigger>(
            "start_printing",
            std::bind(&AdditiveManufacturingNode::handleStartPrinting, this,
                      std::placeholders::_1, std::placeholders::_2));
        
        stop_service_ = this->create_service<std_srvs::srv::Trigger>(
            "stop_printing",
            std::bind(&AdditiveManufacturingNode::handleStopPrinting, this,
                      std::placeholders::_1, std::placeholders::_2));
        
        RCLCPP_INFO(this->get_logger(), "增材制造仿真节点已启动");
        RCLCPP_INFO(this->get_logger(), "打印模式: %s", mode_str.c_str());
        RCLCPP_INFO(this->get_logger(), "生成了 %zu 个路径点", print_path_.size());
        RCLCPP_INFO(this->get_logger(), "层高: %.3f mm", printer_controller_->getLayerHeight() * 1000);
        RCLCPP_INFO(this->get_logger(), "打印速度: %.3f m/s", printer_controller_->getMoveSpeed());
        
        // 自动开始打印
        startPrinting();
    }
    
    std::shared_ptr<PrinterController> getPrinterController() {
        return printer_controller_;
    }
    
private:
    enum class PrintMode {
        SQUARE,
        CIRCLE,
        CUSTOM
    };
    
    void initializePathQueue() {
        // 清空队列
        while (!printer_controller_->path_queue_.empty()) {
            printer_controller_->path_queue_.pop();
        }
        
        // 添加初始路径点
        for (size_t i = 0; i < std::min(size_t(100), print_path_.size()); ++i) {
            printer_controller_->path_queue_.push(print_path_[i]);
        }
    }
    
    void startPrinting() {
        if (!is_printing_ && !print_path_.empty()) {
        is_printing_ = true;
        printer_controller_->is_printing_ = true;
        printer_controller_->is_extruding_ = false; // 初始不挤出
        
        // 清空之前的打印点
        printer_controller_->printed_points_.clear();
        
        // 移动到起始位置（抬起喷嘴）
        if (!print_path_.empty()) {
            printer_controller_->moveToPosition(
                print_path_[0].x, 
                print_path_[0].y, 
                print_path_[0].z + 0.01);
        }
        
        RCLCPP_INFO(this->get_logger(), "开始打印...");
    }
    }
    
    void stopPrinting() {
        is_printing_ = false;
        printer_controller_->is_printing_ = false;
        printer_controller_->is_extruding_ = false;
        
        // 抬起喷嘴
        printer_controller_->moveToPosition(
            printer_controller_->getCurrentX(),
            printer_controller_->getCurrentY(),
            printer_controller_->getCurrentZ() + 0.05);
        
        RCLCPP_INFO(this->get_logger(), "打印完成！");
        RCLCPP_INFO(this->get_logger(), "总共打印了 %zu 个点", printer_controller_->printed_points_.size());
    }
    
    void printingProcess() {
        if (!is_printing_ || current_point_index_ >= print_path_.size()) {
            if (is_printing_ && current_point_index_ >= print_path_.size()) {
                stopPrinting();
            }
            return;
        }
        
        // 获取当前目标点
        const auto& target_point = print_path_[current_point_index_];
        
        // 计算到目标点的距离
        double dx = target_point.x - printer_controller_->getCurrentX();
        double dy = target_point.y - printer_controller_->getCurrentY();
        double dz = target_point.z - printer_controller_->getCurrentZ();
        double distance = std::sqrt(dx*dx + dy*dy + dz*dz);
        
        // 移动到目标点
        printer_controller_->moveToPosition(target_point.x, target_point.y, target_point.z);
        
        // 如果接近目标点，移动到下一个点
        if (distance < 0.001) {
            current_point_index_++;
            
            // 检查是否需要抬起喷嘴（层切换或大跳跃）
            if (current_point_index_ < print_path_.size()) {
                const auto& next_point = print_path_[current_point_index_];
                double jump_distance = std::sqrt(
                    std::pow(next_point.x - target_point.x, 2) + 
                    std::pow(next_point.y - target_point.y, 2));
                
                // 判断是否需要回抽和抬起
                if (jump_distance > 0.01 || std::abs(next_point.z - target_point.z) > 0.0001) {
                    printer_controller_->is_extruding_ = false;
                    
                    // 如果是大跳跃，先抬起喷嘴
                    if (jump_distance > 0.02) {
                        printer_controller_->moveToPosition(
                            target_point.x,
                            target_point.y,
                            target_point.z + 0.005);
                    }
                } else {
                    printer_controller_->is_extruding_ = true;
                }
            }
            
            // 更新路径队列（用于可视化）
            updatePathQueue();
            
            // 打印进度报告
            reportProgress();
        }
    }
    
    void updatePathQueue() {
        // 移除已处理的点
        if (!printer_controller_->path_queue_.empty()) {
            printer_controller_->path_queue_.pop();
        }
        
        // 添加新的路径点
        size_t queue_size = printer_controller_->path_queue_.size();
        size_t points_to_add = 100 - queue_size;
        
        for (size_t i = 0; i < points_to_add && (current_point_index_ + queue_size + i) < print_path_.size(); ++i) {
            printer_controller_->path_queue_.push(print_path_[current_point_index_ + queue_size + i]);
        }
    }
    
    void reportProgress() {
        // 每处理1000个点或每层报告一次进度
        static int last_reported_layer = -1;
        int current_layer = static_cast<int>(print_path_[current_point_index_ - 1].z / printer_controller_->getLayerHeight());
        
        if (current_point_index_ % 1000 == 0 || current_layer != last_reported_layer) {
            double progress = 100.0 * current_point_index_ / print_path_.size();
            RCLCPP_INFO(this->get_logger(), 
                        "打印进度: %.1f%% (层 %d, 点 %zu/%zu)", 
                        progress, current_layer, current_point_index_, print_path_.size());
            last_reported_layer = current_layer;
        }
    }
    
    void handleStartPrinting(const std::shared_ptr<std_srvs::srv::Trigger::Request> request,
                             std::shared_ptr<std_srvs::srv::Trigger::Response> response) {
        (void)request; // 未使用参数
        
        if (!is_printing_) {
            current_point_index_ = 0;
            printer_controller_->printed_points_.clear();
            initializePathQueue();
            startPrinting();
            response->success = true;
            response->message = "打印已开始";
        } else {
            response->success = false;
            response->message = "打印已在进行中";
        }
    }
    
    void handleStopPrinting(const std::shared_ptr<std_srvs::srv::Trigger::Request> request,
                            std::shared_ptr<std_srvs::srv::Trigger::Response> response) {
        (void)request; // 未使用参数
        
        if (is_printing_) {
            stopPrinting();
            response->success = true;
            response->message = "打印已停止";
        } else {
            response->success = false;
            response->message = "没有正在进行的打印";
        }
    }
    
    // 成员变量
    std::shared_ptr<PrinterController> printer_controller_;
    std::unique_ptr<PathPlanner> path_planner_;
    std::vector<geometry_msgs::msg::Point> print_path_;
    size_t current_point_index_;
    bool is_printing_;
    PrintMode print_mode_;
    rclcpp::TimerBase::SharedPtr timer_;
    
    // 服务
    rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr start_service_;
    rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr stop_service_;
};

} // namespace additive_sim

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    
    // 创建主节点
    auto main_node = std::make_shared<additive_sim::AdditiveManufacturingNode>();
    
    // 使用多线程执行器以支持并行处理
    rclcpp::executors::MultiThreadedExecutor executor;
    executor.add_node(main_node);
    executor.add_node(main_node->getPrinterController());
    
    RCLCPP_INFO(rclcpp::get_logger("main"), "=================================");
    RCLCPP_INFO(rclcpp::get_logger("main"), "增材制造仿真系统已启动");
    RCLCPP_INFO(rclcpp::get_logger("main"), "=================================");
    RCLCPP_INFO(rclcpp::get_logger("main"), "可用服务:");
    RCLCPP_INFO(rclcpp::get_logger("main"), "  - /start_printing");
    RCLCPP_INFO(rclcpp::get_logger("main"), "  - /stop_printing");
    RCLCPP_INFO(rclcpp::get_logger("main"), "=================================");
    
    try {
        executor.spin();
    } catch (const std::exception& e) {
        RCLCPP_ERROR(rclcpp::get_logger("main"), "发生异常: %s", e.what());
    }
    
    rclcpp::shutdown();
    return 0;
}