
1.工作空间中创建hello_moveit

```ros2 pkg create \
 --build-type ament_cmake \
 --dependencies moveit_ros_planning_interface rclcpp \
 --node-name hello_moveit hello_moveit
```

2.创建 ROS 节点和执行器

修改hello_moveit.cpp代码为

```cpp
#include <memory>

#include <rclcpp/rclcpp.hpp>
#include <moveit/move_group_interface/move_group_interface.hpp>

int main(int argc, char * argv[])
{
  // Initialize ROS and create the Node
  rclcpp::init(argc, argv);
  auto const node = std::make_shared<rclcpp::Node>(
    "hello_moveit",
    rclcpp::NodeOptions().automatically_declare_parameters_from_overrides(true)
  );

  // Create a ROS logger
  auto const logger = rclcpp::get_logger("hello_moveit");

  // Next step goes here

  // Shutdown ROS
  rclcpp::shutdown();
  return 0;
}
 ```
构建并运行
```
colcon build --packages-select hello_moveit

source install/setup.bash

ros2 run hello_moveit hello_moveit
```

顶部包含的标题只是一些标准 C++ 标题和我们稍后将使用的 ROS 和 MoveIt 的标题。

之后，我们通过正常调用来初始化 rclcpp，然后创建我们的 Node。

```cpp
auto const node = std::make_shared<rclcpp::Node>(
  "hello_moveit",
  rclcpp::NodeOptions().automatically_declare_parameters_from_overrides(true)
);
```

第一个参数是 ROS 用于命名唯一节点的字符串。MoveIt 需要第二个参数，因为我们使用 ROS 参数的方式。

接下来，我们创建一个名为“hello_moveit”的记录器，以使我们的日志输出保持有序和可配置。

```cpp
// Create a ROS logger
auto const logger = rclcpp::get_logger("hello_moveit");
```

最后，我们有关闭 ROS 的代码。

```cpp
// Shutdown ROS
rclcpp::shutdown();
return 0;
```

3 使用 MoveGroupInterface 进行计划和执行
在“下一步从这里开始”的注释处添加以下代码：

```cpp
// Create the MoveIt MoveGroup Interface
using moveit::planning_interface::MoveGroupInterface;
auto move_group_interface = MoveGroupInterface(node, "manipulator");

// Set a target Pose
auto const target_pose = []{
  geometry_msgs::msg::Pose msg;
  msg.orientation.w = 1.0;
  msg.position.x = 0.28;
  msg.position.y = -0.2;
  msg.position.z = 0.5;
  return msg;
}();
move_group_interface.setPoseTarget(target_pose);

// Create a plan to that target pose
auto const [success, plan] = [&move_group_interface]{
  moveit::planning_interface::MoveGroupInterface::Plan msg;
  auto const ok = static_cast<bool>(move_group_interface.plan(msg));
  return std::make_pair(ok, msg);
}();

// Execute the plan
if(success) {
  move_group_interface.execute(plan);
} else {
  RCLCPP_ERROR(logger, "Planning failed!");
}
```

构建并运行

```
colcon build --packages-select hello_moveit

source install/setup.bash

ros2 run hello_moveit hello_moveit
```


![first](pics/000first_moveit_c++.mp4)


















