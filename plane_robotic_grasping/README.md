- get_realsense_rgbd_image.py:是获取realsense拍摄到的rgb图和深度图的脚本
- cfg:存放配置文件
- image:存放待检测图片
- results:存放结果图片
- robot.py:机械臂运动库
- plane_robotic_grasping.py:机械臂平面抓取，采用GRCNN算法获取物体最佳抓取点位姿
  - 输入:rbg图、depth图、相机参数、手眼标定矩阵、相机深度比例、GRCNN模型
  - 输出:抓取点xyz,机械夹爪的yaw角、张开大小