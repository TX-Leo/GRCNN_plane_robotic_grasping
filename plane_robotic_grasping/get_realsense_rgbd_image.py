import pyrealsense2 as rs
import numpy as np
import cv2
import time

class RealsenseD435(object):

    def __init__(self):
        self.image_width = 640
        self.image_height = 480
        self.pipe = None

    def open_realsense(self):
        # =======开启相机=======
        # 1. 初始化相机和流配置cd
        self.pipe = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.depth, self.image_width, self.image_height, rs.format.z16, 30)
        cfg.enable_stream(rs.stream.color, self.image_width, self.image_height, rs.format.bgr8, 30)

        # 2. 启动流
        self.pipe.start(cfg)

        time.sleep(2)

    def check_all_realsense(self):
        # =======查看所有可用的realsense相机=======
        context = rs.context()
        devices = context.query_devices()
        time.sleep(3)
        for device in devices:
            if 'Intel RealSense' in device.get_info(rs.camera_info.name):
                print('Found device with serial number:', device.get_info(rs.camera_info.serial_number))

    def get_rgb_and_depth_image(self):
        # =======拍摄rbg照片和depth照片并保存=======
        # 3. 获取一帧数据
        frames = self.pipe.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        # 4. 将深度图像转换为灰度图像，并保存到磁盘上
        depth_image = np.asanyarray(depth_frame.get_data())
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        cv2.imwrite('./image/depth_image.png', depth_colormap)

        # 5. 保存 RGB 图像到磁盘上
        color_image = np.asanyarray(color_frame.get_data())
        cv2.imwrite('./image/color_image.png', color_image)

    def show_rgbd_video(self):
        # =======实时显示rgbd=======
        while True:
            # 3. 获取一帧数据
            frames = self.pipe.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()

            # 4. 将深度图像转换为灰度图像，并保存到磁盘上
            depth_image = np.asanyarray(depth_frame.get_data())
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            # cv2.imwrite('./image/depth_image.jpg', depth_colormap)

            # 5. 保存 RGB 图像到磁盘上
            color_image = np.asanyarray(color_frame.get_data())
            # cv2.imwrite('./image/color_image.jpg', color_image)

            # 6.屏幕上显示RGB和深度图片
            images = np.hstack((color_image, depth_colormap))
            cv2.imshow('RealSense D435i', images)

    def close_realsense(self):
        # =======关闭相机=======
        # 7. 停止流并关闭设备G
        self.pipe.stop()
        cv2.destroyAllWindows()

def main():
    camera = RealsenseD435()
    # 开启
    camera.open_realsense()
    # 检查可用的
    camera.check_all_realsense()
    # 获取一张rgb图和depth图并保存
    camera.get_rgb_and_depth_image()
    # 实时显示rgbd
    # camera.show_rgbd_video()
    # 关闭相机
    camera.close_realsense()

if __name__ == '__main__':
    main()