import csv
import json
from collections import namedtuple
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import asyncio

from utils.data.camera_data import CameraData
from inference.post_process import post_process_output
from utils.dataset_processing.grasp import detect_grasps
from utils.visualisation.plot import plot_grasp

from robot import FR5ROBOT
from get_realsense_rgbd_image import RealsenseD435


def GRCNN_rgb_and_depth_to_target2base_xyzrxryrz(rgb_path,depth_path,is_visualize,saved_model_path,device,camera_matrix,camera_depth_scale,cam2base_H,robot=None):
    # ==================是否可视化==================
    if is_visualize:
        fig = plt.figure(figsize=(6, 6))
    else:
        fig = None

    # ==================导入模型==================
    model = torch.load(saved_model_path, map_location=torch.device(device))
    # device = "MPS" if torch.backends.mps.is_available() else {"cuda:0" if torch.cuda.is_available() else "cpu"}

    # ==================rgb_path,depth_path to x==================
    # 定义照片处理工具
    cam_data = CameraData(include_rgb=True,include_depth=True,output_size=300)
    # 获取rgb和depth
    rgb = np.array(Image.open(rgb_path))
    depth = np.array(Image.open(depth_path)).astype(np.float32)
    # depth变换
    depth = depth * camera_depth_scale
    depth[depth >1.2]=0 # distance > 1.2m ,remove it
    depth = np.expand_dims(depth, axis=2)
    # 由rgb和depth获得x
    x, depth_img, rgb_img = cam_data.get_data(rgb=rgb, depth=depth)
    print(f'========x:\n========{x}')
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

    # ==================x to q_img, ang_img, width_img==================
    with torch.no_grad():
        xc = x.to(device)
        pred = model.predict(xc)
    q_img, ang_img, width_img = post_process_output(pred['pos'], pred['cos'], pred['sin'], pred['width'])
    print(f'========q_img:========\n{q_img}')
    print(f'========ang_img:========\n{ang_img}')
    print(f'========width_img:========\n{width_img}')

    # ==================q_img, ang_img, width_img to grasps==================
    grasps = detect_grasps(q_img, ang_img, width_img)
    print(f'========grasps:========\n{type(grasps)}')
    print(f'========typegrasps:========\n{type(grasps[0])}')
    if len(grasps) == 0:
        print("未检测到抓取位姿!")
        if is_visualize:
            plot_grasp(fig=fig, rgb_img=cam_data.get_rgb(rgb, False), grasps=grasps, save=True)
        # return False

    # ==================grasps to targets(target2cam_xyz)==================
    pos_z = depth[grasps[0].center[0] + cam_data.top_left[0], grasps[0].center[1] + cam_data.top_left[1]]
    pos_x = np.multiply(grasps[0].center[1] + cam_data.top_left[1] - camera_matrix[0][2],
                        pos_z / camera_matrix[0][0])
    pos_y = np.multiply(grasps[0].center[0] + cam_data.top_left[0] - camera_matrix[1][2],
                        pos_z / camera_matrix[1][1])
    # if pos_z == 0:
    #     print("pos_z == 0")
        # return False
    target = np.asarray([pos_x, pos_y, pos_z])
    target.shape = (3, 1)
    print(f'========target:========\n{target}')

    # ==================grasps to target_angle(target2base_rxryrz)==================
    angle = np.asarray([0, 0, grasps[0].angle])
    angle.shape = (3, 1)
    print(f'========angle:========\n{angle}')

    # ==================grasps to width==================
    width = grasps[0].length  # mm
    print(f'========width:========\n{width}')

    # ==================targets(target2cam_xyz) to target_position(target2base_xyz)==================
    target_position = np.dot(cam2base_H[0:3, 0:3], target) + cam2base_H[0:3, 3:]
    target_position = target_position[0:3, 0]
    print(f'========target_position:========\n{target_position}')

    target_angle = np.dot(cam2base_H[0:3, 0:3], angle)
    print(f'========target_angle:========\n{target_angle}')

    # ==================target_position,target_angle 2 grasp_pose(target2base_xyzrxryrz)==================
    grasp_pose = np.append(target_position, target_angle[2])

    # ==================开始抓取target2base_xyzrxryrz==================
    # 但是目前的平面抓取还没有调试（包括机械臂夹爪、位置的测试)
    # success = robot.plane_grasp([grasp_pose[0], grasp_pose[1], grasp_pose[2] - 0.005], yaw=grasp_pose[3],open_size=width / 100)


    if is_visualize:
        plot_grasp(fig=fig, rgb_img=cam_data .get_rgb(rgb, False), grasps=grasps, save=True)

async def main():
    # ======================拍摄rgb图和深度图=======================
    realsense = RealsenseD435()
    realsense.get_rgb_and_depth_image()
    rgb_path = "./image/color_image.jpg"
    depth_path = "./image/depth_image.jpg"

    # num = 1
    # rgb_path = f"image/rgb{num}.png"
    # depth_path = f"image/depth{num}.png"

    # # =====================初始化Robot对象=====================
    print(f'===============初始化robot并连接中.....==============')
    robot = FR5ROBOT(host='192.168.58.2', port=8080)
    await robot.connect()

    # ==================导入camera_params==================
    with open('cfg/camera_params.json', 'r') as f:
        params = json.load(f)
    camera_matrix = np.array(params['camera_matrix'])
    distortion_coefficients = np.array(params['distortion_coefficients'])
    Camera_Params = namedtuple('Camera_Params', ['camera_matrix', 'distortion_coefficients'])
    camera_params = Camera_Params(camera_matrix, distortion_coefficients)

    # ==================导入cam2base_H==================
    with open('cfg/cam2base_H.csv', newline='') as csvfile:
        # 创建CSV读取器
        reader = csv.reader(csvfile, delimiter=',')
        # 读取CSV文件中的数据
        data = []
        for row in reader:
            data.append(row)
    cam2base_H = np.array(data, dtype=np.float32)

    # ==================导入camera_depth_scale==================
    camera_depth_scale = 0
    with open('cfg/camera_depth_scale.txt', 'r') as f:
        camera_depth_scale = float(f.read().strip())

    # ==================定义导入模型路径和device================
    saved_model_path = '../../trained-models/jacquard-rgbd-grconvnet3-drop0-ch32/epoch_48_iou_0.93'
    device = 'cpu'

    # =================是否可视化================
    is_visualize = True
    
    # =================开始抓取==================
    GRCNN_rgb_and_depth_to_target2base_xyzrxryrz(rgb_path,depth_path,is_visualize,saved_model_path,device,camera_params.camera_matrix,camera_depth_scale,cam2base_H)
    
    # ================机械臂断开连接================
    await robot.disconnect()

if __name__ == '__main__':
    asyncio.run(main())