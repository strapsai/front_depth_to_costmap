#!/usr/bin/env python3
"""
ROS 2 node: Traversability → Occupancygrid (body frame)
"""

import os, struct, yaml
from typing import Dict, Optional
import rclpy
import numpy as np
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge
from transforms3d.quaternions import quat2mat
from message_filters import Subscriber, ApproximateTimeSynchronizer
from visualization_msgs.msg import Marker


# iwshim
from nav_msgs.msg import Odometry, OccupancyGrid, MapMetaData #iwshim 25.06.02

import time, torch
from functools import wraps
import torch.nn.functional as F #mhlee 25.07.08
import torchvision.transforms as transforms #mhlee 25.07.10

# from depth_to_pointcloud_pub.models.fastflow_SSL_n import PSPNET_RADIO_SSL #mhlee 25.07.10
# from depth_to_pointcloud_pub.models.proxy_n_c import Proxy #mhlee 25.07.10
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt
from PIL import Image as PILImage
import random
import cv2
from scipy.ndimage import gaussian_filter
from geometry_msgs.msg import PointStamped # RViz 시각화를 위한 PointStamped (header 있음)
from std_msgs.msg import Header # PointStamped의 header를 채우기 위해 필요
from rclpy.executors import MultiThreadedExecutor # MultiThreadedExecutor 임포트


def measure_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        elapse_time = (end - start) * 1000
        # print(f"[{func.__name__}] Elapsed time: {elapse_time:.4f} msec")
        return result
    return wrapper
    
# ────────────────────── utils ──────────────────────
def load_extrinsic(yaml_name: str, key: str) -> np.ndarray:
    pkg = get_package_share_directory("depth_to_pointcloud_pub")
    with open(os.path.join(pkg, "config", yaml_name), "r", encoding="utf-8") as f:
        return np.array(yaml.safe_load(f)[key]["Matrix"]).reshape(4, 4)
    
def load_intrinsic(yaml_name: str, key: str) -> np.ndarray:
    pkg = get_package_share_directory("depth_to_pointcloud_pub")
    with open(os.path.join(pkg, "config", yaml_name), "r", encoding="utf-8") as f:
        return np.array(yaml.safe_load(f)[key]["Matrix"]).reshape(3, 3)


def fix_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ────────────────────── utils for inference ──────────────────────

def prepare_image(image_left, image_right, input_size=(512, 512)):
    original_size_left = image_left.size
    original_size_right = image_right.size
        

    transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[103.939 / 255, 116.779 / 255, 123.68 / 255],
                            std=[0.229, 0.224, 0.225]),
    ])
    tensor_left = transform(image_left).unsqueeze(0).contiguous().cuda()
    tensor_right = transform(image_right).unsqueeze(0).contiguous().cuda()

    batched_tensor = torch.cat([tensor_left, tensor_right], dim=0).contiguous().cuda()

    return batched_tensor, original_size_left, original_size_right

def load_trt_engine(engine_path):
    with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    return engine

def allocate_trt_buffers(engine):
    inputs, outputs, bindings = [], [], []
    stream = cuda.Stream()
    if stream is None:
        print("[ERROR] Failed to create CUDA stream!")
    else:
        print("[INFO] CUDA stream created successfully.")

    for binding in engine:
        shape = engine.get_tensor_shape(binding)  # ✅ 수정: get_binding_shape → get_tensor_shape
        size = trt.volume(shape)
        dtype = trt.nptype(engine.get_tensor_dtype(binding))  # ✅ 수정: get_binding_dtype → get_tensor_dtype
        
        
        print(f"[DEBUG] Binding: {binding}")
        print(f"        Shape: {shape}")
        print(f"        Dtype: {dtype}")
        print(f"        Size: {size}")
        
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        bindings.append(int(device_mem))
        if engine.get_tensor_mode(binding) == trt.TensorIOMode.INPUT:
            inputs.append((host_mem, device_mem))
        else:
            outputs.append((host_mem, device_mem))
    return inputs, outputs, bindings, stream


def infer_trt_image(context, bindings, inputs, outputs, stream, input_tensor):

    # np.copyto(inputs[0][0], input_tensor.cpu().ravel())                             #이거왜 .cpu 붙였지?? (important)
    # cuda.memcpy_htod_async(inputs[0][1], inputs[0][0], stream)
    # context.execute_v2(bindings=bindings)
    # cuda.memcpy_dtoh_async(outputs[0][0], outputs[0][1], stream)
    # stream.synchronize()

    # output = outputs[0][0]
    # return torch.tensor(output).view(1, 1, 512, 512)  # ✅ 올바른 shape

    cuda.memcpy_dtod_async(
        dest=inputs[0][1],
        src=input_tensor.contiguous().data_ptr(),
        size=input_tensor.element_size() * input_tensor.numel(),
        stream=stream
    ) 
    t_2 = time.perf_counter()
    context.execute_v2(bindings=bindings)
    t_3 = time.perf_counter()


    output_tensor = torch.empty((2, 1, 512, 512), dtype=torch.float32, device='cuda')  # ← 원하는 shape으로
    cuda.memcpy_dtod_async(
        dest=output_tensor.data_ptr(),
        src=outputs[0][1],
        size=output_tensor.element_size() * output_tensor.numel(),
        stream=stream
    )
    stream.synchronize()
    
    # print(f"[Timing] real inference: {(t_3 - t_2)*1000:.2f} ms")

    return output_tensor  # ✅ 올바른 shape

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# ────────────────────── main node ──────────────────────
class TraversabilitytoOccupancygridNode(Node):

# ────────────────────────────────────────────────────────────────────────────────────────────────
# ──────────────────────────────── initial 설정 ──────────────────────────────────────────
# ────────────────────────────────────────────────────────────────────────────────────────────────

    def __init__(self):
        super().__init__("traversability_to_occupancygrid_node")

        # -------------------- Parameter  --------------------
        package_share_directory = get_package_share_directory('depth_to_pointcloud_pub')
        self.device = torch.device("cuda:0")
        try:
            self.pycuda_device = cuda.Device(0)
            self.pycuda_context = self.pycuda_device.make_context()
            self.pycuda_context.push() 
            self.get_logger().info(f"PyCUDA Context created and pushed for device {self.pycuda_device.name()}.")
        except Exception as e:
            self.get_logger().error(f"Failed to create/push PyCUDA Context: {e}")
            raise # 컨텍스트 없이는 진행할 수 없으므로 에러 발생

        # -------------------- Parameter for inference --------------------
        self.trt_engine_path = "/home/ros/workspace/src/front_depth_to_costmap/depth_to_pointcloud_pub/depth_to_pointcloud_pub/3_dynamic.plan"
        self.engine = load_trt_engine(self.trt_engine_path)
        self.execution_context = self.engine.create_execution_context()
        self.inputs, self.outputs, self.bindings, self.stream = allocate_trt_buffers(self.engine)

        # -------------------- Parameter  --------------------
        self.depth_camera_ids = ["frontleft_depth", "frontright_depth"] 
        self.rgb_camera_ids = ["frontleft_rgb", "frontright_rgb"]     
        self.all_camera_ids = self.depth_camera_ids + self.rgb_camera_ids 

        self.bridge = CvBridge()

        # Camera Intrinsic & Extrinsic
        # self.camera_parameter = {
        #     "frontleft_depth_extrinsic":  load_camera_parameter("frontleft_info.yaml",  "body_to_frontleft"),
        #     "frontright_depth_extrinsic": load_camera_parameter("frontright_info.yaml", "body_to_frontright"),
        #     "frontleft_rgb_extrinsic":  load_camera_parameter("frontleft_info.yaml",  "body_to_frontleft_fisheye"),
        #     "frontright_rgb_extrinsic": load_camera_parameter("frontright_info.yaml", "body_to_frontright_fisheye"),
        #     "frontleft_depth_intrinsic":  load_camera_parameter("frontleft_info.yaml",  "frontleft_depth_intrinsic"),
        #     "frontright_depth_intrinsic": load_camera_parameter("frontright_info.yaml", "frontright_depth_intrinsic"),
        #     "frontleft_rgb_intrinsic":  load_camera_parameter("frontleft_info.yaml",  "frontleft_fisheye_intrinsic"),
        #     "frontright_rgb_intrinsic": load_camera_parameter("frontright_info.yaml", "frontright_fisheye_intrinsic"),
        # }


        self.T_body_to_frontleft_depth = torch.tensor(load_extrinsic("frontleft_info.yaml",  "body_to_frontleft"), dtype=torch.float32, device=self.device) 
        self.T_body_to_frontleft_fisheye = torch.tensor(load_extrinsic("frontleft_info.yaml",  "body_to_frontleft_fisheye"), dtype=torch.float32, device=self.device) 
        self.T_body_to_frontright_depth = torch.tensor(load_extrinsic("frontright_info.yaml",  "body_to_frontright"), dtype=torch.float32, device=self.device) 
        self.T_body_to_frontright_fisheye = torch.tensor(load_extrinsic("frontright_info.yaml",  "body_to_frontright_fisheye"), dtype=torch.float32, device=self.device) 
        self.K_frontleft_depth = torch.tensor(load_intrinsic("frontleft_info.yaml",  "frontleft_depth_intrinsic"), dtype=torch.float32, device=self.device) 
        self.K_frontleft_fisheye = torch.tensor(load_intrinsic("frontleft_info.yaml",  "frontleft_fisheye_intrinsic"), dtype=torch.float32, device=self.device) 
        self.K_frontright_depth = torch.tensor(load_intrinsic("frontright_info.yaml",  "frontright_depth_intrinsic"), dtype=torch.float32, device=self.device) 
        self.K_frontright_fisheye = torch.tensor(load_intrinsic("frontright_info.yaml",  "frontright_fisheye_intrinsic"), dtype=torch.float32, device=self.device) 
        
        # -------------------- Frame & Topic --------------------

        # Frame ID
        self.body_frame = "spot1/base/spot/body"
        self.odom_frame = "spot1/base/spot/odom"    # iwshim. 25.05.30

        # Topic name 
        self.odom_topic = "/spot1/base/spot/odometry" # iwshim. 25.05.30
        self.accum_topic = "/spot1/base/spot/depth/accum_front"
        self.occup_topic = "/spot1/base/spot/depth/occup_front"


        # -------------------- Publisher  --------------------
        
        self.pub_accum = self.create_publisher(PointCloud2, self.accum_topic, 10)
        self.pub_occup = self.create_publisher(OccupancyGrid, self.occup_topic, 10)
        self.pub_normal = self.create_publisher(Marker, "/spot1/base/spot/depth/normal_front", 10) #mhlee 25.06.23
        self.pub_image_left = self.create_publisher(Image, "/spot1/base/spot/camera/similarity_left", 10)
        self.pub_image_right = self.create_publisher(Image, "/spot1/base/spot/camera/similarity_right", 10)
        self.pub_robot_center_point = self.create_publisher(PointStamped, '/robot/center_point_for_rviz', 10)

        # -------------------- Subscriber & Syncronizer  --------------------    
        # Subscriber for main Data
        self.sub_leftdepth  = Subscriber(self, Image, "/spot1/base/spot/depth/frontleft/image")
        self.sub_rightdepth = Subscriber(self, Image, "/spot1/base/spot/depth/frontright/image")
        self.sub_odom  = Subscriber(self, Odometry, self.odom_topic) # iwshim. 25.05.30
        self.sub_leftrgb = Subscriber(self, Image, "/spot1/base/spot/camera/frontleft/image")
        self.sub_rightrgb = Subscriber(self, Image, "/spot1/base/spot/camera/frontright/image")

        self.sync = ApproximateTimeSynchronizer(
            [self.sub_leftdepth, self.sub_rightdepth, self.sub_leftrgb, self.sub_rightrgb, self.sub_odom], # iwshim. 25.05.30
            queue_size=10,                    
            slop=0.5)
        self.sync.registerCallback(self.occupancy_cb) # mhlee 25.06.19

        
    #     # 주기 기반 동기화 
    #     self.latest_msgs = {}
    #     self.sub_leftdepth.registerCallback(lambda msg: self.store_msg('left_depth', msg))
    #     self.sub_rightdepth.registerCallback(lambda msg: self.store_msg('right_depth', msg))
    #     self.sub_leftrgb.registerCallback(lambda msg: self.store_msg('left_rgb', msg))
    #     self.sub_rightrgb.registerCallback(lambda msg: self.store_msg('right_rgb', msg))
    #     self.sub_odom.registerCallback(lambda msg: self.store_msg('odom', msg))

    #     self.sync_timer = self.create_timer(1.0 / 14.0, self.sync_callback)  


    # def store_msg(self, key, msg):
    #     self.latest_msgs[key] = msg

    # def sync_callback(self):
    #     if len(self.latest_msgs) == 5:
    #         self.occupancy_cb(
    #             self.latest_msgs['left_depth'],
    #             self.latest_msgs['right_depth'],
    #             self.latest_msgs['left_rgb'],
    #             self.latest_msgs['right_rgb'],
    #             self.latest_msgs['odom']
    #         )
    

# ────────────────────────────────────────────────────────────────────────────────────────────────
# ──────────────────────────────── callback 함수 ──────────────────────────────────────────
# ────────────────────────────────────────────────────────────────────────────────────────────────
    # ───────────── occupancy callback ─────────────  # mhlee 25.06.19

    

    def occupancy_cb(self, msg_leftdepth : Image, msg_rightdepth : Image, msg_leftrgb : Image, msg_rightrgb : Image, msg_odom : Odometry): 
        try:
            # 이미 생성된 컨텍스트를 현재 스레드에 push (활성화)
            self.pycuda_context.push() 
            self.get_logger().debug("PyCUDA Context pushed in callback.")
        except Exception as e:
            self.get_logger().error(f"Failed to push PyCUDA Context in callback: {e}")
            # 컨텍스트 활성화 실패 시, GPU 작업은 불가능하므로 여기서 리턴하거나 예외 처리
            return 
        

        self.get_logger().info(f"Left Depth stamp: {msg_leftdepth.header.stamp.sec}.{msg_leftdepth.header.stamp.nanosec}")
        self.get_logger().info(f"Right Depth stamp: {msg_rightdepth.header.stamp.sec}.{msg_rightdepth.header.stamp.nanosec}")
        self.get_logger().info(f"Left RGB stamp: {msg_leftrgb.header.stamp.sec}.{msg_leftrgb.header.stamp.nanosec}")
        self.get_logger().info(f"Right RGB stamp: {msg_rightrgb.header.stamp.sec}.{msg_rightrgb.header.stamp.nanosec}")
        self.get_logger().info(f"Odom stamp: {msg_odom.header.stamp.sec}.{msg_odom.header.stamp.nanosec}")
        # print(f"{msg_odom.header.stamp.sec}.{msg_odom.header.stamp.nanosec}")
        # print(f"{msg_rightrgb.header.stamp.sec}.{msg_rightrgb.header.stamp.nanosec}")

        # initial fofr CPU

        stamp = rclpy.time.Time.from_msg(msg_odom.header.stamp)
        # self.get_logger().info("GPU-Accelerated Callback (PyTorch3D Version)")

                
        depth_raw_left = self.bridge.imgmsg_to_cv2(msg_leftdepth, "passthrough")  ## 16UC1 → np.ndarray
        depth_raw_right = self.bridge.imgmsg_to_cv2(msg_rightdepth, "passthrough")  ## 16UC1 → np.ndarray

        cv_image_left_rgb = self.bridge.imgmsg_to_cv2(msg_leftrgb, "bgr8")
        pil_image_left_rgb = PILImage.fromarray(cv2.cvtColor(cv_image_left_rgb, cv2.COLOR_BGR2RGB))
        # pil_image_left_rgb = PILImage.fromarray(cv_image_left_rgb)

        cv_image_right_rgb = self.bridge.imgmsg_to_cv2(msg_rightrgb, "bgr8")
        pil_image_right_rgb = PILImage.fromarray(cv2.cvtColor(cv_image_right_rgb, cv2.COLOR_BGR2RGB))
        # pil_image_right_rgb = PILImage.fromarray(cv_image_right_rgb)

        depth_m_left   = depth_raw_left.astype(np.float32) / 1000.0  ## mm → m
        depth_m_right   = depth_raw_right.astype(np.float32) / 1000.0  ## mm → m

        # 유효하지 않은 Depth 0으로 설정
        depth_m_left[depth_m_left <  0.1] = 0.0      
        depth_m_left[depth_m_left > 3.0] = 0.0                                                        
        depth_m_right[depth_m_right < 0.1] = 0.0      
        depth_m_right[depth_m_right > 3.0] = 0.0      

        pos = msg_odom.pose.pose.position
        ori = msg_odom.pose.pose.orientation
        T_odom = self.transform_to_matrix(pos, ori) 
        T_odom_gpu = torch.tensor(T_odom, dtype=torch.float32, device=self.device) 
        # self.get_logger().info(f"Odom pose: position=({pos.x:.2f}, {pos.y:.2f}, {pos.z:.2f}), orientation=({ori.x:.2f}, {ori.y:.2f}, {ori.z:.2f}, {ori.w:.2f})")

        # inpaint filter 적용

        # mask_left = ((depth_m_left < 0.1) | (depth_m_left > 5.0)).astype(np.uint8) * 255
        # mask_right = ((depth_m_right < 0.1) | (depth_m_right > 5.0)).astype(np.uint8) * 255

        # depth_m_left = cv2.inpaint(depth_m_left, mask_left, 7, cv2.INPAINT_TELEA)
        # depth_m_right = cv2.inpaint(depth_m_right, mask_right, 7, cv2.INPAINT_TELEA)


        # gaussian filter 적용
        depth_m_left = cv2.GaussianBlur(depth_m_left, (5,5), 0)
        depth_m_right = cv2.GaussianBlur(depth_m_right, (5,5), 0)

        # # median filter 적용
        # depth_m_left = cv2.medianBlur(depth_m_left, 5)
        # depth_m_right = cv2.medianBlur(depth_m_right, 5)

        depth_m_left = torch.tensor(depth_m_left, device=self.device)
        depth_m_right = torch.tensor(depth_m_right, device=self.device)



        ####### image debuging#######
        # depth_np_left = depth_m_left.detach().cpu().numpy()
        # depth_np_right = depth_m_right.detach().cpu().numpy()

        # img_msg_left = self.bridge.cv2_to_imgmsg(depth_np_left, encoding='32FC1')
        # img_msg_right = self.bridge.cv2_to_imgmsg(depth_np_right, encoding='32FC1')
        # self.pub_image_left.publish(img_msg_left)
        # self.pub_image_right.publish(img_msg_right)
        ####### image debuging#######


        traversability_left, traversability_right = self.image_to_similarity(pil_image_left_rgb, pil_image_right_rgb)

        # -- GPU start -- 
        pts_left_body_gpu  = self.depth_to_pts_frame_body(depth_m_left, self.K_frontleft_depth, self.T_body_to_frontleft_depth)
        pts_right_body_gpu = self.depth_to_pts_frame_body(depth_m_right, self.K_frontright_depth, self.T_body_to_frontright_depth)
    
        pts_with_traversability_left = self.merge_traversability_to_pointcloud_frame_body(pts_left_body_gpu, traversability_left, self.K_frontleft_fisheye, self.T_body_to_frontleft_fisheye)
        pts_with_traversability_right = self.merge_traversability_to_pointcloud_frame_body(pts_right_body_gpu, traversability_right, self.K_frontright_fisheye, self.T_body_to_frontright_fisheye)

        pts_with_traversability_body = torch.cat([pts_with_traversability_left, pts_with_traversability_right], dim=0) # 지금은 중복된것을 torch.cat으로 했는데 , 보수적인 값으로 추정하면서도 연산에 방해안되는 것으로 바꿔보기(Important)

        pts_with_traversability_downsampled = self.voxel_downsample_mean_traversability(pts_with_traversability_body, 0.05)

        points_xyz = pts_with_traversability_downsampled[:, :3] 
        points_homo = torch.cat([points_xyz, torch.ones(points_xyz.shape[0], 1, device=self.device)], dim=1) 

        points_world = (T_odom_gpu @ points_homo.T).T[:, :3] 
        points_final = torch.cat([points_world, pts_with_traversability_downsampled[:, 3:4]], dim=1) 
        
        # -- GPU end -- 

        points_final_cpu = points_final.cpu().numpy()

        og = self.pointcloud_with_traversability_to_occupancy_grid(stamp=stamp, 
            frame=self.odom_frame, 
            pts_with_t=points_final_cpu,
            resolution=0.1, 
            grid_size=150, 
            center_xy=(pos.x, pos.y), 
            )

         # 로봇의 현재 위치 (pos.x, pos.y)를 PointStamped 메시지로 발행
        # current_robot_point_msg = PointStamped()
        # current_robot_point_msg.header.stamp = stamp.to_msg() # 깊이 이미지와 동일한 타임스탬프 사용
        # current_robot_point_msg.header.frame_id = self.odom_frame # OccupancyGrid와 동일한 프레임 사용
        # current_robot_point_msg.point.x = pos.x
        # current_robot_point_msg.point.y = pos.y
        # current_robot_point_msg.point.z = -2.0 # z 값도 포함 (바닥에 점이 찍히도록)
        # self.pub_robot_center_point.publish(current_robot_point_msg)
        # self.get_logger().debug(f"Published robot center point: ({pos.x}, {pos.y}, {pos.z}) in {self.odom_frame} frame.")


        # points_final_cpu = pts_with_traversability_downsampled.cpu().numpy()

        # og = self.pointcloud_with_traversability_to_occupancy_grid(stamp=stamp, 
        # frame=self.body_frame, 
        # pts_with_t=points_final_cpu,
        # resolution=0.1, 
        # grid_size=150, 
        # center_xy=(0.0,0.0), 
        # )


        # pc = self.build_pc(msg_leftdepth.header.stamp, self.odom_frame, points_final_cpu[:, :3])
        # self.pub_accum.publish(pc)
        self.pub_occup.publish(og)

        try:
            self.pycuda_context.pop()
            self.get_logger().debug("PyCUDA Context popped in callback.")
        except Exception as e:
            self.get_logger().error(f"Failed to pop PyCUDA Context in callback: {e}")

# ────────────────────────────────────────────────────────────────────────────────────────────────
# ──────────────────────────────── Filtering ──────────────────────────────────────────
# ────────────────────────────────────────────────────────────────────────────────────────────────

    # @measure_time
    # def depth_hole_filling_with_grabcut(depth):
    #     """
    #     depth: np.float32, hole=0, 단위 m
    #     return: hole이 채워진 depth map
    #     """
    #     mask = np.where(depth == 0, 0, 1).astype('uint8')  # 0이면 배경, 아니면 전경

    #     grabcut_mask = np.where(mask==1, 3, 0).astype('uint8')

    #     # 3. 임의의 rect (전체 이미지 영역)
    #     h, w = depth.shape
    #     rect = (1, 1, w-2, h-2)

    #     # 4. bgdModel, fgdModel 초기화 (필요한 크기)
    #     bgdModel = np.zeros((1,65), np.float64)
    #     fgdModel = np.zeros((1,65), np.float64)

    #     # 5. grabCut 실행
    #     cv2.grabCut(depth.astype('uint8'), grabcut_mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)

    #     # 6. 최종 마스크 얻기 (0,2는 배경, 1,3은 전경)
    #     final_mask = np.where((grabcut_mask==2)|(grabcut_mask==0), 0, 1).astype('uint8')

    #     # 7. hole(배경) 부분을 전경의 평균값으로 채우기 (간단 보간)
    #     mean_depth = depth[final_mask == 1].mean()
    #     filled_depth = depth.copy()
    #     filled_depth[final_mask == 0] = mean_depth

    #     return filled_depth


# ────────────────────────────────────────────────────────────────────────────────────────────────
# ──────────────────────────────── depth Image → 3-D 포인트 변환 ────────────────────────────────────
# ────────────────────────────────────────────────────────────────────────────────────────────────
    @measure_time
    def depth_to_pts_frame_body(self, depth_m: torch.Tensor, K : torch.Tensor, T: torch.Tensor) -> torch.Tensor:
       
        # 픽셀 그리드 생성
        h, w = depth_m.shape
        fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
        u = torch.arange(w, device=self.device).float().repeat(h, 1)
        v = torch.arange(h, device=self.device).float().repeat(w, 1).T

        # 핀홀 역변환: (u,v,depth) → (x,y,z)
        z = depth_m
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        
        valid_mask = z > 0.1
        pts_cam = torch.stack((x[valid_mask], y[valid_mask], z[valid_mask]), dim=1)
        pts_cam_homo = torch.cat([pts_cam, torch.ones(pts_cam.shape[0], 1, device=self.device)], dim=1)
        pts_body_homo = T @ pts_cam_homo.T 
        pts_body = pts_body_homo[:3].T 

        return pts_body


# ────────────────────────────────────────────────────────────────────────────────────────────────
# ──────────────────────────────── Traversability 함수 ──────────────────────────────────────────
# ────────────────────────────────────────────────────────────────────────────────────────────────

    # @measure_time
    def image_to_similarity(self, image_left, image_right):

        t_start = time.perf_counter()
        input_tensor, original_size_left, original_size_right = prepare_image(image_left, image_right)
        t_preprocess = time.perf_counter()

        result = infer_trt_image(
            self.execution_context, self.bindings, self.inputs,
            self.outputs, self.stream, input_tensor
        )
        t_infer = time.perf_counter()


        resized_map_left= F.interpolate(result[0:1, :, :, :], size=original_size_left[::-1], mode='bilinear', align_corners=False)
        t_resize_left = time.perf_counter()
    
        resized_map_right = F.interpolate(result[1:2, :, :, :], size=original_size_right[::-1], mode='bilinear', align_corners=False)        
        t_resize_right = time.perf_counter()

        resized_map_left = (resized_map_left + 1.0) / 2.0
        resized_map_right = (resized_map_right + 1.0) / 2.0

        # GPU 텐서를 CPU NumPy 배열로 변환
        # similarity_np_left = resized_map_left.squeeze().cpu().numpy()
        # similarity_np_right = resized_map_right.squeeze().cpu().numpy()

        # # Image 메시지로 변환 및 발행
        # img_msg_left = self.bridge.cv2_to_imgmsg(similarity_np_left, encoding='32FC1')
        # self.pub_image_left.publish(img_msg_left)
        # img_msg_right = self.bridge.cv2_to_imgmsg(similarity_np_right, encoding='32FC1')
        # self.pub_image_right.publish(img_msg_right)
    
        # print(f"similarity map left min/max: {resized_map_left.min().item()} / {resized_map_left.max().item()}")
        # print(f"similarity map right min/max: {resized_map_right.min().item()} / {resized_map_right.max().item()}")

        return resized_map_left.to(self.device), resized_map_right.to(self.device)

# ────────────────────────────────────────────────────────────────────────────────────────────────
# ──────────────────────────────── Traversability projection 함수 ───────────────────────────
# ────────────────────────────────────────────────────────────────────────────────────────────────
    @measure_time
    def merge_traversability_to_pointcloud_frame_body(self, pts_body: torch.Tensor, traversability_img: torch.Tensor, K_rgb : torch.Tensor, T_rgb_to_body : torch.Tensor) -> Optional[torch.Tensor]:
        
        N = pts_body.shape[0]
        pts4 = torch.cat([pts_body, torch.ones(N, 1, device=self.device)], dim=1) 


        T_body_to_rgb = torch.inverse(T_rgb_to_body) 

        pts_rgb = pts4 @ T_body_to_rgb.T[:, :3]

        x, y, z = pts_rgb[:, 0], pts_rgb[:, 1], pts_rgb[:, 2]

        fx, fy = K_rgb[0, 0], K_rgb[1, 1]
        cx, cy = K_rgb[0, 2], K_rgb[1, 2]

        u = (fx * x / z + cx)
        v = (fy * y / z + cy)

        H, W = traversability_img.shape[2], traversability_img.shape[3]
        valid = (z > 0.1) & (z < 5.0) & (u >= 0) & (u < W) & (v >= 0) & (v < H) 

        pts_body_valid = pts_body[valid]
        t = traversability_img[0, 0, v[valid].long(), u[valid].long()]

        pts_with_t = torch.cat([pts_body_valid, t.unsqueeze(1)], dim=1) # (N, 4)
        return pts_with_t

# ────────────────────────────────────────────────────────────────────────────────────────────────
# ──────────────────────────────── Downsampling 관련 함수 ──────────────────────────────────────────
# ────────────────────────────────────────────────────────────────────────────────────────────────  

    @staticmethod
    @measure_time
    def voxel_downsample_mean_traversability(points_with_t_gpu: torch.Tensor, voxel_size: float) -> torch.Tensor:
        """
        PyTorch 텐서 연산만으로 Voxel Grid의 중심점과 트래버서빌리티를 찾아 다운샘플링합니다.
        
        Args:
            points_with_t_gpu: (N, 4) 모양의 포인트 클라우드 텐서 (XYZT) (GPU에 있어야 함).
            voxel_size: 복셀의 크기 (미터 단위).

        Returns:
            다운샘플링된 (M, 4) 모양의 포인트 클라우드 텐서 (XYZT) (GPU에 있음).
        """
        if points_with_t_gpu.shape[0] == 0:
            return points_with_t_gpu

        # XYZ 값만 사용하여 복셀 인덱스 계산
        voxel_indices = torch.floor(points_with_t_gpu[:, :3] / voxel_size).long()

        # 고유한 복셀 인덱스와 역 인덱스, 개수 찾기
        unique_voxel_indices, inverse_indices, counts = torch.unique(
            voxel_indices, dim=0, return_inverse=True, return_counts=True)

        num_unique_voxels = unique_voxel_indices.shape[0]
        
        # 각 고유 복셀에 속한 포인트들의 XYZT 합계 계산
        # sum_points_per_voxel = torch.zeros((num_unique_voxels, 4), device=points_with_t_gpu.device)
        # sum_points_per_voxel.scatter_add_(0, inverse_indices.unsqueeze(1).expand(-1, 4), points_with_t_gpu)
        
        # 합계를 개수로 나누어 평균(중심점) 계산
        # mean_points_per_voxel = sum_points_per_voxel / counts.unsqueeze(1)

        sum_xyz = torch.zeros((num_unique_voxels, 3), device=points_with_t_gpu.device)
        sum_xyz.scatter_add_(0, inverse_indices.unsqueeze(1).expand(-1, 3), points_with_t_gpu[:, :3])
        mean_xyz = sum_xyz / counts.unsqueeze(1)

        min_t = torch.full((num_unique_voxels, ), float('inf'), device=points_with_t_gpu.device)
        min_t = min_t.scatter_reduce(0, inverse_indices, points_with_t_gpu[:, 3], reduce='amin')

        result = torch.cat([mean_xyz, min_t.unsqueeze(1)], dim=1)

        
        return result


# ────────────────────────────────────────────────────────────────────────────────────────────────
# ──────────────────────────────── Occupancygrid 관련 함수 ──────────────────────────────────────────
# ────────────────────────────────────────────────────────────────────────────────────────────────
    @staticmethod
    @measure_time
    def pointcloud_with_traversability_to_occupancy_grid(stamp, frame: str, pts_with_t: np.ndarray, resolution, grid_size, center_xy):
        """ Parameters
        resolution: 0.05m
        grid_size: the number of cells per each side
        coord_center: 
            Origin(left bottom x) = center[0] - 0.5 * grid_size * resolution
        """

        origin_x = center_xy[0] - 0.5 * grid_size * resolution
        origin_y = center_xy[1] - 0.5 * grid_size * resolution

        x = pts_with_t[:, 0]
        y = pts_with_t[:, 1]
        t = pts_with_t[:, 3]


        # Points to grid index
        indx = np.floor((x - origin_x) / resolution).astype(np.int32)
        indy = np.floor((y - origin_y) / resolution).astype(np.int32)
        
         # 사전에 미리 제거 했으나, 만일의 경우를 대비해서(segmentation fault) mask-out
        mask = (indx >= 0) & (indx < grid_size) & (indy >= 0) & (indy < grid_size)
        indx, indy, t = indx[mask], indy[mask], t[mask]

        grid = np.full(grid_size * grid_size, 101, dtype=np.int16)
        traversability_value = np.round((1 - t) * 100).astype(np.int16)  # 0이 주행가능, 100이 주행불가능
        flat_indices = indy * grid_size + indx
        np.minimum.at(grid, flat_indices, traversability_value)
        grid[grid == 101] = -1
        grid = grid.reshape((grid_size, grid_size)).astype(np.int8)

        # Filtering
        # mask = (grid == -1).astype(np.uint8) 
            
        # grid_for_inpaint = grid.copy()
        # grid_for_inpaint[mask == 1] = 0
        # grid_for_inpaint = ((grid_for_inpaint + 1) * 2).clip(0, 255).astype(np.uint8)

        # inpainted = cv2.inpaint(grid_for_inpaint, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
        # result = (inpainted.astype(np.float32) / 2) - 1
        # result = result.astype(np.int8)
        # grid = result
        # Filtering

        og = OccupancyGrid()
        og.header.stamp = stamp.to_msg()
        og.header.frame_id = frame
        og.info.resolution = resolution
        og.info.width = grid_size
        og.info.height = grid_size
        og.info.origin.position.x = origin_x
        og.info.origin.position.y = origin_y
        og.info.origin.position.z = -2.0 
        og.data = grid.flatten(order='C').astype(int).tolist()
        
        return og



# ────────────────────────────────────────────────────────────────────────────────────────────────
# ──────────────────────────────── 기타 ──────────────────────────────────────────
# ────────────────────────────────────────────────────────────────────────────────────────────────     

        
    @staticmethod
    def transform_to_matrix(position, orientation) -> np.ndarray:
        import transforms3d
        T = np.eye(4)

        T[0, 3] = position.x
        T[1, 3] = position.y
        T[2, 3] = position.z

        quat = [orientation.w,
                orientation.x,
                orientation.y,
                orientation.z]
        R = transforms3d.quaternions.quat2mat(quat)

        T[:3, :3] = R

        return T 
    
    
    # ───────────── numpy 포인트 → PointCloud2 메시지 ─────────────
    @staticmethod
    @measure_time
    def build_pc(stamp, frame: str, points: np.ndarray) -> PointCloud2:
        fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        cloud = PointCloud2()
        cloud.header.stamp = stamp
        cloud.header.frame_id = frame  
        cloud.height = 1
        cloud.width = points.shape[0]
        cloud.fields = fields
        cloud.is_bigendian = False
        cloud.point_step = 12
        cloud.row_step = 12 * points.shape[0]
        cloud.is_dense = True
        cloud.data = b"".join(struct.pack("fff", *pt) for pt in points.astype(np.float32))
        return cloud
    
# ───────────── 엔트리 포인트 ─────────────
def main(argv=None):
    rclpy.init(args=argv)                ## rclpy 초기화
    node = TraversabilitytoOccupancygridNode()       ## 노드 인스턴스 생성
    rclpy.spin(node)                     ## 콜백 루프 진입
    node.destroy_node()                  ## 종료 시 정리
    rclpy.shutdown()                     ## rclpy 종료



if __name__ == "__main__":
    main()                               ## python filename.py 실행 시 main 호출

