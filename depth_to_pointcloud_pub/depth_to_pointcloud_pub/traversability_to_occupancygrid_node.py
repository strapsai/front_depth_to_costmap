#!/usr/bin/env python3
"""
ROS 2 node: Traversability → Occupancygrid (body frame)

"""

import os
import struct
import yaml
import time
import threading
from functools import wraps
from typing import Optional
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import cv2
from PIL import Image as PILImage
from scipy.spatial.transform import Rotation as R
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2, PointField
from nav_msgs.msg import Odometry, OccupancyGrid
from cv_bridge import CvBridge
from ament_index_python.packages import get_package_share_directory
from message_filters import Subscriber, Cache
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt


# ────────────────────── Utilities ──────────────────────

def measure_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        elapse_time = (end - start) * 1000
        print(f"[{func.__name__}] Elapsed time: {elapse_time:.4f} msec")
        return result
    return wrapper
    

def load_extrinsic(yaml_name: str, key: str) -> np.ndarray:
    pkg = get_package_share_directory("depth_to_pointcloud_pub")
    with open(os.path.join(pkg, "config", yaml_name), "r", encoding="utf-8") as f:
        return np.array(yaml.safe_load(f)[key]["Matrix"]).reshape(4, 4)

def load_topics(spot_id: str) -> dict:
    pkg = get_package_share_directory("depth_to_pointcloud_pub")
    with open(os.path.join(pkg, "config", "topics.yaml"), "r", encoding="utf-8") as f:
        return yaml.safe_load(f)[spot_id]

def load_intrinsic(yaml_name: str, key: str) -> np.ndarray:
    pkg = get_package_share_directory("depth_to_pointcloud_pub")
    with open(os.path.join(pkg, "config", yaml_name), "r", encoding="utf-8") as f:
        return np.array(yaml.safe_load(f)[key]["Matrix"]).reshape(3, 3)

# ────────────────────── Inference Utilities ──────────────────────

def prepare_image(image_left, image_right, input_size=(512, 512)):
    original_size_left, original_size_right = None, None
    tensor_left, tensor_right = None, None
        

    transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[103.939 / 255, 116.779 / 255, 123.68 / 255],
                            std=[0.229, 0.224, 0.225]),
    ])

    if image_left:
        original_size_left = image_left.size
        tensor_left = transform(image_left).unsqueeze(0).contiguous().cuda()
    else:
        tensor_left = torch.zeros((1, 3, *input_size), dtype=torch.float32, device='cuda')

    if image_right:
        original_size_right = image_right.size
        tensor_right = transform(image_right).unsqueeze(0).contiguous().cuda()
    else:
        tensor_right = torch.zeros((1, 3, *input_size), dtype=torch.float32, device='cuda')

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
        shape = engine.get_tensor_shape(binding)  
        size = trt.volume(shape)
        dtype = trt.nptype(engine.get_tensor_dtype(binding)) 
        
        
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

    cuda.memcpy_dtod_async(
        dest=inputs[0][1],
        src=input_tensor.contiguous().data_ptr(),
        size=input_tensor.element_size() * input_tensor.numel(),
        stream=stream
    ) 
    t_2 = time.perf_counter()
    context.execute_v2(bindings=bindings)
    t_3 = time.perf_counter()


    output_tensor = torch.empty((2, 1, 512, 512), dtype=torch.float32, device='cuda') 
    cuda.memcpy_dtod_async(
        dest=output_tensor.data_ptr(),
        src=outputs[0][1],
        size=output_tensor.element_size() * output_tensor.numel(),
        stream=stream
    )
    stream.synchronize()
    
    return output_tensor 

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# ────────────────────── main node ──────────────────────
class TraversabilitytoOccupancygridNode(Node):

# ────────────────────────────────────────────────────────────────────────────────────────────────
# ──────────────────────────────── Initialization ──────────────────────────────────────────
# ────────────────────────────────────────────────────────────────────────────────────────────────

    def __init__(self):
        super().__init__("traversability_to_occupancygrid_node")

        # -------------------- GPU & TensorRT setup  --------------------
        self.device = torch.device("cuda:0")
        try:
            self.pycuda_device = cuda.Device(0)
            self.pycuda_context = self.pycuda_device.make_context()
            self.pycuda_context.push() 
            self.get_logger().info(f"PyCUDA Context created and pushed for device {self.pycuda_device.name()}.")
        except Exception as e:
            self.get_logger().error(f"Failed to create/push PyCUDA Context: {e}")
            raise 

        # self.trt_engine_path = "/home/ros/workspace/src/front_depth_to_costmap/traversability_model/traversability_model.plan"
        self.trt_engine_path = "/home/dtc/airlab_ws/autonomy_ws/src/costmap/front_depth_to_costmap/traversability_model/traversability_model.plan"
        self.engine = load_trt_engine(self.trt_engine_path)
        self.execution_context = self.engine.create_execution_context()
        self.inputs, self.outputs, self.bindings, self.stream = allocate_trt_buffers(self.engine)

        # -------------------- parameters  --------------------
        self.depth_camera_ids = ["frontleft_depth", "frontright_depth"] 
        self.rgb_camera_ids = ["frontleft_rgb", "frontright_rgb"]     
        self.all_camera_ids = self.depth_camera_ids + self.rgb_camera_ids 

        self.bridge = CvBridge()

        self.T_body_to_frontleft_depth = torch.tensor(load_extrinsic("frontleft_info.yaml",  "body_to_frontleft"), dtype=torch.float32, device=self.device) 
        self.T_body_to_frontleft_fisheye = torch.tensor(load_extrinsic("frontleft_info.yaml",  "body_to_frontleft_fisheye"), dtype=torch.float32, device=self.device) 
        self.T_body_to_frontright_depth = torch.tensor(load_extrinsic("frontright_info.yaml",  "body_to_frontright"), dtype=torch.float32, device=self.device) 
        self.T_body_to_frontright_fisheye = torch.tensor(load_extrinsic("frontright_info.yaml",  "body_to_frontright_fisheye"), dtype=torch.float32, device=self.device) 
        self.K_frontleft_depth = torch.tensor(load_intrinsic("frontleft_info.yaml",  "frontleft_depth_intrinsic"), dtype=torch.float32, device=self.device) 
        self.K_frontleft_fisheye = torch.tensor(load_intrinsic("frontleft_info.yaml",  "frontleft_fisheye_intrinsic"), dtype=torch.float32, device=self.device) 
        self.K_frontright_depth = torch.tensor(load_intrinsic("frontright_info.yaml",  "frontright_depth_intrinsic"), dtype=torch.float32, device=self.device) 
        self.K_frontright_fisheye = torch.tensor(load_intrinsic("frontright_info.yaml",  "frontright_fisheye_intrinsic"), dtype=torch.float32, device=self.device) 
        
        # -------------------- Frame & Topic --------------------
        spot_id = self.declare_parameter("spot_id", "spot1").get_parameter_value().string_value
        spot_topics = load_topics(spot_id)       
        
        # Frame ID
        self.body_frame = spot_topics["body_frame"]
        self.odom_frame = spot_topics["odom_frame"]

        # -------------------- Publisher  --------------------

        self.pub_occup = self.create_publisher(OccupancyGrid, spot_topics["occup_topic"], 10)
        self.pub_combined_image_raw = self.create_publisher(Image, spot_topics["combined_image_raw"], 10)
        self.pub_combined_image_similarity = self.create_publisher(Image, spot_topics["combined_image_similarity"], 10)

        # -------------------- Subscriber & Syncronizer  --------------------    
        self.sub_leftdepth  = Subscriber(self, Image, spot_topics["depth_left"])
        self.sub_rightdepth = Subscriber(self, Image, spot_topics["depth_right"])
        self.sub_odom       = Subscriber(self, Odometry, spot_topics["odom_topic"])
        self.sub_leftrgb    = Subscriber(self, Image, spot_topics["rgb_left"])
        self.sub_rightrgb   = Subscriber(self, Image, spot_topics["rgb_right"])

        self.c_left  = Cache(self.sub_leftdepth,  5)
        self.c_right = Cache(self.sub_rightdepth, 5)
        self.c_rgbL  = Cache(self.sub_leftrgb,    5)
        self.c_rgbR  = Cache(self.sub_rightrgb,   5)
        self.c_odom  = Cache(self.sub_odom,      20)
        self.is_processing = False
        self.timer_handle = self.create_timer(0.1, self.sync_timer_cb)  # 10Hz 타이머

        self.message_timeout = self.declare_parameter("message_timeout", 1.0).get_parameter_value().double_value
        self.get_logger().info(f"Message timeout set to: {self.message_timeout} seconds")


    def sync_timer_cb(self):
        if self.is_processing:
            return

        now = self.get_clock().now()
        
        def get_valid_msg(cache: Cache, topic_name: str) -> Optional[Image | Odometry]:
            msg = cache.getElemBeforeTime(now)
            if msg is None:
                return None
            
            msg_time = rclpy.time.Time.from_msg(msg.header.stamp)
            age = (now - msg_time).nanoseconds / 1e9
            if age > self.message_timeout:
                self.get_logger().warn(f"Message on topic '{topic_name}' is too old ({age:.2f}s > {self.message_timeout}s). Discarding.")
                return None
            return msg
        
        # 1. Check the odom message
        od_msg = get_valid_msg(self.c_odom, "Odom")
        if od_msg is None:
            self.get_logger().error("Odom message is missing or stale. Cannot proceed.")
            return

        # 2. Check the validity of each camera message.
        dl_msg = get_valid_msg(self.c_left, "Left Depth")
        dr_msg = get_valid_msg(self.c_right, "Right Depth")
        rl_msg = get_valid_msg(self.c_rgbL, "Left RGB")
        rr_msg = get_valid_msg(self.c_rgbR, "Right RGB")

        # 3. Check if each side's 'set' (Depth + RGB) is complete.
        is_left_set_valid = (dl_msg is not None and rl_msg is not None)
        is_right_set_valid = (dr_msg is not None and rr_msg is not None)

        # 4. If neither side has a complete set, abort the callback for this cycle.
        if not is_left_set_valid and not is_right_set_valid:
            self.get_logger().warn("A complete data set (Depth & RGB) is not available for either side. Skipping cycle.")
            return

        # 5. Pass only data from complete sets to the main callback (pass None for invalid sets).
        final_dl_msg = dl_msg if is_left_set_valid else None
        final_rl_msg = rl_msg if is_left_set_valid else None
        final_dr_msg = dr_msg if is_right_set_valid else None
        final_rr_msg = rr_msg if is_right_set_valid else None
        
        self.is_processing = True

        threading.Thread(target=self.process_occupancy, args=(
            final_dl_msg, final_dr_msg, final_rl_msg, final_rr_msg, od_msg
        )).start()

    def process_occupancy(self, dl, dr, rl, rr, od):
        try:
            self.occupancy_cb(dl, dr, rl, rr, od)
        finally:
            self.is_processing = False

# ────────────────────────────────────────────────────────────────────────────────────────────────
# ──────────────────────────────── Main Callback ──────────────────────────────────────────
# ────────────────────────────────────────────────────────────────────────────────────────────────
    @measure_time        
    def occupancy_cb(self, msg_leftdepth : Optional[Image], msg_rightdepth : Optional[Image], msg_leftrgb : Optional[Image], msg_rightrgb : Optional[Image], msg_odom : Odometry):        
        try:
            self.pycuda_context.push() 
        except Exception as e:
            self.get_logger().error(f"Failed to push PyCUDA Context in callback: {e}")
            return 
                    
        stamp = rclpy.time.Time.from_msg(msg_odom.header.stamp)

        image_left_cpu, image_right_cpu, traversability_left, traversability_right = self.image_to_similarity(msg_leftrgb, msg_rightrgb)

        pts_with_traversability_body_list = []


        if msg_leftdepth and msg_leftrgb is not None:
            depth_raw_left = self.bridge.imgmsg_to_cv2(msg_leftdepth, "passthrough")
            depth_m_left = depth_raw_left.astype(np.float32) / 1000.0
            depth_m_left[(depth_m_left < 0.1) | (depth_m_left > 3.0)] = 0.0
            depth_m_left = cv2.GaussianBlur(depth_m_left, (5, 5), 0)
            depth_m_left_gpu = torch.tensor(depth_m_left, device=self.device)

            pts_left_body_gpu = self.depth_to_pts_frame_body(depth_m_left_gpu, self.K_frontleft_depth, self.T_body_to_frontleft_depth)
            pts_with_traversability_left = self.merge_traversability_to_pointcloud_frame_body(pts_left_body_gpu, traversability_left, self.K_frontleft_fisheye, self.T_body_to_frontleft_fisheye)
            
            if pts_with_traversability_left is not None and pts_with_traversability_left.shape[0] > 0:
                pts_with_traversability_body_list.append(pts_with_traversability_left)
        
        if msg_rightdepth and msg_rightrgb is not None:
            depth_raw_right = self.bridge.imgmsg_to_cv2(msg_rightdepth, "passthrough")
            depth_m_right = depth_raw_right.astype(np.float32) / 1000.0
            depth_m_right[(depth_m_right < 0.1) | (depth_m_right > 3.0)] = 0.0
            depth_m_right = cv2.GaussianBlur(depth_m_right, (5, 5), 0)
            depth_m_right_gpu = torch.tensor(depth_m_right, device=self.device)
        
            pts_right_body_gpu = self.depth_to_pts_frame_body(depth_m_right_gpu, self.K_frontright_depth, self.T_body_to_frontright_depth)
            pts_with_traversability_right = self.merge_traversability_to_pointcloud_frame_body(pts_right_body_gpu, traversability_right, self.K_frontright_fisheye, self.T_body_to_frontright_fisheye)

            if pts_with_traversability_right is not None and pts_with_traversability_right.shape[0] > 0:
                pts_with_traversability_body_list.append(pts_with_traversability_right)

        if not pts_with_traversability_body_list:
            self.get_logger().warn("No valid point cloud could be generated from available data. Skipping publication.")
            try:
                self.pycuda_context.pop()
            except Exception as e:
                self.get_logger().error(f"Failed to pop PyCUDA Context in empty callback: {e}")
            return


        pos = msg_odom.pose.pose.position
        ori = msg_odom.pose.pose.orientation
        T_odom = self.transform_to_matrix(pos, ori) 
        T_odom_gpu = torch.tensor(T_odom, dtype=torch.float32, device=self.device) 

        # -- GPU start -- 

        pts_with_traversability_body = torch.cat(pts_with_traversability_body_list, dim=0) 

        pts_with_traversability_downsampled = self.voxel_downsample_mean_traversability(pts_with_traversability_body, 0.05)

        points_xyz = pts_with_traversability_downsampled[:, :3] 
        points_homo = torch.cat([points_xyz, torch.ones(points_xyz.shape[0], 1, device=self.device)], dim=1) 

        points_world = (T_odom_gpu @ points_homo.T).T[:, :3] 
        points_final = torch.cat([points_world, pts_with_traversability_downsampled[:, 3:4]], dim=1) 
        # -- GPU end -- 
        
        # self.resize_map_cpu(image_left_cpu, image_right_cpu, traversability_left, traversability_right )
        points_final_cpu = points_final.cpu().numpy()

        og = self.pointcloud_with_traversability_to_occupancy_grid(stamp=stamp, 
            frame=self.odom_frame, 
            pts_with_t=points_final_cpu,
            resolution=0.1, 
            grid_size=150, 
            center_xy=(pos.x, pos.y), 
            z = pos.z
            )

        self.pub_occup.publish(og)

        try:
            self.pycuda_context.pop()

        except Exception as e:
            self.get_logger().error(f"Failed to pop PyCUDA Context in callback: {e}")


# ────────────────────────────────────────────────────────────────────────────────────────────────
# ──────────────────────────────── Depth → Point Cloud ────────────────────────────────────
# ────────────────────────────────────────────────────────────────────────────────────────────────
    # # @measure_time
    def depth_to_pts_frame_body(self, depth_m: torch.Tensor, K : torch.Tensor, T: torch.Tensor) -> torch.Tensor:
       
        h, w = depth_m.shape
        fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
        u = torch.arange(w, device=self.device).float().repeat(h, 1)
        v = torch.arange(h, device=self.device).float().repeat(w, 1).T

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
# ──────────────────────────────── Traversability ──────────────────────────────────────────
# ────────────────────────────────────────────────────────────────────────────────────────────────

    # @measure_time
    def image_to_similarity(self, msg_leftrgb: Optional[Image], msg_rightrgb: Optional[Image]):       
       
        if not msg_leftrgb and not msg_rightrgb:
            return None, None, None, None

        image_left, image_right = None, None
        image_left_cv, image_right_cv = None, None

        if msg_leftrgb:
                    image_left_cv = self.bridge.imgmsg_to_cv2(msg_leftrgb, "bgr8")
                    image_left = PILImage.fromarray(cv2.cvtColor(image_left_cv, cv2.COLOR_BGR2RGB))
                
        if msg_rightrgb:
            image_right_cv = self.bridge.imgmsg_to_cv2(msg_rightrgb, "bgr8")
            image_right = PILImage.fromarray(cv2.cvtColor(image_right_cv, cv2.COLOR_BGR2RGB))

        input_tensor, original_size_left, original_size_right = prepare_image(image_left, image_right)

        result = infer_trt_image(
            self.execution_context, self.bindings, self.inputs,
            self.outputs, self.stream, input_tensor
        )

        resized_map_left, resized_map_right = None, None

        if original_size_left:
            map_left = F.interpolate(result[0:1, :, :, :], size=original_size_left[::-1], mode='bilinear', align_corners=False)
            resized_map_left = ((map_left + 1.0) / 2.0)

        if original_size_right:
            map_right = F.interpolate(result[1:2, :, :, :], size=original_size_right[::-1], mode='bilinear', align_corners=False)
            resized_map_right = ((map_right + 1.0) / 2.0)

        return image_left_cv, image_right_cv, resized_map_left, resized_map_right

# ────────────────────────────────────────────────────────────────────────────────────────────────
# ──────────────────────────────── Traversability projection ───────────────────────────
# ────────────────────────────────────────────────────────────────────────────────────────────────
    # @measure_time
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
# ──────────────────────────────── Downsampling ──────────────────────────────────────────
# ────────────────────────────────────────────────────────────────────────────────────────────────  

    @staticmethod
    # @measure_time
    def voxel_downsample_mean_traversability(points_with_t_gpu: torch.Tensor, voxel_size: float) -> torch.Tensor:
        if points_with_t_gpu.shape[0] == 0:
            return points_with_t_gpu

        voxel_indices = torch.floor(points_with_t_gpu[:, :3] / voxel_size).long()

        unique_voxel_indices, inverse_indices, counts = torch.unique(
            voxel_indices, dim=0, return_inverse=True, return_counts=True)

        num_unique_voxels = unique_voxel_indices.shape[0]

        sum_xyz = torch.zeros((num_unique_voxels, 3), device=points_with_t_gpu.device)
        sum_xyz.scatter_add_(0, inverse_indices.unsqueeze(1).expand(-1, 3), points_with_t_gpu[:, :3])
        mean_xyz = sum_xyz / counts.unsqueeze(1)
        
        sum_t = torch.zeros((num_unique_voxels,), device=points_with_t_gpu.device)
        sum_t.scatter_add_(0, inverse_indices, points_with_t_gpu[:, 3])
        mean_t = sum_t / counts

        result = torch.cat([mean_xyz, mean_t.unsqueeze(1)], dim=1)
        
        return result


# ────────────────────────────────────────────────────────────────────────────────────────────────
# ──────────────────────────────── PointCloud → OccupancyGrid ──────────────────────────────────────────
# ────────────────────────────────────────────────────────────────────────────────────────────────
    @staticmethod
    # @measure_time
    def pointcloud_with_traversability_to_occupancy_grid(stamp, frame: str, pts_with_t: np.ndarray, resolution, grid_size, center_xy, z):
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

        indx = np.floor((x - origin_x) / resolution).astype(np.int32)
        indy = np.floor((y - origin_y) / resolution).astype(np.int32)
        
        mask = (indx >= 0) & (indx < grid_size) & (indy >= 0) & (indy < grid_size)
        indx, indy, t = indx[mask], indy[mask], t[mask]

        grid = np.full(grid_size * grid_size, 101, dtype=np.int16)
        traversability_value = np.round((1 - t) * 100).astype(np.int16)  # 0: traversable, 100: non-traversable
        flat_indices = indy * grid_size + indx
        np.minimum.at(grid, flat_indices, traversability_value)
        grid[grid == 101] = -1
        grid = grid.reshape((grid_size, grid_size)).astype(np.int8)

        og = OccupancyGrid()
        og.header.stamp = stamp.to_msg()
        og.header.frame_id = frame
        og.info.resolution = resolution
        og.info.width = grid_size
        og.info.height = grid_size
        og.info.origin.position.x = origin_x
        og.info.origin.position.y = origin_y
        og.info.origin.position.z = z
        og.data = grid.flatten(order='C').astype(int).tolist()
        
        return og

    
# ────────────────────────────────────────────────────────────────────────────────────────────────
# ──────────────────────────────── Other Utilities ──────────────────────────────────────────
# ────────────────────────────────────────────────────────────────────────────────────────────────     

    # ───────────── (Optional) similarity map visualization & publish for debugging  ─────────────
    # @measure_time
    def resize_map_cpu(self, image_left_cv, image_right_cv, resized_map_left, resized_map_right):
        overlays = []
        raws = []

        def add_label_cv2(img_cv, text):
            labeled_img = img_cv.copy()
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.0
            color = (0, 0, 0)  
            thickness = 2
            position = (10, 30)    

            cv2.putText(labeled_img, text, position, font, font_scale, color, thickness, lineType=cv2.LINE_AA)
            return labeled_img

        if image_right_cv is not None and resized_map_right is not None:
            similarity_np = resized_map_right.squeeze().cpu().numpy()
            similarity_colored = cv2.applyColorMap((255 - similarity_np * 255).astype(np.uint8), cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(image_right_cv, 0.6, similarity_colored, 0.4, 0)
            
            overlays.append(add_label_cv2(cv2.rotate(overlay, cv2.ROTATE_90_CLOCKWISE), "Frontright similarity"))
            raws.append(add_label_cv2(cv2.rotate(image_right_cv, cv2.ROTATE_90_CLOCKWISE), "Frontright image"))

        if image_left_cv is not None and resized_map_left is not None:
            similarity_np = resized_map_left.squeeze().cpu().numpy()
            similarity_colored = cv2.applyColorMap((255 - similarity_np * 255).astype(np.uint8), cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(image_left_cv, 0.6, similarity_colored, 0.4, 0)

            overlays.append(add_label_cv2(cv2.rotate(overlay, cv2.ROTATE_90_CLOCKWISE), "Frontleft similarity"))
            raws.append(add_label_cv2(cv2.rotate(image_left_cv, cv2.ROTATE_90_CLOCKWISE), "Frontleft image"))
            

        if overlays:
            combined_similarity = cv2.hconcat(overlays)
            img_msg_combined_similarity = self.bridge.cv2_to_imgmsg(combined_similarity, encoding='bgr8')
            self.pub_combined_image_similarity.publish(img_msg_combined_similarity)
        
        if raws:
            combined_raw = cv2.hconcat(raws)
            img_msg_combined_raw = self.bridge.cv2_to_imgmsg(combined_raw, encoding='bgr8')
            self.pub_combined_image_raw.publish(img_msg_combined_raw)

    # ───────────── Convert geometry_msgs/Pose to a 4×4 homogeneous transform  ─────────────
    @staticmethod
    def transform_to_matrix(position, orientation) -> np.ndarray:
        T = np.eye(4)

        T[0, 3] = position.x
        T[1, 3] = position.y
        T[2, 3] = position.z

        quat = [orientation.x,
                orientation.y,
                orientation.z,
                orientation.w]
        rot_mat = R.from_quat(quat).as_matrix()        

        T[:3, :3] = rot_mat

        return T 
    
# ───────────── Entry Point ─────────────
def main(argv=None):
    rclpy.init(args=argv)                
    node = TraversabilitytoOccupancygridNode()      
    rclpy.spin(node)                    
    node.destroy_node()                 
    rclpy.shutdown()                    

if __name__ == "__main__":
    main()                               

