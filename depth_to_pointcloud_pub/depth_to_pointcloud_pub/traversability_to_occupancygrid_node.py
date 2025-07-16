#!/usr/bin/env python3
"""
ROS 2 node: Traversability → Occupancygrid (body frame)
"""

import os, struct, yaml
from typing import Dict, Optional
import rclpy, numpy as np
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

from depth_to_pointcloud_pub.models.fastflow_SSL_n import PSPNET_RADIO_SSL #mhlee 25.07.10
from depth_to_pointcloud_pub.models.proxy_n_c import Proxy #mhlee 25.07.10
from PIL import Image as PILImage
import random


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


# ────────────────────── main node ──────────────────────
class TraversabilitytoOccupancygridNode(Node):

# ────────────────────────────────────────────────────────────────────────────────────────────────
# ──────────────────────────────── initial 설정 ──────────────────────────────────────────
# ────────────────────────────────────────────────────────────────────────────────────────────────

    def __init__(self):
        super().__init__("traversability_to_occupancygrid_node")

        # -------------------- Parameter  --------------------
        package_share_directory = get_package_share_directory('depth_to_pointcloud_pub')
        # model_path = os.path.join(package_share_directory, 'models', '45.pth') #모델파일 경로 추후에 바꿔야함(important)
        model_path = '/home/ros/workspace/src/front_depth_to_costmap/depth_to_pointcloud_pub/depth_to_pointcloud_pub/models/1_0715.pth'
        self.model, self.proxy_model = self.load_model(model_path) 
        self.device = torch.device("cuda:0")
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[103.939/255, 116.779/255, 123.68/255],
                                std=[0.229, 0.224, 0.225]),
        ]) # 이거 정규화 수치 나중에 바꿔야할듯(important)


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


        # -------------------- Subscriber & Syncronizer  --------------------    
        # Subscriber for main Data
        self.sub_leftdepth  = Subscriber(self, Image, "/spot1/base/spot/depth/frontleft/image")
        self.sub_rightdepth = Subscriber(self, Image, "/spot1/base/spot/depth/frontright/image")
        self.sub_odom  = Subscriber(self, Odometry, self.odom_topic) # iwshim. 25.05.30
        self.sub_leftrgb = Subscriber(self, Image, "/spot1/base/spot/camera/frontleft/image")
        self.sub_rightrgb = Subscriber(self, Image, "/spot1/base/spot/camera/frontright/image")

        self.sync = ApproximateTimeSynchronizer(
            [self.sub_leftdepth, self.sub_rightdepth, self.sub_leftrgb, self.sub_rightrgb, self.sub_odom], # iwshim. 25.05.30
            queue_size=30,                    
            slop=1.0)
        self.sync.registerCallback(self.occupancy_cb) # mhlee 25.06.19



# ────────────────────────────────────────────────────────────────────────────────────────────────
# ──────────────────────────────── callback 함수 ──────────────────────────────────────────
# ────────────────────────────────────────────────────────────────────────────────────────────────
    # ───────────── occupancy callback ─────────────  # mhlee 25.06.19

    def occupancy_cb(self, msg_leftdepth : Image, msg_rightdepth : Image, msg_leftrgb : Image, msg_rightrgb : Image, msg_odom : Odometry): 
        # initial fofr CPU
        stamp = rclpy.time.Time.from_msg(msg_leftdepth.header.stamp)
        self.get_logger().info("GPU-Accelerated Callback (PyTorch3D Version)")

                
        depth_raw_left = self.bridge.imgmsg_to_cv2(msg_leftdepth, "passthrough")  ## 16UC1 → np.ndarray
        depth_raw_right = self.bridge.imgmsg_to_cv2(msg_rightdepth, "passthrough")  ## 16UC1 → np.ndarray

        cv_image_left_rgb = self.bridge.imgmsg_to_cv2(msg_leftrgb, "bgr8")
        pil_image_left_rgb = PILImage.fromarray(cv_image_left_rgb)

        cv_image_right_rgb = self.bridge.imgmsg_to_cv2(msg_rightrgb, "bgr8")
        pil_image_right_rgb = PILImage.fromarray(cv_image_right_rgb)

        pos = msg_odom.pose.pose.position
        ori = msg_odom.pose.pose.orientation
        T_odom = self.transform_to_matrix(pos, ori) 
        T_odom_gpu = torch.tensor(T_odom, dtype=torch.float32, device=self.device) 

        depth_m_left   = torch.tensor(depth_raw_left.astype(np.float32) / 1000.0, device=self.device) ## mm → m
        depth_m_right   = torch.tensor(depth_raw_right.astype(np.float32) / 1000.0, device=self.device) ## mm → m
        depth_m_left[depth_m_left > 3.0] = 0.0                               ## 5 m 초과 마스킹
        depth_m_right[depth_m_right > 3.0] = 0.0                               ## 5 m 초과 마스킹
               
        image_tensor_left_rgb = self.transform(pil_image_left_rgb).unsqueeze(0).to(self.device)
        image_tensor_right_rgb = self.transform(pil_image_right_rgb).unsqueeze(0).to(self.device)

        # -- GPU start -- 
        pts_left_body_gpu  = self.depth_to_pts_frame_body(depth_m_left, self.K_frontleft_depth, self.T_body_to_frontleft_depth)
        pts_right_body_gpu = self.depth_to_pts_frame_body(depth_m_right, self.K_frontright_depth, self.T_body_to_frontright_depth)
    
        traversability_left = self.traversability(image_tensor_left_rgb , self.model, self.proxy_model)
        traversability_right = self.traversability(image_tensor_right_rgb, self.model, self.proxy_model) # output 1xHxW image

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
            grid_size=80, 
            center_xy=(pos.x, pos.y), 
        )

        pc = self.build_pc(msg_leftdepth.header.stamp, self.odom_frame, points_final_cpu[:, :3])
        self.pub_accum.publish(pc)
        self.pub_occup.publish(og)




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

    @measure_time
    def load_model(self, model_path):
        fix_seed(1)
        model = PSPNET_RADIO_SSL(in_channels=256, flow_steps=8, freeze_backbone=True, flow='fast') # 이 파라미터들 어떤게 optimized한지 생각해보기 (important)
        proxy_model = Proxy(num_proxies=256, dim=256)
        checkpoint = torch.load(model_path, map_location='cpu')
        model.local_extractor.load_state_dict(checkpoint['local_extractor'], strict=False)
        model.nf_flows.load_state_dict(checkpoint['nf_flows'], strict=False)
        proxy_model.positive_center.data = checkpoint['positive_center']
        # proxy_model.proxy_u.data = checkpoint['proxy_u']
        # proxy_model.negative_centers.data = checkpoint['negative_centers']
        model.eval()
        proxy_model.eval()
        return model.cuda(), proxy_model.cuda()
    
    @measure_time
    def traversability(self, image_tensor : torch.Tensor , model, proxy_model): # 세로로 넣었다가, 세로로 출력되는 것으로 바꾸고, 시작과 끝에 세로-> 가로 / 가로 -> 세로 넣어야함(important)
        use_dummy = True  

        if use_dummy:
            self.get_logger().warn("Using dummy traversability data!")
            H, W = 480, 640
            dummy_map = torch.zeros((H, W), device=self.device)
            dummy_map[:, W // 2 :] = 1.0 
            
            return dummy_map
    
        similarity_map = model.inference(image_tensor, proxy_model)
        
        resized_map = transforms.functional.resize(similarity_map, image_tensor.shape[2:], interpolation=transforms.InterpolationMode.BILINEAR) # 이것도 Interpolation 방법 생각해보기(Important)
        
        sim_np = resized_map.squeeze().cpu().numpy()
        sim_norm = (sim_np - sim_np.min()) / (sim_np.max() - sim_np.min() + 1e-8)

        return sim_norm



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

        H, W = traversability_img.shape
        valid = (z > 0.1) & (z < 5.0) & (u >= 0) & (u < W) & (v >= 0) & (v < H) 

        pts_body_valid = pts_body[valid]
        t = traversability_img[v[valid].long(), u[valid].long()] 

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


