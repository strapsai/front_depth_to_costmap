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
from geometry_msgs.msg import TransformStamped
from tf2_ros import Buffer, TransformListener, TransformBroadcaster

from cv_bridge import CvBridge
from transforms3d.quaternions import quat2mat
from geometry_msgs.msg import PoseStamped
from sklearn.neighbors import NearestNeighbors



from message_filters import Subscriber, ApproximateTimeSynchronizer

# iwshim
from nav_msgs.msg import Odometry, OccupancyGrid, MapMetaData #iwshim 25.06.02
#from sklearn.neighbors import NearestNeighbors#KDTree #iwshim 25.06.02

import time, torch
import open3d as o3d
from functools import wraps


from visualization_msgs.msg import Marker # mhlee 25.06.23
from geometry_msgs.msg import Point, Vector3 # mhlee 25.06.23
import std_msgs.msg # mhlee 25.06.23
import kornia.filters as KF #mhlee 25.07.08
import torch.nn.functional as F #mhlee 25.07.08
from pytorch3d.ops import knn_points, estimate_pointcloud_normals  #mhlee 25.07.09

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
def load_extrinsic_matrix(yaml_name: str, key: str) -> np.ndarray:
    pkg = get_package_share_directory("depth_to_pointcloud_pub")
    with open(os.path.join(pkg, "config", yaml_name), "r", encoding="utf-8") as f:
        return np.array(yaml.safe_load(f)[key]["Matrix"]).reshape(4, 4)

# ────────────────────── main node ──────────────────────
class TraversabilitytoOccupancygridNode(Node):

# ────────────────────────────────────────────────────────────────────────────────────────────────
# ──────────────────────────────── initial 설정 ──────────────────────────────────────────
# ────────────────────────────────────────────────────────────────────────────────────────────────

    def __init__(self):
        super().__init__("traversability_to_occupancygrid_node")

        # -------------------- Parameter  --------------------
        self.prefixes = ["frontleft_depth", "frontright_depth", "frontleft_rgb", "frontright_rgb"]
        self.bridge = CvBridge()

        # Camera Intrinsic & Extrinsic
        self.K: Dict[str, Optional[np.ndarray]] = {p: None for p in self.prefixes}
        self.extr = {
            "frontleft_depth":  load_extrinsic_matrix("frontleft_info.yaml",  "body_to_frontleft"),
            "frontright_depth": load_extrinsic_matrix("frontright_info.yaml", "body_to_frontright"),
            "frontleft_rgb":  load_extrinsic_matrix("frontleft_info.yaml",  "body_to_frontleft_fisheye"),
            "frontright_rgb": load_extrinsic_matrix("frontright_info.yaml", "body_to_frontright_fisheye"),
        }
        
        self.device = torch.device("cuda:0")
        self.clouds = np.zeros((1,3))
        self.clouds_time = np.empty((0,4)) #mhlee. 25.06.21

        # -------------------- Frame & Topic --------------------

        # Frame ID
        self.body_frame = "spot1/base/spot/body"
        self.odom_frame = "spot1/base/spot/odom"    # iwshim. 25.05.30

        # Topic name 
        self.depth_base = "/spot1/base/spot/depth"
        self.rgb_base = "/spot1/base/spot/camera"
        self.odom_topic = "/spot1/base/spot/odometry" # iwshim. 25.05.30
        self.merge_topic = "/spot1/base/spot/depth/merge_front"
        self.accum_topic = "/spot1/base/spot/depth/accum_front"
        self.occup_topic = "/spot1/base/spot/depth/occup_front"


        # -------------------- Publisher  --------------------
        
        self.pub_merge = self.create_publisher(PointCloud2, self.merge_topic, 10)
        self.pub_accum = self.create_publisher(PointCloud2, self.accum_topic, 10)
        self.pub_occup = self.create_publisher(OccupancyGrid, self.occup_topic, 10)
        self.pub_normal = self.create_publisher(Marker, "/spot1/base/spot/depth/normal_front", 10) #mhlee 25.06.23


        # -------------------- Subscriber & Syncronizer  --------------------

        # Subscriber for main Data
        self.sub_leftdepth  = Subscriber(self, Image, f"{self.depth_base}/frontleft/image")
        self.sub_rightdepth = Subscriber(self, Image, f"{self.depth_base}/frontright/image")
        self.sub_odom  = Subscriber(self, Odometry, self.odom_topic) # iwshim. 25.05.30
        self.sub_leftrgb = Subscriber(self, Image, f"{self.rgb_base}/frontleft/image")
        self.sub_rightrgb = Subscriber(self, Image, f"{self.rgb_base}/frontright/image")

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

        stamp = rclpy.time.Time.from_msg(msg_leftdepth.header.stamp)
        self.get_logger().info("GPU-Accelerated Callback (PyTorch3D Version)")

        traversability_left = traversability(msg_leftrgb)
        traversability_right = traversability(msg_rightrgb) # output 1xHxW image

        pts_left  = self.depth_to_pts_frame_body(msg_leftdepth,  "frontleft_depth")
        pts_right = self.depth_to_pts_frame_body(msg_rightdepth, "frontright_depth")

        pts_body = np.vstack((pts_left, pts_right))

        pts_with_traversability_left = self.merge_traversability_to_pointcloud_frmae_body(pts_body, traversability_left, "frontleft_rgb")
        pts_with_traversability_right = self.merge_traversability_to_pointcloud_frmae_body(pts_body, traversability_right, "frontright_rgb")

        pts_dict = {}

        for pt in np.vstack([pts_with_traversability_left, pts_with_traversability_right]):
            key = tuple(np.round(pt[:3], 4))  # (x, y, z), 소수점 4자리 기준
            if key not in pts_dict:
                pts_dict[key] = pt

        points_body = np.array(list(pts_dict.values()))


        pos = msg_odom.pose.pose.position
        ori = msg_odom.pose.pose.orientation
        T = self.transform_to_matrix(pos, ori)
        points_xyz = points_body[:, :3]
        points_homo = np.hstack([points_xyz, np.ones((points_xyz.shape[0], 1))])
        points_world = (T @ points_homo.T).T[:, :3]
        points_final = np.hstack([points_world, points_body[:, 3:4]])

        og = self.pointcloud_with_traversability_to_occupancy_grid(stamp=stamp, 
            frame=self.odom_frame, 
            points=points_final,
            resolution=0.1, 
            grid_size=150, 
            center_xy=(pos.x, pos.y), 
            traversability_threshold = 200
        )

        pc = self.build_pc(msg_leftdepth.header.stamp, self.odom_frame, points_final)
        self.pub_accum.publish(pc)
        self.pub_occup.publish(og)

        # 1. traversability 중복 문제 조금 더 디테일한 함수 쓰기
        # 2. pointcloud downsampling 과정 하기



# ────────────────────────────────────────────────────────────────────────────────────────────────
# ──────────────────────────────── Downsampling 관련 함수 ──────────────────────────────────────────
# ────────────────────────────────────────────────────────────────────────────────────────────────


    # iwshim. 25.05.30
    @staticmethod
    @measure_time
    def voxel_downsample_mean(points: np.ndarray, voxel_size: float) -> np.ndarray:
        voxel_indices = np.floor(points / voxel_size).astype(np.int32)
        # Centroid Point Calculation per each unique voxel
        keys, inverse, counts = np.unique(voxel_indices, axis=0, return_inverse=True, return_counts=True)

        grid_sum = np.zeros_like(keys, dtype=np.float64)
        pts_mean = np.zeros((keys.shape[0], 3), dtype=np.float32)
        for dim in range(3):
            grid_sum = np.bincount(inverse, weights=points[:,dim], minlength=keys.shape[0])
            pts_mean[:,dim] = grid_sum / counts
        return pts_mean
    
    @staticmethod
    @measure_time
    def voxel_downsample_max_elevation_vec(points: np.ndarray, voxel_size: float) -> np.ndarray:
        if points.shape[0] == 0:
            return points

        # 2D grid index
        xy_idx = np.floor(points[:, :2] / voxel_size).astype(np.int32)
        key_arr, inv, counts = np.unique(xy_idx, axis=0, return_inverse=True, return_counts=True)

        # Extract max height
        # Sorting per each grid
        sort_idx = np.lexsort((points[:,2], inv))      # inv 기준, 그 안에서 z 오름차순
        inv_sorted = inv[sort_idx]
        # max z == last index
        _, last_idx = np.unique(inv_sorted, return_index=True)

        max_idx = sort_idx[last_idx]
        return points[max_idx]
    
    # mhlee. 25.06.22
    @staticmethod
    @measure_time
    def _voxel_downsample_mean_pytorch(points_gpu: torch.Tensor, voxel_size: float) -> torch.Tensor:
        """
        PyTorch 텐서 연산만으로 Voxel Grid의 중심점을 찾아 다운샘플링합니다.
        
        Args:
            points_gpu: (N, 3) 모양의 포인트 클라우드 텐서 (GPU에 있어야 함).
            voxel_size: 복셀의 크기 (미터 단위).

        Returns:
            다운샘플링된 (M, 3) 모양의 포인트 클라우드 텐서 (GPU에 있음).
        """
        if points_gpu.shape[0] == 0:
            return points_gpu

        # 1. 각 포인트의 복셀 인덱스 계산
        voxel_indices = torch.floor(points_gpu / voxel_size).long()

        # 2. 고유한 복셀 인덱스와 역 인덱스, 개수 찾기
        # unique_voxel_indices: 중복이 제거된 복셀 인덱스들 (M, 3)
        # inverse_indices: 각 원본 포인트가 어떤 고유 복셀에 속하는지에 대한 인덱스 (N,)
        # counts: 각 고유 복셀에 속한 포인트의 개수 (M,)
        unique_voxel_indices, inverse_indices, counts = torch.unique(
            voxel_indices, dim=0, return_inverse=True, return_counts=True)

        # 3. 각 고유 복셀에 속한 포인트들의 합계 계산
        # scatter_add_를 사용하여 그룹별 합계를 매우 효율적으로 계산합니다.
        num_unique_voxels = unique_voxel_indices.shape[0]
        sum_points_per_voxel = torch.zeros((num_unique_voxels, 3), device=points_gpu.device)
        
        # inverse_indices를 사용하여 points_gpu의 각 포인트를 해당하는 그룹에 더합니다.
        sum_points_per_voxel.scatter_add_(0, inverse_indices.unsqueeze(1).expand(-1, 3), points_gpu)

        # 4. 합계를 개수로 나누어 평균(중심점) 계산
        mean_points_per_voxel = sum_points_per_voxel / counts.unsqueeze(1)
        
        return mean_points_per_voxel


# ────────────────────────────────────────────────────────────────────────────────────────────────
# ──────────────────────────────── Traversability 관련 함수 ──────────────────────────────────────────
# ────────────────────────────────────────────────────────────────────────────────────────────────

# mhlee 25.07.10 
# def traversability (image):
#     return traversability_image
# 아마 1xHxW (traversability값이 0~255로 표현된) 으로 예상



# ────────────────────────────────────────────────────────────────────────────────────────────────
# ──────────────────────────────── Occupancygrid 관련 함수 ──────────────────────────────────────────
# ────────────────────────────────────────────────────────────────────────────────────────────────
    # iwshim. 25.06.02
    @staticmethod
    @measure_time
    def pointcloud_to_occupancy_grid(stamp, frame: str, points: np.ndarray, resolution, grid_size, center_xy, normals):
        """ Parameters
        resolution: 0.05m
        grid_size: the number of cells per each side
        coord_center: 
            Origin(left bottom x) = center[0] - 0.5 * grid_size * resolution
        """
        origin_x = center_xy[0] - 0.5 * grid_size * resolution
        origin_y = center_xy[1] - 0.5 * grid_size * resolution

        # Points to grid index
        indx = np.floor((points[:,0] - origin_x) / resolution).astype(np.int32)
        indy = np.floor((points[:,1] - origin_y) / resolution).astype(np.int32)
        
         # 사전에 미리 제거 했으나, 만일의 경우를 대비해서(segmentation fault) mask-out
        mask = (indx >= 0) & (indx < grid_size) & (indy >= 0) & (indy < grid_size)
        indx, indy = indx[mask], indy[mask]
        normals = normals[mask]

        #valz = points[:,2][mask]
        #occ_mask = valz > -1.7
        #print(f"Occupancy mask count: {occ_mask.sum()}")
        up_vector = np.array([0,0,1], dtype=np.float32)
        similarity = normals @ up_vector
        occ_mask = similarity <= 0.1
        
        indx_occ = indx[occ_mask]
        indy_occ = indy[occ_mask]
        
        grid = np.zeros((grid_size, grid_size), dtype=np.int8)
        grid[indy_occ, indx_occ] = 100
        
        og = OccupancyGrid()
        og.header.stamp = stamp.to_msg()
        og.header.frame_id = frame
        og.info.resolution = resolution
        og.info.width = grid_size
        og.info.height = grid_size
        og.info.origin.position.x = origin_x
        og.info.origin.position.y = origin_y
        # og.info.origin.position.z = -1.7
        og.info.origin.position.z = -2.0 # mhlee 25.06.23
        og.data = grid.flatten(order='C').astype(int).tolist()
        
        return og
    

    @staticmethod
    @measure_time
    def pointcloud_with_traversability_to_occupancy_grid(stamp, frame: str, points: np.ndarray, resolution, grid_size, center_xy, traversability_threshold):
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
        indx = np.floor((points[:,0] - origin_x) / resolution).astype(np.int32)
        indy = np.floor((points[:,1] - origin_y) / resolution).astype(np.int32)
        
         # 사전에 미리 제거 했으나, 만일의 경우를 대비해서(segmentation fault) mask-out
        mask = (indx >= 0) & (indx < grid_size) & (indy >= 0) & (indy < grid_size)
        indx, indy, t = indx[mask], indy[mask], t[mask]

        grid = np.zeros((grid_size, grid_size), dtype=np.int8)
        occ_mask = t < traversability_threshold # mhlee (important : 나중에 traversability code에 따라 255값 변경해야함)
        grid[indy[occ_mask], indx[occ_mask]] = 100

        og = OccupancyGrid()
        og.header.stamp = stamp.to_msg()
        og.header.frame_id = frame
        og.info.resolution = resolution
        og.info.width = grid_size
        og.info.height = grid_size
        og.info.origin.position.x = origin_x
        og.info.origin.position.y = origin_y
        # og.info.origin.position.z = -1.7
        og.info.origin.position.z = -2.0 # mhlee 25.06.23
        og.data = grid.flatten(order='C').astype(int).tolist()
        
        return og



# ────────────────────────────────────────────────────────────────────────────────────────────────
# ──────────────────────────────── 기타 ──────────────────────────────────────────
# ────────────────────────────────────────────────────────────────────────────────────────────────

    # ───────────── depth Image → 3-D 포인트 변환 ─────────────
    @measure_time
    def depth_to_pts_frame_body(self, msg: Image, prefix: str) -> Optional[np.ndarray]:
        
        K = self.K[prefix]

        depth_raw = self.bridge.imgmsg_to_cv2(msg, "passthrough")  ## 16UC1 → np.ndarray
        depth_m   = depth_raw.astype(np.float32) / 1000.0          ## mm → m
        depth_m[depth_m > 5.0] = 0.0                               ## 5 m 초과 마스킹

        # 픽셀 그리드 생성
        h, w = depth_m.shape
        fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
        u, v = np.meshgrid(np.arange(w), np.arange(h))

        # 핀홀 역변환: (u,v,depth) → (x,y,z)
        z = depth_m
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        
        pts4 = np.vstack((x.ravel(), y.ravel(), z.ravel(), np.ones(z.size)))  ## 4×N
        pts4 = pts4[:, pts4[2] > 0.1]            ## z(깊이) 0.1 m 이하 필터링

        # 카메라 → 바디 좌표 변환
        T = self.extr[prefix]                    ## 4×4 변환 행렬
        pts_body = (T @ pts4)[:3].T              ## 결과 (N,3)
        
        return pts_body.astype(np.float32)


    @measure_time
    def merge_traversability_to_pointcloud_frmae_body(pts_body: np.darray, traversability_img: np.darray, prefix_rgb: str) -> Optional[np.ndarray]:

        N = pts_body.shape[0]
        pts4 = np.hstack([pts_body, np.ones((N, 1))])

        T_rgb_to_body = self.extr[prefix_rgb]
        T_body_to_rgb = np.linalg.inv(T_rgb_to_body)

        pts_rgb = (T_body_to_rgb @ pts4.T).T[:, :3]

        x, y, z = pts_rgb[:, 0], pts_rgb[:, 1], pts_rgb[:, 2]

        K_rgb = self.K[prefix_rgb]

        fx, fy = K_rgb[0, 0], K_rgb[1, 1]
        cx, cy = K_rgb[0, 2], K_rgb[1, 2]

        u = (fx * x / z + cx).astype(np.int32)
        v = (fy * y / z + cy).astype(np.int32)

        H, W = traversability_img.shape
        valid = (5 > z > 0.1) & (u >= 0) & (u < W) & (v >= 0) & (v < H) 

        t = np.zeros(N, dtype=np.float32)
        t[valid] = traversability_img[v[valid], u[valid]] / 255.0 # mhlee (important : 나중에 traversability code에 따라 255값 변경해야함)

        pts_with_t = np.hstack([pts_body, t[:, None]])
        return pts_with_t
        

        
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


