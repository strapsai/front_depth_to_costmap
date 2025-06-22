#!/usr/bin/env python3
"""
ROS 2 node: frontleft+frontright depth → 단일 PointCloud2 (body frame)
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

from message_filters import Subscriber, ApproximateTimeSynchronizer

# iwshim
from nav_msgs.msg import Odometry, OccupancyGrid, MapMetaData #iwshim 25.06.02
#from sklearn.neighbors import NearestNeighbors#KDTree #iwshim 25.06.02

import time, torch
# import open3d as o3d
from functools import wraps


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
class DepthToPointCloudNode(Node):

# ────────────────────────────────────────────────────────────────────────────────────────────────
# ──────────────────────────────── initial 설정 ──────────────────────────────────────────
# ────────────────────────────────────────────────────────────────────────────────────────────────

    def __init__(self):
        super().__init__("depth_to_pointcloud_node")

        # -------------------- Parameter  --------------------
        self.prefixes = ["frontleft", "frontright"]
        self.bridge = CvBridge()

        # Camera Intrinsic & Extrinsic
        self.K: Dict[str, Optional[np.ndarray]] = {p: None for p in self.prefixes}
        self.extr = {
            "frontleft":  load_extrinsic_matrix("frontleft_info.yaml",  "body_to_frontleft"),
            "frontright": load_extrinsic_matrix("frontright_info.yaml", "body_to_frontright"),
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
        self.odom_topic = "/spot1/base/spot/odometry" # iwshim. 25.05.30
        self.merge_topic = "/spot1/base/spot/depth/merge_front"
        self.accum_topic = "/spot1/base/spot/depth/accum_front"
        self.occup_topic = "/spot1/base/spot/depth/occup_front"


        # -------------------- Publisher  --------------------
        
        self.pub_merge = self.create_publisher(PointCloud2, self.merge_topic, 10)
        self.pub_accum = self.create_publisher(PointCloud2, self.accum_topic, 10)
        self.pub_occup = self.create_publisher(OccupancyGrid, self.occup_topic, 10)


        # -------------------- Subscriber & Syncronizer  --------------------

        # Accumulation Parameters and Published Topics, iwshim 25.05.30
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # CameraInfo Subscriber 
        for p in self.prefixes:
            self.create_subscription(
                CameraInfo,                                    ## 타입
                f"{self.depth_base}/{p}/camera_info",          ## 토픽 이름
                lambda m, pr=p: self._camera_info_cb(m, pr),   ## 콜백 (prefix 캡쳐)
                10)                                                     ## 큐 길이

        # Subscriber for main Data
        self.sub_left  = Subscriber(self, Image, f"{self.depth_base}/frontleft/image")
        self.sub_right = Subscriber(self, Image, f"{self.depth_base}/frontright/image")
        self.sub_odom  = Subscriber(self, Odometry, self.odom_topic) # iwshim. 25.05.30
        
        # Time Synchronization 
        #self.sync = ApproximateTimeSynchronizer(
        #    [self.sub_left, self.sub_right, self.sub_pose], # iwshim. 25.05.30
        #    queue_size=30,                    
        #    slop=0.15)                        ## 50 ms
        #self.sync.registerCallback(self._synced_depth_cb)
        self.sync = ApproximateTimeSynchronizer(
            [self.sub_left, self.sub_right, self.sub_odom], # iwshim. 25.05.30
            queue_size=30,                    
            slop=1.0)
        #self.sync.registerCallback(self._synced_costmap)
        # self.sync.registerCallback(self._debug_cb)

        # self.sync.registerCallback(self.occupancy_cb) # mhlee 25.06.19
        self.sync.registerCallback(self.occupancy_cb_gpu) 




# ────────────────────────────────────────────────────────────────────────────────────────────────
# ──────────────────────────────── callback 함수 ──────────────────────────────────────────
# ────────────────────────────────────────────────────────────────────────────────────────────────


    # ───────────── CameraInfo 콜백 ─────────────
    def _camera_info_cb(self, msg: CameraInfo, prefix: str):
        self.K[prefix] = np.array(msg.k).reshape(3, 3)   ## 내부 파라미터 저장
        self.get_logger().info(f"[{prefix}] CameraInfo OK\n", once=True)

    # ───────────── 동기화된 Costmap 콜백 ─────────────
    def _debug_cb(self, msg_left: Image, msg_right: Image, odom: Odometry):
        self.get_logger().warning("HIT THE DEPTH CALLBACK\n")
        t_l = msg_left.header.stamp.sec + msg_left.header.stamp.nanosec*1e-9
        t_r = msg_right.header.stamp.sec + msg_right.header.stamp.nanosec*1e-9
        t_o = odom.header.stamp.sec + odom.header.stamp.nanosec*1e-9
        self.get_logger().info(f"stamps:\n L={t_l:.3f} sec, \n R={t_r:.3f} sec, \n O={t_o:.3f} sec")


    def _synced_costmap(self, msg_left: Image, msg_right: Image, odom: Odometry):
        
        stamp = rclpy.time.Time.from_msg(msg_left.header.stamp)
        self.get_logger().warning("HIT THE DEPTH CALLBACK\n")
        
        pts_left  = self._depth_to_pts(msg_left,  "frontleft")
        pts_right = self._depth_to_pts(msg_right, "frontright")
        pts = np.vstack((pts_left, pts_right))           ## (N,3) 행렬 합치기 *속도 최적화시 Check Point.

        pos = odom.pose.pose.position
        ori = odom.pose.pose.orientation
        T = self.transform_to_matrix(pos, ori)
        pts_tf = (T @ np.hstack([pts, np.ones((pts.shape[0],1))]).T).T[:,:3]
        
        self.clouds = pts_tf
        self.clouds = self.voxel_downsample_mean(self.clouds, 0.1)
        pc = self._build_pc(msg_left.header.stamp, self.odom_frame, self.clouds) #Only for display
        self.pub_accum.publish(pc)
        

    # ───────────── occupancy callback ─────────────  # mhlee 25.06.19

    # def occupancy_cb(self, msg_left : Image, msg_right : Image, msg_odom : Odometry):

    #     stamp = rclpy.time.Time.from_msg(msg_left.header.stamp)
    #     stamp_sec = stamp.nanoseconds * 1e-9
    #     self.get_logger().warning("HIT THE DEPTH CALLBACK for occupancygrid\n")

    #     pts_left  = self._depth_to_pts(msg_left,  "frontleft")
    #     pts_right = self._depth_to_pts(msg_right, "frontright")
    #     pts = np.vstack((pts_left, pts_right)) ## (N,3) 행렬 합치기 *속도 최적화시 Check Point.


    #     pos = msg_odom.pose.pose.position
    #     ori = msg_odom.pose.pose.orientation
    #     T = self.transform_to_matrix(pos, ori)
    #     pts_tf = (T @ np.hstack([pts, np.ones((pts.shape[0],1))]).T).T[:,:3]

    #     self.clouds = self.voxel_downsample_mean(pts_tf, 0.1)    

    #     normals = self.estimation_normals(self.clouds)
    #     # normals = self.estimate_normals_half_random_open3d(clouds, k=20, k_search=40, deterministic_k=8 )

    #     og = self.pointcloud_to_occupancy_grid(stamp=stamp, frame=self.odom_frame, points=self.clouds, resolution = 0.1, grid_size= 150, center_xy = (pos.x, pos.y), normals=normals )
        
    #     self.pub_occup.publish(og)
    

    # ───────────── occupancy callback using gpu ─────────────  # mhlee 25.06.22

    def occupancy_cb_gpu(self, msg_left : Image, msg_right : Image, msg_odom : Odometry):


        ########################## Version For Open3d#########################3
        # stamp = rclpy.time.Time.from_msg(msg_left.header.stamp)
        # stamp_sec = stamp.nanoseconds * 1e-9
        # self.get_logger().warning("HIT THE DEPTH CALLBACK for occupancygrid\n")

        # pts_left  = self._depth_to_pts(msg_left,  "frontleft")
        # pts_right = self._depth_to_pts(msg_right, "frontright")
        # pts = np.vstack((pts_left, pts_right)) ## (N,3) 행렬 합치기 *속도 최적화시 Check Point.

        # if pts.shape[0] == 0:
        #     return # 처리할 포인트가 없으면 종료

        # pos = msg_odom.pose.pose.position
        # ori = msg_odom.pose.pose.orientation
        # T_cpu = self.transform_to_matrix(pos, ori)

        # points_torch = torch.from_numpy(pts).to(self.device, torch.float32)
        # T_gpu = torch.from_numpy(T_cpu).to(self.device, torch.float32)  

        # num_pts = points_torch.shape[0]
        # pts_h = torch.cat([points_torch, torch.ones((num_pts, 1), device=self.device)], dim=1)
        # pts_tf_h = torch.matmul(T_gpu, pts_h.T).T
        # pts_tf_torch = pts_tf_h[:, :3] 

        # pcd = o3d.t.geometry.PointCloud(pts_tf_torch)

        # pcd = pcd.voxel_down_sample(voxel_size=0.1)

        # search_param = o3d.t.geometry.KDTreeSearchParamHybrid(radius=0.3, max_nn=30)
        # pcd.estimate_normals(search_param)
        
        # points_final = pcd.point.positions.cpu().numpy()
        # normals_final = pcd.point.normals.cpu().numpy()

        # # 기존 함수를 그대로 호출하되, GPU에서 계산된 결과물을 전달
        # og = self.pointcloud_to_occupancy_grid(
        #     stamp=stamp, 
        #     frame=self.odom_frame, 
        #     points=points_final,  # GPU에서 처리된 포인트
        #     resolution=0.1, 
        #     grid_size=150, 
        #     center_xy=(pos.x, pos.y), 
        #     normals=normals_final # GPU에서 처리된 Normal
        # )
        
        # self.pub_occup.publish(og)      


        ########################## Version For Pytorch3d#########################3
        stamp = rclpy.time.Time.from_msg(msg_left.header.stamp)
        self.get_logger().info("GPU-Accelerated Callback (PyTorch3D Version)")

        pts_left  = self._depth_to_pts(msg_left,  "frontleft")
        pts_right = self._depth_to_pts(msg_right, "frontright")
        pts_cpu = np.vstack((pts_left, pts_right))

        if pts_cpu.shape[0] == 0:
            return


        points_torch = torch.from_numpy(pts_cpu).to(self.device, torch.float32)

        pos = msg_odom.pose.pose.position
        ori = msg_odom.pose.pose.orientation
        T_cpu = self.transform_to_matrix(pos, ori)
        T_gpu = torch.from_numpy(T_cpu).to(self.device, torch.float32)  
        
        num_pts = points_torch.shape[0]
        pts_h = torch.cat([points_torch, torch.ones((num_pts, 1), device=self.device)], dim=1)
        pts_tf_h = torch.matmul(T_gpu, pts_h.T).T
        pts_tf_gpu = pts_tf_h[:, :3]  # (N, 3) 최종 변환된 포인트 (GPU 텐서)
        
        points_down_gpu = self._voxel_downsample_mean_pytorch(points_gpu = pts_tf_gpu, voxel_size=0.1)
        normals_gpu = self._estimate_normals_pytorch3d(points_down_gpu, k=30)
        
        points_final = points_down_gpu.cpu().numpy()
        normals_final = normals_gpu.cpu().numpy()

        og = self.pointcloud_to_occupancy_grid(
            stamp=stamp, 
            frame=self.odom_frame, 
            points=points_final,
            resolution=0.1, 
            grid_size=150, 
            center_xy=(pos.x, pos.y), 
            normals=normals_final
        )
        
        self.pub_occup.publish(og)

    # ───────────── 동기화된 Depth 이미지 콜백 ─────────────
    def _synced_depth_cb(self, msg_left: Image, msg_right: Image, msg_odom: Odometry):
        """
        frontleft·frontright 깊이 이미지가 거의 동시에 도착하면 호출
        두 이미지를 각각 포인트로 변환 후 합쳐 하나의 PointCloud2로 publish
        """
        # iwshim. 25.05.30
        stamp = rclpy.time.Time.from_msg(msg_left.header.stamp)

        try:
            # 1. 바로 TF가 있으면 interpolation 포함해서 반환됨 (내부적으로 지원)
            trans = self.tf_buffer.lookup_transform(
                self.odom_frame,
                self.body_frame,
                stamp,
                timeout=rclpy.duration.Duration(seconds=0.1)
            )
            self.get_logger().info(f"TF found for msg_left at {stamp.nanoseconds * 1e-9:.3f}s")
        except Exception as e:
            self.get_logger().warning(f"TF exact lookup failed: {e}")
            # 2. fallback: 최신 TF로라도 변환 (정확도는 떨어질 수 있음)
            try:
                trans = self.tf_buffer.lookup_transform(
                    self.odom_frame,
                    self.body_frame,
                    rclpy.time.Time(),  # 최신 TF
                    timeout=rclpy.duration.Duration(seconds=0.05)
                )
                self.get_logger().warning(f"Using latest TF as fallback")
            except Exception as e2:
                self.get_logger().error(f"TF lookup totally failed: {e2}")
                return

    # -----------------------------------------------------------
        self.get_logger().warning("HIT THE DEPTH CALLBACK\n")
        pts_left  = self._depth_to_pts(msg_left,  "frontleft")
        pts_right = self._depth_to_pts(msg_right, "frontright")
        pts = np.vstack((pts_left, pts_right))           ## (N,3) 행렬 합치기 *속도 최적화시 Check Point.

        # 4x4 Transform matrix from msg_left.frame_id -> body_frame, iwshim. 25.05.30
        T = self.transform_to_matrix(trans)
        pts_homo = np.hstack([pts, np.ones((pts.shape[0], 1))])
        #self.get_logger().info( f"\nPTS shape: {pts_homo.shape[0]:d}, {pts_homo.shape[1]:d}\n" )
        pts_tf = (T @ pts_homo.T).T[:,:3]
        # -----------------------------------------------------------

        # Accumulation, iwshim. 25.05.30
        #pos = msg_odom.pose.pose.position
        pos = trans.transform.translation
        center = np.array([pos.x, pos.y, pos.z])

        if self.clouds.shape[0] == 1:
            self.clouds = pts_tf
        else:
            self.clouds = pts_tf
            #self.clouds = np.vstack([self.clouds, pts_tf])
        self.clouds = self.voxel_downsample_mean(self.clouds, 0.1)
        #self.clouds = self.voxel_downsample_max_elevation_vec(self.clouds, 0.05)
        #self.clouds = self.remove_far_points(self.clouds, center, 7)
        
        #nm = self.estimation_normals(self.clouds) # fast, but large noise
        #nm = self.estimate_normals_half_random_open3d(self.clouds) # too slow, more than 4,000ms
        #og = self.pointcloud_to_occupancy_grid(msg_left.header.stamp, 
        #                                       self.origin_frame, 
        ##                                       self.clouds,
        #                                       resolution = 0.1,
        #                                       grid_size = 100,
        #                                       center_xy=(pos.x, pos.y),
        #                                       normals = nm)
                                            
        pc = self._build_pc(msg_left.header.stamp, self.odom_frame, self.clouds) #Only for display
        self.pub_accum.publish(pc)
        #self.pub_occup.publish(og)
        self.get_logger().info("------------------------------------------\n")
        # -----------------------------------------------------------
                
        # Print for debugging, iwshim 25.05.30
        """
        pos = msg_odom.pose.pose.position
        self.get_logger().info(
            f"Odom. Position: x={pos.x:.3f}, y={pos.y:.3f}, z={pos.z:.3f}\n"
            f"Left: {msg_left.header.stamp.sec}.{msg_left.header.stamp.nanosec:09d}\n"
            f"Right: {msg_right.header.stamp.sec}.{msg_right.header.stamp.nanosec:09d}\n"
            f"Odom: {msg_odom.header.stamp.sec}.{msg_odom.header.stamp.nanosec:09d}\n"
        )
        """
        # -----------------------------------------------------------
        
        #pc  = self._build_pc(msg_left.header.stamp, self.body_frame, pts)
        #self.pub_merge.publish(pc)                             ## 최종 퍼블리시
        
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


    # mhlee. 25.06.21
    @staticmethod
    @measure_time
    def voxel_downsample_with_timestamp(points_with_time: np.ndarray, voxel_size: float) -> np.ndarray:
        if points_with_time.shape[0] == 0:
            return points_with_time
        
        points = points_with_time[:, :3]
        timestamps = points_with_time[:, 3]

        voxel_indices = np.floor(points / voxel_size).astype(np.int32)
        keys, inverse, counts = np.unique(voxel_indices, axis=0, return_inverse=True, return_counts=True)
        
        grid_sum = np.zeros_like(keys, dtype=np.float64)
        pts_mean = np.zeros((keys.shape[0], 3), dtype=np.float32)
        for dim in range(3):
            grid_sum = np.bincount(inverse, weights=points[:,dim], minlength=keys.shape[0])
            pts_mean[:,dim] = grid_sum / counts

        max_timestamps = np.full(keys.shape[0], -1.0, dtype=np.float32)
        np.maximum.at(max_timestamps, inverse, timestamps)
        
        return np.hstack((pts_mean, max_timestamps[:, np.newaxis]))

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
# ──────────────────────────────── Remove point 관련 함수 ──────────────────────────────────────────
# ────────────────────────────────────────────────────────────────────────────────────────────────
    # mhlee. 25.06.21
    @staticmethod
    @measure_time
    def filter_points_by_time(points: np.ndarray, timestamp: float, window_sec: float) -> np.ndarray:
        """
        시간 윈도우 내의 포인트만 유지합니다.
        ...
        """
        if points.shape[0] == 0:
            return points

        N = points.shape[0]
        
        current_time_arr = np.full((N, 1), timestamp)

        point_timestamps = points[:, 3].reshape(N, 1)
        
        recent_mask = (current_time_arr - point_timestamps).flatten() <= window_sec
        
        return points[recent_mask]

        
        

    # iwshim. 25.05.30
    @staticmethod
    @measure_time
    def remove_far_points(points: np.ndarray, center: np.ndarray, radius: float) -> np.ndarray:
        dist2 = np.sum((points - center.reshape(1,3))**2, axis=1)
        mask = dist2 < radius**2
        return points[mask]
        


# ────────────────────────────────────────────────────────────────────────────────────────────────
# ──────────────────────────────── Normal Estimation 관련 함수 ──────────────────────────────────────────
# ────────────────────────────────────────────────────────────────────────────────────────────────

    # iwshim. 25.06.02
    @staticmethod
    @measure_time
    def estimation_normals(points: np.ndarray, k: int = 40) -> np.ndarray:
        '''
        tree = KDTree(points)
        _, idx = tree.query(points, k = k+1) # idx: N x k+1
        nns = points[idx[:,1:]] # nns: N x k x 3
        mean_nns = nns.mean(axis = 1, keepdims = True) # N x 1 x 3
        X = nns - mean_nns # N x k x 3

        # batch-wise convariance matix 
        convs = np.einsum('nik,nil->nilk',X, X) / (k-1) # N x k x 3 x 3
        cov = convs.sum(axis=1)
        
        eigvals, eigvecs = np.linalg.eigh(cov) # eigvecs: (N, 3, 3)
        normals = eigvecs[:,:,0] # (N,3), the minimum values of eigen vectors
        
        # Only upper sided vector allowed
        normals[normals[:,2] < 0] *= -1
        
        #normalization
        normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
        '''
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn = k))
        
        return np.asarray(pcd.normals)
        
    @staticmethod
    @measure_time
    def estimate_normals_half_random_open3d(points, k=20, k_search=40, deterministic_k=8):
        """
       open3d KDTree로 half-random neighbor 기반 normal estimation
       - points: (N,3) ndarray
       - k: 최종 normal 계산에 사용할 neighbor 개수
       - k_search: KNN pool 개수 (k <= k_search)
       - deterministic_k: 항상 선택할 가장 가까운 이웃 수
       """
        N = points.shape[0]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        kdtree = o3d.geometry.KDTreeFlann(pcd)

        normals = np.zeros((N, 3), dtype=np.float32)

        for i in range(N):
            [_, idx, _] = kdtree.search_knn_vector_3d(points[i], k_search+1)
            idx = np.array(idx[1:])  # 자기 자신 제외
              # 항상 가까운 deterministic_k개 고정
            keep = idx[:deterministic_k]
            rest_pool = idx[deterministic_k:]
              # 나머지는 랜덤하게 뽑음
            if len(rest_pool) >= k - deterministic_k:
                rand = np.random.choice(rest_pool, k - deterministic_k, replace=False)
            else:
                rand = rest_pool
            nn_idx = np.concatenate([keep, rand])
            neighbor_pts = points[nn_idx]

             # normal 추정 (cov/eig)
            mean = neighbor_pts.mean(axis=0, keepdims=True)
            X = neighbor_pts - mean
            cov = (X.T @ X) / (X.shape[0] - 1)
            eigvals, eigvecs = np.linalg.eigh(cov)
            normal = eigvecs[:, 0]
            if normal[2] < 0:
                normal *= -1
            normals[i] = normal / np.linalg.norm(normal)
        return normals
        
    #iwshim. 25.06.02
    #@staticmethod
    #@measure_time
    def bilateral_filter_torch(points_np: np.ndarray, k: int = 20, sigma: float = 0.05):
        points_torch = torch.tensor(points_np, dtype=torch.float32, device='cuda')
        N = points_torch.shape[0]
        
        return np.asarray(pcd.points)
    
    # mhlee. 25.06.22
    @staticmethod
    @measure_time
    def _estimate_normals_pytorch3d(points_gpu: torch.Tensor, k: int) -> torch.Tensor:
        
        """
        PyTorch3D의 빌딩 블록을 사용해 GPU에서 Normal을 추정합니다.
        """
        from pytorch3d.ops import knn_points 

        if points_gpu.ndim == 2:
            points_gpu = points_gpu.unsqueeze(0) # (N, 3) -> (1, N, 3) 배치 차원 추가

        # # 1. GPU에서 k-NN 탐색 (PyTorch3D의 핵심 기능)
        _, _, nn_points = knn_points(points_gpu, points_gpu, K=k, return_nn=True)

        # 2. 주성분 분석 (PCA)을 PyTorch 텐서 연산으로 직접 구현
        centroid = torch.mean(nn_points, dim=2, keepdim=True)
        centered_points = nn_points - centroid
        cov_matrix = torch.matmul(centered_points.transpose(-1, -2), centered_points)
        
        # 고유값/고유벡터 계산 (가장 작은 고유값에 해당하는 벡터가 normal)
        _, eigenvectors = torch.linalg.eigh(cov_matrix)
        normals = eigenvectors[:, :, 0]

        # Normal 방향을 일관성 있게 (예: z축이 양수를 향하도록)
        # 실제로는 센서 위치를 기준으로 방향을 정해야 하지만, 여기서는 간단한 예시를 사용
        z_axis_vector = torch.tensor([0, 0, 1], device=points_gpu.device, dtype=torch.float32).unsqueeze(0).unsqueeze(0) # (1, 1, 3)
        z_axis_similarity = torch.sum(normals * z_axis_vector, dim=-1, keepdim=True) # (B, N_query, 1)
        normals = torch.where(z_axis_similarity < 0, -normals, normals)

        return normals.squeeze(0) # (N, 3) 형태로 반환





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
        og.info.origin.position.z = -1.7
        og.data = grid.flatten(order='C').astype(int).tolist()
        
        return og
        


# ────────────────────────────────────────────────────────────────────────────────────────────────
# ──────────────────────────────── 기타 ──────────────────────────────────────────
# ────────────────────────────────────────────────────────────────────────────────────────────────

    # ───────────── depth Image → 3-D 포인트 변환 ─────────────
    @measure_time
    def _depth_to_pts(self, msg: Image, prefix: str) -> Optional[np.ndarray]:
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
    
    
    
    # geometry_msgs/TransformStamped -> 4x4 Transform Matrix, iwshim 25.05.30
    @staticmethod
    def transform_to_matrix(t):
        import transforms3d
        trans = t.transform.translation
        rot = t.transform.rotation
        T = np.eye(4)
        T[:3, 3] = [trans.x, trans.y, trans.z]
        # Quaternion 2 Rotation Matrix
        R = transforms3d.quaternions.quat2mat([rot.w, rot.x, rot.y, rot.z])
        T[:3, :3] = R
        #print(T)
        return T
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
    def _build_pc(stamp, frame: str, points: np.ndarray) -> PointCloud2:
        fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        cloud = PointCloud2()
        cloud.header.stamp = stamp.to_msg()
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
    node = DepthToPointCloudNode()       ## 노드 인스턴스 생성
    rclpy.spin(node)                     ## 콜백 루프 진입
    node.destroy_node()                  ## 종료 시 정리
    rclpy.shutdown()                     ## rclpy 종료

if __name__ == "__main__":
    main()                               ## python filename.py 실행 시 main 호출


