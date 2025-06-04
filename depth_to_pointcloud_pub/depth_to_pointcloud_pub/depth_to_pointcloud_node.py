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

from transforms3d.quaternions import quat2mat

## 여러 토픽 동기화용
from message_filters import Subscriber, ApproximateTimeSynchronizer

# iwshim
from nav_msgs.msg import Odometry, OccupancyGrid, MapMetaData #iwshim 25.06.02
#from sklearn.neighbors import NearestNeighbors#KDTree #iwshim 25.06.02

import time, torch
import open3d as o3d
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
    """
    config/*.yaml 파일에서 key에 해당하는 4×4 행렬을 읽어온 뒤 numpy 배열로 반환
    """
    pkg = get_package_share_directory("depth_to_pointcloud_pub")        ## 패키지 경로 찾기
    with open(os.path.join(pkg, "config", yaml_name), "r", encoding="utf-8") as f:
        return np.array(yaml.safe_load(f)[key]["Matrix"]).reshape(4, 4)

# ────────────────────── main node ──────────────────────
class DepthToPointCloudNode(Node):
    def __init__(self):
        super().__init__("depth_to_pointcloud_node")  ## 노드 이름 등록

        # 기본 설정 ---------------------------------------------------------
        self.prefixes = ["frontleft", "frontright"]
        self.depth_base = "/spot1/base/spot/depth"
        self.body_frame = "spot1/base/spot/body"
        self.origin_frame = "spot1/base/spot/odom"    # iwshim. 25.05.30
        self.odom_topic = "/spot1/base/spot/odometry" # iwshim. 25.05.30
        self.merge_topic = "/spot1/base/spot/depth/merge_front"
        self.bridge = CvBridge()
        
        # Accumulation Parmas, iwshim 25.05.30
        self.clouds = np.zeros((1,3))
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.accum_topic = "/spot1/base/spot/depth/accum_front"
        self.occup_topic = "/spot1/base/spot/depth/occup_front"
        # -----------------------------------------------------------
        
        # CameraInfo → K 행렬 캐시 ------------------------------------------
        self.K: Dict[str, Optional[np.ndarray]] = {p: None for p in self.prefixes}

        # 고정 extrinsic (cam → body) ---------------------------------------
        self.extr = {
            "frontleft":  load_extrinsic_matrix("frontleft_info.yaml",  "body_to_frontleft"),
            "frontright": load_extrinsic_matrix("frontright_info.yaml", "body_to_frontright"),
        }

        # CameraInfo 구독 ----------------------------------------------------
        for p in self.prefixes:
            self.create_subscription(
                CameraInfo,                                    ## 타입
                f"{self.depth_base}/{p}/camera_info",          ## 토픽 이름
                lambda m, pr=p: self._camera_info_cb(m, pr),   ## 콜백 (prefix 캡쳐)
                10)                                                     ## 큐 길이
                    
        # Depth Image 구독: message_filters 로 동기화 -----------------------
        self.sub_left  = Subscriber(self, Image, f"{self.depth_base}/frontleft/image")
        self.sub_right = Subscriber(self, Image, f"{self.depth_base}/frontright/image")
        self.sub_odom  = Subscriber(self, Odometry, self.odom_topic) # iwshim. 25.05.30
        
        self.sync = ApproximateTimeSynchronizer(
            [self.sub_left, self.sub_right, self.sub_odom],  ## 동기화할 Subscriber 리스트, # iwshim. 25.05.30
            queue_size=10,                    ## 내부 버퍼 크기
            slop=0.05)                        ## 최대 허용 시간차 (초)  = 50 ms
        self.sync.registerCallback(self._synced_depth_cb)  ## 두 이미지가 짝 맞으면 호출

        # Only for debugging 결과 PointCloud2 퍼블리셔 -----------------------------------------
        self.pub_merge = self.create_publisher(PointCloud2, self.merge_topic, 10)
        self.pub_accum = self.create_publisher(PointCloud2, self.accum_topic, 10)
        self.pub_occup = self.create_publisher(OccupancyGrid, self.occup_topic, 10)

    
    # ───────────── CameraInfo 콜백 ─────────────
    def _camera_info_cb(self, msg: CameraInfo, prefix: str):
        self.K[prefix] = np.array(msg.k).reshape(3, 3)   ## 내부 파라미터 저장
        self.get_logger().info(f"[{prefix}] CameraInfo OK\n", once=True)

    # ───────────── 동기화된 Depth 이미지 콜백 ─────────────
    def _synced_depth_cb(self, msg_left: Image, msg_right: Image, msg_odom: Odometry):
        """
        frontleft·frontright 깊이 이미지가 거의 동시에 도착하면 호출
        두 이미지를 각각 포인트로 변환 후 합쳐 하나의 PointCloud2로 publish
        """
        pts_left  = self._depth_to_pts(msg_left,  "frontleft")
        pts_right = self._depth_to_pts(msg_right, "frontright")
        pts = np.vstack((pts_left, pts_right))           ## (N,3) 행렬 합치기 *속도 최적화시 Check Point.
        
        # iwshim. 25.05.30
        try:
            trans = self.tf_buffer.lookup_transform(
                self.origin_frame, # target frame
                self.body_frame, # input frame id
                rclpy.time.Time(),
                timeout = rclpy.duration.Duration(seconds = 0.1)
            )
        except Exception as e:
            self.get_logger().warning(f"TF transform failed: {e}\n")
            return
        # -----------------------------------------------------------
        
        # 4x4 Transform matrix from msg_left.frame_id -> body_frame, iwshim. 25.05.30
        T = self.transform_to_matrix(trans)
        pts_homo = np.hstack([pts, np.ones((pts.shape[0], 1))])
        #self.get_logger().info( f"\nPTS shape: {pts_homo.shape[0]:d}, {pts_homo.shape[1]:d}\n" )
        pts_tf = (T @ pts_homo.T).T[:,:3]
        # -----------------------------------------------------------

        # Accumulation, iwshim. 25.05.30
        pos = msg_odom.pose.pose.position
        center = np.array([pos.x, pos.y, pos.z])

        if self.clouds.shape[0] == 1:
            self.clouds = pts_tf
        else:
            self.clouds = np.vstack([self.clouds, pts_tf])
        #self.clouds = self.voxel_downsample_mean(self.clouds, 0.1)
        self.clouds = self.voxel_downsample_max_elevation_vec(self.clouds, 0.05)
        self.clouds = self.remove_far_points(self.clouds, center, 7)
        nm = self.estimation_normals(self.clouds) # fast, but large noise
        #nm = self.estimate_normals_half_random_open3d(self.clouds) # too slow, more than 4,000ms
        og = self.pointcloud_to_occupancy_grid(msg_left.header.stamp, 
                                               self.origin_frame, 
                                               self.clouds,
                                               resolution = 0.1,
                                               grid_size = 100,
                                               center_xy=(pos.x, pos.y),
                                               normals = nm)
                                               
        pc = self._build_pc(msg_left.header.stamp, self.origin_frame, self.clouds) #Only for display
        self.pub_accum.publish(pc)
        self.pub_occup.publish(og)
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
        
    
    # ───────────── depth Image → 3-D 포인트 변환 ─────────────
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



    # iwshim. 25.05.30
    @staticmethod
    @measure_time
    def remove_far_points(points: np.ndarray, center: np.ndarray, radius: float) -> np.ndarray:
        dist2 = np.sum((points - center.reshape(1,3))**2, axis=1)
        mask = dist2 < radius**2
        return points[mask]
        
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
        og.header.stamp = stamp
        og.header.frame_id = frame
        og.info.resolution = resolution
        og.info.width = grid_size
        og.info.height = grid_size
        og.info.origin.position.x = origin_x
        og.info.origin.position.y = origin_y
        og.info.origin.position.z = -1.7
        og.data = grid.flatten(order='C').astype(int).tolist()
        
        return og
        
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
    node = DepthToPointCloudNode()       ## 노드 인스턴스 생성
    rclpy.spin(node)                     ## 콜백 루프 진입
    node.destroy_node()                  ## 종료 시 정리
    rclpy.shutdown()                     ## rclpy 종료

if __name__ == "__main__":
    main()                               ## python filename.py 실행 시 main 호출


