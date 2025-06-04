import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32
from cv_bridge import CvBridge
import cv2
import torch
from model.model_culane import parsingNet
from utils.image_utils import preprocess_image, postprocess_output
from utils.ransac_utils import fit_ransac_curve
from utils.lane_fallback import LaneHistoryBuffer, SteeringFallbackController
import numpy as np

class Config:
    def __init__(self):
        self.backbone = "18"
        self.num_cell_row = 200
        self.num_row = 72
        self.num_cell_col = 100
        self.num_col = 81
        self.num_lanes = 4
        self.train_height = 320
        self.train_width = 1600
        self.model_path = "pth/culane_res18.pth"
        self.use_aux = False
        self.fc_norm = True

class DeepLaneNode(Node):
    def __init__(self):
        super().__init__('deep_lane_node')
        qos = rclpy.qos.QoSProfile(depth=10, reliability=rclpy.qos.ReliabilityPolicy.BEST_EFFORT)

        self.cfg = Config()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.bridge = CvBridge()
        self.orig_width = 640
        self.orig_height = 480

        self.sub = self.create_subscription(Image, '/truck0/front_camera', self.image_callback, qos)
        self.pub_steer = self.create_publisher(Float32, '/truck0/steer_control', 10)

        self.buffer = LaneHistoryBuffer(max_length=5)
        self.fallback = SteeringFallbackController(max_missing=5, decay_rate=0.8)

        self.model = parsingNet(
            pretrained=False,
            backbone=self.cfg.backbone,
            num_grid_row=self.cfg.num_cell_row,
            num_cls_row=self.cfg.num_row,
            num_grid_col=self.cfg.num_cell_col,
            num_cls_col=self.cfg.num_col,
            num_lane_on_row=self.cfg.num_lanes,
            num_lane_on_col=self.cfg.num_lanes,
            input_height=self.cfg.train_height,
            input_width=self.cfg.train_width,
            use_aux=self.cfg.use_aux,
            fc_norm=self.cfg.fc_norm
        ).to(self.device)

        state_dict = torch.load(self.cfg.model_path, map_location=self.device)
        new_state_dict = {k.replace("module.", ""): v for k, v in state_dict['model'].items()}
        self.model.load_state_dict(new_state_dict)
        self.model.eval()
        self.get_logger().info("✅ 딥러닝 차선 모델 로드 완료 (CULane)")

    def calculate_steering_flexible(self, lanes):
        center_x = self.orig_width // 2

        if not lanes:
            return 0.0

        if len(lanes) == 1:
            lane = lanes[0]
            if not lane:
                return 0.0
            lane.sort(key=lambda pt: pt[1], reverse=True)
            bottom_pts = lane[:5]
            avg_x = np.mean([pt[0] for pt in bottom_pts])
            estimated_center = avg_x + 100 if avg_x < center_x else avg_x - 100
        else:
            all_pts = []
            for lane in lanes:
                if lane:
                    lane.sort(key=lambda pt: pt[1], reverse=True)
                    all_pts.extend(lane[:5])
            if not all_pts:
                return 0.0
            estimated_center = np.mean([pt[0] for pt in all_pts])

        error = center_x - estimated_center
        gain = 0.005
        steer_angle = error * gain
        return float(np.clip(steer_angle, -10.0, 10.0))  # 조향 범위 클립 (설정에 맞게 조정)

    def publish_steering(self, angle):
        msg = Float32()
        msg.data = angle
        self.pub_steer.publish(msg)

    def image_callback(self, msg):
        img_raw = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        input_tensor = preprocess_image(img_raw, self.cfg, self.device)

        with torch.no_grad():
            pred = self.model(input_tensor)

        lanes = postprocess_output(pred, self.cfg, self.orig_width, self.orig_height, fit_ransac_curve)
        detected = len(lanes) > 0

        if detected:
            self.buffer.update(lanes)
        else:
            lanes = self.buffer.get_latest()

        steer_angle = self.calculate_steering_flexible(lanes)
        steer_angle = self.fallback.update(steer_angle, detected=detected)
        self.publish_steering(steer_angle)

        vis = img_raw.copy()
        for lane in lanes:
            for pt in lane:
                cv2.circle(vis, pt, 3, (0, 255, 0), -1)
        cv2.imshow("CULane ROS2 Lane Detection", vis)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = DeepLaneNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    cv2.destroyAllWindows()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
