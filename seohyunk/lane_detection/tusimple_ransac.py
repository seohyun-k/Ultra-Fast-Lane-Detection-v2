import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32
from cv_bridge import CvBridge
import cv2
import torch
import numpy as np

from model.model_tusimple import get_model
from utils.image_utils import preprocess_image, postprocess_output
from utils.ransac_utils import fit_ransac_curve
from utils.lane_fallback import LaneHistoryBuffer, SteeringFallbackController

class Config:
    def __init__(self):
        self.backbone = "18"
        self.num_cell_row = 100
        self.num_row = 56
        self.num_cell_col = 100
        self.num_col = 41
        self.num_lanes = 4
        self.train_height = 320
        self.train_width = 800
        self.model_path = "pth/tusimple_model100.pth"
        self.use_aux = False
        self.fc_norm = False

class LaneDetectionNode(Node):
    def __init__(self):
        super().__init__('lane_detection_node')
        qos = rclpy.qos.QoSProfile(depth=10, reliability=rclpy.qos.ReliabilityPolicy.BEST_EFFORT)

        self.cfg = Config()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.bridge = CvBridge()
        self.orig_width = 640
        self.orig_height = 480

        self.subscription = self.create_subscription(Image, '/truck1/front_camera', self.image_callback, qos)
        self.steer_publisher = self.create_publisher(Float32, '/truck1/steer_control', 10)

        self.model = get_model(self.cfg).to(self.device)
        state_dict = torch.load(self.cfg.model_path, map_location=self.device)
        self.model.load_state_dict(state_dict['model'])
        self.model.eval()

        self.buffer = LaneHistoryBuffer(max_length=5)
        self.fallback = SteeringFallbackController(max_missing=5, decay_rate=0.8)

        self.get_logger().info("✅ TuSimple Lane Detection Node Started")

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
        return float(np.clip(steer_angle, -10.0, 10.0))

    def publish_steering(self, angle):
        msg = Float32()
        msg.data = angle
        #self.steer_publisher.publish(msg)

    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        input_tensor = preprocess_image(frame, self.cfg, self.device)

        with torch.no_grad():
            pred = self.model(input_tensor)

        lanes = postprocess_output(
            pred_dict=pred,
            cfg=self.cfg,
            orig_width=self.orig_width,
            orig_height=self.orig_height,
            fit_ransac_fn=fit_ransac_curve  # ✅ 곡선 대응 RANSAC
        )

        detected = len(lanes) > 0
        if detected:
            self.buffer.update(lanes)
        else:
            lanes = self.buffer.get_latest()

        steer = self.calculate_steering_flexible(lanes)
        steer = self.fallback.update(steer, detected)
        self.publish_steering(steer)

        vis = frame.copy()
        for lane in lanes:
            for pt in lane:
                cv2.circle(vis, pt, 3, (0, 255, 0), -1)
        cv2.imshow("TuSimple Curve-Aware Lane Detection", vis)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = LaneDetectionNode()
    rclpy.spin(node)
    node.destroy_node()
    cv2.destroyAllWindows()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

