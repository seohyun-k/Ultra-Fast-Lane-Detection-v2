import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import torch
import numpy as np
from model.model_curvelanes import get_model  # ✅ 모델 로딩 함수

class Config:
    def __init__(self):
        self.backbone = "18"
        self.num_cell_row = 200
        self.num_row = 72
        self.num_cell_col = 100
        self.num_col = 41
        self.num_lanes = 10
        self.train_height = 800
        self.train_width = 1600
        self.model_path = "models/curvelanes_res18.pth"
        self.use_aux = False
        self.fc_norm = False

class LaneDetectionNode(Node):
    def __init__(self):
        super().__init__('lane_detection_node')
        qos_profile = rclpy.qos.QoSProfile(
            depth=10,
            reliability=rclpy.qos.ReliabilityPolicy.BEST_EFFORT  # ✅ 이미지 퍼블리셔와 QoS 일치
        )
        self.cfg = Config()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = get_model(self.cfg).to(self.device)
        # ✅ 수정 코드: 'module.' 키 제거 후 state_dict 로드
        state_dict = torch.load(self.cfg.model_path, map_location=self.device)

        # 키 이름에서 'module.' 제거
        if 'model' in state_dict:
            new_state_dict = {}
            for k, v in state_dict['model'].items():
                new_key = k.replace('module.', '')  # prefix 제거
                new_state_dict[new_key] = v
            self.model.load_state_dict(new_state_dict)
        else:
            self.model.load_state_dict(state_dict)

        self.model.eval()

        self.bridge = CvBridge()
        self.subscription = self.create_subscription(
            Image,
            '/truck1/front_camera',
            self.image_callback,
            qos_profile
        )

        # 원본 해상도 고정 (1280x720)
        self.orig_width = 640
        self.orig_height = 380

    def preprocess(self, img):
        img = cv2.resize(img, (self.cfg.train_width, self.cfg.train_height))
        img = img / 255.0
        img = (img - 0.5) / 0.5
        img = img.transpose(2, 0, 1)
        return torch.tensor(img, dtype=torch.float32).unsqueeze(0).to(self.device)

    def postprocess(self, pred_dict):
        loc_row = pred_dict['loc_row'].squeeze(0).cpu().numpy()
        exist_row = pred_dict['exist_row'].squeeze(0)[1].cpu().numpy()

        lanes = []

        scale_x = self.orig_width / self.cfg.train_height    # 1280 / 320
        scale_y = self.orig_height / self.cfg.train_width    # 720 / 800

        for lane_idx in range(loc_row.shape[2]):
            if np.sum(exist_row[:, lane_idx]) < 5:
                continue
            points = []
            for r in range(loc_row.shape[0]):
                cls = np.argmax(loc_row[r, :, lane_idx])
                if cls > 0:
                    x = cls * (self.cfg.train_width / self.cfg.num_row)
                    y = r * (self.cfg.train_height / self.cfg.num_cell_row)

                    # 시계방향 90도 회전
                    x_rot = self.cfg.train_height - y
                    y_rot = x

                    # 해상도 보정
                    x_rev = int(x_rot * scale_x)
                    x_final = 640 - x_rev
                    y_final = int(y_rot * scale_y)+100

                    points.append((x_final, y_final))
            lanes.append(points)
        return lanes

    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        input_tensor = self.preprocess(frame)
        with torch.no_grad():
            pred = self.model(input_tensor)
        lanes = self.postprocess(pred)

        vis = frame.copy()
        for lane in lanes:
            for pt in lane:
                cv2.circle(vis, pt, 3, (0, 255, 0), -1)

        cv2.imshow("ROS2 Lane Detection", vis)
        cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = LaneDetectionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
