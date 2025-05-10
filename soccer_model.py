import numpy as np
from ultralytics import YOLO


class BallDetector(object):
    def __init__(self, model_name, cornerPoints, ball_pos_init):
        self.model = YOLO(model_name)  # yolo8足球检测模型
        self.ball_pos = np.array(ball_pos_init)

        # inrange判定
        ymin = min(cornerPoints[0][1], cornerPoints[1][1])
        ymax = max(cornerPoints[2][1], cornerPoints[3][1])
        p = (cornerPoints[2][1] - cornerPoints[0][1]) / (cornerPoints[2][0] - cornerPoints[0][0])
        q = (cornerPoints[3][1] - cornerPoints[1][1]) / (cornerPoints[3][0] - cornerPoints[1][0])
        self.top_bottom_limit = lambda y: ymin < y < ymax
        self.left_limit = lambda x, y: x > cornerPoints[0][0] + (y - cornerPoints[0][1]) / p
        self.right_limit = lambda x, y: x < cornerPoints[1][0] + (y - cornerPoints[1][1]) / q

    """判断人或球是否在球场内"""
    def _inRange(self, point):
        x = point[0]
        y = point[1]
        if self.top_bottom_limit(y) and self.left_limit(x, y) and self.right_limit(x, y):
            return True
        else:
            return False

    def detect(self, img):
        # 检测球
        detect_results = self.model(img)
        # 检测到足球
        if len(detect_results[0].boxes.data) > 0:  # 一帧中检测到一个球
            best_idx = 0
            if len(detect_results[0].boxes.data) > 1:  # 一帧中检测到多个球
                best_idx = detect_results[0].boxes.conf.argmax()  # 找置信度最大的
            conf_ball = detect_results[0].boxes.conf[best_idx].cpu().numpy()  # 检测置信度
            if conf_ball > 0.7:  # 如果置信度大于0.7才认为它是球
                ball_pos = detect_results[0].boxes.xywh[best_idx][:2].cpu().numpy()  # 球的坐标
                if self._inRange(ball_pos):
                    self.ball_pos = ball_pos
        return self.ball_pos

    # def get_point_ball(self):
    #     # 球的位置
    #     if self._inRange(self.ball_pos):
    #         point_ball = self.ball_pos
    #     else:
    #         point_ball = np.zeros(2)
    #     return point_ball

    def reset_ball(self):
        self.ball_pos = np.array([0, 0])