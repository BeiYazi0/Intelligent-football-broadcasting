import numpy as np

from yolox.tracker.byte_tracker import BYTETracker
from yolox.utils.visualize import plot_tracking



def distance(point1, point2):
    return np.linalg.norm(point1 - point2)


class TrackProcesser(object):
    def __init__(self, args, test_size, ckpt, cornerPoints, window_center_data):
        self.tracker = BYTETracker(args, frame_rate=30)
        self.aspect_ratio_thresh = args.aspect_ratio_thresh
        self.min_box_area = args.min_box_area
        self.test_size = test_size
        self.ckpt = ckpt
        self.size_factor = None

        self.track_res = None          # ckpt 个帧中最后一帧的人体跟踪结果
        self.track_res_ckpt = None     # ckpt 个帧中第一帧的人体跟踪结果
        self.track_res_cross = None    # 出现在前 ckpt 帧中的人
        self.img_size = None
        self.frame_shape = None

        # inrange判定
        ymin = min(cornerPoints[0][1], cornerPoints[1][1])
        ymax = max(cornerPoints[2][1], cornerPoints[3][1])
        p = (cornerPoints[2][1] - cornerPoints[0][1]) /  (cornerPoints[2][0] - cornerPoints[0][0])
        q = (cornerPoints[3][1] - cornerPoints[1][1]) / (cornerPoints[3][0] - cornerPoints[1][0])
        self.top_bottom_limit = lambda y: (y > ymin) & (y < ymax)
        self.left_limit = lambda x, y: x > cornerPoints[0][0] + (y - cornerPoints[0][1]) / p
        self.right_limit = lambda x, y:x < cornerPoints[1][0] + (y - cornerPoints[1][1]) / q

        # 用于 pos_diff
        self.top_y = (cornerPoints[0][1] + cornerPoints[1][1]) / 2  # 上边线的y
        self.height_trap = (cornerPoints[2][1] + cornerPoints[3][1]) / 2 - self.top_y  # 梯形的高

        # 记录上一次的 point
        self.point_fast = 0
        self.final_point = None
        self.direct_vec = None
        self.direct_flag =False

        # 窗口
        self.window_size_init = None
        self.window_size = None
        self.window_center = np.memmap(window_center_data, dtype='float32', mode='r+', shape=(2,))
        self.max_v = 10 # 设置最大速度(相隔ckpt帧移动10个像素)

        # 判定
        # self.window_double = None
        self.goal = None # 如果人在球门附近
        self.offest_check = None # 补偿由于透视变换带来的偏移（画面在最右边时向左偏移，在最左边时向右偏移）


    def set_window(self, img_size, window_size_init, size_factor):
        self.window_size_init = window_size_init
        self.img_size = img_size
        self.size_factor = size_factor
        self.frame_shape = img_size[::-1]
        self.window_size = window_size_init.copy()
        # self.window_center[:] = np.array([img_size[0] // 2, img_size[1] // 3])  # 初始化窗口位置（中间偏上
        self.goal = lambda pos: (0.15 * img_size[0] < pos[:, 0]) & (pos[:, 0] < img_size[0] * 0.4) & \
                                (0.2 * img_size[1] < pos[:, 1]) & (pos[:, 1] < 0.5 * img_size[1]) | \
                                (img_size[0] * 0.6 < pos[:, 0]) & (pos[:, 0] < img_size[0] * 0.8) & \
                                (0.2 * img_size[1] < pos[:, 1]) & (pos[:, 1] < 0.4 * img_size[1])
        self.offest_check = lambda x: x > 0.7 * img_size[0] or x < 0.3 * img_size[0]

    """判断人或球是否在球场内"""
    def _inRange(self, points):
        x = points[:, 0]
        y = points[:, 1]
        return self.top_bottom_limit(y) & self.left_limit(x, y) & self.right_limit(x, y)

    """找移动最快的人的所在区域"""
    def _get_pos_diff(self):
        trap_factor = (1 - (self.track_res_cross[:, 2] / 2 - self.top_y) / (self.height_trap * 2))  # 梯形投影的系数
        pos_diff = np.abs(self.track_res_cross[:, 0]) * trap_factor
        return pos_diff

    """将dict按照value从大到小排序，找出在这step帧内移动最快的人的未来平均坐标，计算需要移动到的位置(point_fast)"""
    def _get_point_fast(self, pos_diff):
        n = pos_diff.shape[0]
        max_v_now = 0
        if n > 0:
            if n > 3:
                max_index = np.argpartition(pos_diff, -3)[-3:]
            else:
                max_index = np.argsort(pos_diff)
            max_v_now = pos_diff[max_index[-1]]
            self.max_v = max(max_v_now, self.max_v)  # 更新目前最快的速度
            self.point_fast = self.track_res_cross[max_index, :][:, 3:].mean(axis = 0)
        return max_v_now, self.point_fast

    """画面中所有人的中心点"""
    def _get_point_people(self):
        return self.track_res.mean(axis = 0)

    """生成一张热力图， 找到画面中最密集的区域"""
    def _get_heatmap(self):
        img_size = self.img_size
        n = self.track_res.shape[0]
        # 以该人为中心，周围hp_range个像素的矩阵范围+1
        hp_range = img_size[0] // 30
        heatmap = np.zeros((img_size[1], img_size[0]))

        window_double = lambda pos: (self.window_center[0] - self.window_size[0] // 2 < pos[:, 0]) & (
                pos[:, 0] < self.window_center[0] + self.window_size[0] // 2) & \
                                    (self.window_center[1] - self.window_size[1] // 2 < pos[:, 1]) & (
                                            pos[:, 1] < self.window_center[1] + self.window_size[1] // 2)

        points = np.ones(n)
        points[self.goal(self.track_res)] += 1 # 如果人在球门附近，额外加1
        points[window_double(self.track_res)] += 1 # 如果人出现在上一帧窗口中，额外加1

        start_1 = np.maximum(0, self.track_res[:, 1] - hp_range).astype(np.int16)
        end_1 = np.minimum(img_size[1], self.track_res[:, 1] + hp_range).astype(np.int16)
        start_0 = np.maximum(0, self.track_res[:, 0] - hp_range).astype(np.int16)
        end_0 = np.minimum(img_size[0], self.track_res[:, 0] + hp_range).astype(np.int16)
        for i in range(n):
             heatmap[start_1[i]: end_1[i], start_0[i]: end_0[i]] += points[i]

        return heatmap

    def _get_point_density(self):
        heatmap = self._get_heatmap()
        max_index = np.unravel_index(np.argmax(heatmap, axis=None), heatmap.shape)  # 找热力图中最大值点（最密集的位置）
        point_density = np.array([max_index[1], max_index[0]])
        return point_density

    """根据point_ball， point_people， point_fast， point_density 来找最合适的窗口位置(final_point)"""
    def _get_final_point(self, point_ball):
        window_center = self.window_center
        window_size = self.window_size
        img_size = self.img_size

        # points
        point_people = self._get_point_people() # 画面中所有人的中心点
        pos_diff = self._get_pos_diff() # 找移动最快的人的所在区域
        max_v_now, point_fast = self._get_point_fast(pos_diff)  # 将dict按照value从大到小排序，找出在这step帧内移动最快的人的未来平均坐标，计算需要移动到的位置(point_fast)
        point_density = self._get_point_density() # 生成一张热力图， 找到画面中最密集的区域
        # print(point_ball)
        # print(point_people)
        # print(point_fast)
        # print(point_density)

        # distance
        distance_bp = distance(point_ball, point_people)
        distance_bf = distance(point_ball, point_fast)
        distance_fp = distance(point_fast, point_people)
        distance_fd = distance(point_fast, point_density)
        distance_dp = distance(point_density, point_people)

        window_check = window_size / 10
        if point_ball[1] > 0:  # 如果检测到球了
            if max([distance_bp, distance_bf, distance_fp]) < window_check[0]:  # 三者接近，取三者的中心
                final_point = (point_fast + point_people + point_ball) / 3
            else:  # 否则取与球更为接近的两者中心(当有相近的点时)
                if distance_bp > window_check[0] and distance_bf > window_check[0]:
                    final_point = point_ball
                else:
                    final_point = (point_ball + point_people) / 2 \
                        if distance_bp < distance_bf \
                        else (point_ball + point_fast) / 2
        else:  # 没检测到球
            # 如果有任意两点距离较近，则取其中点为final
            if distance_fd < window_check[1]:
                final_point = (point_fast + point_density) / 2
            elif distance_dp < window_check[1]:
                final_point = (point_density + point_people) / 2
            elif distance_fp < window_check[1]:
                final_point = (point_fast + point_people) / 2
            # 如果有快速移动目标，优先跟速度快的目标
            elif len(pos_diff) > 0 and max_v_now >= 0.9 * self.max_v:
                final_point = point_fast
            elif not self.direct_flag:
                final_point = (point_fast + point_density) / 2
            else:
                # fast, people, density, 以及两两的中点，以及三者的中点中，使画面抖动最小的点作为下一个点
                pre_point = [point_fast, point_people, point_density, (point_fast + point_density) // 2,
                             (point_fast + point_people + point_density) // 3]

                vibe_score = np.zeros(len(pre_point))  # 抖动分数
                for i in range(len(pre_point)):
                    # 对每个点计算一个抖动分数，由两部分组成，一个是与上一个点的移动距离，另一个是与上一次移动向量的方向是否相反
                    distance_score = distance(pre_point[i], self.final_point) / window_size[0]
                    # 单位向量
                    unit_dir_last = self.direct_vec / np.linalg.norm(self.direct_vec)

                    dir_now = pre_point[i] - window_center
                    unit_dir_now = dir_now / np.linalg.norm(dir_now)

                    cos_unit = np.dot(unit_dir_now, unit_dir_last)
                    direct_score = (1 - cos_unit) / 2

                    vibe_score[i] = direct_score + distance_score

                min_idx = np.argmin(np.array(vibe_score))
                final_point = pre_point[min_idx]  # 取最小抖动分数的点
        # 补偿由于透视变换带来的偏移（画面在最右边时向左偏移，在最左边时向右偏移）
        if self.offest_check(final_point[0]):
            offset_x = int(0.25 * (final_point[0] - img_size[0] // 2))
            final_point[0] -= offset_x

        self.final_point = final_point.copy()
        return final_point

    """将目标点(final_point)与当前点(window_center)作差，转化成一个移动向量"""
    def get_direct_vec(self, point_ball, cnt):
        final_point = self._get_final_point(point_ball)
        window_size = self.window_size
        window_center = self.window_center

        direct_vec_pre = self.direct_vec
        direct_vec = np.fix(self.final_point - self.window_center) # 相隔ckpt帧两帧之间的窗口移动向量

        # 限制window移动
        if final_point[0] > window_center[0] - window_size[0] / 4 and \
                final_point[0] < window_center[0] + window_size[0] / 4 and \
                final_point[1] > window_center[1] - window_size[1] / 4 and \
                final_point[1] < window_center[1] + window_size[1] / 4:  # 目标点在上一次的window框的中心部分
            direct_vec *= 0.4
        elif final_point[0] > window_center[0] - window_size[0] * 3 / 8 and \
                final_point[0] < window_center[0] + window_size[0] * 3 / 8 and \
                final_point[1] > window_center[1] - window_size[1] * 3 / 8 and \
                final_point[1] < window_center[1] + window_size[1] * 3 / 8:  # 目标点在上一次的window框的外圈部分
            direct_vec *= 0.7
        else:
            direct_vec *= 0.9
        direct_vec = np.fix(direct_vec)

        # 如果移动向量与上一个向量反向且模长并没远大于上一个向量，则不移动
        if self.direct_flag:
            if (np.dot(direct_vec, direct_vec_pre) < 0 and np.dot(direct_vec, direct_vec) <
                    1.2 * np.dot(direct_vec_pre, direct_vec_pre)):
                direct_vec *= 0

        # 惯性向量
        if self.direct_flag:
            direct_vec += 0.3 * direct_vec_pre

        # 整体人员的移动向量
        person_vec = self.track_res_cross[:, 0:2]
        mv_factor = (window_center[1] + window_size[1] // 2 - self.track_res_cross[:, 4]) / window_size[1]
        person_vec = person_vec * mv_factor.reshape(-1, 1)
        move_vec = person_vec.mean(axis = 0).astype(np.int16)
        if self.track_res_cross.shape[0] > 0:
            direct_vec += 0.5 * move_vec

        direct_vec *= np.log(self.ckpt/cnt) + 1
        if(np.abs(direct_vec).sum() > np.abs(self.img_size).sum()/5):
            direct_vec = self.img_size * np.sign(direct_vec) / 5
        if cnt == self.ckpt:
            self.direct_vec = direct_vec.copy()
            self.direct_flag = True
        return direct_vec

    def update_res(self, outputs):
        track_res_id, track_res_pos = self._update_track(outputs)
        # 排除场外人员
        mask = self._inRange(track_res_pos)
        track_res_id = track_res_id[mask]
        self.track_res = track_res_pos[mask]

        # 取相同的id
        _, index1, index2 = np.intersect1d(self.track_res_ckpt[0], track_res_id, assume_unique=True, return_indices=True)

        track_res_ckpt = self.track_res_ckpt[1]
        self.track_res_cross = np.zeros((len(index1), 3))
        self.track_res_cross[:, 0] = self.track_res[index2, 0] - track_res_ckpt[index1, 0]
        self.track_res_cross[:, 1] = self.track_res[index2, 1] - track_res_ckpt[index1, 1]
        self.track_res_cross[:, 2] = self.track_res[index2, 1] + track_res_ckpt[index1, 1]
        self.track_res_cross = np.concatenate((self.track_res_cross, self.track_res[index2, :]), axis=1)

    def update_res_ckpt(self, outputs):
        track_res_id, track_res_pos = self._update_track(outputs)
        #print(track_res_pos)
        # 排除场外人员
        mask = self._inRange(track_res_pos)
        self.track_res_ckpt = (track_res_id[mask], track_res_pos[mask])

    def _update_track(self, outputs):
        if outputs[0] is not None:
            track_res_id = []
            trak_res_pos = []
            online_targets = self.tracker.update(outputs[0], self.frame_shape, self.test_size)
            # online_tlwhs = []
            # online_ids = []
            for t in online_targets:
                tlwh = t.tlwh
                tid = t.track_id
                vertical = tlwh[2] / tlwh[3] > self.aspect_ratio_thresh
                if tlwh[2] * tlwh[3] > self.min_box_area and not vertical:
                    # online_tlwhs.append(tlwh)
                    # online_ids.append(tid)
                    # results.append(
                    #     f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
                    # )
                    if t.score >= 0.5:
                        track_res_id.append(tid)
                        trak_res_pos.append([tlwh[0] + tlwh[2] / 2, tlwh[1] + tlwh[3] / 2]) # 获取人体多目标跟踪结果
            # online_im = plot_tracking(
            #     frame, online_tlwhs, online_ids, frame_id = frame_id + 1, fps=30)
            if track_res_id == []:
                track_res_id = [-1]
                trak_res_pos = [[self.img_size[0] // 2, self.img_size[1] // 3]]  # 初始化窗口位置（中间偏上
        else:
            track_res_id = [-1]
            trak_res_pos = [[self.img_size[0] // 2, self.img_size[1] // 3]]  # 初始化窗口位置（中间偏上
        #     online_im = frame


        return np.array(track_res_id), np.array(trak_res_pos)

    def update_track(self, outputs):
        if outputs[0] is not None:
            self.tracker.update(outputs[0], self.frame_shape, self.test_size)

    def update_window(self):
        min_size_factor, max_size_factor = self.size_factor
        # self.window_center[:] = self.window_center[:] + self.ckpt * (self.direct_vec * 0.9 / self.ckpt).astype(np.int16)
        window_x = np.min((np.abs(self.window_center[0] - self.img_size[0] // 2), self.img_size[0] // 2))
        window_size_factor = max_size_factor - \
                             window_x * (max_size_factor - min_size_factor) / (self.img_size[0] / 3)
        if self.window_center[1] >= 0.5 * self.img_size[1]:  # 如果窗口落在视频下半部分。则扩大视频窗口
            window_size_factor = window_size_factor * (1.5 * self.window_center[1] + 0.25 * self.img_size[1]) \
                                 / self.img_size[1]
        self.window_size = (self.window_size_init * window_size_factor).astype(np.int16)