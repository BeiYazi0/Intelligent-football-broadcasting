import time
import cv2
import os
import numpy as np

from get_pipe import FfmPipe


class ImgProcesser(object):
    def __init__(self, ckpt, size_factor, target_shape, original_points, left_final_points, right_final_points, window_center_data):
        self.frame_id = 0
        self.img_list = None
        self.save_path = None
        self.pipe = None

        self.ckpt = ckpt
        self.size_factor = size_factor

        # 透视变换参数
        self.original_points = original_points  # 球场四个角点
        self.left_final_points = left_final_points
        self.right_final_points = right_final_points

        self.img_size = None
        self.target_shape = target_shape
        self.window_size_init = None
        self.window_center = np.memmap(window_center_data, dtype='float32', mode='w+', shape=(2,))

        self.img_push = None

    def set_window_init(self, img_size, window_size_init):
        self.img_size = img_size
        self.window_center[:] = np.array([img_size[0] // 2, img_size[1] // 3])  # 初始化窗口位置（中间偏上
        self.window_size_init = window_size_init

    def set_share_img(self, img_duplicate_data, img_duplicate_data_size):
        self.img_list = np.memmap(img_duplicate_data, dtype='uint8', mode='r+', shape=img_duplicate_data_size)

    def set_save_dir(self, save_img_path):
        self.save_path = save_img_path

    def set_pipe(self, rtmp, fps):
        self.pipe = FfmPipe(rtmp, self.target_shape, fps)  # 子程序推流管道

    def set_push_method(self, perspective_flag):
        if perspective_flag:
            self.img_push = lambda direct_vec: self._process_img_perspective(direct_vec)
        else:
            self.img_push = lambda direct_vec: self._process_img(direct_vec)

    def _process_img(self, direct_vec):
        img_size = self.img_size
        min_size_factor, max_size_factor = self.size_factor
        print(direct_vec)
        # 将direct_vec按ckpt帧分配给每帧去移动
        direct_per_frame = np.fix(direct_vec * 0.9 / self.ckpt)
        direct = np.tile(direct_per_frame, (self.ckpt, 1))  # 复制 ckpt 次
        window_centers = np.cumsum(direct, axis=0) + self.window_center[:]
        window_centers_x = window_centers[:, 0]
        window_centers_y = window_centers[:, 1]

        img_size_x_half = img_size[0] // 2
        img_size_y_half = img_size[1] // 2
        # 根据 center 的位置，重新调整window_size，越靠近两边越小 (max_size_factor - x)
        window_x = np.minimum(np.abs(window_centers_x - img_size_x_half), img_size_x_half)
        window_size_factor = max_size_factor - \
                             window_x * (max_size_factor - min_size_factor) / (img_size[0] / 3)
        window_check = window_centers_y < img_size_y_half
        window_size_factor = window_size_factor * window_check + \
                             window_size_factor * (1.5 * window_centers_y + 0.25 * img_size[1]) / img_size[
                                 1] * ~window_check
        window_sizes = (window_size_factor.reshape(-1, 1) * self.window_size_init.reshape(1, -1)).astype(np.int16)

        # 修正防止越界
        limit_x = int(img_size[0] / 20)
        window_sizes[:, 0][window_sizes[:, 0] >= img_size_x_half - limit_x] = img_size_x_half - limit_x
        window_sizes[:, 1][window_sizes[:, 1] >= img_size_y_half - 100] = img_size_y_half - 100

        left_x = window_sizes[:, 0] / 2 + limit_x
        right_x = img_size[0] - window_sizes[:, 0] / 2 - limit_x
        left_mask = window_centers_x <= left_x
        right_mask = window_centers_x >= right_x
        window_centers_x = (left_mask * left_x + right_mask * right_x
                            + ~(left_mask | right_mask) * window_centers_x).astype(np.int16)
        left_y = window_sizes[:, 1] / 2 + 100
        right_y = img_size[1] - window_sizes[:, 1] / 2
        left_mask = window_centers_y <= left_y
        right_mask = window_centers_y >= right_y
        window_centers_y = (left_mask * left_y + right_mask * right_y
                            + ~(left_mask | right_mask) * window_centers_y).astype(np.int16)

        self.window_center[:] = window_centers[-1].copy()

        window_sizes //= 2
        for i in range(self.ckpt):
            try:
                # 根据window_center和window_size取画面
                img_crop = self.img_list[i][
                           window_centers_y[i] - window_sizes[i, 1]:window_centers_y[i] + window_sizes[i, 1],
                           window_centers_x[i] - window_sizes[i, 0]:window_centers_x[i] + window_sizes[i, 0]]

                # 统一到同一尺度
                resize_img = cv2.resize(img_crop, self.target_shape, interpolation=cv2.INTER_AREA)
            except Exception:
                print(window_centers_x[i], window_centers_y[i], window_sizes[i], img_crop.shape)

            # save_img_file = os.path.join(self.save_path,
            #                              str(self.frame_id + 1).zfill(5) + '.jpg')
            # print("process: %d" % self.frame_id)
            # cv2.imwrite(save_img_file, resize_img)
            self.pipe.write(resize_img)
            self.frame_id += 1


    def _process_img_perspective(self, direct_vec):
        # start_time = time.time()
        img_size = self.img_size
        min_size_factor, max_size_factor = self.size_factor
        # 将direct_vec按ckpt帧分配给每帧去移动
        direct_per_frame = np.fix(direct_vec * 0.9 / self.ckpt)
        direct = np.tile(direct_per_frame, (self.ckpt, 1)) # 复制 ckpt 次
        window_centers = np.cumsum(direct, axis = 0) + self.window_center[:]
        window_centers_x = window_centers[:, 0]
        window_centers_y = window_centers[:, 1]

        img_size_x_half = img_size[0] // 2
        img_size_y_half = img_size[1] // 2
        # 根据 center 的位置，重新调整window_size，越靠近两边越小 (max_size_factor - x)
        window_x = np.minimum(np.abs(window_centers_x - img_size_x_half),  img_size_x_half)
        window_size_factor = max_size_factor - \
                             window_x * (max_size_factor - min_size_factor) / (img_size[0] / 3)
        window_check = window_centers_y < img_size_y_half
        window_size_factor = window_size_factor * window_check + \
            window_size_factor * (1.5 * window_centers_y + 0.25 * img_size[1]) / img_size[1] * ~window_check
        window_sizes = (window_size_factor.reshape(-1, 1) * self.window_size_init.reshape(1, -1)).astype(np.int16)

        # 根据窗口移动的位置来生成新的四个点作透视变换
        factor = np.abs(window_centers_x - img_size_x_half).reshape(-1, 1, 1) / (img_size_x_half // 2)
        window_check = (window_centers_x < img_size_x_half).reshape(-1, 1, 1)
        diff_to_left = np.tile(np.expand_dims(self.left_final_points - self.original_points, axis=0),
                               (self.ckpt, 1, 1)) * factor
        diff_to_right = np.tile(np.expand_dims(self.right_final_points - self.original_points, axis=0),
                               (self.ckpt, 1, 1)) * factor
        points = window_check * np.float32(self.original_points + diff_to_left) + \
                 ~window_check * np.float32(self.original_points + diff_to_right)

        # 修正防止越界
        limit_x = int(img_size[0] / 20)
        left_x = window_sizes[:, 0] / 2 + limit_x
        right_x = img_size[0] - window_sizes[:, 0] / 2 - limit_x
        left_mask = window_centers_x <= left_x
        right_mask = window_centers_x >= right_x
        window_centers_x = (left_mask * left_x + right_mask * right_x
                            + ~(left_mask | right_mask) * window_centers_x).astype(np.int16)
        left_y = window_sizes[:, 1] / 2 + 100
        right_y = img_size[1] - window_sizes[:, 1] / 2
        left_mask = window_centers_y <= left_y
        right_mask = window_centers_y >= right_y
        window_centers_y = (left_mask * left_y + right_mask * right_y
                            + ~(left_mask | right_mask) * window_centers_y).astype(np.int16)

        self.window_center[:] = window_centers[-1].copy()

        window_sizes //= 2
        # end_time = time.time()
        # print("img准备所用时间：%.4f" % (end_time - start_time))
        # time_use = 0
        # time_use_crop = 0
        for i in range(self.ckpt):
            # print("process %d" % self.frame_id)
            # start_time = time.time()
            dst = cv2.warpPerspective(self.img_list[i][:],
                                      cv2.getPerspectiveTransform(self.original_points, points[i]), img_size)
            # end_time = time.time()
            # time_use += end_time - start_time

            # 根据window_center和window_size取画面
            # start_time = time.time()
            img_crop = dst[window_centers_y[i] - window_sizes[i, 1] :window_centers_y[i] + window_sizes[i, 1],
                       window_centers_x[i] - window_sizes[i, 0] :window_centers_x[i] + window_sizes[i, 0]]

            # 统一到同一尺度
            resize_img = cv2.resize(img_crop, self.target_shape, interpolation=cv2.INTER_AREA)
            # end_time = time.time()
            # time_use_crop += end_time - start_time

            save_img_file = os.path.join(self.save_path,
                                         str(self.frame_id + 1).zfill(5) + '.jpg')
            cv2.imwrite(save_img_file, resize_img)
            # self.pipe.write(resize_img)
            self.frame_id += 1
        # print("透视变换单帧所用时间：%.4f" % (time_use / self.ckpt))
        # print("crop+resizes单帧所用时间：%.4f" % (time_use_crop / self.ckpt))

    def close(self):
        if self.pipe is not None:
            self.pipe.close()