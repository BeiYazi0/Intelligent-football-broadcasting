import os
import cv2
import numpy as np
import torch
import time
import ffmpeg
import shutil
import torch.multiprocessing as mp
from os import path as osp
from loguru import logger

from yolox.utils import fuse_model, get_model_info

from get_map import map_create
from bytetrack_model import Predictor
from soccer_model import BallDetector
from bytetrack_process import TrackProcesser
from img_process import ImgProcesser
from helper import Timer #, remap, undistort

def write_results(filename, results):
    save_format = '{frame},{id},{x1},{y1},{w},{h},{s},-1,-1,-1\n'
    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids, scores in results:
            for tlwh, track_id, score in zip(tlwhs, track_ids, scores):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                line = save_format.format(frame=frame_id, id=track_id, x1=round(x1, 1), y1=round(y1, 1), w=round(w, 1),
                                          h=round(h, 1), s=round(score, 2))
                f.write(line)
    logger.info('save results to {}'.format(filename))


def Pic2Video(im_dir, video_dir):
    imglist = sorted(os.listdir(im_dir))  # 将排序后的路径返回到imglist列表中
    img = cv2.imread(os.path.join(im_dir, imglist[0]))  # 合并目录与文件名生成图片文件的路径,随便选一张图片路径来获取图像大小
    H, W, D = img.shape  # 获取视频高\宽\深度
    print('height:' + str(H) + '--' + 'width:' + str(W) + '--' + 'depth:' + str(D))
    fps = 30  # 帧率一般选择20-30
    img_size = (W, H)  # 图片尺寸宽x高,必须是原图片的size,否则合成失败
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    videoWriter = cv2.VideoWriter(video_dir, fourcc, fps, img_size)
    for image in imglist:
        img_name = os.path.join(im_dir, image)
        frame = cv2.imread(img_name)
        videoWriter.write(frame)
        # print('合成==>' + img_name)
    videoWriter.release()
    print('finish!')


def get_points(pts):
    ptsr = np.zeros((4, 2))
    ptsl = np.zeros((4, 2))

    w1 = pts[1][0] - pts[0][0]
    w2 = pts[3][0] - pts[2][0]
    h1 = pts[2][1] - pts[0][1]
    h2 = pts[3][1] - pts[1][1]

    ptsr[0][0] = pts[0][0]
    ptsr[0][1] = pts[0][1] + h1 * 0.09
    ptsr[1][0] = pts[1][0] - w1 * 0.12
    ptsr[1][1] = pts[1][1] + h2 * 0.05
    ptsr[2][0] = pts[2][0] + w2 * 0.08
    ptsr[2][1] = pts[2][1] + h1 * 0.07
    ptsr[3][0] = pts[3][0] - w2 * 0.01
    ptsr[3][1] = pts[3][1] - h2 * 0.11

    ptsl[0][0] = pts[0][0] + w1 * 0.2
    ptsl[0][1] = pts[0][1]
    ptsl[1][0] = pts[1][0] + w1 * 0.1
    ptsl[1][1] = pts[1][1] + h2 * 0.14
    ptsl[2][0] = pts[2][0] + w2 * 0.025
    ptsl[2][1] = pts[2][1] - h1 * 0.24
    ptsl[3][0] = pts[3][0] - w2 * 0.1
    ptsl[3][1] = pts[3][1] + h2 * 0.14

    return ptsl, ptsr


class StreamProcessor(object):
    def __init__(self, args, exp):
        try:
            self.app = args.source.split("/")[-2]
        except IndexError:
            self.app = 'localtest'
        self.stream_source = args.source
        self.stream_destination = args.dest
        self.ckpt = 60  # 利用未来ckpt帧的信息
        self.fps = args.fps
        self.interval = args.interval

        self.vis_folder = None
        self.exp = exp
        self.args = args
        self._params_process()

        setting_parameters = np.loadtxt(args.setting)
        # 矫正参数
        self.width = None
        self.height = None
        self.undistort_parameters = setting_parameters[0][:-2].tolist()
        self.img_size = setting_parameters[0][-2:].astype(np.int32).tolist()

        # window
        self.target_shape = setting_parameters[1][:2].astype(np.int32).tolist()  # (960, 540)
        self.window_size_init = setting_parameters[1][:2].astype(np.int32)  # 设置原始窗口尺寸(即画面在正中央时)
        # 窗口最大为window_size_init的1倍(画面中间)
        # 窗口最大为window_size_init的0.6倍(窗口位于最左，最右时)
        self.size_factor = setting_parameters[1][2:-1].tolist()

        # 透视变换参数
        self.perspective_flag = (setting_parameters[1][-1] == 1)
        self.original_points = None
        self.left_final_points = None
        self.right_final_points = None

        # 处理器
        self.soccer_model = args.name

        # save
        self.vid_writer = None
        self.timestamp = None

        ## 共享内存
        # 事件
        self.start_ev = None
        self.stop_ev = None
        self.model_prepare_ev = None
        self.direct_vec_ev = None

        # 原始帧
        self.stream_data = None
        self.stream_data_size = None

        # 矫正后的帧
        self.undistort_data = None
        self.undistort_data_size = None

        # 矫正结果的拷贝
        self.img_duplicate_data = None
        self.img_duplicate_data_size = None

        self.ball_data = None
        self.direct_vec_data = None
        self.window_center_data = None

    def _params_process(self):
        args, exp = self.args, self.exp
        if not args.experiment_name:
            args.experiment_name = exp.exp_name

        output_dir = osp.join(exp.output_dir, args.experiment_name)
        os.makedirs(output_dir, exist_ok=True)
        self.vis_folder = osp.join(output_dir, "track_vis")

        if args.save_result:
            os.makedirs(self.vis_folder, exist_ok=True)

        if args.trt:
            args.device = "gpu"
        args.device = torch.device("cuda" if args.device == "gpu" else "cpu")

        logger.info("Args: {}".format(args))

        if args.conf is not None:
            exp.test_conf = args.conf
        if args.nms is not None:
            exp.nmsthre = args.nms
        if args.tsize is not None:
            exp.test_size = (args.tsize, args.tsize)

    def set_points(self, original_points):
        self.original_points = original_points
        self.left_final_points, self.right_final_points = get_points(original_points)

    def set_window(self, window_size_init, size_factor):
        self.window_size_init = np.array(window_size_init)  # 设置原始窗口尺寸(即画面在正中央时)
        # 窗口最大为window_size_init的1倍(画面中间)
        # 窗口最大为window_size_init的0.6倍(窗口位于最左，最右时)
        self.size_factor = size_factor

    def set_perspective(self, flag):
        self.perspective_flag = flag

    def _prepare(self):
        cap = cv2.VideoCapture(self.stream_source)
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # float
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # float
        fps = cap.get(cv2.CAP_PROP_FPS)
        self.fps = min(self.fps, fps)
        cap.release()

        current_time = time.localtime()
        self.timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
        if self.args.save_result:
            save_folder = osp.join(self.vis_folder, self.timestamp)
            os.makedirs(save_folder, exist_ok=True)
            if self.args.demo == "video":
                save_path = osp.join(save_folder, self.app)
            else:
                save_path = osp.join(save_folder, "camera.mp4")
            logger.info(f"video save_path is {save_path}")
            self.vid_writer = cv2.VideoWriter(
                save_path, cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (self.width, self.height)
            )

    def _cerate_share_memoery_file(self):
        save_path = f'~/.stream_data/{self.app}'
        if not osp.exists(save_path):
            os.makedirs(save_path)

        self.stream_data_size = (self.height, self.width, 3)
        self.stream_data = osp.join(save_path, f"stream_data.bin")
        data = np.memmap(self.stream_data, dtype='uint8', mode='w+', shape=self.stream_data_size)
        data[:] = np.zeros(self.stream_data_size, dtype='uint8')
        del data

        self.undistort_data_size = (self.img_size[1], self.img_size[0], 3)
        self.undistort_data = osp.join(save_path, f"undistort_data.bin")
        data = np.memmap(self.undistort_data, dtype='uint8', mode='w+', shape=self.undistort_data_size)
        data[:] = np.zeros(self.undistort_data_size, dtype='uint8')
        del data

        self.ball_data = osp.join(save_path, "ball_data.bin")
        data = np.memmap(self.ball_data, dtype='float32', mode='w+', shape=(2,))
        data[:] = np.zeros(2, dtype='float32')
        del data

        self.direct_vec_data = osp.join(save_path, "direct_vec_data.bin")
        data = np.memmap(self.direct_vec_data, dtype='float32', mode='w+', shape=(2,))
        data[:] = np.zeros(2, dtype='float32')
        del data

        self.window_center_data = osp.join(save_path, "window_center_data.bin")
        data = np.memmap(self.window_center_data, dtype='float32', mode='w+', shape=(2,))
        data[:] = np.zeros(2, dtype='float32')
        del data

        self.img_duplicate_data_size = (self.ckpt, self.img_size[1], self.img_size[0], 3)
        self.img_duplicate_data = osp.join(save_path, "img_duplicate_data.bin")
        data = np.memmap(self.img_duplicate_data, dtype='uint8', mode='w+', shape=self.img_duplicate_data_size)
        data[:] = np.zeros(self.img_duplicate_data_size, dtype='uint8')
        del data

    def _cap_task(self):
        cap = cv2.VideoCapture(self.stream_source)
        if not cap.isOpened():
            return

        stream_data_put = np.memmap(self.stream_data, dtype='uint8', mode='w+', shape=self.stream_data_size)

        timer = Timer(self.fps)

        if not self.model_prepare_ev.is_set():
            self.model_prepare_ev.wait()
        self.start_ev.set()

        while True:
            timer.start()
            ret, frame = cap.read()
            if ret:
                stream_data_put[:] = frame
                timer.suspend()
            else:
                self.stop_ev.set()
                break
        print("cap_task" + timer.info())

    def _undistort_task(self):
        col, row = map_create(self.width, self.height, *self.undistort_parameters, *self.img_size)

        stream_data_get = np.memmap(self.stream_data, dtype='uint8', mode='r+', shape=self.stream_data_size)
        undistort_data_put = np.memmap(self.undistort_data, dtype='uint8', mode='w+', shape=self.undistort_data_size)

        timer = Timer(self.fps)
        # (A, B, C, D), (I, J) = remap(row, col, self.height, self.width)

        if not self.start_ev.is_set():
            self.start_ev.wait()
        time.sleep(0.01)

        while not self.stop_ev.is_set():
            timer.start()
            # undistort_data_put[:] = cv2.remap(stream_data_get[:]/255, col, row, cv2.INTER_LINEAR)*255

            # img = stream_data_get[:]/255
            # undistort_data_put[:] = (A * img[I, J] + B * img[I, J+1] + C * img[I+1, J] + D * img[I+1, J+1])*255

            cv2.remap(stream_data_get, col, row, cv2.INTER_LINEAR, undistort_data_put)
            # undistort(undistort_data_put, stream_data_get, A, B, C, D, I, J)
            timer.suspend()
        print("undistort_task" + timer.info())

    def _ball_task(self):
        torch.cuda.set_device(1)
        torch.cuda.is_available()
        soccer_detector = BallDetector(self.soccer_model, self.original_points,
                                       [self.img_size[0] // 2, self.img_size[1] // 3])  # 足球检测

        undistort_data_get = np.memmap(self.undistort_data, dtype='uint8', mode='r+', shape=self.undistort_data_size)
        ball_data_put = np.memmap(self.ball_data, dtype='float32', mode='w+', shape=(2,))

        timer = Timer(self.fps)
        frame_id = 0

        if not self.start_ev.is_set():
            self.start_ev.wait()
        time.sleep(1)

        while not self.stop_ev.is_set():
            timer.start()
            ball_data_put[:] = soccer_detector.detect(undistort_data_get[:])
            timer.suspend()
            frame_id += 1
            if (frame_id + 1) % self.ckpt == 0:
                soccer_detector.reset_ball()
        print("ball_task" + timer.info())

    def _detection_task(self):
        torch.cuda.set_device(0)
        torch.cuda.is_available()
        model = self.exp.get_model().to(self.args.device)
        # logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))
        model.eval()

        if not self.args.trt:
            if self.args.ckpt is None:
                output_dir = osp.join(self.exp.output_dir, self.args.experiment_name)
                ckpt_file = osp.join(output_dir, "best_ckpt.pth.tar")
            else:
                ckpt_file = self.args.ckpt
            # logger.info("loading checkpoint")
            ckpt = torch.load(ckpt_file, map_location="cpu")
            # load the model state dict
            model.load_state_dict(ckpt["model"])
            # logger.info("loaded checkpoint done.")

        if self.args.fuse:
            logger.info("\tFusing model...")
            model = fuse_model(model)

        if self.args.fp16:
            model = model.half()  # to FP16

        if self.args.trt:
            assert not self.args.fuse, "TensorRT model is not support model fusing!"
            trt_file = osp.join(output_dir, "model_trt.pth")
            assert osp.exists(
                trt_file
            ), "TensorRT model is not found!\n Run python3 tools/trt.py first!"
            model.head.decode_in_inference = False
            decoder = model.head.decode_outputs
            logger.info("Using TensorRT to inference")
        else:
            trt_file = None
            decoder = None

        interval = self.interval
        timer = Timer(self.fps, interval)
        ckpt = self.ckpt

        predictor = Predictor(model, self.exp, trt_file, decoder, self.args.device, self.args.fp16)
        byte_processer = TrackProcesser(self.args, self.exp.test_size, ckpt,
                                        self.original_points, self.window_center_data)  # 每 ckpt 个帧的处理器

        # 开头帧的操作
        byte_processer.set_window(self.img_size, self.window_size_init, self.size_factor)

        ball_data_get = np.memmap(self.ball_data, dtype='float32', mode='r+', shape=(2,))
        undistort_data_get = np.memmap(self.undistort_data, dtype='uint8', mode='r+', shape=self.undistort_data_size)
        direct_vec_data_put = np.memmap(self.direct_vec_data, dtype='float32', mode='w+', shape=(2,))

        self.model_prepare_ev.set()
        if not self.start_ev.is_set():
            self.start_ev.wait()
        time.sleep(1)

        frame_id = 0
        while not self.stop_ev.is_set():
            # 获取人追踪结果
            timer.start()
            if frame_id % ckpt == 0:
                outputs = predictor.inference(undistort_data_get[:])
                byte_processer.update_res_ckpt(outputs)  # ckpt 个帧的开头帧
            elif (frame_id + interval) % ckpt == 0:
                outputs = predictor.inference(undistort_data_get[:])
                byte_processer.update_res(outputs)  # ckpt 个帧的结尾帧

                point_ball = ball_data_get[:].copy()

                # 将目标点(final_point)与当前点(window_center)作差，转化成一个移动向量
                direct_vec_data_put[:] = byte_processer.get_direct_vec(point_ball, ckpt)
                # print("predictor_task: %d" % (frame_id + interval))
                # print("%d predictor_task" % (frame_id + 1))
                # 更新window
                byte_processer.update_window()
            else:  # elif frame_id % interval == 0:
                outputs = predictor.inference(undistort_data_get[:])
                # byte_processer.update_track(outputs)
                byte_processer.update_res(outputs)
                # 将目标点(final_point)与当前点(window_center)作差，转化成一个移动向量
                direct_vec_data_put[:] = byte_processer.get_direct_vec(ball_data_get[:], (frame_id + interval) % ckpt)
            timer.suspend()
            frame_id += interval
        print("predictor_task" + timer.info())

    def _copy_task(self):
        undistort_data_get = np.memmap(self.undistort_data, dtype='uint8', mode='r+', shape=self.undistort_data_size)
        img_duplicate_data_put = np.memmap(self.img_duplicate_data, dtype='uint8', mode='w+',
                                           shape=self.img_duplicate_data_size)

        frame_id = 0
        ckpt = self.ckpt
        buffer_size = ckpt // 2
        img_buffer = [np.zeros(self.undistort_data_size, dtype='uint8') for _ in range(buffer_size)]
        timer = Timer(self.fps)

        if not self.start_ev.is_set():
            self.start_ev.wait()
        time.sleep(1)

        while not self.stop_ev.is_set():
            idx = frame_id % ckpt

            timer.start()
            if idx < buffer_size:
                np.copyto(img_buffer[idx], undistort_data_get)
                # print("copy: undistort_data_get -> img_buffer[%d]" % frame_id)
            else:
                # print("copy %d" % frame_id)
                np.copyto(img_duplicate_data_put[idx - buffer_size], img_buffer[idx - buffer_size])
                # np.copyto(img_duplicate_data_put[ckpt - idx - 1], img_buffer[ckpt - idx - 1])
                # print("copy: img_buffer[%d] -> img_duplicate_data_put[%d]" % (idx - buffer_size, frame_id - buffer_size))
                np.copyto(img_duplicate_data_put[idx], undistort_data_get)
                # print("copy: undistort_data_get -> img_duplicate_data_put[%d]" % frame_id)

            if (idx + 1) == ckpt:
                self.direct_vec_ev.set()
                # print("%d copy_task" % (frame_id + 1))
            timer.suspend()
            # if (frame_id + 1) % ckpt == 0:
            #     print("buffer copy：%.4f" % (time_use / buffer_size))

            frame_id += 1
            # if frame_id >= 6550:
            #     break
        self.direct_vec_ev.set()
        print("copy_task" + timer.info())

    def _img_task(self):
        ckpt = self.ckpt
        save_img_path = 'save_image' + self.app # 裁剪图片保存路径
        if not os.path.exists(save_img_path):
            os.makedirs(save_img_path)

        direct_vec_data_get = np.memmap(self.direct_vec_data, dtype='float32', mode='r+', shape=(2,))

        img_processer = ImgProcesser(ckpt, self.size_factor, self.target_shape,  # 图像处理器
                                     self.original_points, self.left_final_points, self.right_final_points, self.window_center_data)

        # 开头帧的操作
        img_processer.set_window_init(self.img_size, self.window_size_init)
        img_processer.set_share_img(self.img_duplicate_data, self.img_duplicate_data_size)
        img_processer.set_save_dir(save_img_path)
        img_processer.set_pipe(self.stream_destination, self.fps)
        img_processer.set_push_method(self.perspective_flag)

        if not self.start_ev.is_set():
            self.start_ev.wait()
        time.sleep(1)

        # frame_id = ckpt

        while True:
            if not self.direct_vec_ev.is_set():
                self.direct_vec_ev.wait()
            self.direct_vec_ev.clear()
            # time.sleep(time_expect)
            # print("%d img_task" % frame_id)
            # start_time = time.time()
            if self.stop_ev.is_set():
                img_processer.close()
                break
            img_processer.img_push(direct_vec_data_get[:])
            # end_time = time.time()
            # sleep(time_expect - end_time + start_time)
            # print("process_img：%.4f" % (end_time -start_time))
            # frame_id += ckpt

    def _mutiply_task_start(self):
        mp.set_start_method(method='spawn')  #  init
        self._cerate_share_memoery_file()
        self.start_ev = mp.Event()
        self.stop_ev = mp.Event()
        self.model_prepare_ev = mp.Event()
        self.direct_vec_ev = mp.Event()
        processes = [mp.Process(target=self._undistort_task),
                     mp.Process(target=self._ball_task),
                     mp.Process(target=self._detection_task),
                     mp.Process(target=self._copy_task),
                     mp.Process(target=self._img_task)]

        for process in processes:
            process.daemon = True
            process.start()

        self._cap_task()

        time.sleep(3)

        # 保存的视频为avi格式，转化为mp4
        # vid_name = self.app
        # save_img_path = 'save_image' + vid_name
        # Pic2Video(save_img_path, vid_name + '.avi')
        # input_file = vid_name + '.avi'
        # output_file = vid_name + '.mp4'
        # input_stream = ffmpeg.input(input_file)
        # output_stream = ffmpeg.output(input_stream, output_file)
        # ffmpeg.run(output_stream)
        # os.remove(input_file)
        # shutil.rmtree(save_img_path)

        # if self.args.save_result:
        #     res_file = osp.join(self.vis_folder, f"{self.timestamp}.txt")
        #     with open(res_file, 'w') as f:
        #         f.writelines(results)
        #     logger.info(f"save results to {res_file}")

    def run(self):
        self._prepare()
        self._mutiply_task_start()
        # if self.args.demo == "image":
        #     pass #image_demo()
        # elif self.args.demo == "video" or self.args.demo == "webcam":
        #     self.mutiply_task_start()
