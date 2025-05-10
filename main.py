import argparse
import os
import numpy as np

from yolox.exp import get_exp

from stream_process import StreamProcessor

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]

# os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ['YOLO_VERBOSE'] = str(False)

rtmp_source = "rtmp://61.160.195.115/test/origin"

# 推流地址
rtmp = "rtmp://61.160.195.115/test/live"
# rtmp = "rtmp://47.115.221.91/live/livestream"
# rtmp = "rtmp://live-push.bilivideo.com/live-bvc/?streamname=live_345593374_62084381&key=e1f741310a75ea540b5ae9fa1553fa0d&schedule=rtmp&pflag=1"

original_points = np.float32([[966, 192], [1756, 174], [27, 756], [2700, 536]])  # 球场四个角点


def make_parser():
    parser = argparse.ArgumentParser("Stream Processor")

    # Stream app
    parser.add_argument("--source", default=rtmp_source, type=str, help="Stream source")
    parser.add_argument("--dest", default=rtmp, type=str, help="Stream dest")
    # 抽帧数
    parser.add_argument("--interval", default=5, type=int, help="interval")
    # 矫正参数和窗口以及透视变换配置
    parser.add_argument("--setting", default='setting.txt', type=str, help="setting parameters")

    parser.add_argument("-expn", "--experiment-name", type=str, default=None)

    parser.add_argument(
        "--save_result",
        action="store_true",
        help="whether to save the inference result of image/video",
    )

    # exp file
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="pls input your expriment description file",
    )
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt for eval")
    parser.add_argument("-n", "--name", type=str, default='yolov8/best.pt', help="model name")

    parser.add_argument(
        "--device",
        default="gpu",
        type=str,
        help="device to run our model, can either be cpu or gpu",
    )
    parser.add_argument("--conf", default=None, type=float, help="test conf")
    parser.add_argument("--nms", default=None, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument("--fps", default=30, type=int, help="frame rate (fps)")

    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision evaluating.",
    )
    parser.add_argument(
        "--fuse",
        dest="fuse",
        default=False,
        action="store_true",
        help="Fuse conv and bn for testing.",
    )
    parser.add_argument(
        "--trt",
        dest="trt",
        default=False,
        action="store_true",
        help="Using TensorRT model for testing.",
    )

    # tracking args
    parser.add_argument("--track_thresh", type=float, default=0.5, help="tracking confidence threshold")
    parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.8, help="matching threshold for tracking")
    parser.add_argument(
        "--aspect_ratio_thresh", type=float, default=1.6,
        help="threshold for filtering out boxes of which aspect ratio are above the given value."
    )
    parser.add_argument('--min_box_area', type=float, default=10, help='filter out tiny boxes')
    parser.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="test mot20.")
    return parser

if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)

    stream_process = StreamProcessor(args, exp)
    stream_process.set_points(original_points)
    stream_process.run()
