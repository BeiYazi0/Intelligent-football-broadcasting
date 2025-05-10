import subprocess
import os
import psutil


class FfmPipe():
    def __init__(self, rtmp, imgsize, fps):
        sizeStr = str(imgsize[0]) + 'x' + str(imgsize[1])
        self.my_env = os.environ.copy()
        self.my_env["CUDA_VISIBLE_DEVICES"] = "0"
        self.command = ['ffmpeg',
                   #'-hwaccel', 'cuda',
                   '-y', '-an',
                   '-f', 'rawvideo',
                   '-vcodec', 'rawvideo',
                   '-pix_fmt', 'bgr24',
                   '-s', sizeStr,
                   '-r', '%s' % fps,
                   '-i', '-',
                   '-c:v', 'libx264', #'h264_nvenc', #'libx264',
                   '-pix_fmt', 'yuv420p',
                   '-preset', 'ultrafast',#'medium', #'ultrafast',
                   '-f', 'flv',
                   rtmp]
        self.pipe = subprocess.Popen(self.command, shell=False, stdin=subprocess.PIPE, env = self.my_env)
        self.proc = psutil.Process(self.pipe.pid)

    def _check(self):
        # poll()返回该子进程的状态，0正常结束，1sleep，2子进程不存在，-15 kill，None正在运行
        if self.pipe.poll() is not None:
            # time.sleep(3)
            print("pipe sleep")
            # print("the popen of ffmpeg not run, restart this:%s" % self.name)
            # self.pipe = subprocess.Popen(self.command, stdin=subprocess.PIPE, env = self.my_env)
            self.pipe.kill()
            self.pipe.wait()
            self.pipe = subprocess.Popen(self.command, stdin=subprocess.PIPE, env=self.my_env)
            self.proc = psutil.Process(self.pipe.pid)

    def write(self, data):
        try:
            self.pipe.stdin.write(data.tobytes())
        except BrokenPipeError:
            self._check()
            self.pipe.stdin.write(data.tobytes())

    def close(self):
        self.pipe.kill()

    def suspend(self):
        self.proc.suspend()

    def resume(self):
        self.proc.resume()
