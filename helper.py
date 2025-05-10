import time
import numpy as np

# import numba
# from numba import jit

class Timer(object):
    def __init__(self, fps, interval=1):
        self.unit_time = interval / fps
        self.expect_time = 0
        self.time_sum = 0
        self.time_sum_true = 0

        self.start_time = 0
        self.cnt = 0

    def start(self):
        self.expect_time += self.unit_time
        self.cnt += 1
        self.start_time = time.time()

    def suspend(self):
        _time = time.time() - self.start_time
        self.time_sum_true += _time
        self.time_sum += _time
        if self.expect_time > self.time_sum:
            time.sleep(self.expect_time - self.time_sum)
            self.time_sum = self.expect_time

    def average(self):
        return self.time_sum / self.cnt

    def sum(self):
        return self.time_sum

    def surpass(self):
        return self.time_sum > self.expect_time

    def info(self):
        fps = self.cnt / self.time_sum
        fps_t = self.cnt / self.time_sum_true
        return "帧数: %d    约束总用时: %.2f seconds    约束fps: %.2f    真实总用时: %.2f seconds    真实fps: %.2f" % (
            self.cnt, self.time_sum, fps, self.time_sum_true, fps_t)


# @jit(nopython=True)
# def undistort(res, src, A, B, C, D, I, J):
#     m, n = A.shape[:2]
#     for i in range(m):
#         for j in range(n):
#             p = I[i, j]
#             q = J[i, j]
#             res[i, j] = A[i, j] * src[p, q] + B[i, j] * src[p, q+1] + C[i, j] * src[p+1, q] + D[i, j] * src[p+1, q+1]


def remap(X, Y, H, W):
    I = np.fix(X)         # 对应点在原图邻近点行索引
    J = np.fix(Y);        # 对应点在原图邻近点列索引

    U = X-I
    V = Y-J

    D = U*V
    A = 1-U-V+D
    B = V-D
    C = U-D

    I = I.astype(np.int32)
    J = J.astype(np.int32)

    #边界处理
    I[I > H-2] -= H
    J[J > W-2] -= W

    # img = img.astype(np.float64)   #将img矩阵转化为双精度类型
    # Mnew = (np.expand_dims(A, axis=2) * img[I,J] +
    #         np.expand_dims(B, axis=2) * img[I,J+1] +
    #         np.expand_dims(C, axis=2) * img[I+1,J] +
    #         np.expand_dims(D, axis=2) * img[I+1,J+1]).astype(np.uint8)
    return (np.expand_dims(A, axis=2), np.expand_dims(B, axis=2), np.expand_dims(C, axis=2), np.expand_dims(D, axis=2)), (I, J)
