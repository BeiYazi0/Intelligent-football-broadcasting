import numpy as np
import cv2
import tkinter.filedialog
import tkinter.messagebox
import re

filenames='right'
def get_files():
    global filenames
    filenames = tkinter.filedialog.askopenfilenames(title="选择图片", filetypes=[('图片', 'jpg'), ('图片', 'png')])
    CN_Pattern = re.compile(u'[\u4E00-\u9FBF]+')

    if filenames:
        CN_Match = CN_Pattern.search(str(filenames))
        if CN_Match:
            filenames=None
            tkinter.messagebox.showinfo('提示','文件路径或文件名不能含有中文,请修改!')
            return

def mouse(event, x, y, flags, param):
    image = param[0]
    pts = param[1]

    if event == cv2.EVENT_LBUTTONDOWN:
        pts.append([x, y])
        xy = "%d,%d" % (x, y)
        cv2.circle(image, (x, y), 4, (0, 255, 255), thickness = -1)
        cv2.putText(image, xy, (x, y), cv2.FONT_HERSHEY_PLAIN,
                    1.0, (0, 255, 255), thickness = 2)
        cv2.imshow("image", image)

def auto_perspective():
    if filenames:
        for filename in filenames:
            if filename:
                print("file:", filename)
                image = cv2.imread(filename)
    # 原图中卡片的四个角点
    cv2.namedWindow("image", cv2.WINDOW_NORMAL)

    cv2.imshow("image", image)
    pts = []
    cv2.setMouseCallback("image", mouse, param=(image, pts))

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    assert len(pts) == 4, "每个只允许四个点"

    ptsr = np.zeros((4,2))
    ptsl = np.zeros((4,2))

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

    print("pts:", pts)
    pts = np.float32(pts[:4])
    print("ptsr:", ptsr)
    ptsr = np.float32(ptsr[:4])
    print("ptsl:", ptsl)
    ptsl = np.float32(ptsl[:4])

    # 生成透视变换矩阵
    M1 = cv2.getPerspectiveTransform(pts, ptsr)
    M2 = cv2.getPerspectiveTransform(pts, ptsl)
    # 进行透视变换
    dstr = cv2.warpPerspective(image, M1, (image.shape[1], image.shape[0]))
    cv2.imwrite('dstr.jpg', dstr)
    dstl = cv2.warpPerspective(image, M2, (image.shape[1], image.shape[0]))
    cv2.imwrite('dstl.jpg', dstl)

    return dstr, dstl

root = tkinter.Tk()
root.title('批量处理')
button = tkinter.Button(root, text="上传图片", command=get_files,width=20,height=2)
button.grid(row=0, column=0, padx=80, pady=20)
button1 = tkinter.Button(root, text="透视变换", command=auto_perspective,width=20,height=2)
button1.grid(row=5, column=0, padx=80, pady=20)

root.geometry('300x200+600+300')
root.mainloop()