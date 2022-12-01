import math
import os
import shutil

import cv2
import numpy as np
from matplotlib import pyplot as plt
import argparse

'''
现有功能概述：
依据约定差异区域阈值，识别不同大小、形状、颜色、被遮盖区域，
降噪、最小半径、参数可调、自动调节颜色差异精确度、多线程、极速绘图
输出区域json与结果图片，方便人工复检

优化点：
利用pysider形成exe程序，简化阈值调节方式；
图片轮播、热键复检，人工点选圈定，提高大量数据源识别效率
，识别大量数据实时显示识别图片提前人工复检；

'''

def matchAB(gray_threlod, fileA, fileB):
    # 读取图像数据
    imgA = cv2.imread(fileA)
    imgB = cv2.imread(fileB)
    # 不适用灰色对比，防止颜色差异而形状无差异，颜色差异需由灰度阈值调节以扩大差异区域
    resImg = cv2.absdiff(imgA, imgB)
    gray_resImg = cv2.cvtColor(resImg, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("resImg ", gray_resImg)
    # cv2.waitKey(0)
    # 获取图片A的大小
    whole_height, whole_width = gray_resImg.shape
    # # 用四边形圈出不同部分
    _, result_window_bin = cv2.threshold(gray_resImg, gray_threlod, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(result_window_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # "."：点# ","：像素点# "o"：圆形
    # "v"：朝下三角形# "^"：朝上三角形# "<"：朝左三角形# ">"：朝右三角形
    # "s"：正方形# "p"：五边星# "*"：星型# "h"：1
    # 号六角形# "H"：2# 号六角形# "+"：+号标记# "x"：x号标记# "D"：菱形
    # "d"：小型菱形# "|"：垂直线形# "_"：水平线形
    marks = ["o", "v", "+", "^", "x", "D", "|", "_", "s", "p", "H", "h", ".", ]
    target_list = []
    for contour in contours:
        min = np.nanmin(contour, 0)
        max = np.nanmax(contour, 0)
        loc1 = (min[0][0], min[0][1])
        loc2 = (max[0][0], max[0][1])
        width = abs(loc1[0] - loc2[0])
        height = abs(loc1[1] - loc2[1])
        area = width * height
        # 面积降噪
        if area > 6:
            center_point = (int((loc2[0] + loc1[0]) / 2), int((loc2[1] + loc1[1]) / 2))
            radius = math.sqrt(width * width + height * height)
            radius = int(radius / 2)
            # 四角差异点过滤
            if center_point[0] < 15:
                if center_point[1] < 15 or center_point[1] > whole_height - 15:
                    continue
            elif center_point[0] > whole_width - 15:
                if center_point[1] < 15 or center_point[1] > whole_height - 15:
                    continue
            # 半径降噪
            if radius < radius_min:
                continue
            #     设置差异区域最小值
            if radius < merge_bigger_radius and (not debug):
                radius = merge_bigger_radius
            # print(center_point, radius)
            # plt.scatter(loc1[0], loc1[1], marker=marks[index])
            # plt.scatter(loc2[0], loc2[1], marker=marks[index])
            # index = index + 1
            target_bean = target_Circle(center_point[0], center_point[1], radius)
            target_list.append(target_bean)
    #         相近差异区域融合，以满足实际差异判定，防止被其他物体分割的整个差异区域拆分
    target_list = merge_circle_list(target_list)
    # 依据约定阈值，判断识别结果调节灰度阈值
    if len(target_list) > target_max_len:
        higher_gray_threlod_png_list.append((fileA, fileB))
    elif len(target_list) < target_min_len:
        lower_gray_threlod_png_list.append((fileA, fileB))
    elif not debug:
        # 输出结果
        print_png_list.append((fileA, fileB, target_list))


def print_png(fileA, fileB, target_list):
    # 读取图像数据
    imgA = cv2.imread(fileA)
    imgB = cv2.imread(fileB)
    imgC = imgA.copy()
    #         输出图片，人工复检
    if plt_debug and len(target_list) >= target_min_len and len(target_list) <= target_max_len:
        for target_bean in target_list:
            center_point = (target_bean.m_center_x, target_bean.m_center_y)
            radius = target_bean.m_radius
            if radius < final_bigger_radius:
                radius = final_bigger_radius
            if debug:
                print(center_point, radius)
            # cv2.circle(img,center,radius,color[,thickness[,lineType[,shift]]])
            cv2.circle(imgC, center_point, radius, circle_color, 2)
        plt.subplot(2, 2, 1), plt.imshow(cv2.cvtColor(imgA, cv2.COLOR_BGR2RGB)), plt.title(
            str(os.path.basename(fileA))), plt.xticks(
            []), plt.yticks(
            [])
        plt.subplot(2, 2, 2), plt.imshow(cv2.cvtColor(imgB, cv2.COLOR_BGR2RGB)), plt.title(
            str(os.path.basename(fileB))), plt.xticks(
            []), plt.yticks(
            [])
        plt.subplot(2, 2, 3), plt.imshow(cv2.cvtColor(imgC, cv2.COLOR_BGR2RGB)), plt.title('Answer'), plt.xticks(
            []), plt.yticks([])
        # plt.imshow(cv2.cvtColor(imgC, cv2.COLOR_BGR2RGB))
        plt.savefig(output_png_path + str(os.path.basename(fileA)))
        # 清空缓存，加快速度
        plt.clf()
        # plt.show()
    # if os.path.exists(fileA):
    #     os.remove(fileA)
    # if os.path.exists(fileB):
    #     os.remove(fileB)


def merge_circle_list(target_bean_list):
    # 合并中间集
    target_bean_list_new = []
    is_loop = False
    # 舍弃集
    skip_index_list = []
    for i in range(0, len(target_bean_list)):
        if i in skip_index_list:
            continue
        target_bean_1 = target_bean_list[i]
        is_pass = True
        for j in range(i + 1, len(target_bean_list)):
            if j in skip_index_list:
                continue
            target_bean_2 = target_bean_list[j]
            # 判断两区域是否可合并并返回区域集
            target_bean_list_temp = merge_circle(target_bean_1, target_bean_2)
            # 合并则将对比区域放入舍弃集
            if len(target_bean_list_temp) == 1:
                skip_index_list.append(j)
                target_bean_list_new.append(target_bean_list_temp[0])
                is_pass = False
                is_loop = True
                break
        #         当前区域没有可合并的对比区域
        if is_pass:
            target_bean_list_new.append(target_bean_1)
    #     存在合并区域，循环合并
    if is_loop:
        return merge_circle_list(target_bean_list_new)
    return target_bean_list


def merge_circle(target_bean_1, target_bean_2):
    width = abs(target_bean_1.m_center_x - target_bean_2.m_center_x)
    height = abs(target_bean_1.m_center_y - target_bean_2.m_center_y)
    distance = math.sqrt(width * width + height * height + 3)
    # 两区域宽度和加上距离阈值，兼容遮盖物体宽度
    th_distance = ((target_bean_1.m_radius + target_bean_2.m_radius) + th_distance_between)
    # print(distance, th_distance)
    if distance < th_distance:
        center_x = int((target_bean_1.m_center_x + target_bean_2.m_center_x) / 2)
        center_y = int((target_bean_1.m_center_y + target_bean_2.m_center_y) / 2)
        radius = int((distance + target_bean_1.m_radius + target_bean_2.m_radius) / 2)
        target_bean = target_Circle(center_x, center_y, radius)
        return [target_bean, ]
    else:
        return [target_bean_1, target_bean_2]


# 区域变量
class target_Circle:
    def __init__(self):
        super().__init__()

    def __init__(self, center_x, center_y, radius):
        super().__init__()
        self.m_center_x = center_x
        self.m_center_y = center_y
        self.m_radius = radius


# 输出文件
def add_location_txt(filenameA, filenameB, png_width, png_height, target_list, output_path_json):
    result = "{"
    result += "\"filenameA\":\"" + str(os.path.basename(filenameA)) + "\""
    result += ",\"filenameB\":\"" + str(os.path.basename(filenameB)) + "\""
    result += ",\"png_width\":" + str(png_height)
    result += ",\"png_height\":" + str(png_width)
    # 转化区域变量
    result += ",\"target_location\":[" + get_location_str(target_list) + "]"
    result += "},"
    with open(output_path_json, mode='a+', encoding='utf-8') as f:
        f.writelines(result + "\n")
        f.flush()


def get_location_str(target_list):
    result = ""
    for target_Circle in target_list:
        result += "{\"center_x\":" + str(target_Circle.m_center_x)
        result += ",\"center_y\":" + str(target_Circle.m_center_y)
        radius = target_Circle.m_radius
        # 进一步扩大差异区域
        if radius < final_bigger_radius:
            radius = final_bigger_radius
        result += ",\"radius\":" + str(radius)
        result += "},"
    return result[0:-1]


def matchABList(file_list, gray_threlod):
    for file in file_list:
        (fileA, fileB) = file
        if os.path.exists(fileA):
            matchAB(gray_threlod, fileA, fileB)


def match_path_list(gray_threlod, file_list):
    if debug:
        print(lamp + str(gray_threlod) + lamp + str(len(file_list)) + lamp)
    global higher_gray_threlod_png_list
    global lower_gray_threlod_png_list
    global radius_min
    global print_png_list
    #
    radius_min = random.randint(radius_radom[0], radius_radom[1])
    # 匹配
    print(lamp + "file_list=" + str(len(file_list)) + "，" + str(
        radius_min) + "，" + str(gray_threlod))
    tk = []
    step = int(len(file_list) / tk_sum) + 1
    for i in range(0, tk_sum + 1):
        list = file_list[i * step:(i + 1) * step]
        t = threading.Thread(target=matchABList, name=str(i),
                             args=(list, gray_threlod,))
        t.start()
        tk.append(t)
    for t in tk:
        t.join()
    print("higher_gray_threlod_png_list=" + str(len(higher_gray_threlod_png_list)))
    print("lower_gray_threlod_png_list=" + str(len(lower_gray_threlod_png_list)))
    # 输出文件
    print("add_location_txt=" + str(len(print_png_list)))
    for fileA, fileB, target_list in print_png_list:
        #
        gray_resImg = cv2.cvtColor(cv2.imread(fileA), cv2.COLOR_BGR2GRAY)
        png_width, png_height = gray_resImg.shape
        #
        add_location_txt(fileA, fileB, png_width, png_height, target_list, output_path)
    # 输出图片
    print("print_png=" + str(len(print_png_list)))
    for fileA, fileB, target_list in print_png_list:
        print_png(fileA, fileB, target_list)
    print_png_list = []
    # 舍弃循环多次不符合要求，防止嵌套过深
    mix_png_min_size = len(higher_gray_threlod_png_list) + len(lower_gray_threlod_png_list)
    if mix_png_min_size > 0 and mix_png_min_size < mix_png_min_size_th:
        print(higher_gray_threlod_png_list)
        print(lower_gray_threlod_png_list)
        return
        #         提高灰度阈值
    if len(higher_gray_threlod_png_list) > 0:
        if len(higher_gray_threlod_png_list) > 30:
            gray_threlod = gray_threlod + 3
        elif len(higher_gray_threlod_png_list) > 10:
            gray_threlod = gray_threlod + 2
        elif len(higher_gray_threlod_png_list) > 3:
            gray_threlod = gray_threlod + 1.5
        elif len(higher_gray_threlod_png_list) > 0:
            gray_threlod = gray_threlod + 0.2
        higher_gray_threlod_png_list_temp = higher_gray_threlod_png_list
        higher_gray_threlod_png_list = []
        match_path_list(gray_threlod, higher_gray_threlod_png_list_temp)
    #     降低灰度阈值
    if len(lower_gray_threlod_png_list) > 0:
        if len(lower_gray_threlod_png_list) > 30:
            gray_threlod = gray_threlod - 3
        elif len(lower_gray_threlod_png_list) > 10:
            gray_threlod = gray_threlod - 2
        elif len(lower_gray_threlod_png_list) > 3:
            gray_threlod = gray_threlod - 1
        elif len(lower_gray_threlod_png_list) > 0:
            gray_threlod = gray_threlod - 0.2
        if gray_threlod > 0:
            lower_gray_threlod_png_list_temp = lower_gray_threlod_png_list
            lower_gray_threlod_png_list = []
            match_path_list(gray_threlod, lower_gray_threlod_png_list_temp)


lamp = "*************************************"
debug = False
plt_debug = True
output_path = "find_diff_target_circle.json"
output_png_path = "output/"
# 图片标识颜色
circle_color = (255, 0, 0)
# 灰度阈值待提高列表
higher_gray_threlod_png_list = []
# 灰度阈值待降低列表
lower_gray_threlod_png_list = []
# 降噪半径
radius_min = 0
# 降噪半径随机范围
radius_radom = [2, 15]
# 结果集
print_png_list = []
# 单个图片结果数量范围
target_max_len = 7
target_min_len = 5
# 扩大区域半径用于合并
merge_bigger_radius = 18
# 距离合并阈值
th_distance_between = 5
# 最终扩大区域半径
final_bigger_radius = 35
# 最大多线程数量
tk_sum = 50
# 多次循环不符合，图片舍弃，最大数量
mix_png_min_size_th = 5
#
if __name__ == '__main__':
    if os.path.exists(output_path):
        os.remove(output_path)
    if os.path.exists(output_png_path):
        shutil.rmtree(output_png_path)
    if not os.path.exists(output_png_path):
        os.mkdir(output_png_path)
    file_list = []

    for i in range(0, 600):
        filename_a = 'source/' + str(i) + 'a.png'
        filename_b = 'source/' + str(i) + 'b.png'
        if os.path.exists(filename_a):
            file_list.append((filename_a, filename_b))
        filename_a = 'source/' + str(i) + 'a.jpg'
        filename_b = 'source/' + str(i) + 'b.jpg'
        if os.path.exists(filename_a):
            file_list.append((filename_a, filename_b))
        # else:
        #     break
    match_path_list(30, file_list)
    # print("********************add_location_txt_all=" + str(len(print_png_list)))
    # for fileA, fileB, target_list in print_png_list:
    #     #
    #     gray_resImg = cv2.cvtColor(cv2.imread(fileA), cv2.COLOR_BGR2GRAY)
    #     png_width, png_height = gray_resImg.shape
    #     #
    #     add_location_txt(fileA, fileB, png_width, png_height, target_list, output_path)
    # print("********************print_png=" + str(len(print_png_list)))
    # for fileA, fileB, target_list in print_png_list:
    #     print_png(fileA, fileB, target_list)

    # match_path('source', (), (".png", ))
    # match_path_list(30, [('source/6a.png', 'source/6b.png'), ])
    # matchAB(16, 'source/19a.png', 'source/19b.png')
