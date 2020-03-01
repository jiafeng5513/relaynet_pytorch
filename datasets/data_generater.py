"""
通过2015_BOE_Chiu数据集生成本文所需的训练数据
取人工分层标记,做B样条插值
"""
import scipy.io as scio
import numpy as np
import cv2
import math
import pylab as pl
from scipy import interpolate
import matplotlib.pyplot as plt


filenames = ['./Subject_01.mat','./Subject_02.mat','./Subject_03.mat','./Subject_04.mat',
            './Subject_05.mat','./Subject_06.mat','./Subject_07.mat','./Subject_08.mat',
            './Subject_09.mat','./Subject_10.mat']

color = [[0, 0, 255], [0, 255, 0], [111, 145, 138], [0, 153, 255],
         [0, 255, 255], [107, 162, 94], [255, 255, 0], [255, 0, 0]]

window_name = 'duke_2015_BOE_Chiu'

class data_processer:
    def __init__(self):
        self.SLICE_INDEX = 0
        self.LAYER_ID = 0
        self.FLUID_ID = 0
        self.HUD_SWITCH = True
        self.FILE_INDEX = 0
        pass

    def open_mat(self, file_index):
        data = scio.loadmat(filenames[file_index])
        self.images = data['images']  # (496, 768, 61)
        self.automaticFluidDME = data['automaticFluidDME']  # (496, 768, 61)
        self.manualFluid1 = data['manualFluid1']  # (496, 768, 61)
        self.manualFluid2 = data['manualFluid2']  # (496, 768, 61)
        self.automaticLayersDME = data['automaticLayersDME']  # (8, 768, 61)
        self.automaticLayersNormal = data['automaticLayersNormal']  # (8, 768, 61)
        self.manualLayers1 = data['manualLayers1']  # (8, 768, 61)
        self.manualLayers2 = data['manualLayers2']  # (8, 768, 61)
        self.num_of_images = self.images.shape[2]

    def show_layers(self, will_export=False):
        img = self.images[:, :, self.SLICE_INDEX]
        img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        layers = [self.automaticLayersDME[:, :, self.SLICE_INDEX], self.automaticLayersNormal[:, :, self.SLICE_INDEX],
                  self.manualLayers1[:, :, self.SLICE_INDEX], self.manualLayers2[:, :, self.SLICE_INDEX]]
        layers_name = ['automaticLayersDME', 'automaticLayersNormal', 'manualLayers1', 'manualLayers2']
        layer = layers[self.LAYER_ID]

        fluid_masks = [self.automaticFluidDME[:, :, self.SLICE_INDEX], self.manualFluid1[:, :, self.SLICE_INDEX],
                       self.manualFluid2[:, :, self.SLICE_INDEX]]
        fluid_masks_name = ['automaticFluidDME', 'manualFluid1', 'manualFluid2']
        fluid_mask = np.array(fluid_masks[self.FLUID_ID], np.uint8)
        # 创建一张纯色图,copyto
        temp = np.ones((496, 768), dtype=np.uint8)
        bgr_temp = cv2.cvtColor(temp, cv2.COLOR_GRAY2BGR)
        bgr_temp[:, :, 0] = 128
        bgr_temp[:, :, 1] = 0
        bgr_temp[:, :, 2] = 0

        mask_bgr = cv2.bitwise_and(bgr_temp, bgr_temp, mask=fluid_mask)
        img_bgr = img_bgr + mask_bgr
        # 横轴为x,纵轴为y

        layer_text = ''

        for t in range(8):   # 对当前这张图,检索8条分层标记
            x_set = []
            y_set = []
            for j in range(768):  # 对于每条分层标记,按照x坐标遍历
                if not math.isnan(layer[t][j]):  # 如果第t条分层标记的j位置不为NaN,则此处存储的是这个标记点的y坐标
                    i = int(layer[t][j])
                    x_set.append(j)
                    y_set.append(i)
            # 非空判断
            if len(x_set) > 0 and len(y_set) > 0:
                # 如果x_set的长度不足,则分层线有断开的地方,需要插值
                if max(x_set)-min(x_set)+1>len(x_set):
                    # 先创建新的x坐标集合
                    x_new = np.arange(min(x_set), max(x_set)+1, 1)
                    tck = interpolate.splrep(x_set, y_set)
                    y_new = interpolate.splev(x_new, tck)
                    x_set = x_new
                    y_set = y_new
                    layer_text = '* bspline '
                # 此处得到了一条连续的分层线,画到图片上
                for index in range(len(x_set)):
                    img_bgr.itemset((int(y_set[index]), x_set[index], 0), color[t][0])
                    img_bgr.itemset((int(y_set[index]), x_set[index], 1), color[t][1])
                    img_bgr.itemset((int(y_set[index]), x_set[index], 2), color[t][2])
        #layer_text = layer_text + 'layer x: [' + str(min(x_set)) + ',' + str(max(x_set)) + ']'



        if self.HUD_SWITCH:
            HUD_text_0 = 'file : ' + filenames[self.FILE_INDEX] + ' slice number : ' + str(self.SLICE_INDEX)
            HUD_text_1 = 'layer label : ' + layers_name[self.LAYER_ID] + ' ' + layer_text
            HUD_text_2 = 'fluid label : ' + fluid_masks_name[self.FLUID_ID]
            img_bgr = cv2.putText(img_bgr, HUD_text_0, (50, img_bgr.shape[0] - 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                  (0, 0, 255), 1)
            img_bgr = cv2.putText(img_bgr, HUD_text_1, (50, img_bgr.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                  (0, 0, 255), 1)
            img_bgr = cv2.putText(img_bgr, HUD_text_2, (50, img_bgr.shape[0] - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                  (0, 0, 255), 1)
            # cv2.putText(图像, 文字, (x, y), 字体, 大小, (b, g, r), 宽度)
        cv2.imshow(window_name, img_bgr)
        pass

    def onTrackbarMove(self,object):
        self.SLICE_INDEX = cv2.getTrackbarPos('num', window_name)
        self.LAYER_ID = cv2.getTrackbarPos('layer_id', window_name)
        self.FLUID_ID = cv2.getTrackbarPos('fluid_id', window_name)
        self.HUD_SWITCH = cv2.getTrackbarPos('HUD', window_name) == 1
        self.show_layers()
        pass

    def onTrackbarMove_changefile(self,object):
        self.FILE_INDEX = cv2.getTrackbarPos('file_index', window_name)
        self.open_mat(self.FILE_INDEX)
        cv2.setTrackbarMax('num', window_name,self.num_of_images - 1)
        self.show_layers()
        pass

    def onMouse(self, event, x, y, flags, params):
        if event is cv2.EVENT_LBUTTONDBLCLK:
            # 导出当前图片
            self.show_layers(True)
            pass
        pass

    def viewer(self):
        self.open_mat(self.FILE_INDEX)
        cv2.namedWindow(window_name)
        cv2.createTrackbar('num', window_name, 0, self.num_of_images - 1, self.onTrackbarMove)
        cv2.createTrackbar('layer_id', window_name, 0, 3, self.onTrackbarMove)
        cv2.createTrackbar('fluid_id', window_name, 0, 2, self.onTrackbarMove)
        cv2.createTrackbar('HUD', window_name, 0, 1, self.onTrackbarMove)
        cv2.setTrackbarPos('HUD', window_name, 1)
        cv2.createTrackbar('file_index', window_name, 0, 9, self.onTrackbarMove_changefile)
        cv2.setMouseCallback(window_name, self.onMouse)
        self.show_layers()
        cv2.waitKey(0)
        cv2.destroyWindow(window_name)
        pass

if __name__ == '__main__':
    processer = data_processer()
    processer.viewer()
    pass