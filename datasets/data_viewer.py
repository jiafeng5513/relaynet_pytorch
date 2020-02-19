# viewer of 2015_BOE_Chiu_Analyzer
# data can be download from : http://people.duke.edu/~sf59/Chiu_BOE_2014_dataset.htm
# there are 10 Subject_*.mat files in the zip
import scipy.io as scio
import numpy as np
import cv2
import math

filenames = ['./Subject_01.mat','./Subject_02.mat','./Subject_03.mat','./Subject_04.mat',
            './Subject_05.mat','./Subject_06.mat','./Subject_07.mat','./Subject_08.mat',
            './Subject_09.mat','./Subject_10.mat']

color = [[0, 0, 255], [0, 255, 0], [111, 145, 138], [0, 153, 255],
         [0, 255, 255], [107, 162, 94], [255, 255, 0], [255, 0, 0]]

window_name = 'duke_2015_BOE_Chiu,double hit for export'


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

        label_set = []
        range_text = ''
        for j in range(768):
            for t in range(8):
                if not math.isnan(layer[t][j]):
                    i = int(layer[t][j])
                    # 给img[i][j]上t色
                    img_bgr.itemset((i, j, 0), color[t][0])
                    img_bgr.itemset((i, j, 1), color[t][1])
                    img_bgr.itemset((i, j, 2), color[t][2])
                    label_set.append(j)
                    range_text = 'layer x: [' + str(min(label_set)) + ',' + str(max(label_set)) + ']'
        if self.HUD_SWITCH:
            HUD_text_0 = 'file : ' + filenames[self.FILE_INDEX] + ' slice number : ' + str(self.SLICE_INDEX)
            HUD_text_1 = 'layer label : ' + layers_name[self.LAYER_ID] + ' ' + range_text
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
# 1.编程,挑选出110张有标记的图片,并将图片和标签的两侧无关区域进行裁剪
# 2.存储原始数据并进行编号,要保存好每一张图的来源,即文件名和slice number
# 3.对每一张slice导出一张边缘图,用RGB格式,边缘的颜色要取好
# 4.用其他软件手动标记这张边缘图
# 5.读取手动标记的图片,还原成二值标签
# 6.转储成h5py文件