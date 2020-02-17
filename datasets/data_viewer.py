import scipy.io as scio
import numpy as np
import cv2
import math

dataFile = './Subject_01.mat'
data = scio.loadmat(dataFile)

images = data['images']  # (496, 768, 61)

automaticFluidDME = data['automaticFluidDME']  # (496, 768, 61)
manualFluid1 = data['manualFluid1']  # (496, 768, 61)
manualFluid2 = data['manualFluid2']  # (496, 768, 61)

automaticLayersDME = data['automaticLayersDME']  # (8, 768, 61)
automaticLayersNormal = data['automaticLayersNormal']  # (8, 768, 61)
manualLayers1 = data['manualLayers1']  # (8, 768, 61)
manualLayers2 = data['manualLayers2']  # (8, 768, 61)

color = [[0, 0, 255], [0, 255, 0], [111, 145, 138], [0, 153, 255],
         [0, 255, 255], [107, 162, 94], [255, 255, 0], [255, 0, 0]]

num_of_imgs = images.shape[2]
INDEX = 0
LAYER_ID = 0
FLUID_ID = 0
HUD_switch = True


def onTrackbarMove(object):
    INDEX = cv2.getTrackbarPos('num', dataFile)
    LAYER_ID = cv2.getTrackbarPos('layer_id', dataFile)
    FLUID_ID = cv2.getTrackbarPos('fluid_id', dataFile)
    HUD_switch = cv2.getTrackbarPos('HUD', dataFile) == 1
    show_layers(INDEX, LAYER_ID, FLUID_ID, HUD_switch)
    pass


def show_layers(index, layer_id, fluid_id, HUD_switch):
    """
    显示图片
    :param index:    第几张图片
    :param layer_id: 使用何种分层标记
    :param fluid_id: 使用何种积液标记
    :return:
    """
    # img_name = 'img_' + str(index)
    img = images[:, :, index]
    img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    layers = [automaticLayersDME[:, :, index], automaticLayersNormal[:, :, index], manualLayers1[:, :, index], manualLayers2[:, :, index]]
    layers_name = ['automaticLayersDME', 'automaticLayersNormal', 'manualLayers1', 'manualLayers2']
    layer = layers[layer_id]

    fluid_masks = [automaticFluidDME[:, :, index], manualFluid1[:, :, index], manualFluid2[:, :, index]]
    fluid_masks_name = ['automaticFluidDME', 'manualFluid1', 'manualFluid2']
    fluid_mask = np.array(fluid_masks[fluid_id], np.uint8)
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
                label_set.append(i)
                range_text='layer x: ['+str(min(label_set))+','+str(max(label_set))+']'
    if HUD_switch:
        HUD_text_1 = 'layer label :'+layers_name[layer_id]+' '+range_text
        HUD_text_2 = 'fluid label : ' + fluid_masks_name[fluid_id]
        img_bgr = cv2.putText(img_bgr, HUD_text_1, (50, img_bgr.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
        img_bgr = cv2.putText(img_bgr, HUD_text_2, (50, img_bgr.shape[0] - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
        #cv2.putText(图像, 文字, (x, y), 字体, 大小, (b, g, r), 宽度)
    cv2.imshow(dataFile, img_bgr)
    pass


if __name__ == '__main__':
    cv2.namedWindow(dataFile)
    cv2.createTrackbar('num', dataFile, 0, num_of_imgs - 1, onTrackbarMove)
    cv2.createTrackbar('layer_id', dataFile, 0, 3, onTrackbarMove)
    cv2.createTrackbar('fluid_id', dataFile, 0, 2, onTrackbarMove)
    cv2.createTrackbar('HUD',dataFile, 0, 1, onTrackbarMove)
    cv2.setTrackbarPos('HUD',dataFile,1)
    show_layers(INDEX, LAYER_ID, FLUID_ID,HUD_switch)
    cv2.waitKey(0)
    cv2.destroyWindow(dataFile)
