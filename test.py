import numpy as np
import matplotlib.pyplot as plt
import torch
import h5py
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch.nn.functional as F
from networks.relay_net import ReLayNet
from networks.data_utils import get_imdb_data, ImdbData

SEG_LABELS_LIST = [
    {"id": -1, "name": "void", "rgb_values": [0, 0, 0]},
    {"id": 0, "name": "Region above the retina (RaR)", "rgb_values": [128, 0, 0]},
    {"id": 1, "name": "ILM: Inner limiting membrane", "rgb_values": [0, 128, 0]},
    {"id": 2, "name": "NFL-IPL: Nerve fiber ending to Inner plexiform layer", "rgb_values": [128, 128, 0]},
    {"id": 3, "name": "INL: Inner Nuclear layer", "rgb_values": [0, 0, 128]},
    {"id": 4, "name": "OPL: Outer plexiform layer", "rgb_values": [128, 0, 128]},
    {"id": 5, "name": "ONL-ISM: Outer Nuclear layer to Inner segment myeloid", "rgb_values": [0, 128, 128]},
    {"id": 6, "name": "ISE: Inner segment ellipsoid", "rgb_values": [128, 128, 128]},
    {"id": 7, "name": "OS-RPE: Outer segment to Retinal pigment epithelium", "rgb_values": [64, 0, 0]},
    {"id": 8, "name": "Region below RPE (RbR)", "rgb_values": [192, 0, 0]}];


# {"id": 9, "name": "Fluid region", "rgb_values": [64, 128, 0]}];

def label_img_to_rgb(label_img):
    label_img = np.squeeze(label_img)
    labels = np.unique(label_img)
    label_infos = [l for l in SEG_LABELS_LIST if l['id'] in labels]

    label_img_rgb = np.array([label_img,
                              label_img,
                              label_img]).transpose(1, 2, 0)
    for l in label_infos:
        mask = label_img == l['id']
        label_img_rgb[mask] = l['rgb_values']

    return label_img_rgb.astype(np.uint8)


def my_imdb_loader():
    # Load DATA
    f = h5py.File('datasets/dataset.hdf5', 'r')
    images = f['images']
    layers = f['layers']

    Data = []
    Label = []

    for i in iter(list(images.keys())):
        Data.append(images[list(images.keys())[int(i)]][:])

    for i in iter(list(layers.keys())):
        Label.append(layers[list(layers.keys())[int(i)]][:])

    Tr_Dat = Data[0:25]
    Tr_Label = Label[0:25]

    Te_Dat = Data[26:]
    Te_Label = Label[26:]

    return Tr_Dat, Tr_Label, Te_Dat, Te_Label
    # return (ImdbData(Tr_Dat, Tr_Label), ImdbData(Te_Dat, Te_Label))


def test_model():
    # original data loader:
    # train_data, test_data = get_imdb_data()
    # print("Train size: %i" % len(train_data))
    # print("Test size: %i" % len(test_data))

    # my data loader:
    train_data, train_label, test_data, test_label = my_imdb_loader()
    print("Train size: %i" % len(train_data))
    print("Test size: %i" % len(test_data))

    # convert to NCHW tensor
    input_data = torch.Tensor(test_data[0])
    H,W = input_data.shape
    input_tensor = input_data.reshape([1,1,H,W])
    print(input_tensor.shape)

    relaynet_model = torch.load('models/Exp01/relaynet_epoch20.model')
    with torch.no_grad():
        out = relaynet_model(Variable(input_tensor.cuda(), volatile=True))
    out = F.softmax(out, dim=1)
    max_val, idx = torch.max(out, 1)
    idx = idx.data.cpu().numpy()
    idx = label_img_to_rgb(idx)
    plt.imshow(idx)
    plt.show()

    img_test = test_data.X[0:1]
    img_test = np.squeeze(img_test)
    plt.imshow(img_test)
    plt.show()
    pass


if __name__ == '__main__':
    test_model()
