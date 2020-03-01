"""ClassificationCNN"""
import torch
import torch.nn as nn


class EncoderBlock(nn.Module):
    def __init__(self, channels):
        super(EncoderBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels=channels, out_channels=64, kernel_size=(3, 7), padding=(1, 3), stride=1)
        self.batchnorm = nn.BatchNorm2d(num_features=64)
        self.prelu = nn.PReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

    def forward(self, input):
        out_conv = self.conv(input)
        out_bn = self.batchnorm(out_conv)
        out_block = self.prelu(out_bn)
        out_encoder, indices = self.maxpool(out_block)
        return out_encoder, out_block, indices


class DecoderBlock(nn.Module):
    def __init__(self, channels):
        super(DecoderBlock, self).__init__()
        self.unpool = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.conv = nn.Conv2d(in_channels=channels, out_channels=64, kernel_size=(3, 7), padding=(1, 3), stride=1)
        self.batchnorm = nn.BatchNorm2d(num_features=64)
        self.prelu = nn.PReLU()

    def forward(self, input, out_block, indices):
        unpool = self.unpool(input, indices)
        concat = torch.cat((out_block, unpool), dim=1)
        out_conv = self.conv(concat)
        out_bn = self.batchnorm(out_conv)
        out_block = self.prelu(out_bn)
        return out_block


class ReLayNet(nn.Module):
    def __init__(self, params):
        super(ReLayNet, self).__init__()

        self.encode1 = EncoderBlock(channels=1)
        self.encode2 = EncoderBlock(channels=64)
        self.encode3 = EncoderBlock(channels=64)

        self.conv = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 7), padding=(1, 3), stride=1)
        self.batchnorm = nn.BatchNorm2d(num_features=64)
        self.prelu = nn.PReLU()

        self.decode1 = DecoderBlock(channels=128)
        self.decode2 = DecoderBlock(channels=128)
        self.decode3 = DecoderBlock(channels=128)

        self.classifier = nn.Conv2d(in_channels=64, out_channels=9, kernel_size=1, stride=1)
        self.softmax = nn.Softmax2d()


    def forward(self, input):
        e1, out1, ind1 = self.encode1.forward(input)
        e2, out2, ind2 = self.encode2.forward(e1)
        e3, out3, ind3 = self.encode3.forward(e2)

        out_conv = self.conv(e3)
        out_bn = self.batchnorm(out_conv)
        out_prelu = self.prelu(out_bn)

        d3 = self.decode1.forward(out_prelu, out3, ind3)
        d2 = self.decode2.forward(d3, out2, ind2)
        d1 = self.decode3.forward(d2, out1, ind1)
        prob = self.classifier(d1)

        # out_logit = self.softmax(prob)
        return prob

    @property
    def is_cuda(self):
        """
        Check if model parameters are allocated on the GPU.
        """
        return next(self.parameters()).is_cuda

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)
