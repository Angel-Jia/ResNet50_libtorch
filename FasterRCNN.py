from torch import nn
import torch

from collections import OrderedDict


# 按顺序加载各层参数
def load_model(model: torch.nn.Module, model_path):
    model_weights = torch.load(model_path)['state_dict']
    # assert len(model.state_dict()) == len(model_weights['state_dict'])
    _model_weights = OrderedDict()
    for this_model_key, weights_key in zip(list(model.state_dict().keys()), list(model_weights.keys())):
        _model_weights[this_model_key] = model_weights[weights_key]
        if isinstance(model_weights[weights_key], dict):
            print(this_model_key, "  |  ", weights_key)

    model.load_state_dict(_model_weights)


class Bottleneck(nn.Module):
    def __init__(self, inplane, mid_plane, use_downsample, downsample_stride):
        super(Bottleneck, self).__init__()
        self.inplane = inplane
        self.mid_plane = mid_plane
        self.downsample_stride = downsample_stride
        self.use_downsample = use_downsample
        expansion = 4

        self.conv1 = nn.Conv2d(inplane, mid_plane, kernel_size=1,
                               stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_plane)

        self.conv2 = nn.Conv2d(mid_plane, mid_plane, kernel_size=3,
                               stride=downsample_stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_plane)

        self.conv3 = nn.Conv2d(mid_plane, mid_plane * expansion,
                               kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(mid_plane * expansion)
        self.relu = nn.ReLU(True)

        if use_downsample:
            self.conv_downsample = nn.Conv2d(inplane, mid_plane * expansion, kernel_size=1,
                                             stride=downsample_stride, bias=False)
            self.bn_downsample = nn.BatchNorm2d(mid_plane * expansion)

    def forward(self, input):
        identity = input
        x = self.relu(self.bn1(self.conv1(input)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))

        if self.use_downsample:
            identity = self.bn_downsample(self.conv_downsample(input))
        x = x + identity

        return self.relu(x)


class Resnet50(nn.Module):
    def __init__(self):
        super(Resnet50, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7,
                               stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1)
        self.relu = nn.ReLU(True)
        self.layer1 = self._make_layer(1, 3, 64, 64)
        self.layer2 = self._make_layer(2, 4, 256, 128)
        self.layer3 = self._make_layer(3, 6, 512, 256)
        self.layer4 = self._make_layer(4, 3, 1024, 512)

    def forward(self, input):
        x = self.maxpool(self.relu(self.bn1(self.conv1(input))))
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return x1, x2, x3, x4

    def _make_layer(self, layer_id, layer_num, inplane, mid_plane):
        downsample_stride = 2
        expansion = 4
        if layer_id == 1:
            downsample_stride = 1

        layer_list = [Bottleneck(inplane, mid_plane, True, downsample_stride)]
        inplane = mid_plane * expansion
        for i in range(1, layer_num):
            layer_list.append(Bottleneck(inplane, mid_plane, False, 1))

        return nn.Sequential(*layer_list)


class FPN(nn.Module):
    def __init__(self):
        super(FPN, self).__init__()
        self.lateral_convs0 = nn.Conv2d(256, 256, 1, 1)
        self.lateral_convs1 = nn.Conv2d(512, 256, 1, 1)
        self.lateral_convs2 = nn.Conv2d(1024, 256, 1, 1)
        self.lateral_convs3 = nn.Conv2d(2048, 256, 1, 1)

        self.fpn_convs0 = nn.Conv2d(256, 256, 3, 1, padding=1)
        self.fpn_convs1 = nn.Conv2d(256, 256, 3, 1, padding=1)
        self.fpn_convs2 = nn.Conv2d(256, 256, 3, 1, padding=1)
        self.fpn_convs3 = nn.Conv2d(256, 256, 3, 1, padding=1)

    def forward(self, input):
        x1, x2, x3, x4 = input
        laters0 = self.lateral_convs0(x1)
        laters1 = self.lateral_convs1(x2)
        laters2 = self.lateral_convs2(x3)
        laters3 = self.lateral_convs3(x4)

        laters2 = laters2 + nn.functional.interpolate(laters3, laters2.shape[2:], mode='nearest')
        laters1 = laters1 + nn.functional.interpolate(laters2, laters1.shape[2:], mode='nearest')
        laters0 = laters0 + nn.functional.interpolate(laters1, laters0.shape[2:], mode='nearest')

        out0 = self.fpn_convs0(laters0)
        out1 = self.fpn_convs1(laters1)
        out2 = self.fpn_convs2(laters2)
        out3 = self.fpn_convs3(laters3)
        out4 = nn.functional.max_pool2d(out3, 1, stride=2)

        return out0, out1, out2, out3, out4


class RPNHead(nn.Module):
    def __init__(self):
        super(RPNHead, self).__init__()
        self.rpn_conv = nn.Conv2d(256, 256, 3, 1, padding=1)
        self.rpn_cls = nn.Conv2d(256, 3, 1, 1)
        self.rpn_reg = nn.Conv2d(256, 12, 1, 1)
        self.relu = nn.ReLU(True)

    def forward(self, input):
        rpn_cls_score_list = []
        rpn_bbox_pred_list = []
        for x in input:
            _x = self.relu(self.rpn_conv(x))
            rpn_cls_score = self.rpn_cls(_x)
            rpn_bbox_pred = self.rpn_reg(_x)
            rpn_cls_score_list.append(rpn_cls_score)
            rpn_bbox_pred_list.append(rpn_bbox_pred)

        return rpn_cls_score_list, rpn_bbox_pred_list


class ROIHEAD(nn.Module):
    def __init__(self, cls_num):
        super(ROIHEAD, self).__init__()
        self.fc_cls = nn.Linear(1024, cls_num)
        self.fc_reg = nn.Linear(1024, (cls_num - 1) * 4)
        self.shared_fc1 = nn.Linear(7*7*256, 1024)
        self.shared_fc2 = nn.Linear(1024, 1024)
        self.relu = nn.ReLU(True)

    def forward(self, input):
        x = input.flatten(1)
        x = self.relu(self.shared_fc1(x))
        x = self.relu(self.shared_fc2(x))

        cls_score = self.fc_cls(x)
        bbox_pred = self.fc_reg(x)

        return cls_score, bbox_pred


class FasterRcnn(nn.Module):
    def __init__(self, cls_num):
        super(FasterRcnn, self).__init__()
        self.backbone = Resnet50()
        self.fpn = FPN()
        self.rpnhead = RPNHead()
        self.roi_head = ROIHEAD(cls_num)

    def forward(self, input):
        # FasterRcnn_cpp.pt
        # x = self.backbone(input)
        # feats = self.fpn(x)
        # rpn_cls_score_list, rpn_bbox_pred_list = self.rpnhead(feats)
        # return feats, rpn_cls_score_list, rpn_bbox_pred_list

        # ROIHead_cpp.pt
        # 从mmdetection中导出bbox_feats，或者用torch.randn生成与bbox_feats一样的tensor
        # bbox_feats = torch.load('bbox_feats.pt')
        bbox_feats = torch.randn((1000, 256, 7, 7), dtype=torch.float32).to('cuda:0')
        self.roi_head.eval()
        traced = torch.jit.trace(self.roi_head, bbox_feats)
        traced.save('ROIHead_cpp.pt')
        x = self.roi_head(bbox_feats)

        return x


model = FasterRcnn(81)
load_model(model, './mmdetection2/faster_rcnn_r50_fpn_2x.pth')
model.to('cuda:0')
model.eval()
# torch.save(model.state_dict(), 'model_state_dict.pth')

# 从mmdetection中导出bbox_feats，或者用torch.randn生成与bbox_feats一样的tensor
# img = torch.load('img.pt').to('cuda:0')
img = torch.randn((1, 3, 800, 800), dtype=torch.float32).to('cuda:0')

#traced = torch.jit.trace(model, img)
#traced.save('FasterRcnn_cpp.pt')
