from skimage import color
import numpy as np

import torch
import torch.nn.functional as F
from torchvision.utils import make_grid

from dataset.taskonomy_constants import *


def visualize_batch(
    X=None,  # X是输入的图像数据，默认为None
    Y=None,  # Y是真实标签，默认为None
    M=None,  # M是掩码数据，默认为None，用于区分图像中的关注区域
    Y_preds=None,  # Y_preds是预测标签，默认为None
    channels=None,  # channels指定要可视化的通道，默认为None
    size=None,  # size指定输出图像的尺寸，默认为None
    postprocess_fn=None,  # postprocess_fn是后处理函数，默认为None
    **kwargs,  # 接受额外的关键字参数
):
    """
    Visualize a global batch consists of N-shot images and labels for T channels.
    It is assumed that images are shared by all channels, thus convert channels into RGB and visualize at once.
    """
    vis = []  # 初始化一个列表，用于存储要可视化的数据

    # shape check
    assert X is not None or Y is not None or Y_preds is not None  # 断言至少有一个输入不为None

    # visualize image
    if X is not None:
        img = X.cpu().float()  # 将图像数据移至CPU并转换为浮点类型
        vis.append(img)  # 将图像添加到可视化列表中
    else:
        img = None  # 如果没有图像数据，则设置img为None

    # flatten labels and masks
    Ys = []  # 初始化一个列表，用于存储标签数据
    Ms = []  # 初始化一个列表，用于存储掩码数据
    if Y is not None:
        Ys.append((Y, None))  # 添加真实标签到列表，没有对应的预测标签
        Ms.append(M)  # 添加掩码到列表
    if Y_preds is not None:
        if isinstance(Y_preds, torch.Tensor):  # 如果预测标签是张量
            Ys.append((Y_preds, Y))  # 添加预测标签和真实标签到列表
            Ms.append(None)  # 没有对应的掩码
        elif isinstance(Y_preds, (tuple, list)):  # 如果预测标签是元组或列表
            if Y is not None:
                for Y_pred in Y_preds:
                    Ys.append((Y_pred, Y))  # 对每个预测标签，添加与真实标签的组合
                    Ms.append(None)  # 没有对应的掩码
            else:
                for Y_pred in Y_preds:
                    Ys.append((Y_pred, None))  # 对每个预测标签，添加没有真实标签的组合
                    Ms.append(None)  # 没有对应的掩码
        else:
            ValueError(f"unsupported predictions type: {type(Y_preds)}")  # 不支持的预测类型，抛出错误

    # visualize labels
    if len(Ys) > 0:
        for Y, Y_gt in Ys:
            Y = Y.cpu().float()  # 将标签移至CPU并转换为浮点类型
            if Y_gt is not None:
                Y_gt = Y_gt.cpu().float()  # 如果有真实标签，同样处理

            if channels is None:
                channels = list(range(Y.size(1)))  # 如果没有指定通道，使用所有通道

            label = Y[:, channels].clip(0, 1)  # 裁剪标签值到0到1之间
            if Y_gt is not None:
                label_gt = Y_gt[:, channels].clip(0, 1)  # 裁剪真实标签值到0到1之间
            else:
                label_gt = None  # 没有真实标签

            # fill masked region with random noise
            if M is not None:
                assert Y.shape == M.shape  # 断言标签和掩码形状相同
                M = M.cpu().float()  # 将掩码移至CPU并转换为浮点类型
                label = torch.where(
                    M[:, channels].bool(), label, torch.rand_like(label)
                )  # 使用掩码更新标签区域，未掩码区域填充随机噪声
                if Y_gt is not None:
                    label_gt = Y_gt[:, channels].clip(0, 1)
                    label_gt = torch.where(
                        M[:, channels].bool(), label_gt, torch.rand_like(label_gt)
                    )  # 同样处理真实标签

            if postprocess_fn is not None:
                label = postprocess_fn(label, img, label_gt=label_gt)  # 应用后处理函数

            label = visualize_label_as_rgb(label)  # 将标签可视化为RGB图像
            vis.append(label)  # 添加到可视化列表

    nrow = len(vis[0])  # 获取可视化列表中第一个元素的长度
    vis = torch.cat(vis)  # 拼接所有可视化数据
    if size is not None:
        vis = F.interpolate(vis, size)  # 如果指定了尺寸，调整可视化数据的尺寸
    vis = make_grid(vis, nrow=nrow, **kwargs)  # 使用网格格式组织可视化数据
    vis = vis.float()  # 转换为浮点类型

    return vis  # 返回最终的可视化数据


def postprocess_depth(label, img=None, **kwargs):
    # 乘以0.6，加上0.4，然后对结果做指数运算
    label = 0.6 * label + 0.4
    # 做对数运算，然后除以11.09
    label = torch.exp(label * np.log(2.0**16.0)) - 1.0
    # 除以0.18，做对数运算，然后减去0.64
    label = torch.log(label) / 11.09
    # 除以0.18，加上1.0，然后做归一化
    label = (label - 0.64) / 0.18
    # 乘以2，然后除以255，取值为0-1
    label = (label + 1.0) / 2
    # 乘以255，然后除以255，取值为0-255
    label = (label * 255).byte().float() / 255.0
    # 返回处理后的label
    return label


def postprocess_semseg(label, img=None, **kwargs):
    # 定义颜色元组
    COLORS = (
        "red",
        "blue",
        "yellow",
        "magenta",
        "green",
        "indigo",
        "darkorange",
        "cyan",
        "pink",
        "yellowgreen",
        "black",
        "darkgreen",
        "brown",
        "gray",
        "purple",
        "darkviolet",
    )

    # 如果label的维度为4，则将第1个维度 squeeze 掉
    if label.ndim == 4:
        label = label.squeeze(1)

    # 创建一个空列表，用于存放处理后的label
    label_vis = []
    # 如果img不为空，则遍历img和label，并将它们转换为rgb格式
    if img is not None:
        for img_, label_ in zip(img, label):
            for c in range(len(COLORS) + 1):
                label_[0, c] = c

            label_vis.append(
                torch.from_numpy(
                    color.label2rgb(
                        label_.numpy(),
                        image=img_.permute(1, 2, 0).numpy(),
                        colors=COLORS,
                        kind="overlay",
                    )
                ).permute(2, 0, 1)
            )
    else:
        # 如果img为空，则遍历label，并将它们转换为rgb格式
        for label_ in label:
            for c in range(len(COLORS) + 1):
                label_[0, c] = c

            label_vis.append(
                torch.from_numpy(
                    color.label2rgb(label_.numpy(), colors=COLORS, kind="overlay")
                ).permute(2, 0, 1)
            )

    # 将处理后的label转换为torch.tensor，并返回
    label = torch.stack(label_vis)

    return label


def visualize_label_as_rgb(label):
#    如果label的通道数为1，则将label重复3次，以便和输入的图像可以进行拼接
    if label.size(1) == 1:
        label = label.repeat(1, 3, 1, 1)
#    如果label的通道数为2，则将第二个通道补0，以便和输入的图像可以进行拼接
    elif label.size(1) == 2:
        label = torch.cat((label, torch.zeros_like(label[:, :1])), 1)
#    如果label的通道数为5，则将label的三个通道求平均，以便和输入的图像可以进行拼接
    elif label.size(1) == 5:
        label = torch.stack(
            (label[:, :2].mean(1), label[:, 2:4].mean(1), label[:, 4]), 1
        )
#    如果label的通道数不为1、2、3，则抛出未实现错误
    elif label.size(1) != 3:
        assert NotImplementedError

#    返回拼接后的label
    return label