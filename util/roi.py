import cv2
import torch
import numpy as np


def roi_transform(feature, box, size=(32, 100)):
    resize_h, resize_w = size
    x1, y1, x2, y2, x3, y3, x4, y4 = box
    # rotated_rect = cv2.minAreaRect(np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]]))
    # box_w, box_h = rotated_rect[1][0], rotated_rect[1][1]

    width = feature.shape[2]
    height = feature.shape[1]

    mapped_x1, mapped_y1 = (0, 0)
    mapped_x4, mapped_y4 = (0, resize_h)

    mapped_x2, mapped_y2 = (resize_w, 0)

    src_pts = np.float32([(x1, y1), (x2, y2), (x4, y4)])
    dst_pts = np.float32([
        (mapped_x1, mapped_y1), (mapped_x2, mapped_y2), (mapped_x4, mapped_y4)
    ])

    affine_matrix = cv2.getAffineTransform(src_pts.astype(np.float32), dst_pts.astype(np.float32))
    affine_matrix = param2theta(affine_matrix, width, height)

    affine_matrix *= 1e20  # cancel the error when type conversion
    affine_matrix = torch.tensor(affine_matrix, device=feature.device, dtype=torch.float)
    affine_matrix /= 1e20

    grid = torch.nn.functional.affine_grid(affine_matrix.unsqueeze(0), feature.unsqueeze(0).size())
    feature_rotated = torch.nn.functional.grid_sample(feature.unsqueeze(0), grid)
    feature_rotated = feature_rotated[:, :, 0:resize_h, 0:resize_w]

    feature_rotated = feature_rotated.squeeze(0)
    gray_scale_img = rgb_to_grayscale(feature_rotated).unsqueeze(0)

    return gray_scale_img

def param2theta(param, w, h):
    param = np.vstack([param, [0, 0, 1]])
    param = np.linalg.inv(param)

    theta = np.zeros([2, 3])
    theta[0, 0] = param[0, 0]
    theta[0, 1] = param[0, 1] * h / w
    theta[0, 2] = param[0, 2] * 2 / w + theta[0, 0] + theta[0, 1] - 1
    theta[1, 0] = param[1, 0] * w / h
    theta[1, 1] = param[1, 1]
    theta[1, 2] = param[1, 2] * 2 / h + theta[1, 0] + theta[1, 1] - 1
    return theta

def batch_roi_transform(feature_map, boxes, mapping, size=(32, 100)):
    # print("box2roi",boxes)
    # print(boxes.shape)
    scale_factor = 1.2

    rois = []
    for img_index, box in zip(mapping, boxes):
        feature = feature_map[img_index]
        # 获取原始ROI的坐标
        x1, y1, x2, y2, x3, y3, x4, y4 = box
        # 计算原始ROI的中心点坐标
        center_x = (x1 + x2 + x3 + x4) / 4
        center_y = (y1 + y2 + y3 + y4) / 4
        # 计算新的ROI的宽度和高度
        width = (x2 - x1 + x3 - x4) / 2
        height = (y3 - y1 + y4 - y2) / 2
        # 计算新的宽度和高度
        new_width = width * scale_factor
        new_height = height * scale_factor
        # 将ROI区域往右边扩充
        x1_new = center_x - new_width / 2.3
        y1_new = center_y - new_height / 2.1
        x2_new = center_x + new_width / 1.5
        y2_new = center_y - new_height / 2.1
        x3_new = center_x + new_width / 1.5
        y3_new = center_y + new_height / 2.1
        x4_new = center_x - new_width /2.3
        y4_new = center_y + new_height / 2.1
        # 构建新的ROI坐标
        expanded_box = [x1_new, y1_new, x2_new, y2_new, x3_new, y3_new, x4_new, y4_new]
        # 调用roi_transform函数来执行ROI的转换
        rois.append(roi_transform(feature, expanded_box, size))
    rois = torch.stack(rois, dim=0)
    # print("2roi", rois)
    # print(rois.shape)
    return rois

def rgb_to_grayscale(img):
    # type: (Tensor) -> Tensor
    """Convert the given RGB Image Tensor to Grayscale.
    For RGB to Grayscale conversion, ITU-R 601-2 luma transform is performed which
    is L = R * 0.2989 + G * 0.5870 + B * 0.1140
    Args:
        img (Tensor): Image to be converted to Grayscale in the form [C, H, W].
    Returns:
        Tensor: Grayscale image.
    """
    if img.shape[0] != 3:
        raise TypeError('Input Image does not contain 3 Channels')
    img = 0.2989 * img[0] + 0.5870 * img[1] + 0.1140 * img[2]
    contrast_factor = 2.0
    enhanced_img = (img - img.mean()) * contrast_factor + img.mean()

    return enhanced_img.to(img.dtype)
