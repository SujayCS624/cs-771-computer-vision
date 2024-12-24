import math
import torch
import torchvision

from torchvision.models import resnet
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork, LastLevelP6P7
from torchvision.ops.boxes import batched_nms

import torch
from torch import nn
from torch.nn.functional import binary_cross_entropy_with_logits, one_hot

# point generator
from .point_generator import PointGenerator

# input / output transforms
from .transforms import GeneralizedRCNNTransform

# loss functions
from .losses import sigmoid_focal_loss, giou_loss


class FCOSClassificationHead(nn.Module):
    """
    A classification head for FCOS with convolutions and group norms

    Args:
        in_channels (int): number of channels of the input feature.
        num_classes (int): number of classes to be predicted
        num_convs (Optional[int]): number of conv layer. Default: 3.
        prior_probability (Optional[float]): probability of prior. Default: 0.01.
    """

    def __init__(self, in_channels, num_classes, num_convs=3, prior_probability=0.01):
        super().__init__()
        self.num_classes = num_classes

        conv = []
        for _ in range(num_convs):
            conv.append(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
            )
            conv.append(nn.GroupNorm(16, in_channels))
            conv.append(nn.ReLU())
        self.conv = nn.Sequential(*conv)

        # A separate background category is not needed, as later we will consider
        # C binary classfication problems here (using sigmoid focal loss)
        self.cls_logits = nn.Conv2d(
            in_channels, num_classes, kernel_size=3, stride=1, padding=1
        )
        torch.nn.init.normal_(self.cls_logits.weight, std=0.01)
        # see Sec 3.3 in "Focal Loss for Dense Object Detection'
        torch.nn.init.constant_(
            self.cls_logits.bias, -math.log((1 - prior_probability) / prior_probability)
        )

    def forward(self, x):
        """
        Fill in the missing code here. The head will be applied to all levels
        of the feature pyramid, and predict a single logit for each location on
        every feature location.

        Without pertumation, the results will be a list of tensors in increasing
        depth order, i.e., output[0] will be the feature map with highest resolution
        and output[-1] will the featuer map with lowest resolution. The list length is
        equal to the number of pyramid levels. Each tensor in the list will be
        of size N x C x H x W, storing the classification logits (scores).

        Some re-arrangement of the outputs is often preferred for training / inference.
        You can choose to do it here, or in compute_loss / inference.
        """

        logits = []

        # Iterate over list of feature maps
        for feature_map in x:
            # Apply convolution layers to obtain classification logits
            conv_results = self.conv(feature_map)
            cls_logit = self.cls_logits(conv_results)

            # Reshape logits from (N, C, H, W) to (N, H*W, C)
            N, C, H, W = cls_logit.shape
            cls_logit = cls_logit.view(N, C, H, W).permute(0, 2, 3, 1)
            cls_logit = cls_logit.reshape(N, -1, C)

            # Append logits of each feature map to list of logits
            logits.append(cls_logit)

        return logits


class FCOSRegressionHead(nn.Module):
    """
    A regression head for FCOS with convolutions and group norms.
    This head predicts
    (a) the distances from each location (assuming foreground) to a box
    (b) a center-ness score

    Args:
        in_channels (int): number of channels of the input feature.
        num_convs (Optional[int]): number of conv layer. Default: 3.
    """

    def __init__(self, in_channels, num_convs=3):
        super().__init__()
        conv = []
        for _ in range(num_convs):
            conv.append(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
            )
            conv.append(nn.GroupNorm(16, in_channels))
            conv.append(nn.ReLU())
        self.conv = nn.Sequential(*conv)

        # regression outputs must be positive
        self.bbox_reg = nn.Sequential(
            nn.Conv2d(in_channels, 4, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.bbox_ctrness = nn.Conv2d(
            in_channels, 1, kernel_size=3, stride=1, padding=1
        )

        self.apply(self.init_weights)
        # The following line makes sure the regression head output a non-zero value.
        # If your regression loss remains the same, try to uncomment this line.
        # It helps the initial stage of training
        # torch.nn.init.normal_(self.bbox_reg[0].bias, mean=1.0, std=0.1)

    def init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            torch.nn.init.normal_(m.weight, std=0.01)
            torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Fill in the missing code here. The logic is rather similar to
        FCOSClassificationHead. The key difference is that this head bundles both
        regression outputs and the center-ness scores.

        Without pertumation, the results will be two lists of tensors in increasing
        depth order, corresponding to regression outputs and center-ness scores.
        Again, the list length is equal to the number of pyramid levels.
        Each tensor in the list will be of size N x 4 x H x W (regression)
        or N x 1 x H x W (center-ness).

        Some re-arrangement of the outputs is often preferred for training / inference.
        You can choose to do it here, or in compute_loss / inference.
        """

        regression_outputs = []
        centerness_outputs = []

        # Iterate over list of feature maps
        for feature_map in x:
            # Apply convolution layers to obtain conv_results
            conv_results = self.conv(feature_map)

            # Apply bounding box layers to obtain bounding box regression outputs
            bbox_reg_output = self.bbox_reg(conv_results)

            # Apply centerness layers to obtain centerness outputs
            centerness_output = self.bbox_ctrness(conv_results)

            # Reshape bounding box regression outputs from (N, 4, H, W) to (N, H*W, 4)
            N, _, H, W = bbox_reg_output.shape
            bbox_reg_output = bbox_reg_output.view(N, 4, H, W).permute(0, 2, 3, 1)
            bbox_reg_output = bbox_reg_output.reshape(N, -1, 4)

            # Reshape centerness outputs from (N, 1, H, W) to (N, H*W, 1)
            N, _, H, W = centerness_output.shape
            centerness_output = centerness_output.view(N, 1, H, W).permute(0, 2, 3, 1)
            centerness_output = centerness_output.reshape(N, -1)
            
            # Append outputs of each feature map to corresponding list of outputs
            regression_outputs.append(bbox_reg_output)
            centerness_outputs.append(centerness_output)
        
        return regression_outputs, centerness_outputs


class FCOS(nn.Module):
    """
    Implementation of Fully Convolutional One-Stage (FCOS) object detector,
    as desribed in the journal paper: https://arxiv.org/abs/2006.09214

    Args:
        backbone (string): backbone network, only ResNet is supported now
        backbone_freeze_bn (bool): if to freeze batch norm in the backbone
        backbone_out_feats (List[string]): output feature maps from the backbone network
        backbone_out_feats_dims (List[int]): backbone output features dimensions
        (in increasing depth order)

        fpn_feats_dim (int): output feature dimension from FPN in increasing depth order
        fpn_strides (List[int]): feature stride for each pyramid level in FPN
        num_classes (int): number of output classes of the model (excluding the background)
        regression_range (List[Tuple[int, int]]): box regression range on each level of the pyramid
        in increasing depth order. E.g., [[0, 32], [32 64]] means that the first level
        of FPN (highest feature resolution) will predict boxes with width and height in range of [0, 32],
        and the second level in the range of [32, 64].

        img_min_size (List[int]): minimum sizes of the image to be rescaled before feeding it to the backbone
        img_max_size (int): maximum size of the image to be rescaled before feeding it to the backbone
        img_mean (Tuple[float, float, float]): mean values used for input normalization.
        img_std (Tuple[float, float, float]): std values used for input normalization.

        train_cfg (Dict): dictionary that specifies training configs, including
            center_sampling_radius (int): radius of the "center" of a groundtruth box,
            within which all anchor points are labeled positive.

        test_cfg (Dict): dictionary that specifies test configs, including
            score_thresh (float): Score threshold used for postprocessing the detections.
            nms_thresh (float): NMS threshold used for postprocessing the detections.
            detections_per_img (int): Number of best detections to keep after NMS.
            topk_candidates (int): Number of best detections to keep before NMS.

        * If a new parameter is added in config.py or yaml file, they will need to be defined here.
    """

    def __init__(
        self,
        backbone,
        backbone_freeze_bn,
        backbone_out_feats,
        backbone_out_feats_dims,
        fpn_feats_dim,
        fpn_strides,
        num_classes,
        regression_range,
        img_min_size,
        img_max_size,
        img_mean,
        img_std,
        train_cfg,
        test_cfg,
    ):
        super().__init__()
        assert backbone in ("resnet18", "resnet34", "resnet50", "resnet101", "resnet152")
        self.backbone_name = backbone
        self.backbone_freeze_bn = backbone_freeze_bn
        self.fpn_strides = fpn_strides
        self.num_classes = num_classes
        self.regression_range = regression_range

        return_nodes = {}
        for feat in backbone_out_feats:
            return_nodes.update({feat: feat})

        # backbone network
        backbone_model = resnet.__dict__[backbone](weights="IMAGENET1K_V1")
        self.backbone = create_feature_extractor(
            backbone_model, return_nodes=return_nodes
        )

        # feature pyramid network (FPN)
        self.fpn = FeaturePyramidNetwork(
            backbone_out_feats_dims,
            out_channels=fpn_feats_dim,
            extra_blocks=LastLevelP6P7(fpn_feats_dim, fpn_feats_dim)
        )

        # point generator will create a set of points on the 2D image plane
        self.point_generator = PointGenerator(
            img_max_size, fpn_strides, regression_range
        )

        # classification and regression head
        self.cls_head = FCOSClassificationHead(fpn_feats_dim, num_classes)
        self.reg_head = FCOSRegressionHead(fpn_feats_dim)

        # image batching, normalization, resizing, and postprocessing
        self.transform = GeneralizedRCNNTransform(
            img_min_size, img_max_size, img_mean, img_std
        )

        # other params for training / inference
        self.center_sampling_radius = train_cfg["center_sampling_radius"]
        self.score_thresh = test_cfg["score_thresh"]
        self.nms_thresh = test_cfg["nms_thresh"]
        self.detections_per_img = test_cfg["detections_per_img"]
        self.topk_candidates = test_cfg["topk_candidates"]

    """
    We will overwrite the train function. This allows us to always freeze
    all batchnorm layers in the backbone, as we won't have sufficient samples in
    each mini-batch to aggregate the bachnorm stats.
    """
    @staticmethod
    def freeze_bn(module):
        if isinstance(module, nn.BatchNorm2d):
            module.eval()

    def train(self, mode=True):
        self.training = mode
        for module in self.children():
            module.train(mode)
        # additionally fix all bn ops (affine params are still allowed to update)
        if self.backbone_freeze_bn:
            self.apply(self.freeze_bn)
        return self

    """
    The behavior of the forward function depends on if the model is in training
    or evaluation mode.

    During training, the model expects both the input images
    (list of tensors within the range of [0, 1]),
    as well as a targets (list of dictionary), containing the following keys
        - boxes (``FloatTensor[N, 4]``): the ground-truth boxes in
          ``[x1, y1, x2, y2]`` format, with ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
        - labels (Int64Tensor[N]): the class label for each ground-truth box
        - other keys such as image_id are not used here
    The model returns a Dict[Tensor] during training, containing the classification, regression
    and centerness losses, as well as a final loss as a summation of all three terms.

    During inference, the model requires only the input tensors, and returns the post-processed
    predictions as a List[Dict[Tensor]], one for each input image. The fields of the Dict are as
    follows:
        - boxes (``FloatTensor[N, 4]``): the predicted boxes in ``[x1, y1, x2, y2]`` format,
          with ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
        - labels (Int64Tensor[N]): the predicted labels for each image
        - scores (Tensor[N]): the scores for each prediction

    See also the comments for compute_loss / inference.
    """

    def forward(self, images, targets):
        # sanity check
        if self.training:
            if targets is None:
                torch._assert(False, "targets should not be none when in training")
            else:
                for target in targets:
                    boxes = target["boxes"]
                    torch._assert(
                        isinstance(boxes, torch.Tensor),
                        "Expected target boxes to be of type Tensor.",
                    )
                    torch._assert(
                        len(boxes.shape) == 2 and boxes.shape[-1] == 4,
                        f"Expected target boxes of shape [N, 4], got {boxes.shape}.",
                    )

        # record the original image size, this is needed to decode the box outputs
        original_image_sizes = []
        for img in images:
            val = img.shape[-2:]
            original_image_sizes.append((val[0], val[1]))

        # transform the input
        images, targets = self.transform(images, targets)

        # get the features from the backbone
        # the result will be a dictionary {feature name : tensor}
        features = self.backbone(images.tensors)

        # send the features from the backbone into the FPN
        # the result is converted into a list of tensors (list length = #FPN levels)
        # this list stores features in increasing depth order, each of size N x C x H x W
        # (N: batch size, C: feature channel, H, W: height and width)
        fpn_features = self.fpn(features)
        fpn_features = list(fpn_features.values())

        # classification / regression heads
        cls_logits = self.cls_head(fpn_features)
        reg_outputs, ctr_logits = self.reg_head(fpn_features)

        # 2D points (corresponding to feature locations) of shape H x W x 2
        points, strides, reg_range = self.point_generator(fpn_features)

        # training / inference
        if self.training:
            # training: generate GT labels, and compute the loss
            losses = self.compute_loss(
                targets, points, strides, reg_range, cls_logits, reg_outputs, ctr_logits
            )
            # return loss during training
            return losses

        else:
            # inference: decode / postprocess the boxes
            detections = self.inference(
                points, strides, cls_logits, reg_outputs, ctr_logits, images.image_sizes
            )
            # rescale the boxes to the input image resolution
            detections = self.transform.postprocess(
                detections, images.image_sizes, original_image_sizes
            )
            # return detectrion results during inference
            return detections

    """
    Fill in the missing code here. This is probably the most tricky part
    in this assignment. Here you will need to compute the object label for each point
    within the feature pyramid. If a point lies around the center of a foreground object
    (as controlled by self.center_sampling_radius), its regression and center-ness
    targets will also need to be computed.

    Further, three loss terms will be attached to compare the model outputs to the
    desired targets (that you have computed), including
    (1) classification (using sigmoid focal for all points)
    (2) regression loss (using GIoU and only on foreground points)
    (3) center-ness loss (using binary cross entropy and only on foreground points)

    Some of the implementation details that might not be obvious
    * The output regression targets are divided by the feature stride (Eq 1 in the paper)
    * All losses are normalized by the number of positive points (Eq 2 in the paper)
    * You might want to double check the format of 2D coordinates saved in points

    The output must be a dictionary including the loss values
    {
        "cls_loss": Tensor (1)
        "reg_loss": Tensor (1)
        "ctr_loss": Tensor (1)
        "final_loss": Tensor (1)
    }
    where the final_loss is a sum of the three losses and will be used for training.
    """

    def compute_loss(
        self, targets, points, strides, reg_range, cls_logits, reg_outputs, ctr_logits
    ):
        # Compute batch size and number of feature pyramid levels
        batch_size = len(targets)
        num_fpn_levels = len(cls_logits)

        # Flatten points from (H,W,2) to (H*W,2)
        for level in range(num_fpn_levels):
            points[level] = points[level].reshape([-1, 2])

        # Compute number of points and number of objects in the batch
        num_points_per_level = [len(points_per_level) for points_per_level in points]
        num_objects_per_img = [len(target['boxes']) for target in targets]
        points_all_level = torch.cat(points, dim=0)
        num_points = len(points_all_level)
        num_objects = sum(num_objects_per_img)

        boxes = []
        labels = []
        areas = []

        for img_idx in range(batch_size):
            # Extract target bounding box dimensions, labels, and areas from targets dictionary 
            boxes_per_img = targets[img_idx]['boxes']
            labels_per_img = targets[img_idx]['labels']
            areas_per_img = targets[img_idx]['area']
            boxes.append(boxes_per_img)
            labels.append(labels_per_img)
            areas.append(areas_per_img)
        
        # Concatenate across the entire batch
        boxes = torch.cat(boxes, dim=0)
        labels = torch.cat(labels, dim=0)
        areas = torch.cat(areas, dim=0)
        
        # Expand shape of targets to be (num_points, num_objects)
        boxes = boxes[None].expand(num_points, num_objects, 4) # (num_objects, 4) -> (num_points, num_objects, 4)
        labels = labels[None].expand(num_points, num_objects)
        areas = areas[None].expand(num_points, num_objects)

        # Extract x and y co-ordinates from all points and expand their shape to be (num_points, num_objects)
        xs, ys = points_all_level[:, 1], points_all_level[:, 0]
        xs = xs[:, None].expand(num_points, num_objects) # (num_points) -> (num_points, num_objects)
        ys = ys[:, None].expand(num_points, num_objects)
        

        # Expand shape of regression range to be (num_points, num_objects, 2)
        expanded_reg_range = []
        for level, points_per_level in enumerate(points):
            expanded_reg_range.append(reg_range[level][None].expand(len(points_per_level), -1)) # (2) -> (num_points, 2)
        
        expanded_reg_range = torch.cat(expanded_reg_range, dim=0)
        expanded_reg_range = expanded_reg_range[:, None, :].expand(num_points, num_objects, 2) # (num_points, 2) -> (num_points, num_objects, 2)

        # Expand shape of strides to be (num_points, num_objects)
        strides_expand = torch.zeros_like(xs,requires_grad =False)
        strides_expand_orig = torch.zeros_like(xs,requires_grad =False)

        # Compute center of target bounding boxes
        center_x = (boxes[..., 0] + boxes[..., 2]) / 2
        center_y = (boxes[..., 1] + boxes[..., 3]) / 2

        # Project the points on current level back to the `original` sizes
        # Start index for current level of feature pyramid
        p_start = 0
        for level, points_per_level in enumerate(num_points_per_level):
            # End index for current level of feature pyramid
            p_end = p_start + points_per_level
            # Scaled stride values for current level of feature pyramid
            strides_expand[p_start:p_end] = self.center_sampling_radius * strides[level]
            # Original stride values for current level of feature pyramid
            strides_expand_orig[p_start:p_end] = strides[level]
            # Set start index for next level of feature pyramid
            p_start = p_end

        # Compute lower and upper bounds of center region of the bounding boxes
        x_min = center_x - strides_expand
        y_min = center_y - strides_expand
        x_max = center_x + strides_expand
        y_max = center_y + strides_expand

        # Boolean mask to identify points within the bounding boxes
        is_in_box = (xs<boxes[..., 2]) & (xs>boxes[..., 0]) & (ys<boxes[..., 3]) & (ys>boxes[..., 1]) 
        # Boolean mask to identify points within the center region of the bounding boxes
        is_in_center = (xs<x_max) & (xs>x_min) & (ys<y_max) & (ys>y_min) 

        # Compute training regression targets as per equation 1
        left_dist = (xs - boxes[..., 0])/strides_expand_orig
        top_dist = (ys - boxes[..., 1])/strides_expand_orig
        right_dist = (boxes[..., 2] - xs)/strides_expand_orig
        bottom_dist = (boxes[..., 3] - ys)/strides_expand_orig

        # Combine the 4 regression targets obtained and reshape into (num_points, num_objects, 4)
        reg_targets = torch.stack([left_dist, top_dist, right_dist, bottom_dist], dim=2) # (num_points, num_objects) -> (num_points, num_objects, 4)

        # Find the max of the 4 regression target values for each point and bounding box
        max_reg_targets = reg_targets.max(dim=2)[0]
        # Boolean mask to identify points within the regression range 
        is_in_reg_range = ((max_reg_targets > expanded_reg_range[..., 0]) & (max_reg_targets < expanded_reg_range[..., 1]))

        # Apply the 3 boolean masks to find list of valid points
        valid_points = is_in_box & is_in_reg_range & is_in_center

        # Set label to background class for invalid points
        labels = torch.where(valid_points, labels, self.num_classes)
        # Set areas to a large value for invalid points (10^9)
        areas = torch.where(valid_points, areas, 1000000000)

        # Start index for current image in batch of images
        obj_start = 0
        reg_targets_list = []
        labels_list = []
        for img_idx in range(batch_size):
            # End index for current image in batch of images
            obj_end = obj_start + num_objects_per_img[img_idx]
            
            # Extract regression targets, labels, and areas corresponding to objects in the current image
            reg_targets_per_img = reg_targets[:,obj_start:obj_end,:]
            labels_per_img = labels[:,obj_start:obj_end]
            areas_per_img = areas[:,obj_start:obj_end]

            # Select the labels and regression targets corresponding to the minimum area bounding box for each point
            _, min_area_idx = areas_per_img.min(dim=1) # (num_points, num_objects_per_img) -> (num_points)
            labels_per_img = labels_per_img[range(num_points),min_area_idx]
            reg_targets_per_img = reg_targets_per_img[range(num_points), min_area_idx] 
            labels_list.append(torch.split(labels_per_img, num_points_per_level, dim=0))
            reg_targets_list.append(torch.split(reg_targets_per_img, num_points_per_level, dim=0))
            
            # Set start index for next image in batch of images
            obj_start = obj_end
        
        # Reorganize labels and regression targets by feature pyramid level
        labels_level_first = []
        reg_targets_level_first = []

        for level in range(num_fpn_levels):
            labels_level_first.append(torch.cat([labels_tmp[level] for labels_tmp in labels_list], dim=0))
            reg_targets_level_first.append(torch.cat([reg_targets_tmp[level]for reg_targets_tmp in reg_targets_list], dim=0))

        # Flatten labels, regression targets, regression and classification head outputs across all images in the batch
        labels_flatten = [labels_level.reshape(-1) for labels_level in labels_level_first]
        reg_targets_flatten = [reg_targets_level.reshape(-1, 4) for reg_targets_level in reg_targets_level_first]
        cls_logits_flatten = [cls_logit.reshape(-1, self.num_classes) for cls_logit in cls_logits]
        reg_outputs_flatten = [reg_output.reshape(-1, 4) for reg_output in reg_outputs]
        ctr_logits_flatten = [ctr_logit.reshape(-1) for ctr_logit in ctr_logits]

        # Flatten labels, regression targets, regression and classification head outputs across all levels of the feature pyramid
        labels_flatten = torch.cat(labels_flatten, dim=0)
        reg_targets_flatten = torch.cat(reg_targets_flatten, dim=0)
        cls_logits_flatten = torch.cat(cls_logits_flatten, dim=0)
        reg_outputs_flatten = torch.cat(reg_outputs_flatten, dim=0)
        ctr_logits_flatten = torch.cat(ctr_logits_flatten, dim=0)

        # Boolean mask to identify foreground points
        positive_mask = ((labels_flatten >= 0) & (labels_flatten < self.num_classes)).nonzero().reshape(-1)
        # Filter out only the foreground points for calculating regression and centerness losses
        reg_outputs_flatten = reg_outputs_flatten[positive_mask]
        ctr_logits_flatten = ctr_logits_flatten[positive_mask]
        reg_targets_flatten = reg_targets_flatten[positive_mask]
        
        # One hot encode the labels to compute the sigmoid focal loss
        labels_flatten_num_classes = one_hot(labels_flatten,num_classes=self.num_classes+1)
        labels_flatten_num_classes = labels_flatten_num_classes[:,:self.num_classes]
        
        # Compute the classification loss for all points
        cls_loss = sigmoid_focal_loss(cls_logits_flatten,labels_flatten_num_classes)

        # Check if there are foreground points
        if len(positive_mask) > 0:
            # Flatten points across all levels of the feature pyramid
            points_flatten = torch.cat([points[None].expand(batch_size, 2) for points in points_all_level], dim=0)
            foreground_points = points_flatten[positive_mask]
            
            # Predicted bounding box for foreground points
            tmp_l = foreground_points[...,1] - reg_outputs_flatten[..., 0]
            tmp_r = foreground_points[...,1] + reg_outputs_flatten[..., 2]
            tmp_t = foreground_points[...,0] - reg_outputs_flatten[..., 1]
            tmp_b = foreground_points[...,0] + reg_outputs_flatten[..., 3]
            decoded_reg_outputs_flatten = torch.stack((tmp_l, tmp_t, tmp_r, tmp_b), -1)
            
            # Target bounding box for foreground points
            tmp_l = foreground_points[...,1] - reg_targets_flatten[..., 0]
            tmp_r = foreground_points[...,1] + reg_targets_flatten[..., 2]
            tmp_t = foreground_points[...,0] - reg_targets_flatten[..., 1]
            tmp_b = foreground_points[...,0] + reg_targets_flatten[..., 3]
            decoded_reg_targets_flatten = torch.stack((tmp_l, tmp_t, tmp_r, tmp_b), -1)
            
            # Compute the regression loss for foreground points
            reg_loss = giou_loss(decoded_reg_outputs_flatten, decoded_reg_targets_flatten)

            # Compute centerness score as per equation 3
            left_right = reg_targets_flatten[:, [0, 2]]
            top_bottom = reg_targets_flatten[:, [1, 3]]
            centerness = (left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * (top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
            centerness_targets =  torch.sqrt(centerness)
            # Compute centerness loss for foreground points
            ctr_loss = binary_cross_entropy_with_logits(ctr_logits_flatten, centerness_targets,reduction='none') 

        # If there are no foreground points, set regression and centerness losses to 0
        else:
            reg_loss = torch.zeros_like(cls_loss,requires_grad =True)
            ctr_loss = torch.zeros_like(cls_loss,requires_grad =True)

        # Compute mean of all 3 losses for only the foreground points
        cls_loss = cls_loss.sum() / max(len(positive_mask), 1.0)
        reg_loss = reg_loss.sum()/ max(len(positive_mask), 1.0)
        ctr_loss = ctr_loss.sum()/ max(len(positive_mask), 1.0)
        final_loss = cls_loss + reg_loss + ctr_loss
        
        return {
            "cls_loss": cls_loss,
            "reg_loss": reg_loss,
            "ctr_loss": ctr_loss,
            "final_loss": final_loss,
        }

    """
    Fill in the missing code here. The inference is also a bit involved. It is
    much easier to think about the inference on a single image
    (a) Loop over every pyramid level
        (1) compute the object scores
        (2) filter out boxes with low object scores (self.score_thresh)
        (3) select the top K boxes (self.topk_candidates)
        (4) decode the boxes and their labels
        (5) clip boxes outside of the image boundaries (due to padding) / remove small boxes
    (b) Collect all candidate boxes across all pyramid levels
    (c) Run non-maximum suppression to remove any duplicated boxes
    (d) keep a fixed number of boxes after NMS (self.detections_per_img)

    Some of the implementation details that might not be obvious
    * As the output regression target is divided by the feature stride during training,
    you will have to multiply the regression outputs by the stride at inference time.
    * Most of the detectors will allow two overlapping boxes from different categories
    (e.g., one from "shirt", the other from "person"). That means that
        (a) one can decode two same boxes of different categories from one location;
        (b) NMS is only performed within each category.
    * Regression range is not used, as the range is not enforced during inference.
    * image_shapes is needed to remove boxes outside of the images.
    * Output labels should be offseted by +1 to compensate for the input label transform

    The output must be a list of dictionary items (one for each image) following
    [
        {
            "boxes": Tensor (N x 4) with each row in (x1, y1, x2, y2)
            "scores": Tensor (N, )
            "labels": Tensor (N, )
        },
    ]
    """

    def inference(
        self, points, strides, cls_logits, reg_outputs, ctr_logits, image_shapes
    ):
        
        detections = []
        
        # Compute batch size and number of feature pyramid levels
        batch_size = len(image_shapes)
        num_fpn_levels = len(cls_logits)

        # Apply sigmoid for classification and centerness logits
        for level in range(num_fpn_levels):
            cls_logits[level] = cls_logits[level].sigmoid()
            ctr_logits[level] = ctr_logits[level].sigmoid()
        
        # Iterate over each image in the batch
        for img_idx in range(batch_size):
            image_boxes = []
            image_scores = []
            image_labels = []
            
            # Get height and width of image
            H, W = image_shapes[img_idx]

            for level in range(num_fpn_levels):
                # Flatten points from (H,W,2) to (H*W,2)
                points[level] = points[level].reshape([-1, 2])
                # Retrieve the ctr_logits corresponding to a level of the feature pyramid for an image
                ctr_logits_expanded = ctr_logits[level][img_idx][:,None] # (batch_size, H*W) -> (H*W, 1)
            
                # Compute the object scores
                object_scores = torch.sqrt(cls_logits[level][img_idx]*ctr_logits_expanded)

                # Boolean mask to keep only boxes with object scores above the threshold
                keep_indices = object_scores > self.score_thresh
                object_scores = object_scores[keep_indices]

                # Extract indices of kept candidate boxes
                per_candidate_nonzeros = keep_indices.nonzero()
                # Extract location of kept candidate boxes
                per_box_loc = per_candidate_nonzeros[:, 0]
                # Extract class for each detection and offset by 1 due to zero-indexing
                per_class = per_candidate_nonzeros[:, 1] + 1 
                # Extract the regression predictions of kept candidate boxes
                per_box_regression = reg_outputs[level][img_idx][per_box_loc]
                # Extract points corresponding to kept candidate boxes
                per_locations = points[level][per_box_loc]
                
                # If there are more than K candidate boxes, select only the top-K boxes
                if keep_indices.sum().item() > self.topk_candidates:
                    object_scores, top_k_indices = object_scores.topk(self.topk_candidates, sorted=False)
                    per_class = per_class[top_k_indices]
                    per_box_regression = per_box_regression[top_k_indices]
                    per_locations = per_locations[top_k_indices]
    
                # Compute top left and bottom right corners of the predicted bounding box
                box_x0 = per_locations[:, 1] - per_box_regression[:, 0] * strides[level]
                box_y0 = per_locations[:, 0] - per_box_regression[:, 1] * strides[level]
                box_x1 = per_locations[:, 1] + per_box_regression[:, 2] * strides[level]
                box_y1 = per_locations[:, 0] + per_box_regression[:, 3] * strides[level]

                # Clip boxes outside the image bound
                box_x0 = box_x0.clamp(min = 0, max = W-1)
                box_x1 = box_x1.clamp(min = 0, max = W-1)
                box_y0 = box_y0.clamp(min = 0, max = H-1)
                box_y1 = box_y1.clamp(min = 0, max = H-1)

                # Remove small boxes
                new_keep_indices = ((box_x1 - box_x0)>0)
                new_keep_indices &= ((box_y1 - box_y0)>0)

                boxes = torch.stack([box_x0, box_y0, box_x1, box_y1], dim=-1)

                boxes = boxes[new_keep_indices]
                scores = object_scores[new_keep_indices]
                labels = per_class[new_keep_indices]

                # Append detections for this level
                image_boxes.append(boxes)
                image_scores.append(scores)
                image_labels.append(labels)

            # Concatenate boxes, scores, and labels across all levels of the feature pyramid
            image_boxes = torch.cat(image_boxes, dim = 0)
            image_scores = torch.cat(image_scores, dim = 0)
            image_labels = torch.cat(image_labels, dim = 0)

            # Remove duplicate boxes using Non-Maximum Suppression
            keep_indices = batched_nms(image_boxes, image_scores, image_labels, self.nms_thresh)

            # Keep top-K boxes overall
            keep_indices = keep_indices[:self.detections_per_img]

            # Append the final detections for this image
            detections.append(
                {
                    "boxes": image_boxes[keep_indices],
                    "scores": image_scores[keep_indices],
                    "labels": image_labels[keep_indices],  
                }
            )

        return detections
