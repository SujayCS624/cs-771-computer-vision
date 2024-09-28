import math
import random
import glob
import os
import numpy as np

import cv2
import numbers
import collections
try:
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable

import torch
from torch.utils import data

from utils import resize_image, load_image

# default list of interpolations
_DEFAULT_INTERPOLATIONS = [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC]

#################################################################################
# These are helper functions or functions for demonstration
# You won't need to modify them
#################################################################################


class Compose(object):
    """Composes several transforms together.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    Example:
        >>> Compose([
        >>>     Scale(320),
        >>>     RandomSizedCrop(224),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

    def __repr__(self):
        repr_str = ""
        for t in self.transforms:
            repr_str += t.__repr__() + "\n"
        return repr_str


class RandomHorizontalFlip(object):
    """Horizontally flip the given numpy array randomly
    (with a probability of 0.5).
    """

    def __call__(self, img):
        """
        Args:
            img (numpy array): Image to be flipped.

        Returns:
            numpy array: Randomly flipped image
        """
        if random.random() < 0.5:
            img = cv2.flip(img, 1)
            return img
        return img

    def __repr__(self):
        return "Random Horizontal Flip"


#################################################################################
# You will need to fill in the missing code in these classes
#################################################################################
class Scale(object):
    """Rescale the input numpy array to the given size.

    This class will resize an input image based on its shortest side.

    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (w, h), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size, size * height / width)

        interpolations (list of int, optional): Desired interpolation.
            Default is ``CV2.INTER_NEAREST|CV2.INTER_LINEAR|CV2.INTER_CUBIC``
            Pass None during testing: always use CV2.INTER_LINEAR
    """

    def __init__(self, size, interpolations=_DEFAULT_INTERPOLATIONS):
        assert isinstance(size, int) or (
            isinstance(size, Iterable) and len(size) == 2
        )
        self.size = size
        # use bilinear if interpolation is not specified
        if interpolations is None:
            interpolations = [cv2.INTER_LINEAR]
        assert isinstance(interpolations, Iterable)
        self.interpolations = interpolations

    def __call__(self, img):
        """
        Args:
            img (numpy array): Image to be scaled.

        Returns:
            numpy array: Rescaled image
        """
        # sample interpolation method
        interpolation = random.sample(self.interpolations, 1)[0]

        # get current height and width
        curr_h, curr_w = img.shape[:2]

        # print("Current Image Dimensions: ")
        # print(curr_w, curr_h)

        # scale the image
        if isinstance(self.size, int):
            # Set the smaller dimension to the size property 
            # and proportionally rescale the larger dimension
            if curr_h > curr_w:
                new_w = self.size
                new_h = round(self.size * curr_h/curr_w)
            else:
                new_h = self.size
                new_w = round(self.size * curr_w/curr_h)
        else:
            # Set the width and height dimensions to the size property
            new_w, new_h = self.size
        
        # print("Scale function")
        # print(new_w, new_h)
        
        # Resize image to the new dimensions
        resized_img = resize_image(img, (new_w, new_h), interpolation=interpolation)
        return resized_img

    def __repr__(self):
        if isinstance(self.size, int):
            target_size = (self.size, self.size)
        else:
            target_size = self.size
        return "Scale [Exact Size ({:d}, {:d})]".format(target_size[0], target_size[1])


class RandomSizedCrop(object):
    """Crop the given numpy array to random area and aspect ratio.

    This class will crop a random region with in an input image. The target area
    / aspect ratio (width/height) of the region is first sampled from a uniform
    distribution. A region satisfying the area / aspect ratio is then sampled
    and cropped. This crop is finally resized to a fixed given size. This is
    widely used as data augmentation for training image classification models.

    Args:
        size (sequence or int): size of target image. If size is a sequence like
            (w, h), output size will be matched to this. If size is an int,
            output size will be (size, size).
        interpolations (list of int, optional): Desired interpolation.
            Default is ``CV2.INTER_NEAREST|CV2.INTER_LINEAR|CV2.INTER_CUBIC``
        area_range (list of int): range of areas to sample from
        ratio_range (list of int): range of aspect ratios to sample from
        num_trials (int): number of sampling trials
    """

    def __init__(
        self,
        size,
        interpolations=_DEFAULT_INTERPOLATIONS,
        area_range=(0.25, 1.0),
        ratio_range=(0.8, 1.2),
        num_trials=10,
    ):
        self.size = size
        if interpolations is None:
            interpolations = [cv2.INTER_LINEAR]
        assert isinstance(interpolations, Iterable)
        self.interpolations = interpolations
        self.num_trials = int(num_trials)
        self.area_range = area_range
        self.ratio_range = ratio_range

    def __call__(self, img):
        # sample interpolation method
        interpolation = random.sample(self.interpolations, 1)[0]

        for attempt in range(self.num_trials):

            # sample target area / aspect ratio from area range and ratio range
            area = img.shape[0] * img.shape[1]
            target_area = random.uniform(self.area_range[0], self.area_range[1]) * area
            aspect_ratio = random.uniform(self.ratio_range[0], self.ratio_range[1])

            # print(area, "Area")
            # print(target_area, "Target Area")
            # print(aspect_ratio, "Aspect Ratio")

            # Calculate width and height of cropping
            w = img.shape[1]
            h = img.shape[0]
            cropped_w = round(math.sqrt(target_area * aspect_ratio))
            cropped_h = round(math.sqrt(target_area / aspect_ratio))

            # print(w, "Original Width")
            # print(h, "Original Height")
            # print(cropped_w, "Cropped Width")
            # print(cropped_h, "Cropped Height")

            # Check if dimensions of the cropping fit within the original image dimensions
            if cropped_w <= w and cropped_h <= h:
                # Randomly select top left corner for the cropped image
                start_x = random.randint(0, w - cropped_w)
                start_y = random.randint(0, h - cropped_h)

                # print(start_x, "Start X")
                # print(start_y, "Start Y")
                
                # Crop the image
                img = img[start_y:start_y + cropped_h, start_x:start_x + cropped_w, :]

                # Scale to (size, size) if size is an int
                if isinstance(self.size, int):
                    im_scale = Scale((self.size, self.size), interpolations=self.interpolations)
                # Scale to provided size dimensions
                else:
                    im_scale = Scale(self.size, interpolations=self.interpolations)
                
                cropped_img = im_scale(img)
                return cropped_img
            # Move onto the next trial
            else:
                continue

        # Fall back
        if isinstance(self.size, int):
            im_scale = Scale((self.size, self.size), interpolations=self.interpolations)
        else:
            im_scale = Scale(self.size, interpolations=self.interpolations)
        cropped_img = im_scale(img)
        return cropped_img

    def __repr__(self):
        if isinstance(self.size, int):
            target_size = (self.size, self.size)
        else:
            target_size = self.size
        return (
            "Random Crop"
            + "[Size ({:d}, {:d}); Area {:.2f} - {:.2f}; Ratio {:.2f} - {:.2f}]".format(
                target_size[0],
                target_size[1],
                self.area_range[0],
                self.area_range[1],
                self.ratio_range[0],
                self.ratio_range[1],
            )
        )


class RandomColor(object):
    """Perturb color channels of a given image.

    This class will apply random color perturbation to an input image. An alpha
    value is first sampled uniformly from the range of (-r, r). 1 + alpha is
    further multiply to a color channel. The sampling is done independently for
    each channel. An efficient implementation can be achieved using a LuT.

    Args:
        color_range (float): range of color jitter ratio (-r ~ +r) max r = 1.0
    """

    def __init__(self, color_range):
        self.color_range = color_range

    def __call__(self, img):
        # Generate alphas for each color channel
        alpha_r = random.uniform(-self.color_range, self.color_range)
        alpha_g = random.uniform(-self.color_range, self.color_range)
        alpha_b = random.uniform(-self.color_range, self.color_range)
        # Convert image to a float to store floating values
        pert_img = np.array(img).astype(np.float32)

        # Apply perturbations to the colors
        pert_img[:,:,0] = img[:,:,0]*(1.+alpha_r)
        pert_img[:,:,1] = img[:,:,1]*(1.+alpha_g)
        pert_img[:,:,2] = img[:,:,2]*(1.+alpha_b)

        # Clip values of pixels to be within valid range
        pert_img = np.clip(pert_img, 0, 255).astype(np.uint8)

        return pert_img

    def __repr__(self):
        return "Random Color [Range {:.2f} - {:.2f}]".format(
            1 - self.color_range, 1 + self.color_range
        )


class RandomRotate(object):
    """Rotate the given numpy array (around the image center) by a random degree.

    This class will randomly rotate an image and further crop a local region with
    maximum area. A rotation angle is first sampled and then applied to the input.
    A region with maximum area and without any empty pixel is further determined
    and cropped.

    Args:
        degree_range (float): range of degree (-d ~ +d)
    """

    def __init__(self, degree_range, interpolations=_DEFAULT_INTERPOLATIONS):
        self.degree_range = degree_range
        if interpolations is None:
            interpolations = [cv2.INTER_LINEAR]
        assert isinstance(interpolations, Iterable)
        self.interpolations = interpolations

    # Rotate the image without cutting off the corners of the image
    def rotate_image(self, img, angle_degrees):
        # Get original height and width of the image
        orig_h, orig_w = img.shape[:2]
        # Get co-ordinates of the center of the original image
        orig_center_x = orig_w // 2
        orig_center_y = orig_h // 2

        # Get the rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D((orig_center_x, orig_center_y), angle_degrees, 1.0)

        # Get the cosine and sine of the rotation angle from the rotation matrix
        cos_angle = np.abs(rotation_matrix[0, 0])
        sin_angle = np.abs(rotation_matrix[0, 1])

        # Compute the bounding width and height after rotation 
        bound_w = int((orig_h * sin_angle) + (orig_w * cos_angle))
        bound_h = int((orig_h * cos_angle) + (orig_w * sin_angle))

        # Adjust the rotation matrix to take into account translation
        rotation_matrix[0, 2] += bound_w/2 - orig_center_x
        rotation_matrix[1, 2] += bound_h/2 - orig_center_y

        # Rotate the image
        rotated_img = cv2.warpAffine(img, rotation_matrix, (bound_w, bound_h)) 
        
        return rotated_img
    
    # Find the width and height of the maximal area rectangle within the rotated image
    def find_max_area_rect(self, img, angle_radians):
        # Get original height and width of the image
        orig_h, orig_w = img.shape[:2]

        # Find the dimension of the longer and shorter side of the image
        if (orig_w > orig_h):
            longer_side = orig_w
            shorter_side = orig_h
        else:
            longer_side = orig_h
            shorter_side = orig_w

        # Compute the cosine and sine of the rotation angle
        sin_angle = abs(math.sin(angle_radians)) 
        cos_angle = abs(math.cos(angle_radians))

        # Case when two corners of the cropped image are parallel to the longer
        # side of the rotated image and equidistant from the two longer sides
        if shorter_side <= 2.*sin_angle*cos_angle*longer_side:
            x = shorter_side/2
            if(orig_w > orig_h):
                crop_w = x/sin_angle
                crop_h = x/cos_angle
            else:
                crop_w = x/cos_angle
                crop_h = x/sin_angle
        # Case when the cropped image touches all four sides of the rotated image
        else:
            cos_2_angle = cos_angle*cos_angle - sin_angle*sin_angle
            crop_w = (orig_w*cos_angle - orig_h*sin_angle)/cos_2_angle
            crop_h = (orig_h*cos_angle - orig_w*sin_angle)/cos_2_angle
        
        return crop_w, crop_h



    def __call__(self, img):
        # sample interpolation method
        interpolation = random.sample(self.interpolations, 1)[0]
        # sample rotation
        degree = random.uniform(-self.degree_range, self.degree_range)
        # ignore small rotations
        if np.abs(degree) <= 1.0:
            return img

        #################################################################################
        # Fill in the code here
        #################################################################################
        # Rotate the image without cutting off the original image
        rotated_img = self.rotate_image(img, degree)
        # Calculate the new bounding width and height of the image after rotation
        bound_h, bound_w = rotated_img.shape[:2]

        # Calculate the width and height of the maximal area rectangle within the rotated image
        crop_w, crop_h = self.find_max_area_rect(img, math.radians(degree))

        # Calculate the coordinates of the corners of the new cropped rectangle
        y1 = bound_h//2 - int(crop_h/2)
        y2 = y1 + int(crop_h)
        x1 = bound_w//2 - int(crop_w/2)
        x2 = x1 + int(crop_w)

        # Crop the rectangle
        return rotated_img[y1:y2, x1:x2]

    def __repr__(self):
        return "Random Rotation [Range {:.2f} - {:.2f} Degree]".format(
            -self.degree_range, self.degree_range
        )


#################################################################################
# Additional helper functions. No need to modify.
#################################################################################
class ToTensor(object):
    """Convert a ``numpy.ndarray`` image to tensor.
    Converts a numpy.ndarray (H x W x C) image in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """

    def __call__(self, img):
        assert isinstance(img, np.ndarray)
        # convert image to tensor
        assert (img.ndim > 1) and (img.ndim <= 3)
        if img.ndim == 2:
            img = img[:, :, None]
            tensor_img = torch.from_numpy(
                np.ascontiguousarray(img.transpose((2, 0, 1)))
            )
        if img.ndim == 3:
            tensor_img = torch.from_numpy(
                np.ascontiguousarray(img.transpose((2, 0, 1)))
            )
        # backward compatibility
        if isinstance(tensor_img, torch.ByteTensor):
            return tensor_img.float().div(255.0)
        else:
            return tensor_img


class SimpleDataset(data.Dataset):
    """
    A simple dataset using PyTorch dataloader
    """

    def __init__(self, root_folder, file_ext, transforms=None):
        # root folder, split
        self.root_folder = root_folder
        self.transforms = transforms
        self.file_ext = file_ext

        # load all labels
        file_list = glob.glob(os.path.join(root_folder, "*.{:s}".format(file_ext)))
        self.file_list = file_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        # load image and its label (from file name)
        filename = self.file_list[index]
        img = load_image(filename)
        label = os.path.basename(filename)
        label = label.rstrip(".{:s}".format(self.file_ext))
        # apply data augmentations
        if self.transforms is not None:
            img = self.transforms(img)
        return img, label
