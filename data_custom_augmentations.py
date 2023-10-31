import albumentations as A
import cv2
import numpy as np

from utils import GRID_SIZE


class SafeHorizontalFlip(A.HorizontalFlip):
    """
    Horizontal Flip that changes the order of the keypoints so that the top left one remains in the top left position.
    """

    def __init__(self, gridsize=GRID_SIZE, always_apply: bool = False, p: float = 0.5):
        super().__init__(always_apply, p)
        self.gridsize = gridsize

    def apply_to_keypoints(self, keypoints, **params):
        keypoints = super().apply_to_keypoints(keypoints, **params)

        keypoints = np.array(keypoints).reshape(*self.gridsize, -1)[:, ::-1, :]
        keypoints = keypoints.reshape(np.product(self.gridsize), -1)
        return keypoints

    def get_transform_init_args_names(self):
        return ("gridsize",)


class SafePerspective(A.Perspective):
    """
    Perspective augmentation that keeps all keypoints in the image visible.
    Mostly copied from the original Perspective augmentation from Albumentation.
    """

    def __init__(
        self,
        scale=(0.05, 0.1),
        keep_size=True,
        pad_mode=cv2.BORDER_CONSTANT,
        pad_val=0,
        mask_pad_val=0,
        fit_output=False,
        interpolation=cv2.INTER_LINEAR,
        always_apply=False,
        p=0.5,
    ):
        super().__init__(
            scale,
            keep_size,
            pad_mode,
            pad_val,
            mask_pad_val,
            fit_output,
            interpolation,
            always_apply,
            p,
        )

    @property
    def targets_as_params(self):
        return ["image", "keypoints"]

    def get_params_dependent_on_targets(self, params):
        h, w = params["image"].shape[:2]
        keypoints = np.array(params["keypoints"])[:, :2] / np.array([w, h])
        left = np.min(keypoints[:, 0])
        right = np.max(keypoints[:, 0])
        top = np.min(keypoints[:, 1])
        bottom = np.max(keypoints[:, 1])

        points = np.zeros([4, 2])
        # Top Left point
        points[0, 0] = A.random_utils.uniform(0, max(left - 0.01, left / 2))
        points[0, 1] = A.random_utils.uniform(0, max(top - 0.01, top / 2))
        # Top right point
        points[1, 0] = A.random_utils.uniform(min(right + 0.01, (right + 1) / 2), 1)
        points[1, 1] = A.random_utils.uniform(0, max(top - 0.01, top / 2))
        # Bottom Right point
        points[2, 0] = A.random_utils.uniform(min(right + 0.01, (right + 1) / 2), 1)
        points[2, 1] = A.random_utils.uniform(min(bottom + 0.01, (bottom + 1) / 2), 1)
        # Bottom Left point
        points[3, 0] = A.random_utils.uniform(0, max(left - 0.01, left / 2))
        points[3, 1] = A.random_utils.uniform(min(bottom + 0.01, (bottom + 1) / 2), 1)

        points[:, 0] *= w
        points[:, 1] *= h

        # Obtain a consistent order of the points and unpack them individually.
        # Warning: don't just do (tl, tr, br, bl) = _order_points(...)
        # here, because the reordered points is used further below.
        points = self._order_points(points)
        tl, tr, br, bl = points

        # compute the width of the new image, which will be the
        # maximum distance between bottom-right and bottom-left
        # x-coordiates or the top-right and top-left x-coordinates
        min_width = None
        max_width = None
        while min_width is None or min_width < 2:
            width_top = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
            width_bottom = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
            max_width = int(max(width_top, width_bottom))
            min_width = int(min(width_top, width_bottom))
            if min_width < 2:
                step_size = (2 - min_width) / 2
                tl[0] -= step_size
                tr[0] += step_size
                bl[0] -= step_size
                br[0] += step_size

        # compute the height of the new image, which will be the maximum distance between the top-right
        # and bottom-right y-coordinates or the top-left and bottom-left y-coordinates
        min_height = None
        max_height = None
        while min_height is None or min_height < 2:
            height_right = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
            height_left = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
            max_height = int(max(height_right, height_left))
            min_height = int(min(height_right, height_left))
            if min_height < 2:
                step_size = (2 - min_height) / 2
                tl[1] -= step_size
                tr[1] -= step_size
                bl[1] += step_size
                br[1] += step_size

        # now that we have the dimensions of the new image, construct
        # the set of destination points to obtain a "birds eye view",
        # (i.e. top-down view) of the image, again specifying points
        # in the top-left, top-right, bottom-right, and bottom-left order
        # do not use width-1 or height-1 here, as for e.g. width=3, height=2
        # the bottom right coordinate is at (3.0, 2.0) and not (2.0, 1.0)
        dst = np.array(
            [[0, 0], [max_width, 0], [max_width, max_height], [0, max_height]],
            dtype=np.float32,
        )

        # compute the perspective transform matrix and then apply it
        m = cv2.getPerspectiveTransform(points, dst)

        if self.fit_output:
            m, max_width, max_height = self._expand_transform(m, (h, w))

        return {
            "matrix": m,
            "max_height": max_height,
            "max_width": max_width,
            "interpolation": self.interpolation,
        }
