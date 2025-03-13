# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from os.path import join
import numpy as np
import torch
from scipy.spatial import ConvexHull, Delaunay
from scipy.spatial.qhull import QhullError

from nerfactor.third_party.xiuminglib import xiuminglib as xm
# from third_party.xiuminglib import xiuminglib as xm
from . import math as mathutil  # make sure mathutil uses torch-based operations


# -------------------------------------------------------------------
# I/O helper functions
# -------------------------------------------------------------------

def write_lvis(lvis, fps, out_dir):
    """
    Saves the light visibility array (lvis) in various formats:
      - Raw numpy array (lvis.npy)
      - An averaged image (lvis.png)
      - A video visualizing each light pixel (lvis.mp4)
    """
    xm.os.makedirs(out_dir)
    # Dump raw
    raw_out = join(out_dir, 'lvis.npy')
    with open(raw_out, 'wb') as h:
        np.save(h, lvis)
    # Visualize the average across all lights as an image
    vis_out = join(out_dir, 'lvis.png')
    lvis_avg = np.mean(lvis, axis=2)
    xm.io.img.write_arr(lvis_avg, vis_out)
    # Visualize light visibility for each light pixel as a video
    vis_out = join(out_dir, 'lvis.mp4')
    frames = []
    for i in range(lvis.shape[2]):  # for each light pixel
        frame = xm.img.denormalize_float(lvis[:, :, i])
        frame = np.dstack([frame] * 3)
        frames.append(frame)
    xm.vis.video.make_video(frames, outpath=vis_out, fps=fps)


def write_xyz(xyz_arr, out_dir):
    """
    Saves the xyz array to disk and creates a visualization.
    """
    arr = xyz_arr
    if isinstance(arr, torch.Tensor):
        arr = arr.cpu().numpy()
    xm.os.makedirs(out_dir)
    # Dump raw
    raw_out = join(out_dir, 'xyz.npy')
    with open(raw_out, 'wb') as h:
        np.save(h, arr)
    # Visualization (normalize to [0,1])
    vis_out = join(out_dir, 'xyz.png')
    arr_norm = (arr - arr.min()) / (arr.max() - arr.min())
    xm.io.img.write_arr(arr_norm, vis_out, clip=True)


def write_normal(arr, out_dir):
    """
    Saves and visualizes normal vectors.
    """
    if isinstance(arr, torch.Tensor):
        arr = arr.cpu().numpy()
    # Dump raw
    raw_out = join(out_dir, 'normal.npy')
    with open(raw_out, 'wb') as h:
        np.save(h, arr)
    # Visualization: convert normals from [-1, 1] to [0, 1]
    vis_out = join(out_dir, 'normal.png')
    arr = (arr + 1) / 2
    xm.io.img.write_arr(arr, vis_out)


def write_alpha(arr, out_dir):
    """
    Saves an alpha (opacity) image.
    """
    if isinstance(arr, torch.Tensor):
        arr = arr.cpu().numpy()
    vis_out = join(out_dir, 'alpha.png')
    xm.io.img.write_arr(arr, vis_out)


# -------------------------------------------------------------------
# Geometry helper functions
# -------------------------------------------------------------------

def get_convex_hull(pts):
    """
    Computes the convex hull of a set of points.
    If pts is a torch tensor, it is first converted to a numpy array.
    """
    if isinstance(pts, torch.Tensor):
        pts = pts.cpu().numpy()
    try:
        hull = ConvexHull(pts)
    except QhullError:
        hull = None
    return hull


def in_hull(hull, pts):
    """
    Tests if points lie inside the convex hull.
    """
    verts = hull.points[hull.vertices, :]
    # Create a Delaunay triangulation on the hull vertices.
    hull_delaunay = Delaunay(verts)
    return hull_delaunay.find_simplex(pts) >= 0


def rad2deg(rad):
    return 180 / np.pi * rad


# -------------------------------------------------------------------
# Spherical linear interpolation (slerp)
# -------------------------------------------------------------------

def slerp(p0, p1, t):
    """
    Performs spherical linear interpolation between two 2D vectors.
    Vectors are expected to have shape (1, d) or (d, 1).
    """
    assert p0.dim() == p1.dim() == 2, "Vectors must be 2D"

    if p0.shape[0] == 1:
        cos_omega = p0 @ p1.t()
    elif p0.shape[1] == 1:
        cos_omega = p0.t() @ p1
    else:
        raise ValueError("Vectors should have one singleton dimension")

    omega = mathutil.safe_acos(cos_omega)  # assumes mathutil.safe_acos works on torch tensors

    z0 = p0 * torch.sin((1 - t) * omega) / torch.sin(omega)
    z1 = p1 * torch.sin(t * omega) / torch.sin(omega)

    return z0 + z1


# -------------------------------------------------------------------
# Local coordinate system generation and coordinate conversion
# -------------------------------------------------------------------

def gen_world2local(normal, eps=1e-6):
    """
    Generates rotation matrices that transform world normals to local +Z.
    (world tangents map to local +X, and world binormals to local +Y)

    Input:
      normal: Tensor of shape (N, 3)
    Output:
      rot: Tensor of shape (N, 3, 3), where each 3x3 matrix has rows
           [tangent, binormal, normal].
    """
    # Normalize the input normals safely along dim 1
    normal = mathutil.safe_l2_normalize(normal, axis=1)

    # To avoid colinearity with some special normals, add a small epsilon to z.
    z = torch.tensor([0, 0, 1], dtype=torch.float32, device=normal.device) + eps
    z = z.unsqueeze(0).repeat(normal.shape[0], 1)

    # Compute tangents as cross product between the normal and the (perturbed) z axis.
    t = torch.cross(normal, z, dim=1)
    # Assert that none of the tangents are zero-norm.
    assert torch.all(torch.norm(t, dim=1) > 0), (
        "Found zero-norm tangents, either because of colinearity or zero-norm normals")
    t = mathutil.safe_l2_normalize(t, axis=1)

    # Compute binormals as cross product between the normal and the tangent.
    b = torch.cross(normal, t, dim=1)
    b = mathutil.safe_l2_normalize(b, axis=1)

    # Stack tangent, binormal, and normal to form the rotation matrix for each sample.
    rot = torch.stack([t, b, normal], dim=1)
    return rot


def dir2rusink(a, b):
    """
    Converts two directions (a and b, both of shape (N, 3)) into Rusink coordinates.
    Adapted from nielsen2015on/coordinateFunctions.py->DirectionsToRusink().
    """
    a = mathutil.safe_l2_normalize(a, axis=1)
    b = mathutil.safe_l2_normalize(b, axis=1)
    h = mathutil.safe_l2_normalize((a + b) / 2, axis=1)

    theta_h = mathutil.safe_acos(h[:, 2])
    phi_h = mathutil.safe_atan2(h[:, 1], h[:, 0])

    binormal = torch.tensor([0, 1, 0], dtype=torch.float32, device=a.device)
    normal = torch.tensor([0, 0, 1], dtype=torch.float32, device=a.device)

    def rot_vec(vector, axis, angle):
        """
        Rotates `vector` around an arbitrary `axis` by `angle` radians using Rodrigues’ formula.
        Both vector and axis are expected to be torch tensors.
        """
        # Ensure that angle has shape (N,) and vector has shape (N, 3)
        cos_ang = torch.cos(angle).reshape(-1)  # (N,)
        sin_ang = torch.sin(angle).reshape(-1)  # (N,)
        vector = vector.reshape(-1, 3)
        # If the provided axis is not batched, expand it.
        if axis.dim() == 1:
            axis = axis.unsqueeze(0).repeat(vector.shape[0], 1)
        else:
            axis = axis.reshape(-1, 3)
        # Compute the dot product between each vector and the corresponding axis.
        dot = (vector * axis).sum(dim=1, keepdim=True)
        # Rodrigues' rotation formula:
        rotated = (vector * cos_ang.unsqueeze(1) +
                   axis * dot * (1 - cos_ang).unsqueeze(1) +
                   torch.cross(axis, vector, dim=1) * sin_ang.unsqueeze(1))
        return rotated

    # Apply two successive rotations to obtain the difference vector in the Rusink frame.
    diff = rot_vec(rot_vec(b, normal, -phi_h), binormal, -theta_h)
    diff0, diff1, diff2 = diff[:, 0], diff[:, 1], diff[:, 2]
    # When a and b are the same, diff lies along +h (theta_d = 0), so use safe_atan2 to avoid NaNs.
    theta_d = mathutil.safe_acos(diff2)
    phi_d = torch.remainder(mathutil.safe_atan2(diff1, diff0), np.pi)
    # Stack the resulting coordinates into shape (N, 3)
    rusink = torch.stack((phi_d, theta_h, theta_d), dim=1)

    return rusink