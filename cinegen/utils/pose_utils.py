"""
Pose and trajectory math utilities.

Convention: 4x4 matrices are camera-to-world (c2w):
    [R | t]    R = camera axes in world frame
    [0 | 1]    t = camera position in world frame

Velocity decomposition:
    trans_vel[i] = t[i+1] - t[i]                     shape (3,)
    rot_vel[i]   = axis-angle(R[i]^T @ R[i+1])       shape (3,)

Both are split into:
    direction = vel / ||vel||    (unit vector, zero if ||vel|| < eps)
    speed     = ||vel||          (scalar)
"""

import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Rotation conversions
# ---------------------------------------------------------------------------

def rotation_matrix_to_quaternion(R: torch.Tensor) -> torch.Tensor:
    """
    Vectorized rotation matrix -> quaternion [w, x, y, z].
    Args:  R: (..., 3, 3)
    Returns: (..., 4)
    """
    batch_shape = R.shape[:-2]
    R = R.reshape(-1, 3, 3)
    n = R.shape[0]

    trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]

    q = torch.zeros(n, 4, dtype=R.dtype, device=R.device)

    # Case 1: trace > 0
    m1 = trace > 0
    s = torch.sqrt((trace[m1] + 1.0).clamp(min=1e-10)) * 2  # 4w
    q[m1, 0] = 0.25 * s
    q[m1, 1] = (R[m1, 2, 1] - R[m1, 1, 2]) / s
    q[m1, 2] = (R[m1, 0, 2] - R[m1, 2, 0]) / s
    q[m1, 3] = (R[m1, 1, 0] - R[m1, 0, 1]) / s

    # Case 2: R[0,0] largest diagonal
    m2 = (~m1) & (R[:, 0, 0] > R[:, 1, 1]) & (R[:, 0, 0] > R[:, 2, 2])
    s = torch.sqrt((1.0 + R[m2, 0, 0] - R[m2, 1, 1] - R[m2, 2, 2]).clamp(min=1e-10)) * 2
    q[m2, 0] = (R[m2, 2, 1] - R[m2, 1, 2]) / s
    q[m2, 1] = 0.25 * s
    q[m2, 2] = (R[m2, 0, 1] + R[m2, 1, 0]) / s
    q[m2, 3] = (R[m2, 0, 2] + R[m2, 2, 0]) / s

    # Case 3: R[1,1] largest diagonal
    m3 = (~m1) & (~m2) & (R[:, 1, 1] > R[:, 2, 2])
    s = torch.sqrt((1.0 + R[m3, 1, 1] - R[m3, 0, 0] - R[m3, 2, 2]).clamp(min=1e-10)) * 2
    q[m3, 0] = (R[m3, 0, 2] - R[m3, 2, 0]) / s
    q[m3, 1] = (R[m3, 0, 1] + R[m3, 1, 0]) / s
    q[m3, 2] = 0.25 * s
    q[m3, 3] = (R[m3, 1, 2] + R[m3, 2, 1]) / s

    # Case 4: R[2,2] largest diagonal
    m4 = (~m1) & (~m2) & (~m3)
    s = torch.sqrt((1.0 + R[m4, 2, 2] - R[m4, 0, 0] - R[m4, 1, 1]).clamp(min=1e-10)) * 2
    q[m4, 0] = (R[m4, 1, 0] - R[m4, 0, 1]) / s
    q[m4, 1] = (R[m4, 0, 2] + R[m4, 2, 0]) / s
    q[m4, 2] = (R[m4, 1, 2] + R[m4, 2, 1]) / s
    q[m4, 3] = 0.25 * s

    # Normalize
    q = F.normalize(q, dim=-1)
    return q.reshape(*batch_shape, 4)


def quaternion_to_rotation_matrix(q: torch.Tensor) -> torch.Tensor:
    """
    Quaternion [w, x, y, z] -> rotation matrix.
    Args:  q: (..., 4)
    Returns: (..., 3, 3)
    """
    q = F.normalize(q, dim=-1)
    batch_shape = q.shape[:-1]
    q = q.reshape(-1, 4)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

    R = torch.stack([
        1 - 2*y*y - 2*z*z,  2*x*y - 2*w*z,      2*x*z + 2*w*y,
        2*x*y + 2*w*z,      1 - 2*x*x - 2*z*z,  2*y*z - 2*w*x,
        2*x*z - 2*w*y,      2*y*z + 2*w*x,      1 - 2*x*x - 2*y*y,
    ], dim=-1).reshape(-1, 3, 3)

    return R.reshape(*batch_shape, 3, 3)


def rotation_matrix_to_axis_angle(R: torch.Tensor) -> torch.Tensor:
    """
    Rotation matrix -> axis-angle vector (direction=axis, magnitude=angle).
    Args:  R: (..., 3, 3)
    Returns: (..., 3)
    """
    batch_shape = R.shape[:-2]
    R = R.reshape(-1, 3, 3)

    trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    cos_theta = ((trace - 1.0) / 2.0).clamp(-1.0, 1.0)
    theta = torch.acos(cos_theta)  # (N,)

    # Axis from skew-symmetric part of R
    axis = torch.stack([
        R[:, 2, 1] - R[:, 1, 2],
        R[:, 0, 2] - R[:, 2, 0],
        R[:, 1, 0] - R[:, 0, 1],
    ], dim=-1)  # (N, 3), unnormalized = 2*sin(theta)*axis

    sin_theta = torch.sin(theta).clamp(min=1e-7)
    axis = axis / (2.0 * sin_theta.unsqueeze(-1))

    # For near-zero rotations, axis is undefined → zero vector
    near_zero = theta < 1e-6
    axis = torch.where(near_zero.unsqueeze(-1), torch.zeros_like(axis), axis)

    result = axis * theta.unsqueeze(-1)
    return result.reshape(*batch_shape, 3)


def axis_angle_to_rotation_matrix(aa: torch.Tensor) -> torch.Tensor:
    """
    Axis-angle -> rotation matrix (Rodrigues).
    Args:  aa: (..., 3)
    Returns: (..., 3, 3)
    """
    batch_shape = aa.shape[:-1]
    aa = aa.reshape(-1, 3)

    theta = torch.norm(aa, dim=-1, keepdim=True).clamp(min=1e-7)  # (N, 1)
    axis = aa / theta  # unit axis
    theta = theta.squeeze(-1)  # (N,)

    x, y, z = axis[:, 0], axis[:, 1], axis[:, 2]
    c = torch.cos(theta)
    s = torch.sin(theta)
    t = 1 - c

    R = torch.stack([
        t*x*x + c,   t*x*y - s*z, t*x*z + s*y,
        t*x*y + s*z, t*y*y + c,   t*y*z - s*x,
        t*x*z - s*y, t*y*z + s*x, t*z*z + c,
    ], dim=-1).reshape(-1, 3, 3)

    # Near-zero: identity
    near_zero = torch.norm(aa, dim=-1) < 1e-6
    eye = torch.eye(3, dtype=aa.dtype, device=aa.device).unsqueeze(0).expand(aa.shape[0], -1, -1)
    R = torch.where(near_zero.unsqueeze(-1).unsqueeze(-1), eye, R)

    return R.reshape(*batch_shape, 3, 3)


# ---------------------------------------------------------------------------
# Trajectory processing
# ---------------------------------------------------------------------------

def matrices_to_trans_rot(matrices: torch.Tensor):
    """
    Extract translation and rotation from c2w matrices.
    Args:  matrices: (N, 4, 4)
    Returns:
        translations: (N, 3)
        rotations:    (N, 3, 3)
    """
    return matrices[:, :3, 3], matrices[:, :3, :3]


def compute_velocity(translations: torch.Tensor, rotations: torch.Tensor):
    """
    Compute per-frame velocities from pose sequence.
    Args:
        translations: (N, 3)
        rotations:    (N, 3, 3)
    Returns:
        trans_vel: (N-1, 3)   world-space translation delta
        rot_vel:   (N-1, 3)   axis-angle of relative rotation (in frame i coords)
    """
    trans_vel = translations[1:] - translations[:-1]  # (N-1, 3)

    R_i    = rotations[:-1]  # (N-1, 3, 3)
    R_next = rotations[1:]   # (N-1, 3, 3)
    R_rel  = torch.bmm(R_i.transpose(1, 2), R_next)   # relative in camera frame
    rot_vel = rotation_matrix_to_axis_angle(R_rel)     # (N-1, 3)

    return trans_vel, rot_vel


def decompose_velocity(trans_vel: torch.Tensor, rot_vel: torch.Tensor, eps: float = 1e-6):
    """
    Split velocity into direction (unit vector) and speed (magnitude).
    Returns:
        trans_dir:   (N-1, 3)  unit translation direction
        rot_dir:     (N-1, 3)  unit rotation axis
        trans_speed: (N-1,)    translation magnitude
        rot_speed:   (N-1,)    rotation angle magnitude
    """
    trans_speed = torch.norm(trans_vel, dim=-1)          # (N-1,)
    rot_speed   = torch.norm(rot_vel,   dim=-1)          # (N-1,)

    trans_dir = trans_vel / trans_speed.unsqueeze(-1).clamp(min=eps)
    rot_dir   = rot_vel   / rot_speed.unsqueeze(-1).clamp(min=eps)

    # Zero direction where speed is negligible (undefined direction)
    trans_dir = torch.where((trans_speed < eps).unsqueeze(-1), torch.zeros_like(trans_dir), trans_dir)
    rot_dir   = torch.where((rot_speed   < eps).unsqueeze(-1), torch.zeros_like(rot_dir),   rot_dir)

    return trans_dir, rot_dir, trans_speed, rot_speed


def recompose_velocity(trans_dir: torch.Tensor, rot_dir: torch.Tensor,
                       trans_speed: torch.Tensor, rot_speed: torch.Tensor):
    """Reconstruct velocity from direction and speed."""
    return trans_dir * trans_speed.unsqueeze(-1), rot_dir * rot_speed.unsqueeze(-1)


def reconstruct_trajectory(first_matrix: torch.Tensor,
                            trans_vel: torch.Tensor,
                            rot_vel: torch.Tensor) -> torch.Tensor:
    """
    Reconstruct full c2w trajectory from first frame + velocities.
    Args:
        first_matrix: (4, 4)
        trans_vel:    (N-1, 3)
        rot_vel:      (N-1, 3)  axis-angle
    Returns:
        matrices: (N, 4, 4)
    """
    device = first_matrix.device
    dtype  = first_matrix.dtype
    N = trans_vel.shape[0] + 1

    t = first_matrix[:3, 3].clone()
    R = first_matrix[:3, :3].clone()
    matrices = [first_matrix]

    R_deltas = axis_angle_to_rotation_matrix(rot_vel)   # (N-1, 3, 3)

    for i in range(N - 1):
        t = t + trans_vel[i]
        R = R @ R_deltas[i]
        mat = torch.eye(4, dtype=dtype, device=device)
        mat[:3, :3] = R
        mat[:3, 3]  = t
        matrices.append(mat)

    return torch.stack(matrices, dim=0)  # (N, 4, 4)


def encode_first_pose(matrix: torch.Tensor) -> torch.Tensor:
    """
    Encode a 4x4 c2w matrix as a 7-dim vector [tx, ty, tz, qw, qx, qy, qz].
    Args:  matrix: (4, 4)
    Returns: (7,)
    """
    t = matrix[:3, 3]
    R = matrix[:3, :3]
    q = rotation_matrix_to_quaternion(R.unsqueeze(0)).squeeze(0)  # (4,)
    return torch.cat([t, q], dim=0)


# ---------------------------------------------------------------------------
# Numpy helpers (used in dataset loading)
# ---------------------------------------------------------------------------

def np_matrices_to_velocity(matrices: np.ndarray):
    """
    Convert (N, 4, 4) numpy array to direction and speed numpy arrays.
    Returns:
        trans_dir:   (N-1, 3)
        rot_dir:     (N-1, 3)
        trans_speed: (N-1,)
        rot_speed:   (N-1,)
        first_pose:  (7,)    [tx, ty, tz, qw, qx, qy, qz]
    """
    m = torch.from_numpy(matrices.astype(np.float32))
    trans, rots = matrices_to_trans_rot(m)
    tv, rv = compute_velocity(trans, rots)
    td, rd, ts, rs = decompose_velocity(tv, rv)
    fp = encode_first_pose(m[0])
    return td.numpy(), rd.numpy(), ts.numpy(), rs.numpy(), fp.numpy()
