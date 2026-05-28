"""
Differentiable trajectory reconstruction from direction+speed features.

Given first_pose and per-frame dir+spd features, reconstructs the full
3D pose sequence (translation + axis-angle rotation) for loss computation.

All operations are differentiable for backprop through AE/MAR training.
"""

import torch
import torch.nn.functional as F
import numpy as np


def dirspd_to_velocity(dirspd: torch.Tensor, log_speed: bool = True) -> tuple:
    """
    Convert direction+speed features to translation and rotation velocities.

    dirspd: (B, T, 8) = [trans_dir(3), rot_dir(3), log_trans_speed(1), log_rot_speed(1)]
    Returns: trans_vel (B, T, 3), rot_vel (B, T, 3)
    """
    trans_dir = dirspd[..., :3]     # (B, T, 3) unit direction
    rot_dir = dirspd[..., 3:6]      # (B, T, 3) unit direction
    ts = dirspd[..., 6:7]           # (B, T, 1) log trans speed
    rs = dirspd[..., 7:8]           # (B, T, 1) log rot speed

    if log_speed:
        # Tighten clamp to [-12, 2] → max speed = exp(2) = 7.4, prevents 300-frame integration explosion
        ts = torch.exp(ts.clamp(min=-12.0, max=2.0)).clamp(min=0)   # (B, T, 1)
        rs = torch.exp(rs.clamp(min=-12.0, max=2.0)).clamp(min=0)   # (B, T, 1)

    trans_vel = trans_dir * ts       # (B, T, 3)
    rot_vel = rot_dir * rs           # (B, T, 3)

    return trans_vel, rot_vel


def reconstruct_poses_from_dirspd(
    dirspd: torch.Tensor,
    first_pose: torch.Tensor = None,
    seq_lens: torch.Tensor = None,
) -> torch.Tensor:
    """
    Reconstruct 6D pose sequence (trans3D + rot3D) from direction+speed features.

    Args:
        dirspd: (B, T, 8) direction+speed features
        first_pose: (B, 8) initial pose in dir+spd format
                    [trans_dir(3), rot_dir(3), log_trans_speed(1), log_rot_speed(1)]
                    The first 3D = position direction, next 3D = rotation axis-angle direction
        seq_lens: (B,) actual sequence lengths (optional, for masking)

    Returns:
        poses: (B, T, 6) = [cumulative_trans(3), cumulative_rot(3)] per frame
    """
    B, T, _ = dirspd.shape
    device = dirspd.device

    # Convert dir+spd to velocities
    trans_vel, rot_vel = dirspd_to_velocity(dirspd)  # (B, T, 3) each

    # Cumulative sum to get positions and rotations
    # This is a simplified reconstruction — accumulates velocities linearly
    cum_trans = torch.cumsum(trans_vel, dim=1)  # (B, T, 3)
    cum_rot = torch.cumsum(rot_vel, dim=1)      # (B, T, 3)

    # Add initial pose offset if provided
    if first_pose is not None:
        # first_pose: [trans_dir(3), rot_dir(3), log_ts(1), log_rs(1)]
        fp_trans_dir = first_pose[:, :3]     # (B, 3)
        fp_rot_dir = first_pose[:, 3:6]      # (B, 3)
        fp_ts = first_pose[:, 6:7]           # (B, 1)
        fp_rs = first_pose[:, 7:8]           # (B, 1)

        # Initial position = direction * exp(log_speed)
        init_trans = fp_trans_dir * torch.exp(fp_ts).clamp(min=0)  # (B, 3)
        init_rot = fp_rot_dir * torch.exp(fp_rs).clamp(min=0)      # (B, 3)

        cum_trans = cum_trans + init_trans.unsqueeze(1)
        cum_rot = cum_rot + init_rot.unsqueeze(1)

    # Concatenate: (B, T, 6) = [trans(3), rot(3)]
    poses = torch.cat([cum_trans, cum_rot], dim=-1)

    return poses


def compute_pose_loss(
    pred_dirspd: torch.Tensor,
    gt_dirspd: torch.Tensor,
    first_pose: torch.Tensor = None,
    mask: torch.Tensor = None,
) -> torch.Tensor:
    """
    Compute MSE loss in reconstructed pose space.

    Args:
        pred_dirspd: (B, T, 8) predicted direction+speed
        gt_dirspd: (B, T, 8) ground truth direction+speed
        first_pose: (B, 8) initial pose (same for both pred and gt)
        mask: (B, T) bool mask for valid frames

    Returns:
        pose_loss: scalar
    """
    pred_poses = reconstruct_poses_from_dirspd(pred_dirspd, first_pose)
    gt_poses = reconstruct_poses_from_dirspd(gt_dirspd, first_pose)

    diff = (pred_poses - gt_poses) ** 2  # (B, T, 6)

    if mask is not None:
        diff = diff * mask.unsqueeze(-1).float()
        return diff.sum() / (mask.sum() * 6 + 1e-8)
    else:
        return diff.mean()


def compute_traj_pose_loss(
    pred_traj: torch.Tensor,
    gt_traj: torch.Tensor,
    mask: torch.Tensor = None,
) -> torch.Tensor:
    """
    Compute MSE loss for trajectory type (9D: rot6D + rel_trans).
    Already in pose space — just compute MSE directly.
    """
    diff = (pred_traj - gt_traj) ** 2
    if mask is not None:
        diff = diff * mask.unsqueeze(-1).float()
        return diff.sum() / (mask.sum() * diff.shape[-1] + 1e-8)
    else:
        return diff.mean()
