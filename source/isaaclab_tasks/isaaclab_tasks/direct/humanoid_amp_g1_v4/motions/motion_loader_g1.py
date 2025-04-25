# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import os
import torch
from typing import Optional


class MotionLoaderG1:
    """
    Helper class to load and sample motion data from NumPy-file format.
    """

    def __init__(self, motion_file: str, device: torch.device) -> None:
        """Load a motion file and initialize the internal variables.

        Args:
            motion_file: Motion file path to load.
            device: The device to which to load the data.

        Raises:
            AssertionError: If the specified motion file doesn't exist.
        """
        assert os.path.isfile(motion_file), f"Invalid file path: {motion_file}"
        # data = np.load(motion_file)
        data = torch.load(motion_file)
        # keys: 'base_pose', 'base_position', 'base_velocity', 'base_angular_velocity', 'joint_position', 'joint_velocity', 'link_position', 'link_oritentation', 'lin_velocity', 'link_angular_velocity'
        # base_pose: (139, 4)
        # base_position: (139, 3)
        # base_velocity: (139, 3)
        # base angular velocity: (139, 3)
        # joint_postion: (139, 21)
        # joint velocity: (139, 21)
        # link position (139, 17,3)
        # link oritentation (139, 17, 4)
        # link_velocity (139, 17, 3)
        # link_angular_velocity (139, 17, 3)

        # import ipdb; ipdb.set_trace()
        self.device = device
        # self._dof_names = data["dof_names"].tolist()
        self._dof_names_file = open('/home/morning/Robust_Loco/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/humanoid_amp_g1_v4/motions/crawl_train/joint_id.txt', "r")
        self._dof_names = self._dof_names_file.readlines()
        self._dof_names = [name.strip().lstrip('0123456789 ') for name in self._dof_names] # 21

        # ['keyframe_pelvis_link', 'keyframe_left_hip_link', 'keyframe_left_knee_link', 'keyframe_left_ankle_link', 'keyframe_right_hip_link', 'keyframe_right_knee_link', 'keyframe_right_ankle_link', 'keyframe_head_link', 'keyframe_torso_link', 'keyframe_left_collar_link', 'keyframe_left_shoulder_link', 'keyframe_left_elbow_link', 'keyframe_left_wrist_link', 'keyframe_right_collar_link', 'keyframe_right_shoulder_link', 'keyframe_right_elbow_link', 'keyframe_right_wrist_link']
        # self._body_names = data["body_names"].tolist()
        self._body_names = ['pelvis_link', 'left_hip_link', 'left_knee_link', 'left_ankle_link', 'right_hip_link', 'right_knee_link', 'right_ankle_link', 'head_link', 'torso_link', 'left_collar_link', 'left_shoulder_link', 'left_elbow_link', 'left_wrist_link', 'right_collar_link', 'right_shoulder_link', 'right_elbow_link', 'right_wrist_link']
        # import ipdb; ipdb.set_trace()
        self.dof_positions = torch.tensor(data["joint_position"], dtype=torch.float32, device=self.device)
        self.dof_velocities = torch.tensor(data["joint_velocity"], dtype=torch.float32, device=self.device)
        self.body_positions = torch.tensor(data["link_position"], dtype=torch.float32, device=self.device) # (139, 17, 3)
        self.body_rotations = self.quaternion_xyzw_to_wxyz(torch.tensor(data["link_oritentation"], dtype=torch.float32, device=self.device)) # (139, 17, 4)
        # self.body_rotations_euler = self.quaternion_to_euler(self.body_rotations) # (139, 17, 3)
        self.body_linear_velocities = torch.tensor(
            data["lin_velocity"], dtype=torch.float32, device=self.device
        ) # (139, 17, 3)
        self.body_angular_velocities = torch.tensor(
            data["link_angular_velocity"], dtype=torch.float32, device=self.device # (139, 17, 3)
        )
        # import ipdb; ipdb.set_trace()

        # self.dt = 1.0 / data["fps"]
        self.dt = 1.0 / 30
        self.num_frames = self.dof_positions.shape[0]
        self.duration = self.dt * (self.num_frames - 1)
        print(f"Motion loaded ({motion_file}): duration: {self.duration} sec, frames: {self.num_frames}")

    @property
    def dof_names(self) -> list[str]:
        """Skeleton DOF names."""
        return self._dof_names

    @property
    def body_names(self) -> list[str]:
        """Skeleton rigid body names."""
        return self._body_names

    @property
    def num_dofs(self) -> int:
        """Number of skeleton's DOFs."""
        return len(self._dof_names)

    @property
    def num_bodies(self) -> int:
        """Number of skeleton's rigid bodies."""
        return len(self._body_names)

    def _interpolate(
        self,
        a: torch.Tensor,
        *,
        b: Optional[torch.Tensor] = None,
        blend: Optional[torch.Tensor] = None,
        start: Optional[np.ndarray] = None,
        end: Optional[np.ndarray] = None,
    ) -> torch.Tensor:
        """Linear interpolation between consecutive values.

        Args:
            a: The first value. Shape is (N, X) or (N, M, X).
            b: The second value. Shape is (N, X) or (N, M, X).
            blend: Interpolation coefficient between 0 (a) and 1 (b).
            start: Indexes to fetch the first value. If both, ``start`` and ``end` are specified,
                the first and second values will be fetches from the argument ``a`` (dimension 0).
            end: Indexes to fetch the second value. If both, ``start`` and ``end` are specified,
                the first and second values will be fetches from the argument ``a`` (dimension 0).

        Returns:
            Interpolated values. Shape is (N, X) or (N, M, X).
        """
        if start is not None and end is not None:
            return self._interpolate(a=a[start], b=a[end], blend=blend)
        if a.ndim >= 2:
            blend = blend.unsqueeze(-1)
        if a.ndim >= 3:
            blend = blend.unsqueeze(-1)
        return (1.0 - blend) * a + blend * b

    def _slerp(
        self,
        q0: torch.Tensor,
        *,
        q1: Optional[torch.Tensor] = None,
        blend: Optional[torch.Tensor] = None,
        start: Optional[np.ndarray] = None,
        end: Optional[np.ndarray] = None,
    ) -> torch.Tensor:
        """Interpolation between consecutive rotations (Spherical Linear Interpolation).

        Args:
            q0: The first quaternion (wxyz). Shape is (N, 4) or (N, M, 4).
            q1: The second quaternion (wxyz). Shape is (N, 4) or (N, M, 4).
            blend: Interpolation coefficient between 0 (q0) and 1 (q1).
            start: Indexes to fetch the first quaternion. If both, ``start`` and ``end` are specified,
                the first and second quaternions will be fetches from the argument ``q0`` (dimension 0).
            end: Indexes to fetch the second quaternion. If both, ``start`` and ``end` are specified,
                the first and second quaternions will be fetches from the argument ``q0`` (dimension 0).

        Returns:
            Interpolated quaternions. Shape is (N, 4) or (N, M, 4).
        """
        if start is not None and end is not None:
            return self._slerp(q0=q0[start], q1=q0[end], blend=blend)
        if q0.ndim >= 2:
            blend = blend.unsqueeze(-1)
        if q0.ndim >= 3:
            blend = blend.unsqueeze(-1)

        qw, qx, qy, qz = 0, 1, 2, 3  # wxyz
        cos_half_theta = (
            q0[..., qw] * q1[..., qw]
            + q0[..., qx] * q1[..., qx]
            + q0[..., qy] * q1[..., qy]
            + q0[..., qz] * q1[..., qz]
        )

        neg_mask = cos_half_theta < 0
        q1 = q1.clone()
        q1[neg_mask] = -q1[neg_mask]
        cos_half_theta = torch.abs(cos_half_theta)
        cos_half_theta = torch.unsqueeze(cos_half_theta, dim=-1)

        half_theta = torch.acos(cos_half_theta)
        sin_half_theta = torch.sqrt(1.0 - cos_half_theta * cos_half_theta)

        ratio_a = torch.sin((1 - blend) * half_theta) / sin_half_theta
        ratio_b = torch.sin(blend * half_theta) / sin_half_theta

        new_q_x = ratio_a * q0[..., qx : qx + 1] + ratio_b * q1[..., qx : qx + 1]
        new_q_y = ratio_a * q0[..., qy : qy + 1] + ratio_b * q1[..., qy : qy + 1]
        new_q_z = ratio_a * q0[..., qz : qz + 1] + ratio_b * q1[..., qz : qz + 1]
        new_q_w = ratio_a * q0[..., qw : qw + 1] + ratio_b * q1[..., qw : qw + 1]

        new_q = torch.cat([new_q_w, new_q_x, new_q_y, new_q_z], dim=len(new_q_w.shape) - 1)
        new_q = torch.where(torch.abs(sin_half_theta) < 0.001, 0.5 * q0 + 0.5 * q1, new_q)
        new_q = torch.where(torch.abs(cos_half_theta) >= 1, q0, new_q)
        return new_q

    def _compute_frame_blend(self, times: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute the indexes of the first and second values, as well as the blending time
        to interpolate between them and the given times.

        Args:
            times: Times, between 0 and motion duration, to sample motion values.
                Specified times will be clipped to fall within the range of the motion duration.

        Returns:
            First value indexes, Second value indexes, and blending time between 0 (first value) and 1 (second value).
        """
        phase = np.clip(times / self.duration, 0.0, 1.0)
        index_0 = (phase * (self.num_frames - 1)).round(decimals=0).astype(int)
        index_1 = np.minimum(index_0 + 1, self.num_frames - 1)
        blend = ((times - index_0 * self.dt) / self.dt).round(decimals=5)
        return index_0, index_1, blend

    def sample_times(self, num_samples: int, duration: float | None = None) -> np.ndarray:
        """Sample random motion times uniformly.

        Args:
            num_samples: Number of time samples to generate.
            duration: Maximum motion duration to sample.
                If not defined samples will be within the range of the motion duration.

        Raises:
            AssertionError: If the specified duration is longer than the motion duration.

        Returns:
            Time samples, between 0 and the specified/motion duration.
        """
        duration = self.duration if duration is None else duration
        assert (
            duration <= self.duration
        ), f"The specified duration ({duration}) is longer than the motion duration ({self.duration})"
        return duration * np.random.uniform(low=0.0, high=1.0, size=num_samples)

    def sample(
        self, num_samples: int, times: Optional[np.ndarray] = None, duration: float | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample motion data.

        Args:
            num_samples: Number of time samples to generate. If ``times`` is defined, this parameter is ignored.
            times: Motion time used for sampling.
                If not defined, motion data will be random sampled uniformly in time.
            duration: Maximum motion duration to sample.
                If not defined, samples will be within the range of the motion duration.
                If ``times`` is defined, this parameter is ignored.

        Returns:
            Sampled motion DOF positions (with shape (N, num_dofs)), DOF velocities (with shape (N, num_dofs)),
            body positions (with shape (N, num_bodies, 3)), body rotations (with shape (N, num_bodies, 4), as wxyz quaternion),
            body linear velocities (with shape (N, num_bodies, 3)) and body angular velocities (with shape (N, num_bodies, 3)).
        """
        times = self.sample_times(num_samples, duration) if times is None else times
        index_0, index_1, blend = self._compute_frame_blend(times)
        blend = torch.tensor(blend, dtype=torch.float32, device=self.device)

        return (
            self._interpolate(self.dof_positions, blend=blend, start=index_0, end=index_1),
            self._interpolate(self.dof_velocities, blend=blend, start=index_0, end=index_1),
            self._interpolate(self.body_positions, blend=blend, start=index_0, end=index_1),
            self._slerp(self.body_rotations, blend=blend, start=index_0, end=index_1),
            self._interpolate(self.body_linear_velocities, blend=blend, start=index_0, end=index_1),
            self._interpolate(self.body_angular_velocities, blend=blend, start=index_0, end=index_1),
        )

    def get_dof_index(self, dof_names: list[str]) -> list[int]:
        """Get skeleton DOFs indexes by DOFs names.

        Args:
            dof_names: List of DOFs names.

        Raises:
            AssertionError: If the specified DOFs name doesn't exist.

        Returns:
            List of DOFs indexes.
        """

        # import ipdb; ipdb.set_trace()
        indexes = []
        select_robot_indexes = []
        for i, name in enumerate(dof_names):
            if name not in self._dof_names:
                print(f"The specified DOF name ({name}) doesn't exist: {self._dof_names}")
            else:
                indexes.append(self._dof_names.index(name))
                select_robot_indexes.append(i)

        # import ipdb; ipdb.set_trace()
        return indexes, select_robot_indexes

    def get_body_index(self, body_names: list[str]) -> list[int]:
        """Get skeleton body indexes by body names.

        Args:
            dof_names: List of body names.

        Raises:
            AssertionError: If the specified body name doesn't exist.

        Returns:
            List of body indexes.
        """
        # import ipdb; ipdb.set_trace()
        indexes = []
        for name in body_names:
            assert name in self._body_names, f"The specified body name ({name}) doesn't exist: {self._body_names}"
            indexes.append(self._body_names.index(name))
        return indexes
    
    def quaternion_xyzw_to_wxyz(self, quaternion: torch.Tensor) -> torch.Tensor:
        """Convert quaternion from xyzw to wxyz format.
        """
        return torch.cat([quaternion[..., 3:4], quaternion[..., :3]], dim=-1)
    
    def qeuler(self, q, order, epsilon=0):
        """
        Convert quaternion(s) q to Euler angles.
        Expects a tensor of shape (*, 4), where * denotes any number of dimensions.
        Returns a tensor of shape (*, 3).
        """
        assert q.shape[-1] == 4
        
        original_shape = list(q.shape)
        original_shape[-1] = 3
        q = q.view(-1, 4)
        
        q0 = q[:, 0]
        q1 = q[:, 1]
        q2 = q[:, 2]
        q3 = q[:, 3]
        
        if order == 'xyz':
            x = torch.atan2(2 * (q0 * q1 - q2 * q3), 1 - 2*(q1 * q1 + q2 * q2))
            y = torch.asin(torch.clamp(2 * (q1 * q3 + q0 * q2), -1+epsilon, 1-epsilon))
            z = torch.atan2(2 * (q0 * q3 - q1 * q2), 1 - 2*(q2 * q2 + q3 * q3))
        elif order == 'yzx':
            x = torch.atan2(2 * (q0 * q1 - q2 * q3), 1 - 2*(q1 * q1 + q3 * q3))
            y = torch.atan2(2 * (q0 * q2 - q1 * q3), 1 - 2*(q2 * q2 + q3 * q3))
            z = torch.asin(torch.clamp(2 * (q1 * q2 + q0 * q3), -1+epsilon, 1-epsilon))
        elif order == 'zxy':
            x = torch.asin(torch.clamp(2 * (q0 * q1 + q2 * q3), -1+epsilon, 1-epsilon))
            y = torch.atan2(2 * (q0 * q2 - q1 * q3), 1 - 2*(q1 * q1 + q2 * q2))
            z = torch.atan2(2 * (q0 * q3 - q1 * q2), 1 - 2*(q1 * q1 + q3 * q3))
        elif order == 'xzy':
            x = torch.atan2(2 * (q0 * q1 + q2 * q3), 1 - 2*(q1 * q1 + q3 * q3))
            y = torch.atan2(2 * (q0 * q2 + q1 * q3), 1 - 2*(q2 * q2 + q3 * q3))
            z = torch.asin(torch.clamp(2 * (q0 * q3 - q1 * q2), -1+epsilon, 1-epsilon))
        elif order == 'yxz':
            x = torch.asin(torch.clamp(2 * (q0 * q1 - q2 * q3), -1+epsilon, 1-epsilon))
            y = torch.atan2(2 * (q1 * q3 + q0 * q2), 1 - 2*(q1 * q1 + q2 * q2))
            z = torch.atan2(2 * (q1 * q2 + q0 * q3), 1 - 2*(q1 * q1 + q3 * q3))
        elif order == 'zyx':
            x = torch.atan2(2 * (q0 * q1 + q2 * q3), 1 - 2*(q1 * q1 + q2 * q2))
            y = torch.asin(torch.clamp(2 * (q0 * q2 - q1 * q3), -1+epsilon, 1-epsilon))
            z = torch.atan2(2 * (q0 * q3 + q1 * q2), 1 - 2*(q2 * q2 + q3 * q3))
        else:
            raise

        return torch.stack((x, y, z), dim=1).view(original_shape)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True, help="Motion file")
    args, _ = parser.parse_known_args()

    motion = MotionLoader(args.file, "cpu")

    print("- number of frames:", motion.num_frames)
    print("- number of DOFs:", motion.num_dofs)
    print("- number of bodies:", motion.num_bodies)
