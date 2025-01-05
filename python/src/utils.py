import torch
import pymotion.rotations.quat_torch as quat


# ORIGINAL RETURNING QUATERNIONS
def from_root_quat(q: torch.Tensor, parents: torch.Tensor) -> torch.Tensor:
    """
    Convert root-centered quaternions to the skeleton information.

    Parameters
    ----------
    dq: torch.Tensor[..., n_joints, 4]
        Includes as first element the global position of the root joint
    parents: torch.Tensor[n_joints]

    Returns
    -------
    rotations : torch.Tensor[..., n_joints, 4]
    """
    n_joints = q.shape[-2]
    # rotations has shape (frames, n_joints, 4)
    rotations = q.clone()
    # make transformations local to the parents
    # (initially local to the root)
    for j in reversed(range(1, n_joints)):
        parent = parents[j]
        if parent == 0:  # already in root space
            continue
        inv = quat.inverse(rotations[..., parent, :])
        rotations[..., j, :] = quat.mul(inv, rotations[..., j, :])
    return rotations


def to_matrix_4(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to rotation matrices.

    Parameters
    ----------
        quaternions: torch.Tensor[..., [w,x,y,z]]

    Returns
    -------
        rotmats: torch.Tensor[..., 4, 4]. Matrix order: [[r0.x, r0.y, r0.z, r0.w],
                                                         [r1.x, r1.y, r1.z, r1.w],
                                                         [r2.x, r2.y, r2.z, r2.w],
                                                         [r3.x, r3.y, r3.z, r3.w]] where ri is row i.
    """
    qw, qx, qy, qz = torch.unbind(quaternions, -1)

    x2 = qx + qx
    y2 = qy + qy
    z2 = qz + qz
    xx = qx * x2
    yy = qy * y2
    wx = qw * x2
    xy = qx * y2
    yz = qy * z2
    wy = qw * y2
    xz = qx * z2
    zz = qz * z2
    wz = qw * z2

    m = torch.zeros(quaternions.shape[:-1] + (4, 4), device=quaternions.device)
    m[..., 0, 0] = 1.0 - (yy + zz)
    m[..., 0, 1] = xy - wz
    m[..., 0, 2] = xz + wy
    m[..., 1, 0] = xy + wz
    m[..., 1, 1] = 1.0 - (xx + zz)
    m[..., 1, 2] = yz - wx
    m[..., 2, 0] = xz - wy
    m[..., 2, 1] = yz + wx
    m[..., 2, 2] = 1.0 - (xx + yy)
    m[..., 3, 3] = 1.0

    return m


# MODIFIED RETURNING ROTATION MATRICES
def from_root_quat_to_rotmat(q: torch.Tensor, parents: torch.Tensor) -> torch.Tensor:
    """
    Convert root-centered quaternions to the skeleton information.

    Parameters
    ----------
    dq: torch.Tensor[..., n_joints, 4]
        Includes as first element the global position of the root joint
    parents: torch.Tensor[n_joints]

    Returns
    -------
    rotations : torch.Tensor[..., n_joints, 4, 4]
    """
    # rotations has shape (frames, n_joints, 4)
    rotations = to_matrix_4(q)
    rotations_inverse = to_matrix_4(quat.inverse(q))
    rotations_output = torch.empty_like(rotations, device=q.device)
    # Create a mask for parents that are not zero
    mask = parents != 0
    # make transformations local to the parents
    # (initially local to the root)
    rotations_output[..., ~mask, :, :] = rotations[..., ~mask, :, :]
    rotations_output[..., mask, :, :] = torch.matmul(
        rotations_inverse[..., parents[mask], :, :], rotations[..., mask, :, :]
    )
    return rotations_output


def fk_rotmat(
    rot: torch.Tensor,
    global_pos: torch.Tensor,
    offsets: torch.Tensor,
    parents: torch.Tensor,
) -> torch.Tensor:
    """
    Compute forward kinematics for a skeleton.
    From the local rotations, global position and offsets, compute the
    positions and rotation matrices of the joints in world space.

    Parameters
    -----------
        rot: torch.Tensor[..., n_joints, 4, 4]
        global_pos: torch.Tensor[..., 3]
        offsets: torch.Tensor[..., n_joints, 3] or torch.Tensor[n_joints, 3]
        parents: torch.Tensor[n_joints]

    Returns
    --------
        positions: torch.Tensor[..., n_joints, 3]
            positions of the joints
        rotmats: torch.Tensor[..., 3, 3]. Matrix order: [[r0.x, r0.y, r0.z],
                                                         [r1.x, r1.y, r1.z],
                                                         [r2.x, r2.y, r2.z]] where ri is row i.
            rotation matrices of the joints
    """
    rot[..., :3, 3] = offsets
    # first joint is global position
    rot[..., 0, :3, 3] = global_pos
    # other joints are transformed by the transform matrix
    i = 1
    for parent in parents[i:]:
        rot[..., i, :, :] = torch.matmul(
            rot[..., parent, :, :].clone(),
            rot[..., i, :, :].clone(),
        )
        i += 1
    positions = rot[..., :3, 3]
    rotmats = rot[..., :3, :3]
    return positions, rotmats
