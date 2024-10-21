import math

import einops
import mrcfile
import napari
import numpy as np
import torch
import torch.nn.functional as F
from torch_cubic_spline_grids import CubicBSplineGrid1d

from libtilt.alignment import find_image_shift
from libtilt.backprojection import backproject_fourier
from libtilt.fft_utils import dft_center
from libtilt.filters import low_pass_filter
from libtilt.patch_extraction import extract_squares
from libtilt.projection import project_fourier, project_image_real
from libtilt.rescaling.rescale_fourier import rescale_2d
from libtilt.shapes import circle
from libtilt.shift.shift_image import shift_2d
from libtilt.transformations import Ry, Rz, T

IMAGE_FILE = "data/tomo200528_107.st"
IMAGE_PIXEL_SIZE = 1.724
STAGE_TILT_ANGLE_PRIORS = torch.arange(-51, 54, 3)
TILT_AXIS_ANGLE_PRIOR = -90.0  # -88.7 according to mdoc, but I set it faulty to see if
# the optimization works
ALIGNMENT_PIXEL_SIZE = 13.79 * 2
# set 0 degree tilt as reference
REFERENCE_TILT = STAGE_TILT_ANGLE_PRIORS.abs().argmin()

tilt_series = torch.as_tensor(mrcfile.read(IMAGE_FILE))

tilt_series, _ = rescale_2d(
    image=tilt_series,
    source_spacing=IMAGE_PIXEL_SIZE,
    target_spacing=ALIGNMENT_PIXEL_SIZE,
    maintain_center=True,
)

tilt_series -= einops.reduce(tilt_series, "tilt h w -> tilt 1 1", reduction="mean")
tilt_series /= torch.std(tilt_series, dim=(-2, -1), keepdim=True)
n_tilts, h, w = tilt_series.shape
center = dft_center((h, w), rfft=False, fftshifted=True)
center = einops.repeat(center, "yx -> b yx", b=len(tilt_series))
tilt_series = extract_squares(
    image=tilt_series,
    positions=center,
    sidelength=min(h, w),
)

# set tomogram and tilt-series shape
size = min(h, w)
tomogram_dimensions = (size,) * 3
tilt_dimensions = (size,) * 2

# mask for coarse alignment
coarse_alignment_mask = circle(
    radius=size // 3,
    smoothing_radius=size // 6,
    image_shape=tilt_dimensions,
)

ramp_weights = (
    torch.abs(torch.arange(-size // 2 + size % 2, size // 2 + size % 2, 1.0))
    / (size // 2)
).repeat(
    size, 1
)  # tilt images usually rotate around x-axis

# do an IMOD style coarse tilt-series alignment
center = dft_center(tilt_dimensions, rfft=False, fftshifted=True)
coarse_shifts = torch.zeros((len(tilt_series), 2), dtype=torch.float32)
masked_tilts = tilt_series * coarse_alignment_mask

# find coarse alignment for negative tilts
current_shift = torch.zeros(2)
for i in range(REFERENCE_TILT, 0, -1):
    shift = find_image_shift(
        masked_tilts[i],
        masked_tilts[i - 1],
    )
    current_shift += shift
    coarse_shifts[i - 1] = current_shift

# find coarse alignment positive tilts
current_shift = torch.zeros(2)
for i in range(REFERENCE_TILT, tilt_series.shape[0] - 1, 1):
    shift = find_image_shift(
        masked_tilts[i],
        masked_tilts[i + 1],
    )
    current_shift += shift
    coarse_shifts[i + 1] = current_shift

coarse_aligned = shift_2d(tilt_series, shifts=coarse_shifts)


def optimize_tilt_axis_angle(
    aligned_ts, coarse_alignment_mask, initial_tilt_axis_angle
):
    """Optimize tilt axis angles on a spline grid using the LBFGS optimizer."""
    coarse_aligned_masked = aligned_ts * coarse_alignment_mask

    # generate a weighting for the common line ROI by projecting the mask
    mask_weights = project_image_real(
        coarse_alignment_mask, torch.eye(2).reshape(1, 2, 2)
    )
    mask_weights /= mask_weights.max()  # normalise to 0 and 1

    # optimize tilt axis angle
    grid_resolution = 3
    tilt_axis_grid = CubicBSplineGrid1d(resolution=grid_resolution, n_channels=1)
    tilt_axis_grid.data = torch.tensor(
        [
            torch.mean(initial_tilt_axis_angle),
        ]
        * grid_resolution,
        dtype=torch.float32,
    )
    interpolation_points = torch.linspace(0, 1, len(tilt_series))

    lbfgs = torch.optim.LBFGS(
        tilt_axis_grid.parameters(),
        history_size=10,
        max_iter=4,
        line_search_fn="strong_wolfe",
    )

    def closure():

        tilt_axis_angles = tilt_axis_grid(interpolation_points)
        # for common lines each 2d image is projected perpendicular to the tilt axis,
        # thus add 90 degrees
        R = Rz(tilt_axis_angles + 90, zyx=False)[:, :2, :2]
        # R = Rz(x_lbfgs + 90, zyx=False)[:, :2, :2].repeat(len(aligned_ts), 1, 1)

        projections = []
        for i in range(len(coarse_aligned_masked)):
            projections.append(
                project_image_real(coarse_aligned_masked[i], R[i : i + 1]).squeeze()
            )
        projections = torch.stack(projections)
        projections = projections - einops.reduce(
            projections, "tilt w -> tilt 1", reduction="mean"
        )
        projections = projections / torch.std(projections, dim=(-1), keepdim=True)
        # weight the lines by the projected mask
        projections = projections * mask_weights

        lbfgs.zero_grad()
        squared_differences = (
            projections - einops.rearrange(projections, "b d -> b 1 d")
        ) ** 2
        loss = einops.reduce(squared_differences, "b1 b2 d -> 1", reduction="sum")
        loss.backward()
        print(loss.item())
        return loss

        # if not (epoch % 10):
        #     print(epoch, loss.item(), tilt_axis_grid.data.mean())

    for _ in range(3):
        lbfgs.step(closure)

    tilt_axis_angles = tilt_axis_grid(interpolation_points)
    # print(tilt_axis_angles)

    return tilt_axis_angles


def stretch(image, stretch, tilt_axis):
    """Stretch an image along the tilt axis."""
    cos_phi = math.cos(math.radians(tilt_axis))
    sin_phi = math.sin(math.radians(tilt_axis))
    m_rotate_forward = torch.tensor(
        [[cos_phi, -sin_phi, 0], [sin_phi, cos_phi, 0], [0, 0, 1]]
    )
    m_rotate_backward = torch.linalg.inv(m_rotate_forward)
    m_scale = torch.tensor([[stretch, 0, 0], [0, 1, 0], [0, 0, 1]])
    m_affine = m_rotate_backward @ m_scale @ m_rotate_forward
    m_affine = torch.unsqueeze(m_affine[:2], 0)
    flow_grid = F.affine_grid(m_affine, (1, 1, size, size), align_corners=True)
    new = F.grid_sample(
        einops.rearrange(image, "h w -> 1 1 h w"),
        flow_grid,
        mode="bicubic",
        align_corners=True,
    )
    return new[0, 0]


def coarse_align(tilt_series, mask, tilt_angles, tilt_axis_angle):
    """Find coarse shifts of images while stretch each pair along the tilt axis."""
    coarse_shifts = torch.zeros((len(tilt_series), 2), dtype=torch.float32)
    # find coarse alignment for negative tilts
    current_shift = torch.zeros(2)
    for i in range(REFERENCE_TILT, 0, -1):
        shift = find_image_shift(
            tilt_series[i] * mask,
            stretch(
                tilt_series[i - 1],
                math.cos(math.radians(tilt_angles[i - 1]))
                / math.cos(math.radians(tilt_angles[i])),
                tilt_axis_angle[i - 1],
            )
            * mask,
        )
        current_shift += shift
        coarse_shifts[i - 1] = current_shift

    # find coarse alignment positive tilts
    current_shift = torch.zeros(2)
    for i in range(REFERENCE_TILT, tilt_series.shape[0] - 1, 1):
        shift = find_image_shift(
            tilt_series[i] * mask,
            stretch(
                tilt_series[i + 1],
                math.cos(math.radians(tilt_angles[i + 1]))
                / math.cos(math.radians(tilt_angles[i])),
                tilt_axis_angle[i + 1],
            )
            * mask,
        )
        current_shift += shift
        coarse_shifts[i + 1] = current_shift
    return coarse_shifts


tilt_axis_angle = torch.tensor(TILT_AXIS_ANGLE_PRIOR)
shifts = None
for i in range(2):
    print(f"iteration {i}")
    tilt_axis_angle = optimize_tilt_axis_angle(
        coarse_aligned,
        coarse_alignment_mask,
        tilt_axis_angle,
    )
    print("new tilt axis angle:", tilt_axis_angle)

    shifts = coarse_align(
        tilt_series, coarse_alignment_mask, STAGE_TILT_ANGLE_PRIORS, tilt_axis_angle
    )
    print("new shifts:", shifts)
    coarse_aligned = shift_2d(tilt_series, shifts=shifts)

# viewer = napari.Viewer()
# viewer.add_image((tilt_series * full_mask)[1:].detach().numpy(), name='experimental1')
# viewer.add_image((tilt_series * full_mask)[0:-1].detach().numpy(),
# name='experimental2')
# napari.run()

tomogram_center = dft_center(tomogram_dimensions, rfft=False, fftshifted=True)
tilt_image_center = dft_center(tilt_dimensions, rfft=False, fftshifted=True)

s0 = T(-tomogram_center)
r0 = Ry(STAGE_TILT_ANGLE_PRIORS, zyx=True)
r1 = Rz(tilt_axis_angle, zyx=True)
s2 = T(F.pad(tilt_image_center, pad=(1, 0), value=0))
M = s2 @ r1 @ r0 @ s0

# coarse reconstruction
coarse_reconstruction = backproject_fourier(
    images=coarse_aligned,
    rotation_matrices=torch.linalg.inv(M[:, :3, :3]),
    rotation_matrix_zyx=True,
    do_gridding_correction=False,
)

mrcfile.write(
    "data/tomo200528_107.mrc",
    coarse_reconstruction.detach().numpy().astype(np.float32),
    voxel_size=ALIGNMENT_PIXEL_SIZE,
    overwrite=True,
)


# PROJECTION MATCHING
def projection_matching(shifts):
    """Run projection matching."""
    aligned_set = [REFERENCE_TILT]
    init = REFERENCE_TILT
    tilt_ids = torch.arange(n_tilts)
    factor = 1
    while init + 1 * factor < n_tilts or init - 1 * factor >= 0:
        # new_aligned = []
        for x in (-1, 1):
            i = init + x * factor
            if i < 0 or i >= n_tilts:
                continue
            tilt_angle = STAGE_TILT_ANGLE_PRIORS[i]

            weights = einops.rearrange(
                torch.cos(
                    torch.deg2rad(torch.abs(STAGE_TILT_ANGLE_PRIORS - tilt_angle))
                ),
                "n -> n 1 1",
            )
            tilt_mask = torch.logical_and(
                tilt_ids >= min(aligned_set), tilt_ids <= max(aligned_set)
            )
            intermediate_recon = backproject_fourier(
                images=shift_2d(
                    tilt_series[tilt_mask] * weights[tilt_mask],
                    shifts=shifts[tilt_mask],
                ),
                rotation_matrices=torch.linalg.inv(M[:, :3, :3][tilt_mask]),
                rotation_matrix_zyx=True,
                do_gridding_correction=False,  # This is very important for some reason
                pad=True,
            )

            projection = project_fourier(
                volume=intermediate_recon,
                rotation_matrices=torch.linalg.inv(M[i : i + 1, :3, :3]),
                rotation_matrix_zyx=True,
                pad=True,
            ).squeeze()
            projection = (projection - projection.mean()) / projection.std()

            old_shift = shifts[i].clone()
            shifts[i] = find_image_shift(
                projection * coarse_alignment_mask,
                tilt_series[i] * coarse_alignment_mask,
            )
            print(
                f"aligning index {i} at angle {tilt_angle}: "
                f"{torch.abs(old_shift - shifts[i])}"
            )
            # new_aligned.append(i)
            aligned_set.append(i)
        # aligned_set += new_aligned
        factor += 1
    return shifts


# Do weighted back projection
M_ramp = Rz(-tilt_axis_angle, zyx=False)[:, :2, :3]
M_ramp[..., 2:] = 0

grids = F.affine_grid(M_ramp, (n_tilts, 1, size, size), align_corners=True)
ramp_filters = torch.fft.ifftshift(
    F.grid_sample(
        einops.repeat(ramp_weights, "h w -> n 1 h w", n=n_tilts),
        grids,
        mode="bicubic",
        align_corners=True,
    ).squeeze(),
    dim=(-1, -2),
) * low_pass_filter(
    0.45,
    0.05,
    (size, size),
    rfft=False,
    fftshift=False,
)

weighted = torch.fft.ifftn(
    torch.fft.fftn(coarse_aligned, dim=(-1, -2)) * ramp_filters, dim=(-1, -2)
).real

viewer = napari.Viewer()
viewer.add_image(ramp_filters.detach().numpy())
viewer.add_image(weighted.detach().numpy())
napari.run()

r0 = Ry(STAGE_TILT_ANGLE_PRIORS, zyx=False)
r1 = Rz(tilt_axis_angle, zyx=False)
M = r1 @ r0

rec = torch.zeros((100, size, size))
for i in range(n_tilts):
    grid = F.affine_grid(
        M[i : i + 1, :3], (1, 1, 100, size, size), align_corners=False  # N, C, D, H, W
    )
    rec += torch.squeeze(
        F.grid_sample(
            einops.rearrange(weighted[i], "h w -> 1 1 1 h w"),
            grid,
            align_corners=False,
            mode="bilinear",
        )
    )
print(rec.shape)
viewer = napari.Viewer()
viewer.add_image(coarse_reconstruction.detach().numpy())
viewer.add_image(rec.detach().numpy())
napari.run()

for _ in range(2):
    shifts = projection_matching(shifts)

    fine_aligned = shift_2d(tilt_series, shifts=shifts)

    tilt_axis_angle = optimize_tilt_axis_angle(
        fine_aligned,
        coarse_alignment_mask,
        tilt_axis_angle,
    )
    print("new tilt axis angle:", tilt_axis_angle)

    s0 = T(-tomogram_center)
    r0 = Ry(STAGE_TILT_ANGLE_PRIORS, zyx=True)
    r1 = Rz(tilt_axis_angle, zyx=True)
    s2 = T(F.pad(tilt_image_center, pad=(1, 0), value=0))
    M = s2 @ r1 @ r0 @ s0


fine_reconstruction = backproject_fourier(
    images=fine_aligned,
    rotation_matrices=torch.linalg.inv(M[:, :3, :3]),
    rotation_matrix_zyx=True,
    do_gridding_correction=False,
)

mrcfile.write(
    "data/tomo200528_107_fine.mrc",
    fine_reconstruction.detach().numpy().astype(np.float32),
    voxel_size=ALIGNMENT_PIXEL_SIZE,
    overwrite=True,
)

viewer = napari.Viewer()
viewer.add_image(coarse_aligned.detach().numpy(), name="coarse aligned")
viewer.add_image(fine_aligned.detach().numpy(), name="fine aligned")
viewer.add_image(coarse_reconstruction.detach().numpy(), name="coarse recon")
viewer.add_image(fine_reconstruction.detach().numpy(), name="fine recon")
napari.run()


roi_mask = torch.zeros_like(tilt_series)
for i, theta in enumerate(STAGE_TILT_ANGLE_PRIORS):
    offset = int((size // 2) * (1 - torch.abs(torch.cos(theta * math.pi / 180))))
    if offset == 0:
        roi_mask[i] = 1
    else:
        roi_mask[i, offset:-offset, :] = 1
full_mask = roi_mask * coarse_alignment_mask
