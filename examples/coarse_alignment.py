import math
from pathlib import Path
from typing import Sequence

import einops
import mrcfile
import napari
import numpy as np
import torch
import torch.nn.functional as F
from torch_cubic_spline_grids import CubicBSplineGrid1d

from libtilt.alignment import find_image_shift
from libtilt.backprojection import backproject_fourier
from libtilt.coordinate_utils import array_to_grid_sample, homogenise_coordinates
from libtilt.fft_utils import dft_center
from libtilt.grids import coordinate_grid
from libtilt.patch_extraction import extract_squares
from libtilt.projection import project_image_real
from libtilt.rescaling.rescale_fourier import rescale_2d
from libtilt.shapes import circle
from libtilt.shift.shift_image import shift_2d
from libtilt.transformations import R_2d, Ry, Rz, T, T_2d

# TODO write some functions like
# def affine_transform_2d(
#       images: torch.Tensor,  # shape: 'n h w'
#       affine_matrices: torch.Tensor,  # shape: 'n 3 3'
#       interpolation: str = 'bicubic,
# ):
#
# def affine_transform_3d(
#       volumes: torch.Tensor,  # shape: 'n d h w'
#       affine_matrices: torch.Tensor,  # shape: 'n 4 4'
#       interpolation: str = 'bilinear',  # is actually trilinear in grid_sample
# ):
#
# def back_project_real(images, volume_shape, affine_matrices):

IMAGE_FILE = Path("data/tomo200528_100.st")
IMAGE_PIXEL_SIZE = 1.724
STAGE_TILT_ANGLE_PRIORS = torch.arange(-51, 51, 3)  # 107: 54, 100: 51
TILT_AXIS_ANGLE_PRIOR = -90.0  # -88.7 according to mdoc, but I set it faulty to see if
# the optimization works
ALIGNMENT_PIXEL_SIZE = IMAGE_PIXEL_SIZE * 8
# set 0 degree tilt as reference
REFERENCE_TILT = int(STAGE_TILT_ANGLE_PRIORS.abs().argmin())
ALIGN_Z = int(2000 / ALIGNMENT_PIXEL_SIZE)  # number is in A
RECON_Z = int(3000 / ALIGNMENT_PIXEL_SIZE)
WEIGHTING = "hamming"  # weighting scheme for filtered back projection
# the object diameter in number of pixels
OBJECT_DIAMETER = 300 / ALIGNMENT_PIXEL_SIZE

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

    return tilt_axis_angles.detach()


def stretch(image, stretch, tilt_axis):
    """Stretch an image along the tilt axis."""
    m_rotate_backward = R_2d(tilt_axis, yx=True)
    m_rotate_forward = torch.linalg.inv(m_rotate_backward)
    m_scale = torch.tensor([[stretch, 0, 0], [0, 1, 0], [0, 0, 1]])
    # F.affine grid requires 2 x 3 matrices
    m_affine = (m_rotate_backward @ m_scale @ m_rotate_forward)[:, :2]
    flow_grid = F.affine_grid(m_affine, (1, 1, size, size), align_corners=True)
    stretched = F.grid_sample(
        einops.rearrange(image, "h w -> 1 1 h w"),
        flow_grid,
        mode="bicubic",
        align_corners=True,
    ).squeeze()
    return stretched


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


def filtered_back_projection(
    tilt_series,
    tomogram_dimensions,
    tilt_angles,
    tilt_axis_angles,
    shifts,
    weighting: str = "exact",
    object_diameter: float | None = None,
):
    """Run weighted back projection incorporating some alignment parameters.

    weighting: str, default "hamming"
        all filters here start at 1/N (instead of 0 for ramp and hamming) which
        improves the low res signal upon forward projection of the reconstruction
        Options:
            - "ramp": increases linearly from 1/N to 1 from the zero frequency to
                nyquist
            - "exact": is based on and improves low-res signal on forward projection:
                Reference : Optik, Exact filters for general geometry three-dimensional
                reconstruction, vol.73,146,1986.
            - "hamming": modified hamming as used in AreTomo, further modified here to
                also start a 1/N
    object_diameter: float | None, default None
        object diameter specified in number of pixels, only needed for the exact filter

    """
    # initializes sizes
    device = tilt_series.device
    n_tilts, size, _ = tilt_series.shape  # for simplicity assume square images
    tilt_image_dimensions = (size, size)
    tomogram_center = dft_center(tomogram_dimensions, rfft=False, fftshifted=True)
    tilt_image_center = dft_center(tilt_image_dimensions, rfft=False, fftshifted=True)

    # generate the 2d alignment affine matrix
    s0 = T_2d(-tilt_image_center)
    r0 = R_2d(tilt_axis_angles, yx=True)
    s1 = T_2d(-shifts)
    s2 = T_2d(tilt_image_center)
    M = einops.rearrange((s2 @ s1 @ r0 @ s0), "... i j -> ... 1 1 i j").to(device)

    grid = homogenise_coordinates(coordinate_grid(tilt_image_dimensions, device=device))
    grid = einops.rearrange(grid, "h w coords -> h w coords 1")
    grid = M @ grid
    grid = einops.rearrange(grid, "... d h w coords 1 -> ... d h w coords")[
        ..., :2
    ].contiguous()
    grid_sample_coordinates = array_to_grid_sample(grid, tilt_image_dimensions)
    aligned = torch.squeeze(
        F.grid_sample(
            einops.rearrange(tilt_series, "n h w -> n 1 h w"),
            grid_sample_coordinates,
            align_corners=True,
            mode="bicubic",
        )
    )

    # generate weighting function and apply to aligned tilt series
    if weighting == "exact":
        if object_diameter is None:
            raise ValueError(
                "Calculation of exact weighting requires an object " "diameter."
            )
        if len(tilt_angles) == 1:
            filters = 1
        else:  # slice_width could be provided as a function argument it can be
            # calculated as: (pixel_size * 2 * imdim) / object_diameter
            q = einops.rearrange(
                torch.arange(
                    size // 2 + size % 2 + 1, dtype=torch.float32, device=device
                )
                / size,
                "q -> 1 1 q",
            )
            sampling = torch.sin(
                torch.deg2rad(
                    torch.abs(einops.rearrange(tilt_angles, "n -> n 1") - tilt_angles)
                )
            ).to(device)
            sampling = einops.rearrange(sampling, "n m -> n m 1")
            q_overlap_inv = sampling / (2 / object_diameter)
            over_weighting = 1 - torch.clip(q * q_overlap_inv, min=0, max=1)
            filters = 1 / einops.reduce(over_weighting, "n m q -> n q", "sum")
            filters = einops.rearrange(filters, "n w -> n 1 w")
    elif weighting == "ramp":
        filters = torch.arange(
            size // 2 + size % 2 + 1, dtype=torch.float32, device=device
        )
        filters /= filters.max()
        filters = filters * (1 - 1 / n_tilts) + 1 / n_tilts  # start at 1 / N
    elif weighting == "hamming":  # AreTomo3 code uses a modified hamming window
        # 2 * q * (0.55f + 0.45f * cosf(6.2831852f * q))  # with q from 0 to .5 (Ny)
        # https://github.com/czimaginginstitute/AreTomo3/blob/
        #   c39dcdad9525ee21d7308a95622f3d47fe7ab4b9/AreTomo/Recon/GRWeight.cu#L20
        q = (
            torch.arange(size // 2 + size % 2 + 1, dtype=torch.float32, device=device)
            / size
        )
        # filters = 2 * q * (.55 + .45 * torch.cos(2 * torch.pi * q))
        filters = 2 * q * (0.54 + 0.46 * torch.cos(2 * torch.pi * q))
        filters /= filters.max()  # 0-1 normalization
        filters = filters * (1 - 1 / n_tilts) + 1 / n_tilts  # start at 1 / N
    else:
        raise ValueError("Invalid weighting option provided for FBP.")

    weighted = torch.fft.irfftn(
        torch.fft.rfftn(aligned, dim=(-2, -1)) * filters, dim=(-2, -1)
    )
    if len(weighted.shape) == 2:  # rfftn gets rid of batch dimension: add it back
        weighted = einops.rearrange(weighted, "h w -> 1 h w")

    # time for real space back projection
    s0 = T(-tomogram_center)
    r0 = Ry(tilt_angles, zyx=True)
    s1 = T(F.pad(tilt_image_center, pad=(1, 0), value=0))
    M = einops.rearrange(s1 @ r0 @ s0, "... i j -> ... 1 1 i j").to(device)

    reconstruction = torch.zeros(
        tomogram_dimensions, dtype=torch.float32, device=device
    )
    grid = homogenise_coordinates(coordinate_grid(tomogram_dimensions, device=device))
    grid = einops.rearrange(grid, "d h w coords -> d h w coords 1")

    for i in range(n_tilts):
        grid_t = M[i] @ grid
        grid_t = einops.rearrange(grid_t, "... d h w coords 1 -> ... d h w coords")[
            ..., :3
        ].contiguous()
        grid_sample_coordinates = array_to_grid_sample(grid_t, tomogram_dimensions)
        reconstruction += torch.squeeze(
            F.grid_sample(
                einops.rearrange(weighted[i], "h w -> 1 1 1 h w"),
                einops.rearrange(
                    grid_sample_coordinates, "d h w coords -> 1 d h w coords"
                ),
                align_corners=True,
                mode="bilinear",
            )
        )
    return reconstruction, aligned


def predict_projection(
    tomogram,
    tilt_image_dimensions,
    tilt_angles,
    tilt_axis_angles,
    shifts,
):
    """Predict a projection from an intermediate reconstruction.

    For now only assumes to project with a single matrix, but should also work for
    sets of matrices.
    """
    device = tomogram.device
    tomogram_dimensions = tomogram.shape
    tomogram_center = dft_center(tomogram_dimensions, rfft=False, fftshifted=True)
    # tilt_image_center = dft_center(tilt_image_dimensions, rfft=False, fftshifted=True)
    # TODO project to proper image dimensions

    # time for real space projection
    s0 = T(-tomogram_center)
    r0 = Ry(tilt_angles, zyx=True)
    r1 = Rz(tilt_axis_angles, zyx=True)
    s1 = T(F.pad(-shifts, pad=(1, 0), value=0))
    s2 = T(tomogram_center)
    M = einops.rearrange(
        torch.linalg.inv(s2 @ s1 @ r1 @ r0 @ s0), "... i j -> ... 1 1 i j"
    ).to(device)

    grid = homogenise_coordinates(coordinate_grid(tomogram_dimensions, device=device))
    grid = einops.rearrange(grid, "d h w coords -> d h w coords 1")
    torch.matmul(M, grid, out=grid)
    grid = einops.rearrange(grid, "... d h w coords 1 -> ... d h w coords")[
        ..., :3
    ].contiguous()
    grid_sample_coordinates = array_to_grid_sample(grid, tomogram_dimensions)
    rotated = torch.squeeze(
        F.grid_sample(
            einops.rearrange(tomogram, "d h w -> 1 1 d h w"),
            einops.rearrange(grid_sample_coordinates, "d h w coords -> 1 d h w coords"),
            align_corners=True,
            mode="bilinear",
        )
    )
    projection = rotated.mean(axis=-3)
    weights = (rotated != 0).sum(axis=-3)
    torch.div(weights, weights.max(), out=weights)
    return projection, weights


def projection_matching(
    tilt_series: torch.Tensor,
    tomogram_dimensions: Sequence[int],
    reference_tilt_id: int,
    tilt_angles: torch.Tensor,
    tilt_axis_angles: torch.Tensor,
    current_shifts: torch.Tensor,
    alignment_mask: torch.Tensor,
    debug: bool = False,
):
    """Run projection matching."""
    n_tilts, size, _ = tilt_series.shape
    aligned_set = [reference_tilt_id]
    shifts = current_shifts.detach().clone()

    # generate indices by alternating postive/negative tilts
    max_offset = max(reference_tilt_id, len(tilt_angles) - reference_tilt_id - 1)
    index_sequence = []
    for i in range(1, max_offset + 1):  # skip reference
        if reference_tilt_id + i < len(tilt_angles):
            index_sequence.append(reference_tilt_id + i)
        if i > 0 and reference_tilt_id - i >= 0:
            index_sequence.append(reference_tilt_id - i)

    # if debug:  # for debug mode, store the predicted projections
    projections = torch.zeros((n_tilts, size, size))
    projections[reference_tilt_id] = tilt_series[reference_tilt_id]

    for i in index_sequence:
        tilt_angle = tilt_angles[i]
        weights = einops.rearrange(
            torch.cos(torch.deg2rad(torch.abs(tilt_angles - tilt_angle))),
            "n -> n 1 1",
        )
        intermediate_recon, _ = filtered_back_projection(
            (tilt_series[aligned_set,] * weights[aligned_set,]).to("cuda"),
            tomogram_dimensions,
            tilt_angles[aligned_set,],
            tilt_axis_angles[aligned_set,],
            shifts[aligned_set,],
            weighting=WEIGHTING,
            object_diameter=OBJECT_DIAMETER,
        )
        projection, projection_weights = predict_projection(
            intermediate_recon,
            (size, size),
            tilt_angles[[i],],
            tilt_axis_angles[[i],],
            shifts[[i],],
        )  # TODO the volume edges introduce edges in the projected image
        # ensure correlation in relevant area
        projection_weights *= alignment_mask.to("cuda")
        shift = find_image_shift(
            tilt_series[i].to("cuda") * projection_weights,
            projection * projection_weights,
        )
        shifts[i] -= shift.to("cpu")
        aligned_set.append(i)

        print(  # TODO should be some sort of logging?
            f"aligned index {i} at angle {tilt_angle}: " f"{shift}"
        )

        # mainly for debug
        projections[i] = (projection * projection_weights).detach().to("cpu")

    return shifts, projections


# coarse reconstruction
initial_reconstruction, _ = filtered_back_projection(
    tilt_series,
    (RECON_Z, size, size),
    STAGE_TILT_ANGLE_PRIORS,
    tilt_axis_angle,
    shifts,
    weighting=WEIGHTING,
    object_diameter=OBJECT_DIAMETER,
)


# some optimizations parameters
max_iter = 10  # this seems solid
tolerance = 0.1  # should probably be related to pixel size
predicted_tilts = []
for i in range(max_iter):
    print(f"projection matching iteration {i}")
    tilt_axis_angle = optimize_tilt_axis_angle(
        shift_2d(tilt_series, shifts=shifts),
        coarse_alignment_mask,
        tilt_axis_angle,
    )
    print("new tilt axis angle:", tilt_axis_angle)

    new_shifts, pred = projection_matching(
        tilt_series,
        (ALIGN_Z, size, size),
        REFERENCE_TILT,
        STAGE_TILT_ANGLE_PRIORS,
        tilt_axis_angle,
        shifts,
        coarse_alignment_mask,
        debug=False,
    )
    predicted_tilts.append(pred)

    if torch.all(torch.abs(shifts - new_shifts) < tolerance):
        break

    shifts = new_shifts

viewer = napari.Viewer()
viewer.add_image(tilt_series.detach().numpy(), name="raw tilts")
for i, p in enumerate(predicted_tilts):
    viewer.add_image(p.detach().numpy(), name=f"prediction at iter {i}")
napari.run()

final, aligned_ts = filtered_back_projection(
    tilt_series,
    (RECON_Z, size, size),
    STAGE_TILT_ANGLE_PRIORS,
    tilt_axis_angle,
    shifts,
    weighting=WEIGHTING,
    object_diameter=OBJECT_DIAMETER,
)

mrcfile.write(
    IMAGE_FILE.with_name(IMAGE_FILE.stem + "_exact.mrc"),
    final.detach().numpy().astype(np.float32),
    voxel_size=ALIGNMENT_PIXEL_SIZE,
    overwrite=True,
)

# generate a proper fourier inverted reconstruction
tomogram_center = dft_center(tomogram_dimensions, rfft=False, fftshifted=True)
tilt_image_center = dft_center(tilt_dimensions, rfft=False, fftshifted=True)

s0 = T(-tomogram_center)
r0 = Ry(STAGE_TILT_ANGLE_PRIORS, zyx=True)
r1 = Rz(tilt_axis_angle, zyx=True)
s2 = T(F.pad(tilt_image_center, pad=(1, 0), value=0))
M = s2 @ r1 @ r0 @ s0

fine_reconstruction = backproject_fourier(
    images=shift_2d(tilt_series, shifts),
    rotation_matrices=torch.linalg.inv(M[:, :3, :3]),
    rotation_matrix_zyx=True,
    do_gridding_correction=False,
)

mrcfile.write(
    IMAGE_FILE.with_name(IMAGE_FILE.stem + "_fine.mrc"),
    fine_reconstruction.detach().numpy().astype(np.float32),
    voxel_size=ALIGNMENT_PIXEL_SIZE,
    overwrite=True,
)

viewer = napari.Viewer()
viewer.add_image(initial_reconstruction.detach().numpy(), name="initial reconstruction")
viewer.add_image(final.detach().numpy(), name="optimized reconstruction")
viewer.add_image(fine_reconstruction.detach().numpy(), name="fourier inverted")
viewer.add_image(aligned_ts.detach().numpy(), name="aligned_ts")
napari.run()
