import mrcfile
import numpy as np
import torch
import torch.nn.functional as F
import einops
from torch_cubic_spline_grids import CubicBSplineGrid1d
from typing import Optional, Union
from scipy import optimize

from libtilt.backprojection import backproject_fourier, backproject_real
from libtilt.fft_utils import dft_center
from libtilt.patch_extraction import extract_squares
from libtilt.rescaling.rescale_fourier import rescale_2d
from libtilt.shapes import circle
from libtilt.shift.shift_image import shift_2d
from libtilt.transformations import Ry, Rz, T
from libtilt.correlation import correlate_2d
from libtilt.projection import project_image_real
from libtilt.projection import project_fourier, project_volume_real


def find_shift(
        image_a: torch.Tensor,
        image_b: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        upsample_factor: float = 10,
) -> torch.Tensor:
    image_shape = image_a.shape
    center = dft_center(image_shape, rfft=False, fftshifted=True)
    if mask is None:
        correlation = correlate_2d(
            image_a,
            image_b,
            normalize=True
        )
    else:
        correlation = correlate_2d(
            image_a * mask,
            image_b * mask,
            normalize=True
        )
    maximum = torch.tensor(
        np.unravel_index(correlation.argmax().cpu(), shape=image_shape)
    )
    shift = center - maximum

    # find interpolated shift by umsampling the correlation image
    peak_region_y = slice(maximum[0] - 3, maximum[0] + 4)
    peak_region_x = slice(maximum[1] - 3, maximum[1] + 4)
    upsampled = F.interpolate(
        einops.rearrange(correlation[peak_region_y, peak_region_x], 'h w -> 1 1 h w'),
        scale_factor=upsample_factor,
        mode='bicubic',
        align_corners=True
    )
    upsampled = einops.rearrange(upsampled, '1 1 h w -> h w')
    upsampled_center = dft_center(upsampled.shape, rfft=False, fftshifted=True)
    upsampled_shift = (upsampled_center - torch.tensor(
        np.unravel_index(upsampled.argmax().cpu(), shape=upsampled.shape)
    )) / upsample_factor
    print(upsampled.max())

    # add upsampled shift to integer shift
    full_shift = shift + upsampled_shift
    return full_shift


def tilt_axis_angle_optimization(
        image_stack: torch.Tensor,
        alignment_mask: torch.Tensor,
        initial_tilt_axis: Union[torch.Tensor, float] = .0,
        device: str = 'cpu',
) -> torch.Tensor:
    image_stack_masked = (image_stack * alignment_mask).to(device)
    # generate a weighting for the common line ROI by projecting the mask
    mask_weights = project_image_real(
        alignment_mask.to(device),
        torch.eye(2).reshape(1, 2, 2).to(device)
    )
    mask_weights /= mask_weights.max()  # normalise to 0 and 1

    def f(x):
        # for common lines each 2d image is projected perpendicular to the tilt axis, thus add 90 degrees
        R = Rz(x + 90, zyx=False)[:, :2, :2]

        projections = []
        for i in range(len(image_stack_masked)):
            projections.append(
                project_image_real(
                    image_stack_masked[i],
                    R.to(device)
                ).squeeze()
            )
        projections = torch.stack(projections)
        projections = projections - einops.reduce(projections, 'tilt w -> tilt 1', reduction='mean')
        projections = projections / torch.std(projections, dim=(-1), keepdim=True)
        # weight the lines by the projected mask
        projections = projections * mask_weights

        squared_differences = (projections - einops.rearrange(projections, 'b d -> b 1 d')) ** 2
        loss = einops.reduce(squared_differences, 'b1 b2 d -> 1', reduction='sum')
        return loss.item()

    result = optimize.minimize_scalar(
        f, bounds=(initial_tilt_axis-90, initial_tilt_axis+90)
    )
    return result.x


def projection_matching(
        tilt_series,
        tilt_angles,
        reference_tilt,
        tilt_axis_angle,
        mask,
        tomogram_center,
        tilt_image_center,
        device,
):
    # PROJECTION MATCHING!
    # * our reference_tilt remains fixed without a shift
    # * loop through the tilts in order of largest difference from the reference_tilt
    # * weight each projection by cosine of difference angle
    _, indices = torch.sort(torch.abs(tilt_angles - tilt_angles[reference_tilt]))
    aligned_tilts = torch.zeros(len(tilt_angles), dtype=torch.bool)
    aligned_tilts[reference_tilt] = True
    shifts = torch.zeros((len(tilt_series), 2), dtype=torch.float32)
    projections = []
    tomo_shape = (200, tilt_series.shape[-2], tilt_series.shape[-1])
    tomo_center = dft_center(tomo_shape, rfft=False, fftshifted=True)
    for i in indices:
        theta = tilt_angles[i]
        if theta == tilt_angles[reference_tilt]:
            continue
        if aligned_tilts.sum() < 3:
            shifts[i] = find_shift(tilt_series[reference_tilt], tilt_series[i], mask=mask)
            aligned_tilts[i] = True
            continue
        # absolute after cosine because after 90 degrees the weight should go up from 0 again
        # tilt_weights = torch.abs(torch.cos(torch.deg2rad(torch.abs(tilt_angles - theta))))
        aligned = shift_2d(
            tilt_series,
            shifts=-shifts
        )  # * einops.rearrange(tilt_weights, 'x -> x 1 1')

        # create backprojection matrix
        # s0 = T(-tomo_center)
        r0 = Ry(tilt_angles, zyx=True)
        r1 = Rz(tilt_axis_angle, zyx=True)
        # s2 = T(F.pad(tilt_image_center, pad=(1, 0), value=0))
        # M = s2 @ r1 @ r0 @ s0
        M = r1 @ r0 # @ s0

        # coarse reconstruction
        # weighted_reconstruction = backproject_fourier(
        #     images=aligned[aligned_tilts].to(device),
        #     rotation_matrices=torch.linalg.inv(M[:, :3, :3][aligned_tilts]).to(device),
        #     rotation_matrix_zyx=True,
        #     pad=False,
        # )
        weighted_reconstruction = backproject_real(
            projection_images=aligned[aligned_tilts].to(device),
            projection_matrices=torch.linalg.inv(M[aligned_tilts]).to(device),
            output_dimensions=tomo_shape,
        )
        import napari
        viewer = napari.Viewer()
        viewer.add_image(weighted_reconstruction.cpu().detach().numpy())
        napari.run()

        # forward project
        # projection = project_fourier(
        #     volume=weighted_reconstruction,
        #     rotation_matrices=torch.linalg.inv(M[i:i+1, :3, :3]).to(device),
        #     rotation_matrix_zyx=True,
        #     pad=False,
        # ).squeeze().cpu()
        projection = project_volume_real(
            volume=weighted_reconstruction,
            rotation_matrices=torch.linalg.inv(M[i:i+1, :3, :3]).to(device)
        ).squeeze().cpu()
        projections.append(projection)
        print(f'projection {theta}{chr(176)}')
        shifts[i] = find_shift(projection, tilt_series[i], mask=mask)
        aligned_tilts[i] = True
    import napari
    viewer = napari.Viewer()
    viewer.add_image(torch.stack(projections).detach().numpy())
    napari.run()
    return shifts


def areteamtomo_iteration(
        tilt_series,
        tilt_angles,
        reference_tilt,
        initial_tilt_axis_angle,
        mask,
        tomogram_center,
        tilt_image_center,
        device,
):
    print('\nPROJECTION MATCHING CYCLE\n')
    shifts = projection_matching(
        tilt_series,
        tilt_angles,
        reference_tilt,
        initial_tilt_axis_angle,
        mask,
        tomogram_center,
        tilt_image_center,
        device
    )
    fine_aligned = shift_2d(tilt_series, shifts=-shifts)

    print('\nREFINE TILT AXIS ANGLE OPTIMIZATION\n')
    new_tilt_axis_angle = tilt_axis_angle_optimization(
        fine_aligned,
        mask,
        initial_tilt_axis=initial_tilt_axis_angle,
        device=device,
    )
    print('initial tilt axis angle:', initial_tilt_axis_angle)
    print('final tilt axis angle:', new_tilt_axis_angle)
    return new_tilt_axis_angle, shifts


IMAGE_FILE = 'data/tomo200528_100.st'
IMAGE_PIXEL_SIZE = 1.724
STAGE_TILT_ANGLE_PRIORS = torch.arange(-51, 51, 3)
TILT_AXIS_ANGLE_PRIOR = -30  # -88.7 according to mdoc, but I set it faulty to see if the optimization works
ALIGNMENT_PIXEL_SIZE = 13.79 * 2
# set 0 degree tilt as reference
REFERENCE_TILT = STAGE_TILT_ANGLE_PRIORS.abs().argmin()
DEVICE = 'cuda:0'  # for cpu set to 'cpu'

tilt_series = torch.as_tensor(mrcfile.read(IMAGE_FILE), dtype=torch.float32)

tilt_series, _ = rescale_2d(
    image=tilt_series,
    source_spacing=IMAGE_PIXEL_SIZE,
    target_spacing=ALIGNMENT_PIXEL_SIZE,
    maintain_center=True,
)

tilt_series -= einops.reduce(tilt_series, 'tilt h w -> tilt 1 1', reduction='mean')
tilt_series /= torch.std(tilt_series, dim=(-2, -1), keepdim=True)
n_tilts, h, w = tilt_series.shape
center = dft_center((h, w), rfft=False, fftshifted=True)
center = einops.repeat(center, 'yx -> b yx', b=len(tilt_series))
tilt_series = extract_squares(
    image=tilt_series,
    positions=center,
    sidelength=min(h, w),
)

# set tomogram and tilt-series shape
size = min(h, w)
tomogram_dimensions = (size, ) * 3
tilt_dimensions = (size, ) * 2

# mask for coarse alignment
coarse_alignment_mask = circle(
    radius=size // 3,
    smoothing_radius=size // 6,
    image_shape=tilt_dimensions,
)

# do an IMOD style coarse tilt-series alignment
print('\nPAIRWISE XCORR\n')
coarse_shifts = torch.zeros((len(tilt_series), 2), dtype=torch.float32)

# find coarse alignment for negative tilts
current_shift = coarse_shifts[REFERENCE_TILT].clone()
for i in range(REFERENCE_TILT, 0, -1):
    shift = find_shift(tilt_series[i], tilt_series[i - 1], mask=coarse_alignment_mask)
    current_shift += shift
    coarse_shifts[i - 1] = current_shift

# find coarse alignment positive tilts
current_shift = coarse_shifts[REFERENCE_TILT].clone()
for i in range(REFERENCE_TILT, tilt_series.shape[0] - 1, 1):
    shift = find_shift(tilt_series[i], tilt_series[i + 1], mask=coarse_alignment_mask)
    current_shift += shift
    coarse_shifts[i + 1] = current_shift

# create aligned stack for common lines; apply the mask here to prevent recalculation
print('\nTILT AXIS ANGLE OPTIMIZATION\n')
coarse_aligned = shift_2d(tilt_series, shifts=-coarse_shifts)
tilt_axis_angle = tilt_axis_angle_optimization(
    coarse_aligned,
    coarse_alignment_mask,
    initial_tilt_axis=TILT_AXIS_ANGLE_PRIOR,
    device=DEVICE,
)
print('initial tilt axis angle:', TILT_AXIS_ANGLE_PRIOR)
print('final tilt axis angle:', tilt_axis_angle)

# set image and recon center
tomogram_center = dft_center(tomogram_dimensions, rfft=False, fftshifted=True)
tilt_image_center = dft_center(tilt_dimensions, rfft=False, fftshifted=True)

refined_tilt_axis_angle = tilt_axis_angle
for _ in range(3):
    predicted_tilt_axis_angle, refined_shifts = areteamtomo_iteration(
        tilt_series,
        STAGE_TILT_ANGLE_PRIORS,
        REFERENCE_TILT,
        refined_tilt_axis_angle,
        coarse_alignment_mask,
        tomogram_center,
        tilt_image_center,
        DEVICE,
    )
    refined_tilt_axis_angle = predicted_tilt_axis_angle

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
    pad=False,
)

fine_aligned = shift_2d(tilt_series, shifts=-refined_shifts)

s0 = T(-tomogram_center)
r0 = Ry(STAGE_TILT_ANGLE_PRIORS, zyx=True)
r1 = Rz(refined_tilt_axis_angle, zyx=True)
s2 = T(F.pad(tilt_image_center, pad=(1, 0), value=0))
M = s2 @ r1 @ r0 @ s0

# coarse reconstruction
fine_reconstruction = backproject_fourier(
    images=fine_aligned,
    rotation_matrices=torch.linalg.inv(M[:, :3, :3]),
    rotation_matrix_zyx=True,
    pad=False,
)

import napari

viewer = napari.Viewer()
viewer.add_image(tilt_series.detach().numpy(), name='experimental')
viewer.add_image(coarse_aligned.detach().numpy(), name='coarse aligned')
viewer.add_image(fine_aligned.detach().numpy(), name='fine aligned')
viewer.add_image(coarse_reconstruction.detach().numpy(), name='coarse reconstruction')
viewer.add_image(fine_reconstruction.detach().numpy(), name='fine reconstruction')
napari.run()
