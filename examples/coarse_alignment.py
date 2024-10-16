import mrcfile
import torch
import torch.nn.functional as F
import einops
import math
import numpy as np
import napari
from scipy import optimize

from libtilt.backprojection import backproject_fourier
from libtilt.fft_utils import dft_center
from libtilt.patch_extraction import extract_squares
from libtilt.rescaling.rescale_fourier import rescale_2d
from libtilt.shapes import circle
from libtilt.shift.shift_image import shift_2d
from libtilt.transformations import Ry, Rz, T
from libtilt.projection import project_image_real, project_volume_real, project_fourier
from libtilt.alignment import find_image_shift

IMAGE_FILE = 'data/tomo200528_107.st'
IMAGE_PIXEL_SIZE = 1.724
STAGE_TILT_ANGLE_PRIORS = torch.arange(-51, 54, 3)
TILT_AXIS_ANGLE_PRIOR = -30  # -88.7 according to mdoc, but I set it faulty to see if the optimization works
ALIGNMENT_PIXEL_SIZE = 13.79 * 4
# set 0 degree tilt as reference
REFERENCE_TILT = STAGE_TILT_ANGLE_PRIORS.abs().argmin()

tilt_series = torch.as_tensor(mrcfile.read(IMAGE_FILE))

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
reference_shift = torch.tensor([.0, .0])
center = dft_center(tilt_dimensions, rfft=False, fftshifted=True)
coarse_shifts = torch.zeros((len(tilt_series), 2), dtype=torch.float32)

roi_mask = torch.zeros_like(tilt_series)
for i, theta in enumerate(STAGE_TILT_ANGLE_PRIORS):
    offset = int((size // 2) * (1 - torch.abs(torch.cos(theta * math.pi / 180))))
    if offset == 0:
        roi_mask[i] = 1
    else:
        roi_mask[i, offset: -offset, :] = 1
full_mask = roi_mask * torch.unsqueeze(coarse_alignment_mask, 0)

# find coarse alignment for negative tilts
current_shift = reference_shift.clone()
for i in range(REFERENCE_TILT, 0, -1):
    shift = find_image_shift(
        tilt_series[i] * full_mask[i - 1],
        tilt_series[i - 1] * full_mask[i - 1],
    )
    current_shift += shift
    coarse_shifts[i - 1] = current_shift

# find coarse alignment positive tilts
current_shift = reference_shift.clone()
for i in range(REFERENCE_TILT, tilt_series.shape[0] - 1, 1):
    shift = find_image_shift(
        tilt_series[i] * full_mask[i + 1],
        tilt_series[i + 1] * full_mask[i + 1],
    )
    current_shift += shift
    coarse_shifts[i + 1] = current_shift

print(coarse_shifts)

def optimize_tilt_axis_angle(aligned_ts, mask, x0):
    aligned_masked = aligned_ts * mask
    mask_weights = project_image_real(mask,
                                      torch.eye(2).reshape(1, 2, 2))
    mask_weights /= mask_weights.max()  # normalise to 0 and 1

    def f(x):
        # for common lines each 2d image is projected perpendicular to the tilt axis,
        # thus add 90 degrees
        R = Rz(x + 90, zyx=False)[:, :2, :2]

        projections = []
        for i in range(len(aligned_masked)):
            projections.append(
                project_image_real(
                    aligned_masked[i],
                    R
                ).squeeze()
            )
        projections = torch.stack(projections)
        # projections = projections - einops.reduce(projections, 'tilt w -> tilt 1', reduction='mean')
        # projections = projections / torch.std(projections, dim=(-1), keepdim=True)
        # weight the lines by the projected mask
        projections = projections * mask_weights
        squared_differences = (projections - einops.rearrange(projections, 'b d -> b 1 d')) ** 2
        loss = einops.reduce(squared_differences, 'b1 b2 d -> 1', reduction='mean')
        return loss

    # print([(x, f(x)) for x in range(-80, -120, -1)])

    pred = optimize.minimize_scalar(f, bounds=[x0 - 90, x0 + 90])
    #
    # import napari
    # viewer = napari.Viewer()
    # viewer.add_image(f(-91).detach().numpy(), name='-91')
    # viewer.add_image(f(-111).detach().numpy(), name='-111')
    # napari.run()

    return pred

coarse_aligned = shift_2d(tilt_series, shifts=coarse_shifts)
tilt_axis_prediction = float(optimize_tilt_axis_angle(
    coarse_aligned,
    coarse_alignment_mask,
    TILT_AXIS_ANGLE_PRIOR,
).x)
print('final tilt axis angle:', tilt_axis_prediction)

def stretch(image, stretch, tilt_axis):
    cos_phi = math.cos(math.radians(tilt_axis))
    sin_phi = math.sin(math.radians(tilt_axis))
    m_rotate_forward = torch.tensor([[cos_phi, -sin_phi, 0],
                                     [sin_phi, cos_phi, 0],
                                     [0, 0, 1]])
    m_rotate_backward = torch.linalg.inv(m_rotate_forward)
    m_scale = torch.tensor([[stretch, 0, 0],
                            [0, 1, 0],
                            [0, 0, 1]])
    m_affine = m_rotate_backward @ m_scale @ m_rotate_forward
    m_affine = torch.unsqueeze(m_affine[:2], 0)
    flow_grid = F.affine_grid(m_affine, (1, 1, size, size), align_corners=True)
    new = F.grid_sample(torch.unsqueeze(torch.unsqueeze(image, 0), 0), flow_grid,
                  mode='bicubic', align_corners=True)
    return new[0, 0]

coarse_shifts = torch.zeros((len(tilt_series), 2), dtype=torch.float32)
# find coarse alignment for negative tilts
current_shift = reference_shift.clone()
for i in range(REFERENCE_TILT, 0, -1):
    shift = find_image_shift(
        tilt_series[i] * full_mask[i - 1],
        stretch(
            tilt_series[i - 1],
            math.cos(math.radians(STAGE_TILT_ANGLE_PRIORS[i - 1])) /
                math.cos(math.radians(STAGE_TILT_ANGLE_PRIORS[i])),
            tilt_axis_prediction
        ) * full_mask[i - 1]
    )
    current_shift += shift
    coarse_shifts[i - 1] = current_shift

# find coarse alignment positive tilts
current_shift = reference_shift.clone()
for i in range(REFERENCE_TILT, tilt_series.shape[0] - 1, 1):
    shift = find_image_shift(
        tilt_series[i] * full_mask[i + 1],
        stretch(
            tilt_series[i + 1],
            math.cos(math.radians(STAGE_TILT_ANGLE_PRIORS[i + 1])) /
                math.cos(math.radians(STAGE_TILT_ANGLE_PRIORS[i])),
            tilt_axis_prediction
        ) * full_mask[i + 1]
    )

    current_shift += shift
    coarse_shifts[i + 1] = current_shift

print('round2')
print(coarse_shifts)

coarse_aligned = shift_2d(tilt_series, shifts=coarse_shifts)
tilt_axis_prediction = float(optimize_tilt_axis_angle(
    coarse_aligned,
    coarse_alignment_mask,
    tilt_axis_prediction,
).x)
print('final tilt axis angle:', tilt_axis_prediction)
# viewer = napari.Viewer()
# viewer.add_image((tilt_series * full_mask)[1:].detach().numpy(), name='experimental1')
# viewer.add_image((tilt_series * full_mask)[0:-1].detach().numpy(), name='experimental2')
# napari.run()


tomogram_center = dft_center(tomogram_dimensions, rfft=False, fftshifted=True)
tilt_image_center = dft_center(tilt_dimensions, rfft=False, fftshifted=True)

s0 = T(-tomogram_center)
r0 = Ry(STAGE_TILT_ANGLE_PRIORS, zyx=True)
r1 = Rz(tilt_axis_prediction, zyx=True)
s2 = T(F.pad(tilt_image_center, pad=(1, 0), value=0))
M = s2 @ r1 @ r0 @ s0

# coarse reconstruction
coarse_reconstruction = backproject_fourier(
    images=coarse_aligned,
    rotation_matrices=torch.linalg.inv(M[:, :3, :3]),
    rotation_matrix_zyx=True,
)

mrcfile.write(
    'data/tomo200528_107.mrc',
    coarse_reconstruction.detach().numpy().astype(np.float32),
    voxel_size=ALIGNMENT_PIXEL_SIZE,
    overwrite=True,
)

viewer = napari.Viewer()
viewer.add_image(full_mask.detach().numpy())
viewer.add_image(tilt_series.detach().numpy(), name='experimental')
viewer.add_image(coarse_aligned.detach().numpy(), name='coarse aligned')
viewer.add_image(coarse_reconstruction.detach().numpy(), name='coarse reconstruction')
napari.run()


def normalise_under_mask(stack, mask):
    p = torch.sum(mask, dim=(-2, -1), keepdim=True)
    stack_masked = stack * mask
    mean = torch.sum(stack_masked, dim=(-2, -1), keepdim=True) / p
    std = (torch.sum(
        stack_masked ** 2, dim=(-2, -1), keepdim=True
    ) / p - mean ** 2) ** 0.5
    new_stack = (stack - mean) / std
    new_stack = new_stack * mask
    return new_stack


def optimize_shifts(
        tilt_series,
        projection_matrices,
        initial_shifts,
        tilt_angles,
        mask,
):
    roi_mask = torch.zeros_like(tilt_series)
    for i, theta in enumerate(tilt_angles):
        offset = int((size // 2) * (1 - torch.abs(torch.cos(theta * math.pi / 180))))
        if offset == 0:
            roi_mask[i] = 1
        else:
            roi_mask[i, offset: -offset, :] = 1
    full_mask = roi_mask * mask

    predicted_shifts = initial_shifts.detach().clone().requires_grad_(True)
    optimiser = torch.optim.Adam(
        params=[predicted_shifts, ],
        lr=1,  # .1
    )

    tilt_mask = torch.arange(len(tilt_series))
    for epoch in range(50):
        with (torch.no_grad()):
            projections = []
            for i in range(len(tilt_series)):
                # print(i)
                intermediate_recon = backproject_fourier(
                    images=shift_2d(
                        tilt_series[tilt_mask != i], shifts=predicted_shifts[
                            tilt_mask != i]
                    ),
                    rotation_matrices=torch.linalg.inv(
                        projection_matrices[:, :3, :3][tilt_mask != i]
                    ),
                    rotation_matrix_zyx=True,
                )

                projections.append(project_fourier(
                    volume=intermediate_recon,
                    rotation_matrices=torch.linalg.inv(
                        projection_matrices[i: i+1, :3, :3]
                    ),
                    rotation_matrix_zyx=True
                ).squeeze())
            projections = normalise_under_mask(torch.stack(projections), full_mask)

        optimiser.zero_grad()
        experimental = shift_2d(
            tilt_series, shifts=predicted_shifts
        ).squeeze()
        experimental = normalise_under_mask(experimental, full_mask)

        loss = torch.mean(torch.abs(experimental - projections) ** 2).sqrt()
        loss.backward()
        optimiser.step()
        print(epoch, loss)

    viewer = napari.Viewer()
    viewer.add_image(experimental.detach().numpy(), name='experimental')
    viewer.add_image(projections.detach().numpy(), name='predicted')
    viewer.add_image(((experimental - projections) ** 2).detach().numpy())
    napari.run()

    return predicted_shifts.clone().detach().requires_grad_(False)


shifts = coarse_shifts.detach().clone()
for _ in range(1):
    s0 = T(-tomogram_center)
    r0 = Ry(STAGE_TILT_ANGLE_PRIORS, zyx=True)
    r1 = Rz(tilt_axis_prediction, zyx=True)
    s2 = T(F.pad(tilt_image_center, pad=(1, 0), value=0))
    M = s2 @ r1 @ r0 @ s0

    shifts = optimize_shifts(
        tilt_series, M, shifts, STAGE_TILT_ANGLE_PRIORS, coarse_alignment_mask
    )

    fine_aligned = shift_2d(tilt_series, shifts=shifts)

    tilt_axis_prediction = float(optimize_tilt_axis_angle(
        fine_aligned, coarse_alignment_mask, tilt_axis_prediction
    ).x)
    print('final tilt axis angle:', tilt_axis_prediction)

s0 = T(-tomogram_center)
r0 = Ry(STAGE_TILT_ANGLE_PRIORS, zyx=True)
r1 = Rz(TILT_AXIS_ANGLE_PRIOR, zyx=True)
s2 = T(F.pad(tilt_image_center, pad=(1, 0), value=0))
M = s2 @ r1 @ r0 @ s0

# coarse reconstruction
shifts_only_reconstruction = backproject_fourier(
    images=coarse_aligned,
    rotation_matrices=torch.linalg.inv(M[:, :3, :3]),
    rotation_matrix_zyx=True,
)

fine_aligned = shift_2d(tilt_series, shifts=shifts)

s0 = T(-tomogram_center)
r0 = Ry(STAGE_TILT_ANGLE_PRIORS, zyx=True)
r1 = Rz(tilt_axis_prediction, zyx=True)
s2 = T(F.pad(tilt_image_center, pad=(1, 0), value=0))
M = s2 @ r1 @ r0 @ s0

# coarse reconstruction
fine_reconstruction = backproject_fourier(
    images=fine_aligned,
    rotation_matrices=torch.linalg.inv(M[:, :3, :3]),
    rotation_matrix_zyx=True,
)

mrcfile.write(
    'data/tomo200528_107.mrc',
    fine_reconstruction.detach().numpy().astype(np.float32),
    voxel_size=ALIGNMENT_PIXEL_SIZE,
    overwrite=True,
)

viewer = napari.Viewer()
viewer.add_image(tilt_series.detach().numpy(), name='experimental')
viewer.add_image(coarse_aligned.detach().numpy(), name='coarse aligned')
viewer.add_image(fine_aligned.detach().numpy(), name='fine aligned')
# viewer.add_image(shifts_only_reconstruction.detach().numpy(), name='shifts only
# reconstruction')
viewer.add_image(coarse_reconstruction.detach().numpy(), name='coarse reconstruction')
viewer.add_image(fine_reconstruction.detach().numpy(), name='fine reconstruction')
napari.run()
