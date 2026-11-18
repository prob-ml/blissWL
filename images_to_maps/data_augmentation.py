import random

import torch


TILE_MASK_KEYS = (
    "loss_mask",
    "interior_mask",
    "nominal_patch_tile_mask",
    "image_valid_tile_mask",
)

# Order of the last dim of psf_params, from get_bands_flux_and_psf:
# [Ixx, Iyy, Ixy, fwhm] in coadd pixel coordinates, per band.
PSF_IXX, PSF_IYY, PSF_IXY = 0, 1, 2
N_PSF_PARAMS = 4


def rotate_flip_psf_params(psf_params, rotate_id, flip_id):
    """PSF params of an image rotated rotate_id x 90 deg, then flipped if flip_id.

    Each 90 deg rotation swaps Ixx and Iyy and negates Ixy; a flip along either
    axis negates Ixy; fwhm is unchanged.  The rule holds whichever pixel axis
    is called x and in either rotation direction.  Returns a new tensor.
    """
    if psf_params.shape[-1] != N_PSF_PARAMS:
        raise ValueError(
            f"psf_params has {psf_params.shape[-1]} params per band; expected "
            f"{N_PSF_PARAMS} ([Ixx, Iyy, Ixy, fwhm])"
        )
    out = psf_params.clone()
    if rotate_id % 2 == 1:
        out[..., PSF_IXX] = psf_params[..., PSF_IYY]
        out[..., PSF_IYY] = psf_params[..., PSF_IXX]
    if (rotate_id + flip_id) % 2 == 1:
        out[..., PSF_IXY] = -psf_params[..., PSF_IXY]
    return out


class RemoveAnacalData:
    """Remove anacal_data field from batch to avoid collation errors with None values."""

    def __call__(self, datum):
        return {k: v for k, v in datum.items() if k != "anacal_data"}


class LensingRotateFlipTransform(torch.nn.Module):
    def __init__(self, without_replacement=False):
        super().__init__()
        self.rotate_id = -1
        self.flip_id = -1
        self.seen_states = set()
        self.without_replacement = without_replacement

    def __call__(self, datum):
        self.rotate_id = random.randint(0, 3)
        self.flip_id = random.randint(0, 1)

        if self.without_replacement:
            if len(self.seen_states) == 8:
                self.seen_states = set()
            while ((self.rotate_id, self.flip_id)) in self.seen_states:
                self.rotate_id = random.randint(0, 3)
                self.flip_id = random.randint(0, 1)
            self.seen_states.add((self.rotate_id, self.flip_id))

        datum_out = {
            "psf_params": rotate_flip_psf_params(
                datum["psf_params"], self.rotate_id, self.flip_id
            )
        }

        # apply rotation
        datum_out["images"] = datum["images"].rot90(self.rotate_id, [1, 2])
        d = datum["tile_catalog"]
        datum_out["tile_catalog"] = {
            k: v.rot90(self.rotate_id, [0, 1]) for k, v in d.items()
        }
        for key in TILE_MASK_KEYS:
            if key in datum:
                datum_out[key] = datum[key].rot90(self.rotate_id, [0, 1])

        # apply flip
        if self.flip_id == 1:
            datum_out["images"] = datum_out["images"].flip([1])
            d = datum_out["tile_catalog"]
            datum_out["tile_catalog"] = {k: v.flip([0]) for k, v in d.items()}
            for key in TILE_MASK_KEYS:
                if key in datum_out:
                    datum_out[key] = datum_out[key].flip([0])

        # shear requires special logic; other spin-2 tile fields (shear*_sum,
        # shear1_shear2, ellip_*) are moved but not sign-flipped, so they are
        # not valid training targets after this transform
        if all(k in datum["tile_catalog"] for k in ("shear_1", "shear_2")):
            shear1 = datum_out["tile_catalog"]["shear_1"]
            shear2 = datum_out["tile_catalog"]["shear_2"]
            for _ in range(self.rotate_id):
                shear1 = -shear1
                shear2 = -shear2
            if self.flip_id == 1:
                shear2 = -shear2
            datum_out["tile_catalog"]["shear_1"] = shear1
            datum_out["tile_catalog"]["shear_2"] = shear2

        # locations require special logic
        if "locs" in datum["tile_catalog"]:
            locs = datum_out["tile_catalog"]["locs"]
            for _ in range(self.rotate_id):
                # Rotate 90 degrees clockwise (in pixel coordinates)
                locs = torch.stack((1 - locs[..., 1], locs[..., 0]), dim=3)
            if self.flip_id == 1:
                locs = torch.stack((1 - locs[..., 0], locs[..., 1]), dim=3)
            datum_out["tile_catalog"]["locs"] = locs

        return datum_out
