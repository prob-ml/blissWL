"""Mean-field NPE predictions on padded DC2 crops, stitched into one tile map.

Padded crops overlap: each one's interior (the tiles in ``loss_mask``) abuts
its neighbours' interiors, and the border tiles are image context only.  This
module runs a trained ``MassMapEncoder`` over a dataloader of padded crops,
keeps each crop's interior, and places the interiors on a tile grid using the
crop's row and column in its tract's crop grid (row 0 is the south edge).
"""

import os

import torch
from lightning.pytorch.utilities import move_data_to_device

from images_to_maps.dc2.dc2 import padded_crop_position
from images_to_maps.utils.tile_mask import crop_catalog_to_mask, get_tile_mask

FIELDS = ("shear_1", "shear_2", "convergence")


def crop_grid_position(
    batch: dict, file_path: str | None = None
) -> tuple[int, int, int]:
    """Tract and crop row/column of the single crop in ``batch``.

    Read from the batch's own metadata; if ``file_path`` is given, it must
    agree with the position encoded in the file name.
    """
    if batch["images"].shape[0] != 1:
        raise ValueError("expected a batch with one crop")
    crop_row, crop_col = (int(v) for v in batch["crop_row_col"][0])
    position = (int(batch["nominal_tract"][0]), crop_row, crop_col)
    if file_path is not None:
        from_name = padded_crop_position(os.path.basename(file_path))
        if from_name != position:
            raise ValueError(
                f"{file_path}: name gives {from_name}, batch gives {position}"
            )
    return position


def predict_crop_interiors(encoder, dataloader, device) -> dict:
    """Posterior means/SDs and truth on each crop's interior tiles.

    Returns a dict of tensors stacked over crops (on the CPU):
    ``{field}_true``, ``{field}_pred`` (posterior mean) and ``{field}_std``
    for each of ``FIELDS``, plus ``ra`` and ``dec``, each with shape
    ``(n_crops, h, w, n_bins)``; ``loss`` with shape ``(n_crops,)`` (the
    training loss on the interior); and ``positions``, a list of
    ``(tract, row, col)`` tuples.
    """
    file_paths = getattr(dataloader.dataset, "file_paths", None)
    if file_paths is not None and len(file_paths) != len(dataloader.dataset):
        file_paths = None  # files with several crops each: skip the name check

    out = {"positions": [], "loss": []}
    for i, raw_batch in enumerate(dataloader):
        batch = move_data_to_device(raw_batch, device)
        out["positions"].append(
            crop_grid_position(batch, None if file_paths is None else file_paths[i])
        )

        with torch.no_grad():
            inputs = torch.cat(
                [inorm.get_input_tensor(batch) for inorm in encoder.image_normalizers],
                dim=2,
            )
            x_cat = encoder.net(inputs)
            target = batch["tile_catalog"]
            nll = encoder.var_dist.compute_nll(x_cat, target)

            mask = get_tile_mask(batch, encoder.loss_mask_key, *x_cat.shape[1:3])
            if mask is None:
                raise ValueError(
                    "encoder.loss_mask_key is not set; crops have no interior"
                )
            loss = torch.where(mask, nll, 0.0).sum() / mask.sum()

            per_tile = {}
            for factor, params in encoder.var_dist._factor_param_pairs(x_cat):
                dist = factor.get_dist(params).base_dist
                per_tile[f"{factor.name}_pred"] = dist.loc
                per_tile[f"{factor.name}_std"] = dist.scale
            for field in FIELDS:
                per_tile[f"{field}_true"] = target[field]
            per_tile["ra"] = target["ra"]
            per_tile["dec"] = target["dec"]
            per_tile = crop_catalog_to_mask(per_tile, mask)

        out["loss"].append(loss.cpu())
        for key, value in per_tile.items():
            out.setdefault(key, []).append(value[0].cpu())

    return {
        key: value if key == "positions" else torch.stack(value)
        for key, value in out.items()
    }


def stitch_interiors(values: torch.Tensor, positions: list) -> torch.Tensor:
    """Place per-crop interiors ``(n_crops, h, w, ...)`` on one tile grid.

    Rows of the result run south to north and columns east to west (RA
    decreases with column), matching the tiles within each crop.  Raises
    unless the crops are from one tract and fill a rectangle of the crop grid
    exactly once.
    """
    if values.shape[0] != len(positions):
        raise ValueError(f"{values.shape[0]} crops but {len(positions)} positions")
    tracts = {tract for tract, _, _ in positions}
    if len(tracts) != 1:
        raise ValueError(f"crops come from several tracts: {sorted(tracts)}")
    rows = [row for _, row, _ in positions]
    cols = [col for _, _, col in positions]
    row0, col0 = min(rows), min(cols)
    n_rows, n_cols = max(rows) - row0 + 1, max(cols) - col0 + 1
    if len(set(positions)) != len(positions):
        raise ValueError("some crops appear more than once")
    if len(positions) != n_rows * n_cols:
        raise ValueError(
            f"{len(positions)} crops do not fill the {n_rows} x {n_cols} crop rectangle"
        )

    h, w = values.shape[1:3]
    stitched = values.new_empty((n_rows * h, n_cols * w, *values.shape[3:]))
    for value, row, col in zip(values, rows, cols):
        r, c = (row - row0) * h, (col - col0) * w
        stitched[r : r + h, c : c + w] = value
    return stitched
