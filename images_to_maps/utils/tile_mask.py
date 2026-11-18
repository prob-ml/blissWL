"""Per-tile masks that restrict losses and metrics to a subset of tiles.

Padded DC2 crops store boolean ``(n_tiles_h, n_tiles_w)`` masks such as
``loss_mask``, which marks the crop interior; the border tiles are image
context only.
"""

import torch


def get_tile_mask(batch: dict, key: str | None, n_tiles_h: int, n_tiles_w: int):
    """Return the ``(batch, n_tiles_h, n_tiles_w)`` bool mask ``batch[key]``.

    Returns None when ``key`` is None (use all tiles).  Raises if the mask is
    missing, has the wrong shape, or selects no tiles for some batch element.
    """
    if key is None:
        return None
    if key not in batch:
        raise KeyError(f"loss_mask_key={key!r} is not in the batch")
    mask = batch[key].bool()
    batch_size = batch["images"].shape[0]
    if mask.shape != (batch_size, n_tiles_h, n_tiles_w):
        raise ValueError(
            f"{key} has shape {tuple(mask.shape)}; "
            f"expected {(batch_size, n_tiles_h, n_tiles_w)}"
        )
    if not mask.flatten(1).any(dim=1).all():
        raise ValueError(f"{key} selects no tiles for some batch element")
    return mask


def crop_catalog_to_mask(catalog: dict, mask: torch.Tensor) -> dict:
    """Crop ``(batch, h, w, ...)`` tensors in ``catalog`` to the tiles in ``mask``.

    The mask must be the same for every batch element and fill its bounding
    rectangle (e.g. the central interior of a padded crop), so the cropped
    tensors keep their ``(batch, h', w', ...)`` layout.  Tensors without the
    mask's tile dimensions are returned unchanged.
    """
    first = mask[0]
    if not (mask == first).all():
        raise ValueError("tile mask differs across the batch")
    rows = first.any(dim=1).nonzero().flatten()
    cols = first.any(dim=0).nonzero().flatten()
    row0, row1 = int(rows.min()), int(rows.max()) + 1
    col0, col1 = int(cols.min()), int(cols.max()) + 1
    if not first[row0:row1, col0:col1].all():
        raise ValueError("tile mask is not a filled rectangle")

    tile_shape = tuple(mask.shape[1:])
    cropped = {}
    for name, value in catalog.items():
        if (
            isinstance(value, torch.Tensor)
            and value.dim() >= 3
            and tuple(value.shape[1:3]) == tile_shape
        ):
            cropped[name] = value[:, row0:row1, col0:col1]
        else:
            cropped[name] = value
    return cropped
