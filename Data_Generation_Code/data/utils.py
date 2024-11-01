import numpy as np
import pandas as pd
from typing import Optional, Sequence
from ast import literal_eval
import torch


def read_index_row(index: pd.DataFrame, row: int):
    dem = torch.tensor(np.load(index.iloc[row][1]))
    influx = literal_eval(index.iloc[row][2])
    outflux = literal_eval(index.iloc[row][3])
    discharge = index.iloc[row][4]
    time_stamps = literal_eval(index.iloc[row][5])
    return dem, influx, outflux, discharge, time_stamps


def topography_difference(tile: np.ma.masked_array):
    return np.max(tile) - np.min(tile)


def masked_values_percentage(tile: np.ma.MaskedArray):
    return np.sum(tile.mask) / tile.size


def fix_missing_values(tile: np.ma.MaskedArray,
                       masked_value_offset: Optional[float] = 30):
    tile.data[tile.mask] = masked_value_offset + np.max(tile)
    return tile


def divide_to_tiles(image, tile_shape):
    imageSizeRow, imageSizeCol = image.shape
    tile_rows, tile_cols = tile_shape
    numPatchRow = int(imageSizeRow / tile_rows)
    numPatchCol = int(imageSizeCol / tile_cols)
    if ((imageSizeRow > tile_rows * numPatchRow) | (imageSizeCol > tile_cols * numPatchCol)):
        image = np.pad(image, ((0, tile_rows - (imageSizeRow - tile_rows * numPatchRow)),
                               (0, tile_cols - (imageSizeCol - tile_cols * numPatchCol))), 'symmetric')
    height = image.shape[0]
    width = image.shape[1]
    numCol = int(height / tile_rows)
    numRow = int(width / tile_cols)
    tiles = []
    for i in range(0, numCol * numRow):
        tiles.append([])
    count = 0
    for i in range(numCol):
        for j in range(numRow):
            mm = ((i + 1) * tile_rows)
            nn = ((j + 1) * tile_cols)
            img = image[i * tile_rows:mm, j * tile_cols:nn]
            img1 = img[np.newaxis, :, :]
            if count == 0:
                tiles = img1
            else:
                tiles = np.concatenate((tiles, img1), axis=0)
            count += 1
    return tiles


def find_lowest_point(tile: np.ma.MaskedArray):
    rows, cols = tile.shape
    min_index = np.argmin(
        np.ma.concatenate([tile[:, 0], tile[:, -1], tile[0, :],
                           tile[-1, :]]))
    # left, right, up, down
    if min_index < rows:
        return min_index, 0
    if min_index < 2 * rows:
        return min_index - rows, -1
    if min_index < 2 * rows + cols:
        return 0, min_index - 2 * rows
    else:
        return -1, min_index - (2 * rows + cols)
