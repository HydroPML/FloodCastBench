import abc
import enum
from typing import Sequence, Tuple

import torch
import torch.nn.functional as F


G = 9.8
OUTFLUX_SLOPE = 0.2

# range for influx and outflux
def _flux_location_to_indices(dem_shape: int, flux_location: torch.Tensor):
    x, y, length = flux_location
    rows, cols = dem_shape
    index = x if x > 0 else y
    # index = int(index / down_sample_factor)
    dim = rows if x > 0 else cols
    if length > dim:
        raise ValueError(f'cross section length {length} is longer than DEM'
                         f' dimension {dim}')
    indices = torch.arange(index - length // 2, index + length // 2)
    if index - length // 2 < 0:
        indices += abs(index - length // 2)
    if index + length // 2 > dim:
        indices -= index + length // 2 - dim
    # print('indices_______',indices,indices.shape)
    return indices.to(torch.long)


def calculate_boundaries(dem_shape,
                         influx_locations: Sequence[Sequence[int]],
                         outflux_locations: Sequence[Sequence[int]],
                         discharge: Sequence[float]):
    rows = dem_shape[0]
    #print('dem_shape_____',dem_shape)
    cols = dem_shape[1]
    # influx_x_list = []  # left, right
    # influx_y_list = []  # up, down
    # outflux_x_list = []
    # outflux_y_list = []
    # for influx, outflux, discharge in zip(influx_locations, outflux_locations,
    #                                       discharges):
    dischargein, dischargeout = discharge[0], discharge[1]
    influx_x, influx_y, _ = influx_locations
    influx_width = 180
    influx_x_list = torch.zeros(rows, 2)
    influx_y_list = torch.zeros(cols, 2)
    influx_indices = _flux_location_to_indices(dem_shape, influx_locations)
    if influx_x > 0 and influx_y == 0:
        influx_x_list[:, 0][influx_indices] = dischargein / influx_width
    if influx_x > 0 and influx_y < 0:
        influx_x_list[:, 1][influx_indices] = dischargein / influx_width
    if influx_x == 0 and influx_y > 0:
        influx_y_list[:, 0][influx_indices] = dischargein / influx_width
    if influx_x < 0 and influx_y > 0:
        influx_y_list[:, 1][influx_indices] = dischargein / influx_width
    outflux_x, outflux_y, _ = outflux_locations
    outflux_width = 377
    outflux_x_list = torch.zeros(rows, 2)
    outflux_y_list = torch.zeros(cols, 2)
    outflux_indices = _flux_location_to_indices(dem_shape, outflux_locations)
    # print('outflux_indices', outflux_indices)
    # if outflux_x > 0 and outflux_y == 0:
    #     outflux_x_list[:, 0][outflux_indices] = dischargeout / outflux_width
    # if outflux_x > 0 and outflux_y < 0:
    #     outflux_x_list[:, 1][outflux_indices] = dischargeout / outflux_width
    # if outflux_x == 0 and outflux_y > 0:
    #     outflux_y_list[:, 0][outflux_indices] = dischargeout / outflux_width
    # if outflux_x < 0 and outflux_y > 0:
    #     outflux_y_list[:, 1][outflux_indices] = dischargeout / outflux_width
    if outflux_x > 0 and outflux_y == 0:
        outflux_x_list[:, 0][outflux_indices] = 1
    if outflux_x > 0 and outflux_y == -1:
        outflux_x_list[:, 1][outflux_indices] = 1
    if outflux_x == 0 and outflux_y > 0:
        outflux_y_list[:, 0][outflux_indices] = 1
    if outflux_x == -1 and outflux_y > 0:
        outflux_y_list[:, 1][outflux_indices] = 1
    # outflux_x = torch.stack(outflux_x_list)
    # outflux_y = torch.stack(outflux_y_list)
    # influx_x = torch.stack(influx_x_list) #torch.Size([1, 2000, 2])
    # influx_y = torch.stack(influx_y_list)
    #print('iutflux_x________', influx_x, influx_x.shape)
    return influx_x_list, influx_y_list, outflux_x_list, outflux_y_list


class BoundaryType(enum.Enum):
    FLUX, RAIN = range(2)


class BoundaryConditions(abc.ABC):
    """A class for applying boundary conditions."""

    @abc.abstractmethod
    def __call__(self, h_n: torch.Tensor, flux_x: torch.Tensor,
                 flux_y: torch.Tensor
                 ) -> Tuple[float, torch.Tensor, torch.Tensor]:
        """Applies boundary conditions.

         Returns homogeneous water difference, flux_x and flux_y"""
        raise NotImplementedError('Calling an abstract method.')


class FluxBoundaryConditions(BoundaryConditions):
    def __init__(self, dem_shape: [int, int],
                 influx_location: Sequence[Sequence[int]],
                 outflux_location: Sequence[Sequence[int]],
                 dischargein: Sequence[float]):
        influx_x, influx_y, _ = influx_location
        outflux_x, outflux_y, _ = outflux_location
        self.influx_y = influx_y
        self.outflux_x = outflux_x
        influx_x, influx_y, outflux_x, outflux_y = calculate_boundaries(
            dem_shape, influx_location, outflux_location, dischargein)
        self.influx_x = influx_x.unsqueeze(0).unsqueeze(0).cuda()
        self.influx_y = influx_y.unsqueeze(0).unsqueeze(0).cuda()
        self.outflux_x = outflux_x.unsqueeze(0).unsqueeze(0).cuda()
        self.outflux_y = outflux_y.unsqueeze(0).unsqueeze(0).cuda()

    def __call__(self, h_n: torch.Tensor, flux_x: torch.Tensor,
                 flux_y: torch.Tensor
                 ) -> Tuple[float, torch.Tensor, torch.Tensor]:
        flux_x = F.pad(flux_x, pad=[1, 1])
        flux_y = F.pad(flux_y, pad=[0, 0, 1, 1])
        # print('flux_x_____',flux_x.shape)
        flux_x[:, :, :, 0] += self.influx_x[:, :, :, 0].to(
            flux_x.device)  # flux_x:(1,1,2000,2001),self.influx_x:(1,1,2000,2)  下面两行只有一行有用（左/右）
        print(flux_x[:, :, :, 0].shape)
        flux_x[:, :, :, -1] += self.influx_x[:, :, :, 1].to(flux_x.device)
        flux_y[:, :, 0, :] += self.influx_y[:, :, :, 0].to(flux_y.device)
        flux_y[:, :, -1, :] += self.influx_y[:, :, :, 1].to(flux_y.device)
        flux_x[:, :, :, -1] += G * h_n[:, :, :, -1] * OUTFLUX_SLOPE * (
            self.outflux_x[:, :, :, 1].to(flux_x.device))
        flux_x[:, :, :, 0] -= G * h_n[:, :, :, 0] * OUTFLUX_SLOPE * (
            self.outflux_x[:, :, :, 0].to(flux_x.device))
        flux_y[:, :, 0, :] -= G * h_n[:, :, 0, :] * OUTFLUX_SLOPE * (
            self.outflux_y[:, :, :, 0].to(flux_y.device))
        flux_y[:, :, -1, :] += G * h_n[:, :, -1, :] * OUTFLUX_SLOPE * (
            self.outflux_y[:, :, :, 1].to(flux_y.device))
        return 0, flux_x, flux_y


class RainBoundaryConditions(BoundaryConditions):
    def __init__(self, discharge: torch.Tensor):
        self.discharge = discharge  # meters/second
        self.rainfall_per_pixel = self.discharge.reshape(-1, 1, 1, 1)

    def zero_discharge(self, indices_to_zero: torch.Tensor):
        self.discharge[indices_to_zero] = 0
        self.rainfall_per_pixel = self.discharge.reshape(-1, 1, 1, 1)

    def __call__(self, h_n: torch.Tensor, flux_x: torch.Tensor,
                 flux_y: torch.Tensor
                 ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        flux_x = F.pad(flux_x, pad=[1, 1])
        flux_y = F.pad(flux_y, pad=[0, 0, 1, 1])
        return self.rainfall_per_pixel, flux_x, flux_y

# class RainBoundaryConditions(BoundaryConditions):
#     def __init__(self, discharge: torch.Tensor):
#         self.discharge = discharge  # meters/second
#         self.rainfall_per_pixel = self.discharge.reshape(-1, 1, 1, 1)
#         self.zero = torch.Tensor([0]).to(discharge.device)
#         self.init_source = True
#
#     def zero_discharge(self, indices_to_zero: torch.Tensor):
#         self.discharge[indices_to_zero] = 0
#         self.rainfall_per_pixel = self.discharge.reshape(-1, 1, 1, 1)
#
#     def __call__(self, h_n: torch.Tensor, flux_x: torch.Tensor,
#                  flux_y: torch.Tensor
#                  ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
#         #print(self.init_source)
#
#         flux_x = F.pad(flux_x, pad=[1, 1])
#         flux_y = F.pad(flux_y, pad=[0, 0, 1, 1])
#
#         if self.init_source  == True:
#             self.init_source = False
#             return self.rainfall_per_pixel, flux_x, flux_y
#
#         elif self.init_source == False:
#             return self.rainfall_per_pixel , flux_x, flux_y




