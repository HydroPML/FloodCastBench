import abc
import enum
from typing import Sequence, Tuple

import torch
import torch.nn.functional as F

G = 9.8
OUTFLUX_SLOPE = 0.2


def _flux_location_to_indices(dem_shape: int, flux_location: torch.Tensor,
                              down_sample_factor: int):
    x, y, length = flux_location
    rows = dem_shape
    cols = dem_shape
    index = x if x > 0 else y
    index = int(index / down_sample_factor)
    dim = rows if x > 0 else cols
    if length > dim:
        raise ValueError(f'cross section length {length} is longer than DEM'
                         f' dimension {dim}')
    indices = torch.arange(index - length // 2, index + length // 2)
    if index - length // 2 < 0:
        indices += abs(index - length // 2)
    if index + length // 2 > dim:
        indices -= index + length // 2 - dim
    #print('indices_______',indices,indices.shape)
    return indices.to(torch.long)


def calculate_boundaries(dx: float, dem_shape,
                         influx_locations: Sequence[Sequence[int]],
                         outflux_locations: Sequence[Sequence[int]],
                         discharges: Sequence[float], down_sample_factor: int):
    rows = dem_shape
    #print('dem_shape_____',dem_shape)
    cols = dem_shape
    influx_x_list = []  # left, right
    influx_y_list = []  # up, down
    outflux_x_list = []
    outflux_y_list = []
    for influx, outflux, discharge in zip(influx_locations, outflux_locations,
                                          discharges):
        influx_x, influx_y, influx_width = influx
        #print('influx______',influx_x, influx_y, influx_width)
        influx_width /= down_sample_factor
        influx_x_list.append(torch.zeros(rows, 2))
        influx_y_list.append(torch.zeros(cols, 2))
        influx_indices = _flux_location_to_indices(dem_shape, influx,
                                                   down_sample_factor)
        if influx_x > 0 and influx_y == 0:
            influx_x_list[-1][:, 0][
                influx_indices] += discharge / influx_width / dx
        if influx_x > 0 and influx_y == -1:
            influx_x_list[-1][:, 1][
                influx_indices] -= discharge / influx_width / dx
        if influx_x == 0 and influx_y > 0:
            influx_y_list[-1][:, 0][
                influx_indices] += discharge / influx_width / dx
        if influx_x == -1 and influx_y > 0:
            influx_y_list[-1][:, 1][
                influx_indices] -= discharge / influx_width / dx
        outflux_x, outflux_y, outflux_width = outflux
        outflux_width /= down_sample_factor
        outflux_x_list.append(torch.zeros(rows, 2))
        outflux_y_list.append(torch.zeros(cols, 2))
        outflux_indices = _flux_location_to_indices(dem_shape, outflux,
                                                    down_sample_factor)
        if outflux_x > 0 and outflux_y == 0:
            outflux_x_list[-1][:, 0][outflux_indices] = 1
        if outflux_x > 0 and outflux_y == -1:
            outflux_x_list[-1][:, 1][outflux_indices] = 1
        if outflux_x == 0 and outflux_y > 0:
            outflux_y_list[-1][:, 0][outflux_indices] = 1
        if outflux_x == -1 and outflux_y > 0:
            outflux_y_list[-1][:, 1][outflux_indices] = 1
    outflux_x = torch.stack(outflux_x_list)
    outflux_y = torch.stack(outflux_y_list)
    influx_x = torch.stack(influx_x_list) #torch.Size([1, 2000, 2])
    influx_y = torch.stack(influx_y_list)
    #print('iutflux_x________', influx_x, influx_x.shape)
    return influx_x, influx_y, outflux_x, outflux_y


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
    def __init__(self, dx: float, dem_shape: int,
                 influx_location: Sequence[Sequence[int]],
                 outflux_location: Sequence[Sequence[int]],
                 discharge: Sequence[float], down_sample_factor: int = 16):
        self.dx = dx
        self.down_sample_factor = down_sample_factor
        influx_x, influx_y, outflux_x, outflux_y = calculate_boundaries(
            dx, dem_shape, influx_location, outflux_location, discharge,
            self.down_sample_factor)
        self.influx_x = influx_x.unsqueeze(1).cuda()
        self.influx_y = influx_y.unsqueeze(1).cuda()
        self.outflux_x = outflux_x.unsqueeze(1).cuda()
        self.outflux_y = outflux_y.unsqueeze(1).cuda()

    def __call__(self, h_n: torch.Tensor, flux_x: torch.Tensor,
                 flux_y: torch.Tensor
                 ) -> Tuple[float, torch.Tensor, torch.Tensor]:
        flux_x = F.pad(flux_x, pad=[1, 1])
        flux_y = F.pad(flux_y, pad=[0, 0, 1, 1])
        #print('flux_x_____',flux_x.shape)
        flux_x[:, :, :, 0] += self.influx_x[:, :, :, 0].to(flux_x.device)     #flux_x:(1,1,2000,2001),self.influx_x:(1,1,2000,2)  下面两行只有一行有用（左/右）
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




