import argparse
import numpy as np
import random
import torch
from systems_pbc import *
import torch.backends.cudnn as cudnn
from utils import *
from visualize import *
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib.image as pm
import torch.nn as nn
import tifffile
from data import utils
import scipy.ndimage
import boundary
from PIL import Image
import torch.nn.functional as F
from hydraulics import saint_venant
from util import visualization
import io
import os
from scipy import interpolate
from skimage.transform import resize
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
gpu_ids = [0]
output_device = gpu_ids[0]


################
# Arguments
################
parser = argparse.ArgumentParser(description='data generation for FloodCastBench')
#train
parser.add_argument('--theta', type=float, default=0.7, help='q centered weighting. [0,1].')

args = parser.parse_args()
# CUDA support
if torch.cuda.is_available():
    device = torch.device('cuda')
    print(device)
else:
    device = torch.device('cpu')
    print(device)


############################
# Process data
############################
def inter(array, size):
    h, w = array.shape
    new_h, new_w = np.floor_divide((h, w), size)
    x = np.linspace(0, w - 1, w)
    y = np.linspace(0, h - 1, h)
    new_x = np.linspace(0, w - 1, new_w)
    new_y = np.linspace(0, h - 1, new_h)
    f = interpolate.interp2d(x, y, array, kind='linear')
    array_down = f(new_x, new_y)
    # array_down = resize(array, (new_h, new_w), order=1, anti_aliasing=True)
    return array_down
def find_lowest_point(tile):
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

# Parameters
g = torch.tensor(9.80616, dtype=torch.float64)
dem_tif_path = 'Path_to_DEM'
pre_path = 'Path_to_rainfall'
man_path = 'Path_to_manning' # manning is calculted by land use and land cover
ini_height = 'Path_to_ini_height' # initinal conditions (water depth)
dem_map = tifffile.imread(dem_tif_path)
print('dem_map_original', dem_map.shape)
dem_map = dem_map[150:-50, 50:-50]
# DEM z
def process_dem(dem_map):
    # with open(dem_tif_path, 'rb') as f:
    #     tiffbytes = f.read()
    np_ma_map = np.ma.masked_array(dem_map, mask=(dem_map < -2000))
    np_ma_map = utils.fix_missing_values(np_ma_map)
    dem = torch.from_numpy(np_ma_map)
    return dem.float().to(device)
# Precipitation
def precip(pre_path, current_time, downsampling = True):
    pre_list = os.listdir(pre_path)
    pre_list.sort()
    # mp = int((i * t0) / 60)
    # pre_list = pre_list[mp]
    pre_list_train = []
    mp = int(current_time / 1800)
    pre_list_train.append(pre_list[mp])

    print('pre_list', pre_list_train)
    count = 0
    for f in pre_list_train:
        f_path = os.path.join(pre_path, f)
        imag = tifffile.imread(f_path)
        imag = imag[150:-50, 50:-50]
        if downsampling == True:
            imag = inter(imag, 16)
        imag[imag < 0] = 0
        imag = torch.from_numpy(imag)
        imag = torch.unsqueeze(imag, dim=0)
        if count == 0:
            tiles_time = imag
        else:
            tiles_time = torch.cat((tiles_time, imag), dim=0)
        count = count + 1
    tiles_time = torch.unsqueeze(tiles_time, dim=0)
    tiles_time = tiles_time.float().to(device)
    return tiles_time

# Manning Coefficient
def manning(man_path, downsampling = True):
    img = np.load(man_path)
    np_map = np.array(img)
    np_map = np_map[150:-50, 50:-50]
    if downsampling == True:
        np_map = inter(np_map, 16)
    print('max_manning', np.mean(np_map))
    man = torch.from_numpy(np_map)
    man = torch.unsqueeze(man, dim=0)
    man = torch.unsqueeze(man, dim=0)
    return man.float().to(device)

# initial water level h
def gen_init(ini_height, downsampling = False):
    imag = tifffile.imread(ini_height)
    rows, cows = imag.shape
    if downsampling == True:
        imag = inter(imag, 16)
    imag[imag < 0] = 0
    imag = imag * 2.50
    image = torch.from_numpy(imag)
    return image.float().to(device)

def dis_init(ini_discharge, downsampling = True):
    imag = tifffile.imread(ini_discharge)
    rows, cows = imag.shape
    if downsampling == True:
        imag = inter(imag, 16)
    imag[imag < 0] = 0
    imag = imag * 365.2
    print('dis_init',np.min(imag))
    imag = torch.from_numpy(imag)

    return imag.float().to(device)

def boundary_r(dem_map, mm, downsampling = True):
    rows, cols = dem_map.shape
    dem_shape = [rows, cols]
    # outflux = (x, y, OUTFLUX_LENGTH)
    # if downsampling == True:
    #     outflux = (-1, 254, 10)
    # else:
    outflux = (123, 0, 10) # the outflux (x, y, OUTFLUX_LENGTH)
    influx = (61, -1, 10) # the influx (x, y, INFLUX_LENGTH)

    # discharge in influx and outflux
    # print('influx', influx)
    dischargein = [0, 0, 0, 0, 0, 0] # [m^3/s] cubic meters per second
    dischargeiout = [0, 0, 0, 0, 0, 0] # [m^3/s] cubic meters per second
    dischargein_a = np.array(dischargein[mm])
    dischargeout_a = np.array(dischargeiout[mm])
    discharge = [dischargein_a, dischargeout_a]
    # dischargeout = 1
    # FluxBoundaryConditions
    boundary_conditions = boundary.FluxBoundaryConditions(
        dem_shape, influx, outflux, discharge)
    return boundary_conditions

def cfl(dx: float, max_h: torch.Tensor, alpha: float) -> torch.Tensor:
    return alpha * dx / (g + max_h)

def train():
    # physical domain
    days = 6
    # T = 86400
    lmbdleft, lmbdright = 0, dem_map.shape[0] * 30
    thtlower, thtupper = 0, dem_map.shape[1] * 30
    t00, tfinal = 0, days*24*60*60
    dx = 30.0
    dt0 = 30

    # # model
    resolution = dx
    solver = saint_venant.SaintVenantFlux(resolution, args.theta)
    solver.cuda()
    target_time = torch.tensor(tfinal, dtype=torch.float).to(device)
    current_time = torch.tensor(t00, dtype=torch.float).to(device)
    print('current_time', current_time)
    z_n = process_dem(dem_map)
    print('z_n', z_n.shape)
    z_n = torch.unsqueeze(z_n, dim=0)
    z_n = torch.unsqueeze(z_n, dim=0)
    # rainfall
    # manning
    Manning = manning(man_path, downsampling=False)
    h_gt = []
    qx_gt = []
    qy_gt = []
    nn = 1
    while not torch.isclose(current_time, target_time, rtol=0, atol=0.5).all():
        log_dir = 'Path_to_Results/'
        mm = int(current_time / 86400)
        # boundary = boundary_r(dem_map, mm, downsampling=False)
        Rain = precip(pre_path, current_time, downsampling=False)
        if current_time == 0:
            # h_n0 = gen_init(ini_height, downsampling=False)
            # h_n0 = torch.unsqueeze(h_n0, dim=0)
            # h_n0 = torch.unsqueeze(h_n0, dim=0)
            h_n0 = torch.zeros(z_n.shape[0], z_n.shape[1], z_n.shape[2], z_n.shape[3]).float().to(device)
            h_n = torch.zeros(z_n.shape[0], z_n.shape[1], z_n.shape[2], z_n.shape[3]).float().to(device)
            q_x_n = torch.zeros(h_n.shape[0], h_n.shape[1],
                                h_n.shape[2], h_n.shape[3] - 1).float().to(device)
            q_y_n = torch.zeros(h_n.shape[0], h_n.shape[1],
                                h_n.shape[2] - 1, h_n.shape[3]).float().to(device)
            # h_gt.append(h_n0)
            # qx_gt.append(q_x_n)
            # qy_gt.append(q_y_n)
            h, qx, qy = torch.squeeze(h_n0), torch.squeeze(q_x_n), torch.squeeze(q_y_n)
            h, qx, qy = h.cpu().detach().numpy(), qx.cpu().detach().numpy(), qy.cpu().detach().numpy()
            h_aa = Image.fromarray(h)
            qx_aa = Image.fromarray(qx)
            qy_aa = Image.fromarray(qy)
            h_aa.save(os.path.join(log_dir, 'h_gt/%s.tiff' % current_time))
        alpha = 0.7
        MIN_H_N = 0.01
        min_h_n = MIN_H_N * torch.ones(h_n.shape[0]).cuda()
        max_h = torch.max(h_n.reshape(h_n.shape[0], -1).max(dim=1).values, min_h_n)
        dt = cfl(dx, max_h, alpha)
        dt = torch.min(torch.abs(dt0 * nn - current_time), dt.squeeze()).reshape_as(dt)
        h_n, q_x_n, q_y_n = solver(z_n, h_n, q_x_n, q_y_n, dt, Rain, Manning, h_n0)
        if torch.isnan(h_n).any():
            raise RuntimeError('nan values found in coarse solver.')
        h_n = F.threshold(h_n, threshold=0, value=0)
        q_x_n = F.threshold(q_x_n, threshold=0, value=0)
        q_y_n = F.threshold(q_y_n, threshold=0, value=0)
        current_time += dt.squeeze()
        print('current_time', current_time)
        if current_time % dt0 == 0:
            Rain_o = Rain.clone().detach().squeeze().cpu().numpy()
            Rain_out = np.mean(Rain_o)
            file1 = "Path_to_Rseults_Rain.txt"
            with open(file1, 'a', encoding='utf-8') as f:
                f.writelines(str(Rain_out) + '\n')
            nn = nn + 1
            # h_gt.append(h_n)
            # qx_gt.append(q_x_n)
            # qy_gt.append(q_y_n)
            h, qx, qy = torch.squeeze(h_n), torch.squeeze(q_x_n), torch.squeeze(q_y_n)
            h, qx, qy = h.cpu().detach().numpy(), qx.cpu().detach().numpy(), qy.cpu().detach().numpy()
            h_n = F.threshold(h_n, threshold=0, value=0)
            q_x_n = F.threshold(q_x_n, threshold=0, value=0)
            q_y_n = F.threshold(q_y_n, threshold=0, value=0)
            file10 = "Path_to_Rseults_current_time.txt"
            with open(file10, 'a', encoding='utf-8') as f10:
                f10.writelines(str(current_time.item()) + '\n')
            h_n_o = h_n.clone().detach().squeeze().cpu().numpy()
            h_n_out = np.mean(h_n_o)
            file11 = "Path_to_Rseults_water_height.txt"
            with open(file11, 'a', encoding='utf-8') as f11:
                f11.writelines(str(h_n_out) + '\n')
            _EPSILON = 1e-6
            q_x_n_o = q_x_n.clone().detach().squeeze().cpu().numpy()
            q_x_n_out = np.mean(q_x_n_o)
            q_y_n_o = q_y_n.clone().detach().squeeze().cpu().numpy()
            q_y_n_out = np.mean(q_y_n_o)
            q_a = (q_x_n_out ** 2 + q_y_n_out ** 2 + _EPSILON) ** 0.5
            file12 = "Path_to_Rseults_discharge.txt"
            with open(file12, 'a', encoding='utf-8') as f12:
                f12.writelines(str(q_a) + '\n')
            h_aa = Image.fromarray(h)
            qx_aa = Image.fromarray(qx)
            qy_aa = Image.fromarray(qy)
            h_aa.save(os.path.join(log_dir, 'h_gt/%s.tiff' % current_time))
            # qx_aa.save(os.path.join(log_dir, 'qx_gt/%s.tiff' % current_time))
            # qy_aa.save(os.path.join(log_dir, 'qy_gt/%s.tiff' % current_time))
        # print('len_supervised', len(h_gt))

if __name__ == '__main__':
    train()