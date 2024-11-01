import logging
from typing import List, Optional

import matplotlib
import matplotlib.animation as animation
import matplotlib.cm
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LightSource


def render_hillshade_water_image(z: np.ndarray, h: np.ndarray,
                                 vmin: Optional[float] = None,
                                 vmax: Optional[float] = None):
    norm = matplotlib.colors.Normalize(vmin, vmax)
    mappable = matplotlib.cm.ScalarMappable(norm, 'Blues')
    color_h = mappable.to_rgba(h)
    alpha = np.ones_like(h) * 0.85
    alpha = np.dstack([alpha] * 3)
    color_h = color_h[:, :, :3]

    # Shade from the northwest, with the sun 45 degrees from horizontal
    ls = LightSource(azdeg=315, altdeg=45)
    overlay = ls.hillshade(z, vert_exag=1, dx=1, dy=1)
    norm = matplotlib.colors.Normalize(np.min(overlay), np.max(overlay))
    mappable = matplotlib.cm.ScalarMappable(norm, 'gray')
    overlay = mappable.to_rgba(overlay)
    overlay_image = overlay[:, :, :3]

    return (1 - alpha) * overlay_image + alpha * color_h


def render_hillshade_image(z: np.ndarray, vmin: Optional[float] = None,
                           vmax: Optional[float] = None):
    # Shade from the northwest, with the sun 45 degrees from horizontal
    ls = LightSource(azdeg=315, altdeg=45)
    rgb = ls.shade(z, cmap=plt.cm.gist_earth, blend_mode='overlay',
                   vert_exag=1, dx=1, dy=1, vmin=vmin, vmax=vmax)
    return rgb[:, :, :3]


def render_water_simulation_movie(state_list: List[np.ndarray], dem: np.ndarray,
                                  time_list: List[float]):
    matplotlib.rcParams['animation.embed_limit'] = 1000
    fig, ax = plt.subplots(1, 1)
    dem = np.squeeze(dem)
    max_height = -np.inf
    min_height = np.inf
    for value in state_list:
        max_height = max_height if max_height > np.max(value) else np.max(value)
        min_height = min_height if min_height < np.min(value) else np.min(value)
    vmin = 0
    logging.info(f'render water range - [{min_height:.3f},{max_height:.3f}]')
    image = plt.imshow(
        render_hillshade_water_image(dem, state_list[0], vmin=vmin,
                                     vmax=max_height))
    plt.axis('off')

    def update_eta(num):
        ax.set_title(r'Water Surface Height $z+h$ at t = {:.2f} hours'.format(
            time_list[num] / 3600), fontname="serif", fontsize=16)
        image.set_data(render_hillshade_water_image(dem, state_list[num],
                                                    vmin=vmin, vmax=max_height))
        return image

    anim = animation.FuncAnimation(fig, update_eta, frames=len(state_list),
                                   interval=10, blit=False)
    html_movie = anim.to_jshtml(fps=16)
    plt.close()
    return html_movie


def render_dem_evolution_movie(dem_list: List[np.ndarray]):
    fig, ax = plt.subplots(1, 1)
    image = plt.imshow(dem_list[0])
    plt.colorbar()
    plt.axis('off')

    def update_eta(num):
        ax.set_title(f'DEM at step {num}', fontname="serif", fontsize=16)
        image.set_data(dem_list[num])
        plt.clim(np.min(dem_list[num]), np.max(dem_list[num]))
        return image

    anim = animation.FuncAnimation(fig, update_eta, frames=len(dem_list),
                                   interval=10, blit=False)
    html_movie = anim.to_jshtml(fps=16)
    plt.close()
    return html_movie


def plot_dem(dem: np.ndarray, colormap: str = plt.cm.gist_earth,
             colorbar: Optional[bool] = False, vmin: Optional[float] = None,
             vmax: Optional[float] = None):
    figure = plt.figure()
    if not vmax:
        vmax = np.max(dem)
    if not vmin:
        vmin = np.min(dem)
    image = plt.imshow(dem, cmap=colormap, vmax=vmax, vmin=vmin)
    if colorbar:
        plt.colorbar(image)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.tight_layout(pad=0)
    return figure


def plot_difference_map(state_1: np.ndarray, state_2: np.ndarray,
                        label_1: str, label_2: str,
                        vmin: Optional[float] = None,
                        vmax: Optional[float] = None):
    state_1 = state_1.squeeze()
    state_2 = state_2.squeeze()
    if not vmax:
        vmax = np.max(np.abs(state_2 - state_1))
    if not vmin:
        vmin = -vmax
    figure = plt.figure()
    image = plt.imshow(state_1 - state_2, cmap='coolwarm', vmin=vmin, vmax=vmax)
    plt.title(f'Water Difference Map - Red:{label_1}, Blue:{label_2}')
    plt.colorbar(image)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.tight_layout(pad=0)
    plt.close()
    return figure


def plot_loss_histogram(loss_vector: torch.Tensor, path: str, bins: int = 20,
                        min: float = 0, max: float = 0):
    bars = torch.histc(loss_vector, bins=bins, min=min, max=max)
    bars /= torch.sum(bars)
    bin_edges = torch.linspace(min, max, steps=bins)
    plt.figure(figsize=(12, 8))
    plt.bar(bin_edges, bars, width=0.0004)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(path + f'/loss_histogram.png', bbox_inches='tight')
    plt.close()
