import base64
import sys
from dataclasses import dataclass

import cv2
import jinja2
import numpy as np
import torch
from matplotlib import figure as plt_fig
from matplotlib.backends import backend_agg

from utils import visualization as viz


@dataclass
class Sample:
    """Class for storing sample outputs."""
    sample_id: int
    loss: float
    model_dem: str
    baseline_dem: str
    model_gt_difference: str
    model_solution: str
    baseline_solution: str


def rgb_to_png_coded_string(rgb_image: np.ndarray) -> str:
    """Converts RGB numpy array to png decoded string."""
    scaled_rgb_image = (np.clip(rgb_image, 0.0, 1.0) * 255).astype(np.uint8)
    success, png_image = cv2.imencode('.png', scaled_rgb_image[:, :, ::-1])
    if not success:
        raise ValueError('Error encoding PNG image')
    return base64.b64encode(png_image.tostring()).decode('utf8')


def plt_figure_to_rgb(figure: plt_fig.Figure) -> np.ndarray:
    canvas = backend_agg.FigureCanvasAgg(figure)
    width, height = figure.get_size_inches() * figure.get_dpi()
    canvas.draw()  # draw the canvas, cache the renderer
    rgb_image = np.fromstring(canvas.tostring_rgb(), dtype='uint8').reshape(
        int(height), int(width), 3)
    return rgb_image / 255


def export_evaluation_html(model: str, sample_id: int, loss: float,
                           predicted_dem: torch.Tensor,
                           baseline_dem: torch.Tensor,
                           predicted_state: torch.Tensor,
                           true_state: torch.Tensor) -> str:
    baseline_dem = baseline_dem.clone().detach().squeeze().cpu().numpy()
    predicted_dem = predicted_dem.clone().detach().squeeze().cpu().numpy()
    true_state = true_state.clone().detach().squeeze().cpu().numpy()
    predicted_state = predicted_state.clone().detach().squeeze().cpu().numpy()
    vmax = max(np.max(true_state), np.max(predicted_state))
    vmin = min(np.min(true_state), np.min(predicted_state))
    diff_figure = viz.plot_difference_map(predicted_state, true_state, 'model',
                                          'true')
    true_state = viz.render_hillshade_water_image(
        baseline_dem, true_state, vmin, vmax)
    predicted_state = viz.render_hillshade_water_image(
        predicted_dem, predicted_state, vmin, vmax)
    vmax = max(np.max(baseline_dem), np.max(predicted_dem))
    vmin = min(np.min(baseline_dem), np.min(predicted_dem))
    baseline_dem_figure = viz.plot_dem(baseline_dem, colorbar=True, vmin=vmin,
                                       vmax=vmax)
    predicted_dem_figure = viz.plot_dem(predicted_dem, colorbar=True, vmin=vmin,
                                        vmax=vmax)
    baseline_dem_rgb = plt_figure_to_rgb(baseline_dem_figure)
    predicted_dem_rgb = plt_figure_to_rgb(predicted_dem_figure)
    predicted_true_diff = plt_figure_to_rgb(diff_figure)
    samples = [Sample(sample_id=sample_id,
                      loss=loss,
                      model_dem=rgb_to_png_coded_string(predicted_dem_rgb),
                      baseline_dem=rgb_to_png_coded_string(baseline_dem_rgb),
                      model_solution=rgb_to_png_coded_string(predicted_state),
                      baseline_solution=rgb_to_png_coded_string(true_state),
                      model_gt_difference=rgb_to_png_coded_string(
                          predicted_true_diff))]
    resource_loader = jinja2.FileSystemLoader(searchpath="./")
    jinja_env = jinja2.Environment(
        loader=resource_loader,
        autoescape=jinja2.select_autoescape(['html', 'xml'])
    )
    template = jinja_env.get_template('utils/evaluation_template.html')
    model_str = 'Model' if model else 'Baseline'

    return template.render(title=f'Sample Number {sample_id} - {model_str}',
                           samples=samples, command=sys.argv)
