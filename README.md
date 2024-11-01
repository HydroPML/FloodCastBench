![FloodCastBench](https://github.com/HydroPML/FloodCastBench/blob/main/Figures/figures1.png)
# FloodCastBench Dataset
FloodCastBench provides comprehensive low-fidelity and high-fidelity flood forecasting datasets specifically for machine learning-based spatiotemporal, cross-regional, and downscaled flood modeling and forecasting.


## Data Generation
To use the model, execute the following steps:
1. **preparing the necessary input files and parameters, including topography data (DEM), rainfall data, land use and land cover data (Manning coefficients), as well as initial and boundary conditions.**
2. **Please change the 'paths of different input data and results' to your local path.**
3. **Run Data_Generation_Code/main.py**

## Dataset Structure

FloodCastBench
- Low-Fidelity Flood Forecasting
  - 480 m
    - Pakistan flood
    - Mozambique flood
- High-Fidelity Flood Forecasting
  - 30 m
    - Australia flood
    - UK flood
  - 60 m
    - Australia flood
    - UK flood
- Relevant Data
  - DEM
  - Land use and land cover
  - Rainfall
    - Pakistan flood
    - Mozambique flood
    - Australia flood
    - UK flood
  - Georeferenced files
  - Initial conditions

