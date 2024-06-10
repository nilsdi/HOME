#%% imports and definition of class
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import json


from rasterio.transform import from_origin
from rasterio.features import rasterize
from rasterio.plot import show

from shapely.geometry import shape

import matplotlib.cm as cm
from matplotlib import colors
from matplotlib.colors import ListedColormap
from matplotlib.colors import Normalize

from pathlib import Path

class ProjectDensityGrid():
    ''' 
    Handles the project density grid for an area.
    Key methods:
    - plot_density_grid: plots the density grid
    - save_density_grid: saves the density grid
    '''
    def __init__(self, 
                        metadata_all_projects: list,
                        region_shape_file:str,
                        resolution: float = 500, # resolution in meters per pixel
                        region:str = 'Norway') -> None:
        '''
        Args:
        metadata_all_projects (list): metadata for all projects, needs to work with ___
        region_shape (str): path to a geopanda readable shape of the region
        '''
        self.metadata_all_projects = metadata_all_projects
        self.region_shape_utm = gpd.read_file(region_shape_file).to_crs('EPSG:25833')
        
        self.resolution = resolution
        self.region = region
        self.region_grid, self.transform, self.raster_width, self.raster_height = self.create_region_grid()

        self.density_grid = self.get_density_grid()
        self.oldest_grid = self.get_oldest_project_grid()
        #self._display_raster(density_grid, self.transform, cmap='gist_heat_r', 
        #                                plot_boundaries=True, plot_colorbar=True, 
        #                                color_bar_label='Project coverage - # of projects')
    def create_region_grid(self):
        '''
        Creates a grid of the region that we can use to fill it up.
        '''
        bounds, transform = self._get_region_transform()
        width = int(np.ceil((bounds[2] - bounds[0]) / self.resolution))
        height = int(np.ceil((bounds[3] - bounds[1]) / self.resolution))

        # creaty empty grid
        raster = np.zeros((height, width), dtype=np.uint8)
        return raster, transform, width, height
    
    def _get_region_transform(self):
        ''' 
        transform to match other geodata onto the region grid
        '''
        bounds = self.region_shape_utm.total_bounds
        buffer = 10**5 # 100 km buffer
        bounds[0] -= buffer
        bounds[1] -= buffer
        bounds[2] += buffer
        bounds[3] += buffer
        #print(bounds)
        width = int(np.ceil((bounds[2] - bounds[0]) / self.resolution))
        height = int(np.ceil((bounds[3] - bounds[1]) / self.resolution))
        transform = from_origin(bounds[0], bounds[3], self.resolution, self.resolution)
        return bounds, transform
    
    def _get_project_grid(self, 
                                        project_geometry:dict,
                                        transform,
                                        width:int,
                                        height:int,
                                        area_fill:float = 1, 
                                        empty_fill:float = 0):
        '''
        Args:
        project_geometry (dict): geometry of the project in crs ('EPSG:32633')
        area_fill (float): fill value for the project area
        empty_fill (float): fill value for the empty area
        '''
        project_raster = rasterize(project_geometry, out_shape=(height, width), 
                                                fill = empty_fill, default_value=area_fill, 
                                                transform=transform, dtype=np.uint8)
        return project_raster
    
    def _get_project_geometry(self, project_metadata:dict):
        '''
        Get the geometry of the project - may need to be adjusted for different metadata formats
        '''
        # for Norway:
        if self.region.lower() == 'norway':
            project_geometry = gpd.GeoSeries(shape(project_metadata['geometry']))
            project_geometry.crs = 'EPSG:25833'
            project_geometry = project_geometry.to_crs('EPSG:32633')
            return project_geometry
        else:
            raise NotImplementedError('fetching the geometry of metadata for the Region not implemented') 
    
    def _get_project_year(self, project_metadata:dict):
        '''
        Get the year of the project - may need to be adjusted for different metadata formats
        '''
        # for Norway:
        if self.region.lower() == 'norway':
            project_year = int(project_metadata['properties']['aar'])
            return project_year
        else:
            raise NotImplementedError('fetching the year of metadata for the Region not implemented')

    def get_density_grid(self):
        '''
        Get the density grid for the region
        '''
        density_grid = np.zeros((self.raster_height, self.raster_width), dtype=np.uint8)
        for project_metadata in self.metadata_all_projects:
            project_geometry = self._get_project_geometry(project_metadata)
            project_raster = self._get_project_grid(project_geometry, self.transform, self.raster_width, self.raster_height)
            density_grid += project_raster
        return density_grid

    def get_oldest_project_grid(self):
        ''' 
        get the grid of the oldest project for each year
        '''
        coverage_grid = self.density_grid.copy()
        coverage_grid[coverage_grid > 0] = 1
        coverage_grid = coverage_grid*5000
        for project_metadata in self.metadata_all_projects:
            project_geometry = self._get_project_geometry(project_metadata)
            project_raster = self._get_project_grid(project_geometry, self.transform, 
                                                                            self.raster_width, self.raster_height)
            project_year = self._get_project_year(project_metadata)
            
            # we want to make sure only the pixels with the actual project coverage get the year
            # the remaining pixels should get a year that is much higher.
            project_raster = project_raster * -project_year
            project_raster = project_raster + np.ones(np.shape(project_raster)) * project_year*2
            coverage_grid = np.minimum(coverage_grid, project_raster)
        return coverage_grid
    
    def _display_raster(self, raster, transform, 
                                    cmap='viridis',
                                    plot_boundaries:bool = True,
                                    plot_colorbar:bool = True,
                                    color_bar_label:str = None,
                                    ignore_zero:bool = False,
                                    show_plot:bool = True, 
                                    save_as:str = None,
                                    figsize:tuple = None) -> None:

        '''
        Display any raster
        '''
        if figsize:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig, ax = plt.subplots(figsize=(15, 15))
        if ignore_zero:
            '''
            # Create a colormap
            cmap = plt.get_cmap(cmap)
            # Set the color of the zero values to white
            cmap.set_bad('white',0)
            # Create a normalized color map
            norm = colors.Normalize(vmin=np.min(raster[raster > 0]), vmax=np.max(raster))
            #show(raster, transform=transform, ax=ax, cmap=cmap, norm = norm)
            # Mask the zero values in the raster
            raster = np.ma.masked_where(raster == 0, raster)
            # Use imshow with raster.filled() to apply the set_bad color to masked values
            ax.imshow(raster.filled(), cmap=cmap, norm=norm)
            '''
            # Create a colormap
            cmap = plt.get_cmap(cmap)
            # Create a new colormap that maps zero values to white and non-zero values to the original colormap
            colors = ['white'] + [cmap(i) for i in range(1, cmap.N)]
            cmap = ListedColormap(colors)
            # Create a normalized color map
            norm = Normalize(vmin=np.min(raster[raster > 0]), vmax=np.max(raster))
            # Mask the zero values in the raster
            raster = np.ma.masked_where(raster == 0, raster)
            # Use imshow with raster.filled() to apply the set_bad color
            ax.imshow(raster.filled(), cmap=cmap, norm=norm)
        else:
            #show(raster, transform=transform, ax=ax, cmap=cmap)
            ax.imshow(raster, cmap=cmap) # doesnt work with the boundaries 
        if plot_boundaries:
            self.region_shape_utm.boundary.plot(ax=ax, color = 'darkgrey', linewidth=0.5)
            
        if plot_colorbar:
            if ignore_zero:
                sm = cm.ScalarMappable(cmap=cmap, norm=norm)
                cb_ticks = [int(n) for n in np.linspace(np.min(raster[raster > 0]), np.max(raster), 5)]
            else:
                max_value = np.max(raster)
                min_value = np.min(raster)
                # Create a ScalarMappable instance
                sm = cm.ScalarMappable(cmap=cmap, 
                                                        norm=plt.Normalize(vmin=min_value, vmax=max_value))
                cb_ticks = [int(n) for n in np.linspace(np.min(raster), np.max(raster), 5)]

            # Create a colorbar
            cbar = plt.colorbar(sm, ax=ax, orientation='vertical', ticks=cb_ticks)
            if color_bar_label:
                cbar.set_label(color_bar_label) # 'Project coverage - # of projects'

        if save_as:
            plt.savefig(save_as, dpi = 300)
        if show_plot:
            plt.show() 
        return

#%% actually plot

# get the path to the shape file
root_directory = Path().resolve().parents[1]
# add path to Norway shapes in data to path
path_to_shape = root_directory / 'data'/'raw'/'maps'/'Norway_boundaries'/'NOR_adm0.shp'

# load the metadata
metadata_file = 'metadata_all_projects_20240523000926.json'
path_to_data = root_directory / 'data' / 'raw' / 'orthophoto'

with open(path_to_data / metadata_file, 'r') as f:
    metadata_all_projects = json.load(f)

# create the density grid
Norway_density = ProjectDensityGrid(metadata_all_projects['ProjectMetadata'], 
                                                            region_shape_file = path_to_shape, 
                                                            resolution=10000)
# plot the density grid
Norway_density._display_raster(Norway_density.density_grid, Norway_density.transform, 
                                                    cmap='gist_heat_r', plot_boundaries=False, plot_colorbar=True, 
                                                    color_bar_label='Project coverage - # of projects')
Norway_density._display_raster(Norway_density.oldest_grid, Norway_density.transform,
                                                    cmap='viridis', plot_boundaries=False, plot_colorbar=True, 
                                                    ignore_zero=True,
                                                    color_bar_label='Oldest project year', figsize=(15, 15))
# %%
