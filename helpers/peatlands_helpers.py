import sys
sys.path.append("/workspace/DQTools/")
#sys.path.append("/workspace//")

import matplotlib
matplotlib.use('nbagg')
import matplotlib.pyplot as plt
import pandas as pd

import ipywidgets as widgets
from pyproj import Proj, transform

from IPython.display import clear_output
from IPython.lib.display import FileLink

import helpers.helpers as helpers
from DQTools.dataset import Dataset
from DQTools.search import Search


class PeatHelpers(helpers.Helpers):

    def get_data_from_datacube(self, product, subproduct, start, end,
                               latitude, longitude, projection=None):

        ds = Dataset(product=product,
                     subproduct=subproduct)

        ds.get_data(start=start, stop=end, projection=projection,
                    latlon=[latitude, longitude])

        return ds.data

    def reproject_coords(self, y, x, projection):
        if projection == "British National Grid":
            lat, lon = self.bng_to_latlon(y, x)
        else:
            # Assuming coords in lat lon if not BNG
            lat, lon = y, x
        return lat, lon

    def bng_to_latlon(self, northing, easting):
        """ convert British National Grid easting and northing to latitude and longitude"""
        bng_proj = Proj(
            '+proj=tmerc +lat_0=49 +lon_0=-2 +k=0.9996012717 +x_0=400000 +y_0=-100000 +ellps=airy +towgs84=446.448,-125.157,542.06,0.1502,0.247,0.8421,-20.4894 +units=m +no_defs=True')
        latlon_proj = Proj(init='epsg:4326')
        lon, lat = transform(bng_proj, latlon_proj, easting, northing)
        return lat, lon

    def data_to_csv(self, product, subproduct,
                    projection, y, x, start, end):

        with self.out:
            clear_output()
            print("Getting data...")

            print(f'y={y}, x={x}')
            lat, lon = self.reproject_coords(y, x, projection)
            print(f'lat={lat}, lon={lon}')
            data = self.get_data_from_datacube(product,
                                               subproduct,
                                               pd.to_datetime(start),
                                               pd.to_datetime(end),
                                               lat,
                                               lon,
                                               projection)
            st = pd.to_datetime(start)
            en = pd.to_datetime(end)
            filename = f"{product}_{subproduct}_{projection}_{y}_{x}" \
                       f"_{st.date()}_{en.date()}.csv"
            data.to_dataframe().to_csv(filename)
            localfile = FileLink(filename)
            display(localfile)

            plt.figure(figsize=(8, 6))
            data.__getitem__(subproduct).plot()
            plt.show()


class PeatWidgets(helpers.Widgets):

    def get_projection_widgets(self):
        return self.projection()

    def projection(self):
        projection_list = ['WGS84', 'British National Grid', 'Sinusoidal']
        return widgets.Dropdown(
            options=projection_list,
            description='Projection:',
            layout=self.item_layout,
            disabled=False, )

    def get_x_attributes(self, projection):
        if projection == "British National Grid":
            attributes = {"min": -9999999, "max": 9999999, "description": "Easting (x)"}
        else:
            attributes = {"min": -180, "max": 180, "description": "Longitude (x)"}
        return attributes

    def get_y_attributes(self, projection):
        if projection == "British National Grid":
            attributes = {"min": -9999999, "max": 9999999, "description": "Northing (y)"}
        else:
            attributes = {"min": -90, "max": 90, "description": "Latitude (y)"}
        return attributes
