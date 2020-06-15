from __future__ import print_function

import sys
sys.path.append("..")

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import matplotlib
# matplotlib.use('nbagg')
import matplotlib.pyplot as plt
import numpy as np
import os
import datetime
import pandas as pd

from traitlets import traitlets

import ipywidgets as widgets
from ipyleaflet import (
    Map, Marker, basemaps, basemap_to_tiles,
    TileLayer, ImageOverlay, Polyline, Polygon, Rectangle,
    GeoJSON, WidgetControl, DrawControl, LayerGroup, FullScreenControl,
    interactive)

from IPython.display import display, clear_output
from IPython.lib.display import FileLink

from DQTools.DQTools.dataset import Dataset
from DQTools.DQTools.search import Search


class Helpers:

    def __init__(self, out, keyfile=None):

        self.out = out
        if keyfile is None:
            self.keyfile = os.path.join(os.path.expanduser("~"), 
                                        'assimila_dq.txt')
        else:
            self.keyfile = keyfile

    def get_data_from_datacube(self, product, subproduct, start, end,
                               latitude, longitude):

        ds = Dataset(product=product, 
                     subproduct=subproduct,
                     key_file=self.keyfile)

        ds.get_data(start=start, stop=end,
                    latlon=[latitude, longitude])

        return ds.data

    def get_data_from_datacube_nesw(self, product, subproduct, north, east,
                                    south, west, start, end):

        with self.out:
            clear_output()
            print("Getting data...")

            ds = Dataset(product=product, 
                         subproduct=subproduct,
                         key_file=self.keyfile)

            ds.get_data(start=start, stop=end,
                        region=[north, east, south,west])

            return ds.data

    def check(self, north, east, south, west, start, end):
        if str(end) < str(start):
            raise ValueError('End date should not be before start date')

        if east and west and east < west:
            raise ValueError('East value should be greater than west')

        if north and south and north < south:
            raise ValueError('North value should be greater than south')

    def color_map_nesw(self, product, subproduct, north, east, south, west,
                       date, hour):

        with self.out:
            clear_output()
            print("Getting data...")

            start = Helpers.combine_date_hour(self, date, hour)
            end = Helpers.combine_date_hour(self, date, hour)

            Helpers.check(self, north, east, south, west, start, end)

            list_of_results = Helpers.get_data_from_datacube_nesw(
                self, product, subproduct, north, east, south, west, start, end)

            y = list_of_results
            y.__getitem__(subproduct).plot()
            plt.show()

    def color_map_nesw_compare(self, product, subproduct, north, east, south,
                               west, date1, hour1, date2, hour2):

        with self.out:
            clear_output()
            print("Getting data...")

            start1 = Helpers.combine_date_hour(self, date1, hour1)
            end1 = Helpers.combine_date_hour(self, date1, hour1)
            start2 = Helpers.combine_date_hour(self, date2, hour2)
            end2 = Helpers.combine_date_hour(self, date2, hour2)

            Helpers.check(self, north, east, south, west, start1, end1)
            Helpers.check(self, north, east, south, west, start2, end2)

            list_of_results1 = Helpers.get_data_from_datacube_nesw(
                self, product, subproduct, north, east,
                south, west, start1, end1)
            y1 = list_of_results1

            list_of_results2 = Helpers.get_data_from_datacube_nesw(
                self, product, subproduct, north, east,
                south, west, start2, end2)

            y2 = list_of_results2

            fig, axs = plt.subplots(1, 2, figsize=(9, 4))
            y1.__getitem__(subproduct).plot(ax=axs[0])
            y2.__getitem__(subproduct).plot(ax=axs[1])
            plt.tight_layout()
            plt.show()

    def prepare_map(dc, m):

        dc.rectangle = {'shapeOptions': {'color': '#FF0000'}}
        dc.marker = {"shapeOptions": {"fillColor": "#fca45d",
                                      "color": "#fca45d", "fillOpacity": 1.0}}
        dc.polyline = {}
        dc.polygon = {}
        dc.circlemarker = {}

        # Create a group of layers and add it to the Map
        group = LayerGroup()
        m.add_layer(group)

        # given Africa: N: 38.25, S: -36.25, E: 53.25, W: -19.25
        africa = GeoJSON(
            data={'type': 'Feature', 'properties':
                {'name': "Africa", 'style':
                    {'color': '#0000FF', 'clickable': True}},
                  'geometry': {'type': 'Polygon',
                               'coordinates': [[[-19, 38], [53, 38],
                                                [53, -36], [-19, -36]]]}},
            hover_style={'fillColor': '03449e'})

        group.add_layer(africa)

        # given Colombia: N: 13.75, S: -5.25, E: -62.75, W: -83.25
        colombia = GeoJSON(data={'type': 'Feature',
                                 'properties': {'name': "Colombia",
                                                'style': {'color': '#0000FF',
                                                          'clickable': True}},
                                 'geometry': {'type': 'Polygon',
                                              'coordinates': [[[-83, 14],
                                                               [-63, 14],
                                                               [-63, -5],
                                                               [-83, -5]]]}},
                           hover_style={'fillColor': '03449e'})

        group.add_layer(colombia)

    def get_coords_point(self, action, geo_json):

        coords = (geo_json.get('geometry', 'Point'))
        x = coords.get('coordinates')[0]
        y = coords.get('coordinates')[1]
        north = y
        south = y
        east = x
        west = x
        return north, east, south, west

    def get_coords_polygon(self, action, geo_json):

        poly = (geo_json.get('geometry', 'Polygon'))
        coords = poly.get('coordinates')[0]
        SW = coords[0]
        NW = coords[1]
        NE = coords[2]
        SE = coords[3]
        north = (NW[1] + NE[1]) / 2
        east = (NE[0] + SE[0]) / 2
        south = (SW[1] + SW[1]) / 2
        west = (NW[0] + SW[0]) / 2

        return north, east, south, west

    def update_nesw(x):
        def create_wid(a):
                w.observe(on_change)
                return a
        w = interactive(create_wid, a = x)
        def on_change(change):
            if change['type'] == 'change' and change['name'] == 'value':
                x.value = change['new']
                #print(x.value)
        w.observe(on_change)
        return w

    def mouse_interaction(m, label):
        def handle_interaction(**kwargs):
            if kwargs.get('type') == 'mousemove':
                label.value = str(kwargs.get('coordinates'))
        m.on_interaction(handle_interaction)
        display(label)

    def compare_rfe_skt_time(self, longitude, latitude, start, end):
        Helpers.check(self, None, None, None, None, start, end)

        with self.out:
            clear_output()
            print("Getting data...")

            # temperature
            list_of_results = self.get_data_from_datacube('era5', 'skt',
                                                          start, end,
                                                          latitude, longitude)
            x = list_of_results.skt - 273.15

            # rainfall
            list_of_results = self.get_data_from_datacube('tamsat', 'rfe',
                                                          start, end,
                                                          latitude, longitude)
            y = list_of_results.rfe

            fig, ax1 = plt.subplots()

            color = 'tab:red'
            ax1.set_xlabel('time')
            ax1.set_ylabel('skt', color=color)
            ax1.tick_params(axis='y', labelcolor=color)
            ax1.set_title("")
            x.plot(ax=ax1, color=color)
            plt.title('rfe and skt against time')

            # instantiate a second axes that shares the same x-axis
            ax2 = ax1.twinx()
            color = 'tab:blue'
            ax2.set_ylabel('rfe', color=color)
            ax2.tick_params(axis='y', labelcolor=color)
            y.plot(ax=ax2, color=color)
            plt.title('rfe and skt against time')

            # otherwise the right y-label is slightly clipped
            fig.tight_layout()
            plt.show()

    def compare_rainfall_products(self, latitude, longitude, start, end):

        with self.out:

            clear_output()
            print("Getting data...")

            p1 = self.get_data_from_datacube('tamsat',
                                             'rfe',
                                             np.datetime64(start),
                                             np.datetime64(end),
                                             latitude,
                                             longitude)

            p2 = self.get_data_from_datacube('chirps',
                                             'rfe',
                                             np.datetime64(start),
                                             np.datetime64(end),
                                             latitude,
                                             longitude)

            plt.figure(figsize=(8, 6))

            p1.rfe.plot(label='TAMSAT')
            p2.rfe.plot(label='CHIRPS')

            plt.legend()
            plt.show()
    
    def compare_temperature_subproducts(self, latitude, longitude, start, end):

        with self.out:

            clear_output()
            print("Getting data...")

            p1 = self.get_data_from_datacube('era5',
                                             'skt',
                                             np.datetime64(start),
                                             np.datetime64(end),
                                             latitude,
                                             longitude)

            p2 = self.get_data_from_datacube('era5',
                                             't2m',
                                             np.datetime64(start),
                                             np.datetime64(end),
                                             latitude,
                                             longitude)

            plt.figure(figsize=(8, 6))

            p1.rfe.plot(label='skt')
            p2.rfe.plot(label='t2m')

            plt.legend()
            plt.show()

    def compare_rainfall_years(self, product, latitude, longitude,
                               year1, year2):

        with self.out:

            clear_output()
            print("Getting data...")

            product_name = product.lower()

            y1 = self.get_data_from_datacube(
                product_name,
                'rfe',
                np.datetime64(f"{int(year1)}-01-01"),
                np.datetime64(f"{int(year1)}-12-31"),
                latitude,
                longitude).groupby('time.dayofyear').mean()

            y2 = self.get_data_from_datacube(
                product_name,
                'rfe',
                np.datetime64(f"{int(year2)}-01-01"),
                np.datetime64(f"{int(year2)}-12-31"),
                latitude,
                longitude).groupby('time.dayofyear').mean()

            plt.figure(figsize=(8, 6))

            y1.rfe.plot(label=int(year1))
            y2.rfe.plot(label=int(year2))

            plt.legend()
            plt.show()
            
    def compare_temperature_years(self, product, latitude, longitude,
                                  year1, year2):

        with self.out:
            
            clear_output()
            print("Getting data...")
            
            product_name = product.lower()

            y1 = self.get_data_from_datacube(
                product_name,
                'skt',
                np.datetime64(f"{int(year1)}-01-01"),
                np.datetime64(f"{int(year1)}-12-31"),
                latitude,
                longitude).groupby('time.dayofyear').mean()

            y2 = self.get_data_from_datacube(
                product_name,
                'skt',
                np.datetime64(f"{int(year2)}-01-01"),
                np.datetime64(f"{int(year2)}-12-31"),
                latitude,
                longitude).groupby('time.dayofyear').mean()

            plt.figure(figsize=(8, 6))

            y1.skt.plot(label=int(year1))
            y2.skt.plot(label=int(year2))

            plt.legend()
            plt.show()

    def data_to_csv(self, product, subproduct,
                    latitude, longitude, start, end):

        with self.out:

            clear_output()
            print("Getting data...")

            data = self.get_data_from_datacube(product,
                                               subproduct,
                                               pd.to_datetime(start),
                                               pd.to_datetime(end),
                                               latitude,
                                               longitude)

            st = pd.to_datetime(start)
            en = pd.to_datetime(end)
            filename = f"{product}_{subproduct}_{latitude}_{longitude}" \
                       f"_{st.date()}_{en.date()}.csv"
            data.to_dataframe().to_csv(filename)
            localfile = FileLink(filename)
            display(localfile)

            plt.figure(figsize=(8, 6))
            data.__getitem__(subproduct).plot()
            plt.show()

    def plot_rainfall_year_vs_climatology(self, product, latitude, longitude,
                                          start, end):

        with self.out:

            clear_output()
            print("1/3 Getting data for request year...")

            product_name = product.lower()

            y1 = self.get_data_from_datacube(
                product_name,
                'rfe',
                np.datetime64(start),
                np.datetime64(end),
                latitude,
                longitude).groupby('time.dayofyear').mean()

            print("2/3 Calculating climatology...")

            clim = self.get_data_from_datacube(
                product_name,
                'rfe',
                np.datetime64(f"2000-01-01"),
                np.datetime64(f"2019-12-31"),
                latitude,
                longitude)

            std = clim.groupby("time.dayofyear").std()
            mean = clim.groupby("time.dayofyear").mean()

            print("3/3 Plotting data")

            mean = mean.sel(dayofyear=slice(mean.dayofyear.values.min(),
                                            mean.dayofyear.values.max()))

            std = std.sel(dayofyear=slice(std.dayofyear.values.min(),
                                          std.dayofyear.values.max()))

            std_min = mean - std
            std_min.rfe.values[std_min.rfe.values < 0] = 0

            plt.figure(figsize=(8, 6))

            y1.rfe.plot(label="2018")

            mean.rfe.plot(label="Climatology", color='gray', alpha=0.6)
            plt.fill_between(mean.dayofyear.values,
                             (mean + std).rfe.values,
                             std_min.rfe.values,
                             color='Grey', alpha=0.3)

            plt.legend()
            plt.show()

    def calculate_degree_days(self, latitude, longitude, start, end, lower,
                              upper):

        with self.out:

            clear_output()
            print("1/3 Getting data...")


            temp = self.get_data_from_datacube(
                'era5',
                'skt',
                np.datetime64(start),
                np.datetime64(end+datetime.timedelta(hours=23)),
                latitude,
                longitude).skt - 273.15

            print("2/3 Calculating degree days...")

            temp.values[temp.values > upper] = lower
            temp.values[temp.values < lower] = lower
            temp.values = (temp.values - lower) / 24
            temp = temp.resample(time="1D").sum()

            temp.plot()
            plt.show()

    def combine_date_hour(self, date, hour):
        # to get start date and hour
        d = date.value
        x = d.strftime("%Y-%m-%d ")
        h = hour.value
        if h < 10:
            y = ("0" + str(h) + ":00:00")
        else:
            y = (str(h) + ":00:00")
        start = "\"" + x + y + "\""

        # to get end date and hour
        d = date.value
        x = d.strftime("%Y-%m-%d ")
        h = hour.value
        if h < 10:
            y = ("0" + str(h) + ":00:00")
        else:
            y = (str(h) + ":00:00")
        return "\"" + x + y + "\""

    def compare_two_locations(self, product, subproduct, lat1, lon1,
                              lat2, lon2, start_date, start_hour,
                              end_date, end_hour):

        with self.out:
            clear_output()
            print("Getting data...")

            # to get start date and hour
            start = Helpers.combine_date_hour(self, start_date, start_hour)

            # to get end date and hour
            end = Helpers.combine_date_hour(self, end_date, end_hour)

            fig, ax1 = plt.subplots(figsize=(8, 4))

            #latlon1
            list_of_results = Helpers.get_data_from_datacube(
                product, subproduct, start, end, lat1, lon1,)

            x = list_of_results
            x.__getitem__(subproduct).plot(label=(lat1, lon1))

            #latlon2
            list_of_results = Helpers.get_data_from_datacube(
                product, subproduct, start, end, lat2, lon2,)

            y = list_of_results
            y.__getitem__(subproduct).plot(label=(lat2, lon2))

            list_x = x.__getitem__(subproduct).values
            list_y = y.__getitem__(subproduct).values
            max_val = max([list_x.max(), list_y.max()]) + 2
            min_val = min([list_x.min(), list_y.min()]) - 2

            plt.title('comparing two locations')
            plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
            ax1.set_ylim([min_val, max_val])
            fig.tight_layout()
            plt.show()

class Widgets:

    def __init__(self):

        self.search = Search()
        self.item_layout = widgets.Layout(height='30px', width='400px')

    def get_lat_lon_widgets(self):

        return self.latitude(), self.longitude()
    
    def get_product_widgets(self):
        
        return self.product()

    def longitude(self):

        return widgets.BoundedFloatText(value=7.5,
                                        min=-180,
                                        max=180.0,
                                        step=0.0001,
                                        description='Longitude (x):',
                                        disabled=False,
                                        layout=self.item_layout)

    def latitude(self):

        return widgets.BoundedFloatText(value=7.5,
                                        min=-90.0,
                                        max=90.0,
                                        step=0.0001,
                                        description='Latitude (y):',
                                        disabled=False,
                                        layout=self.item_layout)

    def get_point(self, value, description):
        return widgets.BoundedFloatText(value=(value),
                                        min=-180,
                                        max=180,
                                        description=description,
                                        layout=self.item_layout,
                                        disabled=False,
                                        readout=True,
                                        readout_format='d')

    def product(self):

        return widgets.Dropdown(
            options=self.search.products().name.tolist(),
            description='Product:',
            layout=self.item_layout,
            disabled=False, )

    def subproduct(self):

        return widgets.Dropdown(description="Subproduct:",
                                layout=self.item_layout)

    def rainfall_products(self):

        return widgets.Dropdown(options=['TAMSAT', 'CHIRPS', 'GPM'],
                                description='Product:',
                                layout=self.item_layout,
                                disabled=False, )
    
    def temperature_products(self):
        
        return widgets.Dropdown(options=['skt'],
                                description='Product:',
                                layout=self.item_layout,
                                disabled=False, )

    def get_year_widgets(self):

        y1 = widgets.BoundedFloatText(value=2018, min=2000, max=2019, step=1,
                                      description='Year 1 :', disabled=False,
                                      layout=self.item_layout)

        y2 = widgets.BoundedFloatText(value=2019, min=2000, max=2019, step=1,
                                      description='Year 2 :', disabled=False,
                                      layout=self.item_layout)

        return y1, y2

    def get_date(self, value, description):
        return widgets.DatePicker(description=description,
                                  layout=self.item_layout,
                                  value=value,
                                  disabled=False)

    def get_hour(self, value, description):

        return widgets.IntSlider(description=description,
                                 layout=self.item_layout,
                                 value=value,
                                 disabled=False,
                                 min='00',
                                 max='23')

    def degree_day_threshold(self, min_val, max_val, value, string):
        return widgets.BoundedFloatText(value=value,
                                        min=min_val,
                                        max=max_val,
                                        step=0.0001,
                                        description=string,
                                        disabled=False,
                                        layout=self.item_layout)

    def set_up_button(self, method):

        button = LoadedButton(description="Get Data",
                              layout=self.item_layout)
        button.on_click(method)
        button.button_style = 'primary'

        return button

    @staticmethod
    def display_widget(widget_list):

        for w in widget_list:
            display(w)

    @staticmethod
    def display_widgets(product, subproduct, north, east, south,
                        west, date, hour, button, m):

        from ipywidgets import HBox, VBox

        # for w in widget_list:
        #     display(w)
        box1 = VBox([product, subproduct, north, east, south,
                     west, date, hour, button])

        box2 = HBox([box1, m])
        box_layout = widgets.Layout(
            display='flex',
            flex_flow='row',
            align_items='stretch',
            width='100%')
        display(box2)

    @staticmethod
    def display_widget_comparison(product, subproduct, north, east, south,
                                  west, date1, hour1, date2, hour2, button, m):

        from ipywidgets import HBox, VBox

        # for w in widget_list:
        #     display(w)
        box1 = VBox([product, subproduct, north, east, south,
                     west, date1, hour1, date2, hour2, button])
        box2 = HBox([box1, m])
        box_layout = widgets.Layout(
            display='flex',
            flex_flow='row',
            align_items='stretch',
            width='100%')
        display(box2)

    @staticmethod
    def display_output():

        out = widgets.Output()
        display(out)
        return out

    def get_subproduct_list(self, product):

        return self.search.get_subproduct_list_of_product(product)

    def get_date_widgets(self):

        return self.start_date(), self.end_date()

    def start_date(self):

        return widgets.DatePicker(description='Start Date: ',
                                  layout=self.item_layout,
                                  value=datetime.datetime(2000, 1, 1),
                                  disabled=False)

    def end_date(self):
        return widgets.DatePicker(description='EndDate: ',
                                  layout=self.item_layout,
                                  value=datetime.datetime(2000, 2, 1),
                                  disabled=False)


class LoadedButton(widgets.Button):
    """A button that can holds a value as a attribute."""

    def __init__(self, value=None, *args, **kwargs):
        super(LoadedButton, self).__init__(*args, **kwargs)
        self.add_traits(value=traitlets.Any(value))
