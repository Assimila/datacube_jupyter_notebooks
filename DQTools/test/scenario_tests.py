from DQTools import Dataset
import datetime as dt


# Single point extraction for a single timestep
ds = Dataset(product='degree_day_delta', subproduct='ddd_busseola_fusca')
ds.get_data(start=dt.datetime(2019, 1, 1),
            stop=dt.datetime(2019, 1, 1),
            latlon=[0.0, 35.0])
ds = None

# Single point extraction for time period
ds = Dataset(product='degree_day_delta', subproduct='ddd_busseola_fusca')
ds.get_data(start=dt.datetime(2019, 1, 1),
            stop=dt.datetime(2019, 1, 31),
            latlon=[0.0, 35.0])
ds = None

# Extract full tile for single time step
ds = Dataset(product='degree_day_delta', subproduct='ddd_busseola_fusca')
ds.get_data(start=dt.datetime(2019, 1, 1),
            stop=dt.datetime(2019, 1, 1),
            tile='ken_prise')
ds = None

# Extract full tile for time period
ds = Dataset(product='degree_day_delta', subproduct='ddd_busseola_fusca')
ds.get_data(start=dt.datetime(2019, 1, 1),
            stop=dt.datetime(2019, 1, 31),
            tile='ken_prise')
ds = None

# Extract full tile for single time step with resampling
ds = Dataset(product='degree_day_delta', subproduct='ddd_busseola_fusca')
ds.get_data(start=dt.datetime(2019, 1, 1),
            stop=dt.datetime(2019, 1, 1),
            tile='ken_prise', res=0.01)
ds = None

# Extract full tile for time period with resampling
ds = Dataset(product='degree_day_delta', subproduct='ddd_busseola_fusca')
ds.get_data(start=dt.datetime(2019, 1, 1),
            stop=dt.datetime(2019, 1, 31),
            tile='ken_prise', res=0.01)
ds = None

# Extract full tile for single time step with resampling
ds = Dataset(product='degree_day_delta', subproduct='ddd_busseola_fusca')
ds.get_data(start=dt.datetime(2019, 1, 1),
            stop=dt.datetime(2019, 1, 1),
            tile='ken_prise', res=0.01, country='kenya')
ds = None

ds = Dataset(product='tamsat', subproduct='rfe')
ds.get_data(start=dt.datetime(2018, 6, 3),
            stop=dt.datetime(2018, 6, 3),
            region=[10., 10., 0.0, 0.0])
ds = None
