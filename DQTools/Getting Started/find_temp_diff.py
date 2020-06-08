from DQTools import Dataset, Register, Search
import datetime as dt

# Register new tile and product/sub-product
Register('[location of]/new_tiles.yaml')
Register('[location of]/new_subproduct_product.yaml')

# Connect to Datacube product
eraskin = Dataset(product="era5", subproduct="skt", tile="madagascar", res=0.25)
erat2m =  Dataset(product="era5", subproduct="t2m", tile="madagascar", res=0.25)

# Download data
eraskin.get_data(start=eraskin.last_timestep - dt.timedelta(days=10),
                 stop = eraskin.last_timestep)
erat2m.get_data(start=eraskin.last_timestep - dt.timedelta(days=10),
                 stop = eraskin.last_timestep)

# Connect to output datacube product
temp_diff = Dataset(product="bp_test", subproduct="temp_diff")

# Calculate new data
new_data = eraskin.data.skt - erat2m.data.t2m
new_data.name = 'temp_diff'

# Set to be the data attribute
temp_diff.data = new_data.to_dataset()

# Set last gold
temp_diff.last_gold = eraskin.last_timestep

# Push into the datacube. USE WITH CAUTION
temp_diff.put()