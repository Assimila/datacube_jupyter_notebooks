from DQTools.search import Search
from DQTools.dataset import Dataset
import datetime as dt

s = Search()
print(s.tiles())
print(s.products())
print(s.subproducts())

d = Dataset(product="era5", subproduct="skt", tile="era5_africa", res=0.25)
data = d.get_data(start=dt.datetime(2018, 8, 1),
                  stop=dt.datetime(2018, 8, 2),
                  tile='ken_prise')
