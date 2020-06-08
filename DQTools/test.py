import os.path as op
import sys
import datetime as dt

from DQTools import Dataset


tamsat = Dataset(product='tamsat', subproduct='rfe')

tamsat.get_data(start=dt.datetime(2000, 1, 1),
                stop=dt.datetime(2000, 1, 2),
                region=[100.0, 100.0, 110.0, 110.0])

print(tamsat.data)


# tamsat = Dataset(product='tamsat', subproduct='rfe')
#
# tamsat.get_data(start=dt.datetime(1999, 1, 1),
#                 stop=dt.datetime(1999, 1, 2))
#
# print(tamsat.data)
