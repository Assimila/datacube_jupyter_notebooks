import logging
import datetime
import os.path as op
from .connect.connect import Connect
from .connect.connect_log.setup_logger import SetUpLogger


class Search:
    """
    Tools to allow a user to check the contents of the datacube from the
    commandline.

    New search requests can be serviced by adding to the BespokeSearch
    class in the DataCube at: src/datacube/dq_database/db_view.py
    """

    def __init__(self):
        """
        Set up logging.
        """
        try:
            # base, extension = op.splitext('search.log')
            # today = datetime.datetime.today()
            # log_filename = "{}{}{}".format(base,
            #                                today.strftime("_%Y_%m_%d"),
            #                                extension)
            #
            # SetUpLogger.setup_logger(
            #     log_filename=op.abspath(op.join(op.dirname(__file__),
            #                                     log_filename)),
            #     default_config=op.abspath(op.join(op.dirname(__file__),
            #                    "./connect/connect_log/logging_config.yml")))
            self.logger = logging.getLogger("__main__")

        except Exception:
            raise

    def tiles(self):
        """
        Return all the tiles available.
        :return:
        """
        try:
            # Instantiate the datacube connector
            conn = Connect()

            # extract a dataframe of the tile table
            df = conn.get_all_table_data("tile")

            return df

        except Exception as e:
            self.logger.error("Unable to get tiles.\n%s" % e)
            print("Unable to get tiles, "
                  "please see logfile for details.")

    def products(self):
        """
        Return all the products available.
        :return:
        """
        try:
            # Instantiate the datacube connector
            conn = Connect()

            # extract a dataframe of the product table
            df = conn.get_all_table_data("product")

            return df

        except Exception as e:
            self.logger.error("Unable to get products.\n%s" % e)
            print("Unable to get products, "
                  "please see logfile for details.")

    def subproducts(self):
        """
        Return all the sub-products available.
        :return:
        """
        try:
            # Instantiate the datacube connector
            conn = Connect()

            # extract a dataframe of the sub-product table
            df = conn.get_all_table_data("subproduct")

            return df

        except Exception as e:
            self.logger.error("Unable to get sub-products.\n%s" % e)
            print("Unable to get sub-products, "
                  "please see logfile for details.")

    def get_subproduct_list_of_product(self, product):
        """
        Return a list of sub-products based on product selected
        :param product: The name of the product
        :return:
        """
        try:
            # Instantiate the datacube connector
            conn = Connect()

            return conn.get_product_subproducts(product)

            # # Get product's dataframe
            # result = conn.get_product_meta(product)
            #
            # # Initialise list
            # list = []
            #
            # # Gets sub-product of the product
            # # result[index][1].name -- gets indexth sub-product
            # for r in result:
            #     # Identifies unique sub-product names
            #     sub = (r[1].name.unique().tolist())
            #     # Adds to list
            #     list.append(sub)
            #
            # # Use monoids to extract the list
            # subproduct_list = sum(list, [])
            #
            # return subproduct_list

        except Exception as e:
            self.logger.error("Unable to get product's sub-products.\n%s" % e)
            print("Unable to get product's sub-products, "
                  "please see logfile for details.")

    def list_projections(self):
        """
        Return a list of allowed Common Names for projections.
        :return: Common names and instructions
        """
        names = ['WGS84', 'British National Grid', 'Sinusoidal']

        print("The following names are accepted projections:")
        for name in names: print(name)
        print("For any other projection, please use the proj4 string,"
              " details can be found at https://spatialreference.org/ref/"
              " where, for example, WGS84 Mercator is "
              "'+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs'")
