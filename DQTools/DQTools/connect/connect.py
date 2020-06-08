import os.path as op

from .DQclient import AssimilaData


class Connect:
    """
    Establish handshake and data transfer with the DataCube.

    """

    def __init__(self, key_file=None):
        """
        Make connection to the DataCube.

        :param key_file: location of data cube key file

        """

        if not key_file:
            key_file = op.join(op.dirname(__file__), ".assimila_dq")

        self.http_client = AssimilaData(keyfile=key_file)

    def get_product_subproducts(self, product):
        """
        Get the sub-products of this product.
        See DQManager:search_database() for details of returned data; also
        TestDQDataBaseView.test_search_specific_recurse_product_only().

        :param product: The name of the product

        :return: list of sub-products
        """
        try:

            result = self.http_client.get({'command': 'GET_META',
                                           'action': 'search_metadata',
                                           'params': {
                                               'search_terms': {'name': product},
                                               'recurse': 'True'}})

            retval = list()
            for item in result['subproducts']:
                retval.append(item['name'])

            return retval

        except Exception as e:
            raise e

    def get_product_meta(self, product):
        """
        Extract all available metadata for this product and its sub-products.

        :param product: The name of the product

        :return:
        """
        try:

            result = self.http_client.get({'command': 'GET_META',
                                           'action':
                                               'get_metadata_product_and_children',
                                           'params': {'product': product}})

            return result

        except Exception as e:
            raise e

    def get_subproduct_meta(self, product, subproduct, bounds=None, tile=None):
        """
        Extract all available metadata for this product & sub-product and
        specific region or tile if requested.

        :param product: The name of the product
        :param subproduct: The name of the sub-product
        :param bounds: dictionary of n-s-e-w bounds
        :param tile: tilename (must match tile registered in DataCube)
        :return:
        """
        try:

            result = self.http_client.get(
                {'command': 'GET_META',
                 'action': 'get_subproduct_metadata_for_dqtools',
                 'params': {'product': product,
                            'subproduct': subproduct,
                            'bounds': bounds,
                            'tile': tile}})

            return result[0]

        except Exception as e:
            raise e

    def get_subproduct_data(self, product, subproduct, start, stop,
                            bounds, res, tile, country, latlon,
                            projection):
        """
        Extract and return an xarray of data from the datacube

        :param product: The name of the product
        :param subproduct: The name of the sub-product
        :param start: The starting time for extracting data
        :param stop: The ending time for extracting data
        :param bounds: The bounds for the data (dictionary of n-s-e-w bounds)
        :param res: The required resolution of the data. If this is provided
        then the DataCube will enact a gdal Warp to return an array of the
        provided latitude, fitting the given bounds.
        :param tile: A tile name to define the bounds of the data once in
        the DataCube
        :param country: A country over which to carry out zonal averaging
        :param projection: Name or proj4 string to define projection.
        :return:
        """
        try:
            # Prepare the product metadata
            get_request_params = {
                'product': product,
                'subproduct': [subproduct],
                'start_date': start,
                'end_date': stop,
            }

            # If a resolution or projection has been provided then warp
            if res or projection:
                warp_params = dict()
                if res: warp_params = {'xRes': res, 'yRes': res}
                # this next line will add to the dict if 'res' has already
                # populated it or will make a new entry anyway
                if projection: warp_params['dstSRS'] = projection
                get_request_params['warp'] = warp_params
                get_request_params['warptobounds'] = True

            if country:
                if tile:
                    action = 'get_zonal_data'
                    get_request_params['tile'] = tile
                    get_request_params["zonal_stats"] = country
                else:
                    # One cannot expect zonal stats without having a tile
                    # TODO Support bounds for zonal stats in future
                    raise Exception('A tile must be specified to calculate '
                                    'zonal statistics.')

            # Add in area information if specified
            if tile and not country:
                action = 'get_tile_data'
                get_request_params['tile'] = tile
            elif bounds:
                action = 'get_area_data'
                get_request_params['north'] = bounds.north
                get_request_params['south'] = bounds.south
                get_request_params['east'] = bounds.east
                get_request_params['west'] = bounds.west
            elif latlon:
                action = 'get_position_data'
                get_request_params['lat'] = latlon[0]
                get_request_params['lon'] = latlon[1]
            # else:
            #     # get global data
            #     action = 'get_area_data'
            #     get_request_params['north'] = 90
            #     get_request_params['south'] = -90
            #     get_request_params['east'] = 180
            #     get_request_params['west'] = -180

            # Request data
            data = self.http_client.get({
                'command': 'GET_DATA',
                'action': action,
                'params': get_request_params})

            return data

        except Exception as e:
            raise e

    def put_subproduct_data(self, data):
        """
        Write sub-product data to the datacube

        :param data: an xarray DataSet object to be sent to the DataCube
        :return:
        """

        # Prepare put request
        put_request = {
            'command': 'PUT_DATA',
            'action': 'put_data',
            'params': {'overwrite': 'True'}}

        self.http_client.put(put_request, data)

    def get_all_table_data(self, tablename):
        """
        Method to return everything in a single table. Used for Search class
        methods.

        :param tablename: The name of the DataCube database table to be
                          searched
        :return:
        """

        request = {
            'command': 'GET_META',
            'action': 'get_table_contents',
            'params': {'table': tablename}
        }

        result = self.http_client.get(request)

        return result

    def register(self, config_dict):

        # Check what is attempting to be registered based on
        # dictionary keys
        if 'subproducts' in config_dict:
            self.register_product(config_dict)

        else:
            self.register_tile(config_dict)

    def register_tile(self, config_dict):
        """
        Register tiles with the datacube.

        :param config_dict:

        :return: N/A
        """
        # Prepare put request
        put_request = {
            'command': 'PUT_NEW',
            'action': 'register_tile_from_dictionary',
            'params': {'spec': config_dict}}

        self.http_client.put(put_request)

    def register_product(self, config_dict):
        """
        Register products+sub-product groups with the datacube.

        :param config_dict:

        :return: N/A
        """
        # Prepare put request
        put_request = {
            'command': 'PUT_NEW',
            'action': 'register_product_from_dictionary',
            'params': {'spec': config_dict}}

        self.http_client.put(put_request)

    # def register_with_file
    #     use PUT_FILE
