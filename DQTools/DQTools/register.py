import yaml
import logging
import datetime
import os.path as op
from .connect.connect import Connect
from .connect.connect_log.setup_logger import SetUpLogger


class Register:

    """
    Class for registering data to the DataCube
    """

    def __init__(self, config_file_path):
        """
        Register a new product/sub-product or tile from a config.yaml file
        file.

        :param config_file_path: The full path to the config file
        :return:
        """
        try:
            base, extension = op.splitext('register.log')
            today = datetime.datetime.today()
            log_filename = "{}{}{}".format(base,
                                           today.strftime("_%Y_%m_%d"),
                                           extension)

            SetUpLogger.setup_logger(
                log_filename=op.abspath(op.join(op.dirname(__file__),
                                                log_filename)),
                default_config=op.abspath(op.join(op.dirname(__file__),
                               "./connect/connect_log/logging_config.yml")))
            self.logger = logging.getLogger("__main__")

        except Exception:
            raise

        try:
            # Extract the config information from the yaml file
            with open(config_file_path, 'r') as file:
                config_dict = yaml.load(file)

            # Connect to DQ
            conn = Connect()

            # Send to the connector
            conn.register(config_dict)

        except (FileNotFoundError, OSError) as e:
            self.logger.error("Unable to load configuration for registration.\n"
                              "%s" % e)
            print("Unable to load configuration for registration, "
                  "please see logfile for details.")
        except Exception as e:
            self.logger.error("Unable to create Registration object.\n%s" % e)
            print("Unable to create Registration object, "
                  "please see logfile for details.")
