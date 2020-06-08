import logging.config
import logging.handlers
import os
import yaml
import datetime
import shutil

class SetUpLogger:

    @staticmethod
    def setup_handlers(default_config, env_key, logger_id, log_filename):
        """
        Format handlers

        :param default_config:  Path to logging configuration YAML file.
        :param env_key:         Environment variable set to a log config
                                file. Ignored if not set.
        :param logger_id:       User may provide a name for the logger.
        :param log_filename:    Specify to create provider/process
                                specific log files

        :return: N/A
        """

        config_file = default_config
        value = os.getenv(env_key, None)

        if value:
            config_file = value

        # Open the logging configuration YAML file
        if os.path.exists(config_file):
            with open(config_file, "rb") as f:
                config = yaml.safe_load(f.read())

            # Rename the logger (change from __main__) if needs be
            # i.e. if logger_id has been set by client, change the dict
            # to replace the __main__ entry key with the new name given
            # in logger_id. All other information is preserved.
            config['loggers'][logger_id] = config['loggers'].pop('__main__')

            # Loop over each handler in the YAML configuration file
            for i in (config["handlers"].keys()):

                # If handler has a filename option, specify one.
                if "filename" in config['handlers'][i].keys():

                    # If no log filename has been specified, create one.
                    if not log_filename:
                        log_filename = config["handlers"][i]["filename"]
                        base, extension = os.path.splitext(log_filename)
                        today = datetime.datetime.today()
                        log_filename = "{}{}{}".format(base,
                                                       today.strftime(
                                                           "_%Y_%m_%d"),
                                                       extension)

                    # Set the logging filename
                    config["handlers"][i]["filename"] = log_filename

        return config

    @staticmethod
    def setup_logger(log_filename=None,
                     default_config="logging_config.yml",
                     default_level=logging.INFO,
                     logger_id="__main__",
                     env_key="LOG_CFG"):
        """
        Method to read a logging YAML file and extract the formatting
        information required to set up the logger.

        :param log_filename:    Default = None; but can be specified to
                                create provider/process specific log
                                files.

        :param default_config:  Path to logging configuration YAML file.
                                Default = local logging_config.yml

        :param default_level:   Logging level which defines what logging
                                information is written to the log file.
                                Default = INFO
                                See: https://bit.ly/2HLKGCI for more
                                info.

        :param logger_id:       User may provide a name for the logger.
                                Default = "__main__"

        :param env_key:         Environment variable set to a log config
                                file. Ignored if not set.
                                Default = "LOG_CFG"

        :return: N/A
        """
        # Once a logger is created in the Python interpreter process,
        # it is available to all modules because it is a singleton.
        # Thus,this setup should only be done once, otherwise duplicate
        # handlers will just be added to the existing logger.

        # Get the logger for this interpreter.
        # Get exiting one if it's already instantiated, create it if it
        # doesn't.
        logger = logging.getLogger(logger_id)

        # If no logger handlers have been set up, do it.
        if not logger.handlers:

            # If a configuration file has been specified.
            if default_config:

                # Set up logger handlers
                config = SetUpLogger.setup_handlers(default_config, env_key,
                                                    logger_id, log_filename)

                # Set up the logger according the the configuration
                logging.config.dictConfig(config)

            # If no YAML configuration file found, use module defaults.
            else:
                logging.basicConfig(level=default_level)

        # If handler filename is different to log_filename.
        for handler in logger.handlers:

            # If the handler is a file handler, replace the with new.
            if isinstance(handler, logging.FileHandler):

                # If the specified log filename does not match that in
                # the default file handler, replace the handler.
                if log_filename and handler.baseFilename != log_filename:

                    # Set up a new file handler
                    h = logging.FileHandler(log_filename)

                    # Change log file permissions to allow any user
                    # to write to
                    #os.chmod(log_filename, 0o777)

                    h.setLevel(handler.level)
                    h.setFormatter(handler.formatter)

                    # Remove the old handler, add new one.
                    logger.removeHandler(handler)
                    logger.addHandler(h)

        # If needed, change log file permissions to allow any user
        # to write to it.
        status = os.stat(log_filename)
        if oct(status.st_mode)[-3:] != '777':
            os.chmod(log_filename, 0o777)
