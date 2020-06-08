"""
Send requests and receive replies from DQ server.

Please see prototype code in scratch/ for lots of helpful print
statements which are removed here for clarity.

***************************************************************
REQUIRES .assimila_dq file to work - this is in the test/ area.
***************************************************************

Key code modified from ECMWF api.py
(https://software.ecmwf.int/wiki/download/attachments/56664858/ecmwf-
api-client-python.tgz)
which is (C) Copyright 2012-2013 ECMWF.
Their software is licensed under the terms of the Apache Licence
Version 2.0 which can be obtained at
http://www.apache.org/licenses/LICENSE-2.0.

"""
from .connect_log.setup_logger import SetUpLogger

import logging
import os.path as op
import os
import json
import requests
import pickle
import gzip
import datetime
# ======================================================================
# Methods and class to support login credentials.


# Exception class
class DQKeyFetchError(Exception):
    def __init__(self, arg):
        self.message = arg

    def __str__(self):
        return repr(self.message)


def _get_dqkey_from_env():
    """
    Query environment variables for DQ connection details.
    All must be set for this to be valid.
    :return: connection information
    """
    try:
        key = os.environ["ASSIMILA_DQ_KEY"]
        port = os.environ["ASSIMILA_DQ_PORT"]
        url = os.environ["ASSIMILA_DQ_URL"]
        login = os.environ["ASSIMILA_DQ_LOGIN"]
        return key, url, port, login
    except KeyError:
        raise DQKeyFetchError("ERROR: Could not get the DataCube key from "
                              "the environment")


def _get_dqkey_from_file(keyfile=None):
    """
    Obtain DQ connection details from user's file.
    :return: connection information
    """
    # TODO decide on standard location of specific user connection file
    # dq_file = os.path.normpath(os.path.expanduser("~/.assimila_dq"))

    if keyfile is None:
        # this line assumes that the client's password is in the same
        # place as the code creating the AssimilaData() object.
        dq_file = op.join(op.abspath(op.curdir), ".assimila_dq")
    else:
        dq_file = keyfile

    try:
        with open(dq_file) as f:
            config = json.load(f)
    except IOError as e:  # Failed reading from file
        raise DQKeyFetchError("ERROR: File reading failed" + str(e))
    except ValueError:  # JSON decoding failed
        raise DQKeyFetchError("ERROR: Missing or malformed DQ key in '%s'"
                              % dq_file)
    except Exception as e:  # Unexpected error
        raise DQKeyFetchError(str(e))

    try:
        key = config["key"]
        port = config["port"]
        url = config["url"]
        login = config["login"]
        return key, url, port, login

    except Exception:
        raise DQKeyFetchError("ERROR: Missing or malformed DQ key in '%s'"
                              % dq_file)


def get_dqkey_values(keyfile=None):
    """
    Get the DataCube key from the environment or the '.assimila_dq'
    file. The environment is looked at first.

    :return: tuple with the key, url, port and login forming the API
             key.

    :raise: APIKeyFetchError: When unable to get the API key from
            either the environment or the assimila_dq file.
    """
    try:
        key_values = _get_dqkey_from_env()
    except DQKeyFetchError:
        try:
            key_values = _get_dqkey_from_file(keyfile)
        except DQKeyFetchError:
            raise

    return key_values

# ======================================================================
# Methods and classes to support http connection.
# NOTE that this is broken into two classes following the ECMWF design
# where each has a lot more to do. To allow room for expansion, they've
# not been combined here.
#
# Please see prototype code for lots of helpful print statements which
# are removed here for clarity.


class APIRequest(object):
    """
    Http client class.
    Creates headers, sends requests and receives responses. Deals with
    conversion to/from binary for transmission.

    Note that the class is instantiated afresh on each connection so
    cannot keep any instance variables between connections.
    """
    def __init__(self, logger, service, url, login=None, key=None):
        """
        Create DQ client http object.
        Ensure user has correct credentials.
        :param logger: instance of the logger created by the client
                       (currently unused)
        :param service: command from user
        :param url: http connection address
        :param login: user login
        :param key: user key
        :raise ConnectionRefusedError: for any problem with login details
        :raise ConnectionError: for problems connecting to the server
        :raise Exception: anything else
        """
        self.logger = logger
        self.service = service
        self.url = url
        self.login = login
        self.key = key

        try:
            # check credentials with simple connection. Header could
            # contain cookie for future use.
            hdrs = {"From": self.login, "X-DQ-KEY": self.key,
                    "X-DQ-SERVICE": self.service}
            resp = requests.get(self.url, headers=hdrs)

            if resp.status_code != 200:
                raise ConnectionRefusedError(resp.headers)

        except ConnectionRefusedError:
            # catch the error we've just raised to ensure it gets
            # passed up verbatim.
            raise
        except ConnectionError as e:
            print("Connection Error : %s" % e.args)
            raise
        except Exception as e:
            print("Other error : %s" % e.__repr__())
            raise

    def get_from_dq(self, req):
        """
        Retrieve information or data from the datacube.
        Depending on the command, this function either writes a file at
        the specified target, returns an xarray of raw data, or returns
        json format metadata.

        :param req: json request

        :return: x-array data, or metadata. No return if file requested.

        :raise Exception: for any problems
        """
        try:
            payload = pickle.dumps(req, protocol=-1)
            resp = requests.post(self.url, data=payload)

            if resp.status_code != 200:

                # this code replaces the line breaks(\n) mix with the \\ to
                # normal line breaks which fixes the issue of the exception
                # not displaying properly when raised, altough it does not
                # fix the log, also removes brackets at start and end of
                # error message

                # Another possible solution when the only thing that doesnt
                # need to be fixed is the error heading:
                # keys = resp.headers.keys()
                # for i in resp.headers:
                #   try:
                #       resp.headers[keys[i]] = resp.headers[keys[i]].replace(
                #       "\\n",\n").replace("\\", " ").replace("("
                #       ,"").replace(")","")
                #   except TypeError:
                #       pass

                formatted_str = resp.headers['error'].replace(
                    "\\n", "\n").replace("\\", " ").replace("(", "")\
                    .replace(")", "")

                resp.headers.update({'error': formatted_str})

                raise Exception(resp.headers['error'])

            if self.service == "GET_FILE":
                target = req.get("target")
                resp_unpacked = gzip.decompress(resp.content)
                with open(target, "wb") as tgt:
                    # binary content
                    # tgt.write(resp.content)
                    tgt.write(resp_unpacked)

            elif self.service == "GET_DATA":
                resp_unpacked = gzip.decompress(resp.content)
                # use pickle to de-serialize xarray data
                return pickle.loads(resp_unpacked)

            elif self.service == "GET_META":
                return pickle.loads(resp.content)

            else:
                # TODO assume text content or raise Exception
                print(resp.text)

        except Exception:
            raise

    def put_to_dq(self, req, data=None):
        """
        Upload information, data or file to the datacube.
        The service asked for is used to determine the upload location
        on the server (if data or file).

        :param req: json request
        :param data: optional x-array to upload

        :return: no return

        :raise Exception: for any problems
        """
        # POST to server replies with url to send in the
        # PUT request
        # Also send the metadata so the server has access to it later
        # in the PUT request

        try:
            # for metadata and registration, this request contains ALL info.
            payload = pickle.dumps(req, protocol=-1)
            resp_1 = requests.post(self.url, data=payload)

            if resp_1.status_code != 200:
                formatted_str = resp_1.headers['error'].replace(
                        "\\n", "\n").replace("\\", " ").replace("(", "") \
                    .replace(")", "")
                resp_1.headers.update({'error': formatted_str})
                raise Exception(resp_1.headers['error'])

            # extract the response's text for PUT_FILE and PUT_DATA only.
            if self.service == "PUT_FILE":
                put_url = self.url + resp_1.text
                # 'source' will be local upload filepath
                if 'source' in req:
                    # requests library allows files to be sent, but I
                    # can't work out how to extract on the server side so
                    # have made it a payload  instead. If anyone else can
                    # get it to work... please do :)

                    # convert contents of file to bytes and compress
                    with open(req.get('source'), 'rb') as f_in:
                        payload = gzip.compress(f_in.read())

                    resp_2 = requests.put(put_url,
                                          data=payload)
                    if resp_2.status_code != 200:
                        raise Exception(resp_2.headers)
                else:
                    # raise exception if no source provided provided.
                    raise Exception("Non-existent source file.")

            elif self.service == "PUT_DATA":
                put_url = self.url + resp_1.text
                if data is not None:  # data is xarray

                    # For ERA5 registration, the below line caused a
                    # spike of 7.2GB of memory usage from a 1.4GB array
                    payload_pickled = pickle.dumps(data, protocol=-1)
                    payload = gzip.compress(payload_pickled)

                    resp_2 = requests.put(put_url, data=payload)

                    if resp_2.status_code != 200:
                        raise Exception(resp_2.headers)

            elif self.service == "PUT_NEW" or self.service == "PUT_META":
                # for PUT_NEW (registration) we may have a list of items where
                # their yaml definition files will be local on the server
                # or a dictionary registration definition.
                # For registration using a file, use PUT_FILE

                # All information was sent in the first requests.post() where
                # it is actioned by the server.
                pass

            else:
                raise Exception("Unsupported http command.")

        except Exception:
            raise


class AssimilaData(object):
    """
    Class providing a client with methods to interact with the datacube.
    It combines the connection and work requests, extracting
    information from the formatted request data to control the http
    client behaviour.

    The only two methods (get() and put()) always create the APIRequest.
    This then checks the authentication with the call handled by the
    server's do_GET() handler.
    Subsequent requests are handled by do_POST() and do_PUT() where the
    command sent in the request determines the behaviour.
    """
    def __init__(self, url=None, port=None, key=None, login=None,
                 keyfile=None):
        """
        Create the DQ client API object.
        Connection information may be provided. If none, or some is
        missing, the details will be found from either the environment
        or from the user's connection config. file.

        :param url: DQ server address
        :param port: DQ server port
        :param key: User's unique key (as provided by Assimila)
        :param login: User's unique login string with date, access and email
        :param keyfile: optional location of connection file
        """
        # set up the logging output filename here so that no changes are
        # needed in its configuration file.
        base, extension = os.path.splitext('./connect_log/DQClient.log')
        today = datetime.datetime.today()
        log_filename = "{}{}{}".format(base,
                                       today.strftime("_%Y_%m_%d"),
                                       extension)

        SetUpLogger.setup_logger(
            log_filename=op.abspath(op.join(op.dirname(__file__),
                                            log_filename)),
            default_config=op.abspath(op.join(op.dirname(__file__),
                                      "./connect_log/logging_config.yml")))
        self.logger = logging.getLogger("__main__")

        if url is None or port is None or key is None or login is None:
            key, url, port, login = get_dqkey_values(keyfile)

        self.url = url
        self.key = key
        self.login = login
        self.port = port
        self.full_url = self.url + ':' + self.port

        # self.logger.info(f"HTTP Client initialised with keyfile: {keyfile}")
        self.logger.info("HTTP Client initialised with keyfile: %s" % keyfile)

    def __del__(self):
        logging.shutdown()

    def get(self, req):
        """
        Retrieve requested information from the datacube.

        :param req: json formatted command

        :return: xarray data, or metadata. No return if file requested.

        :raise ConnectionRefusedError: if authentication fails
        :raise Exception: for any other problem
        """
        try:
            c = APIRequest(self.logger, req.get('command'),
                           self.full_url, self.login, self.key)

            if req.get('command') == 'GET_FILE':
                c.get_from_dq(req)
            else:
                return c.get_from_dq(req)

        except ConnectionRefusedError as e:
            # print("User not authorised : %s" % e.args)
            self.logger.warning("User not authorised : %s" % e.args)
            raise
        except Exception as e:
            # print("Error in client get : %s" % e.__repr__())
            self.logger.warning("Error in client get : %s" % e.__repr__())
            raise

    def put(self, req, data=None):
        """
        Store given information in the datacube.

        :param req: json formatted command
        :param data: optional xarray data

        :return: no return

        :raise ConnectionRefusedError: if authentication fails
        :raise Exception: for any other problem
        """
        try:
            c = APIRequest(self.logger, req.get('command'),
                           self.full_url, self.login, self.key)

            if req.get('command') == 'PUT_DATA':
                c.put_to_dq(req, data=data)
            else:
                c.put_to_dq(req)

        except ConnectionRefusedError as e:
            # print("User not authorised : %s" % e.strerror)
            self.logger.warning("User not authorised : %s" % e.strerror)
            raise
        except Exception as e:
            # print("Error in client put : %s" % e.__repr__())
            self.logger.warning("Error in client put : %s" % e.__repr__())
            raise
