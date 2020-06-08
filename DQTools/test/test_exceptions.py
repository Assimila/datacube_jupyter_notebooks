import nose.tools as ns
import os.path as op
import os
import sys
import time
from multiprocessing import Process
import signal

wkspace_root = op.normpath(op.join(__file__, '../../../'))
sys.path.insert(0, op.join(wkspace_root,'datacube'))

import src.datacube.dataserver.server as serv
from src.datacube.archiver import Archiver
from src.datacube.dq_database.registration_manager import RegistrationManager
from test.datacube_tests.test_dataset.register_test_dataset import Dummy
import test.database_helpers as db


class TestDQToolExceptions(object):
    dq_permissions_file = None
    connect_file = None
    connect_sub_file = None
    archiver = None
    reg = None
    keyfile = None

    @classmethod
    def setup_class(cls):
        cls.dq_permissions_file = op.abspath(op.join(wkspace_root,
                "datacube/test/datacube_tests/manager_tests/DQ_permissions.yaml"))

        # Set up the test database
        # - to control the connection details, set the two file identities
        cls.connect_file = op.abspath(op.join(wkspace_root,
                "datacube/src/datacube/dq_database/dq_DB_conf/db_settings.yaml"))

        cls.connect_sub_file = op.abspath(op.join(wkspace_root,
                "datacube/test/datacube_tests/db_settings.yaml"))

        # - substitute the DB connection information to use a test server i.e.
        # the local file for the 'real' one - this sets it up for
        # any subsequent connection by other objects.
        # -- get our own connection to the test server
        dbconn = db.setup_test_db_connection(cls.connect_file,
                                             cls.connect_sub_file)

        # - create a clean database
        script = op.abspath(op.join(wkspace_root,
                 "datacube/test/datacube_tests/database_setup.sql"))
        db.execute_sql_file(script, dbconn)

        # - put the dummy product and sub-products into the database
        cls.archiver = Archiver()
        cls.reg = RegistrationManager()
        db.register_and_load_dummy_data(cls.archiver, cls.reg)

        # Set up a test server
        # =================================================================== #
        # comment out the following four lines to allow independent
        # server start, also edit teardown_class() and server.py
        p = Process(target=cls.start_server)
        p.start()
        cls.server_pid = p.pid
        time.sleep(2)   # Allow server to start up
        # =================================================================== #

        # Our tests will need a full permission keyfile
        cls.keyfile = op.abspath(op.join(wkspace_root,
                 "datacube/test/http_tests/.assimila_dq"))

    @classmethod
    def start_server(cls):
        # Start server on localhost
        serv.run(test=True)

    @classmethod
    def teardown_class(cls):
        """This method is run once for each class _after_ all tests are run"""
        # =================================================================== #
        # comment out the following line to allow independent server stop.
        os.kill(cls.server_pid, signal.SIGTERM)
        # =================================================================== #

        # Reinstate the DB connection information
        db.teardown_test_db_connection(cls.connect_file)


# To test, copy failure keyfiles from the http_tests and ensure the correct
# exceptions are thrown
# Also, try to get products etc which don't exist in the dummy database.