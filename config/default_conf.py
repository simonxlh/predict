import configparser
from os import path

cf = configparser.ConfigParser()
cf_file_path = path.join(path.dirname(path.abspath(__file__)), './config.cfg')

# print(cf_file_path)
cf.read(cf_file_path)

MO_ACCESS_URI = cf.get("Monitoring", "access_uri")
MO_RETENTION = cf.getint("Monitoring", "retention")
MO_COLLECT_STEP = cf.getint("Monitoring", "collect_step")