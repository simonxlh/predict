import configparser
from os import path

cf = configparser.ConfigParser()
cf_file_path = path.join(path.dirname(path.abspath(__file__)), './config.cfg')

# print(cf_file_path)
cf.read(cf_file_path)

MO_ACCESS_URI = cf.get("Monitoring", "access_uri")
MO_RETENTION = cf.getint("Monitoring", "retention")
MO_COLLECT_STEP = cf.getint("Monitoring", "collect_step")

PRED_PERIOD = cf.getint("Predict", "predict_period")

PROPHET_WT = cf.getfloat("Prophet", "weight")

LSTM_UNITS = cf.getint("LSTM", "units")
LSTM_BATH_SIZE = cf.getint("LSTM", "batch_size")
LSTM_EPOCHS = cf.getint("LSTM", "epochs")
LSTM_IN_LEN = cf.getint("LSTM", "input_steps")
LSTM_WT = cf.getfloat("LSTM", "weight")

ED_WT = cf.getfloat("Encoder-Decoder", "weight")

AT_WT = cf.getfloat("Attention", "weight")