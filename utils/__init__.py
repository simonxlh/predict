import logging
import logging.config
from os import path

from .visualize import A
from .compare import B

logger = logging.getLogger('usual')

# logging.getLogger("simpleExample")
# logging.basicConfig(level=logging.INFO, \
#     format="%(asctime)s - %(filename)s:%(lineno)s - %(levelno)s %(levelname)s %(pathname)s %(module)s %(funcName)s %(created)f %(thread)d %(threadName)s %(process)d %(name)s - %(message)s")
