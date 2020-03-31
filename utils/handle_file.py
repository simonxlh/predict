import logging
logger = logging.getLogger('usual')

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def get_first_last_line(filename):
    """
    get the first line and the last line in the file

    Args:
        filename: file name

    Returns: (filst line,last line) or (None, None) for empty file
    """
    try:
        filesize = os.path.getsize(filename)
        if filesize == 0:
            return None, None
        else:
            with open(filename, 'rb') as fp: # to use seek from end, must use mode 'rb'
                first_line = fp.readline()
                offset = -8                 # initialize offset
                while -offset < filesize:   # offset cannot exceed file size
                    fp.seek(offset, 2)      # read # offset chars from eof(represent by number '2')
                    lines = fp.readlines()  # read from fp to eof
                    if len(lines) >= 2:     # if contains at least 2 lines
                        return first_line.decode(), lines[-1].decode()  # then last line is totally included
                    else:
                        offset *= 2         # enlarge offset
                fp.seek(0)
                lines = fp.readlines()
                return first_line.decode(), lines[-1].decode()
    except FileNotFoundError:
        logger.error(filename + ' not found!')
        return None, None
