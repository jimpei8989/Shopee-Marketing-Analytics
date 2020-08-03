import sys
import time

def generate_submission(predictions, output_file):
    with open(output_file, 'w') as f:
        print('row_id,open_flag', file=f)
        for i, pred in enumerate(predictions):
            print(f'{i},{pred}', file=f)

_LightGray = '\x1b[38;5;251m'
_Bold = '\x1b[1m'
_Underline = '\x1b[4m'
_Orange = '\x1b[38;5;215m'
_SkyBlue = '\x1b[38;5;38m'
_Reset = '\x1b[0m'

class EventTimer():
    def __init__(self, name = '', verbose = True):
        self.name = name
        self.verbose = verbose

    def __enter__(self):
        if self.verbose:
            print(_LightGray + '------------------ Begin "' + _SkyBlue + _Bold + _Underline + self.name + _Reset + _LightGray + '" ------------------' + _Reset, file = sys.stderr)
        self.beginTimestamp = time.time()
        return self

    def __exit__(self, type, value, traceback):
        elapsedTime = time.time() - self.beginTimestamp
        if self.verbose:
            print(_LightGray + '------------------ End "' + _SkyBlue + _Bold + _Underline + self.name + _Reset + _LightGray + ' (Elapsed ' + _Orange + f'{elapsedTime:.4f}' + _Reset + 's)" ------------------' + _Reset + '\n', file = sys.stderr)

    def gettime(self):
        return time.time() - self.beginTimestamp
