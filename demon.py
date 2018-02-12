from datetime import datetime
import pandas as pd
import numpy as np
import scipy
from constants import CONSTANTS
import matplotlib
import matplotlib.pyplot as plt
import math
import matplotlib.dates as mdates

from plotall import plotdoubleseries
from plotall import plotsingleseries
from averagemandi import mandipriceseries
from averageretail import retailpriceseries

fstart = CONSTANTS['STARTDATE']
fend = CONSTANTS['ENDDATE']

#plotdoubleseries(mandipriceseries,retailpriceseries,'Time','Time','Mandi Price','RetailPrice',fstart,fend)
plotsingleseries(mandipriceseries,'Mandi Price','Time','Price per Quintal',fstart,fend,True,True)
