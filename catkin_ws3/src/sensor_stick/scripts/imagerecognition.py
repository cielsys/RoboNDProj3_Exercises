#!/usr/bin/env python
import matplotlib

def Backend_Switch(whichBackEnd):
    matplotlib.use(whichBackEnd, warn=False, force=True)
    from matplotlib import pyplot as plt
    print "Switched to:", matplotlib.get_backend()

Backend_Switch('QT4Agg')

import matplotlib.pyplot as plt
import numpy as np
import cv2
import math
import os
import sys
import pickle
#import collections
#import types
#import traceback
#import pprint
#pp = pprint.PrettyPrinter(indent=4)