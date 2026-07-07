# AP 2026

from abc import abstractmethod
# from tkinter.font import names
from typing import Callable

import numpy as np
from astropy.io import ascii, fits
from astropy.table import Table

from astropy import units as u
import os

from utilities.fitFileHandler import FitFileHandler

class SpectraHandler(FitFileHandler):
    def __init__(self, path):
        super().__init__(path)
        
        
    @abstractmethod
    def load_fit_file(self, path):
        self.fit_path = path
        pass
