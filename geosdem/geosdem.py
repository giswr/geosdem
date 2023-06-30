"""Main module."""
from .geosdem import *

import string
import random
import ipyleaflet

class Map(ipyleaflet.Map):
    
    def __init__(self) -> None :
        super().__init__()