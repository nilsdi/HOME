"""Handeling orthophotos from norgeibilder and building data from matrikkel.
This package gives functions to deal with the APIs of matrkkelen and norgeibilder, 
as well as some management of the orthophotos (e.g. folder structure, project log etc.)
"""

import HOME.data_acquisition.norgeibilder as norgeibilder
import HOME.data_acquisition.matrikkel as matrikkel
from HOME.data_acquisition.norgeibilder.orthophoto_api.download_project import *
