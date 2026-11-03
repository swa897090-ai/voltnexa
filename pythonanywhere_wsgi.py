# PythonAnywhere: paste this into your WSGI configuration file (Web tab) and change USERNAME.
import os, sys
path = "/home/USERNAME/voltnexa"
sys.path.insert(0, path)
os.chdir(path)
from voltnexa_pro_v3 import app as application
