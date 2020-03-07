
import os
import glob
import sys
import tarfile
import time
from colorama import init
init()

from datetime import datetime
import json

cNrm  = '\033[0m'  # white (normal)
cR    = '\033[31m' # red
cG    = '\033[32m' # green
cY    = '\033[33m' # yellowe
cB    = '\033[34m' # blue
cP    = '\033[35m' # purple
cM    = '\033[35m' # purple
cC    = '\033[36m' # cyan
cW    = '\033[37m' # yellown

cIR="\033[0;91m"         # Red
cIG="\033[0;92m"       # Green
cIY="\033[0;93m"      # Yellow
cIB="\033[0;94m"        # Blue
cIP="\033[0;95m"      # Purple
cIM="\033[0;95m"      # Purple
cIC="\033[0;96m"        # Cyan
cIW="\033[0;97m"       # White
defnodecolor = cNrm
l_allways = 0
l_err = 1
l_error = 1
l_warn = 2
l_info = 3
l_verbose = 4
l_debug = 5
levelstring = ["allways","error","warn","info","verbose","debug"]

class Lgger:

    verbosity = 3
    nodename = ""

    stripcolor = False
    # l_always = 0    
    # l_error = 1
    # l_err = 1
    # l_warn = 2
    # l_info = 3
    # l_debug = 4
    defnodecolor = cNrm

    def __init__(self, nodename="default",verbosity=3,defnodecolor=cNrm):
        self.nodename = nodename
        self.verbosity = verbosity
        self.defnodecolor = defnodecolor

    def setverbosity(self,verbosity):
        self.verbosity = verbosity

    def dolog(self,lev,msg,clr=defnodecolor):
        if (lev<=self.verbosity):
            if (self.stripcolor):
                print(str(msg))
            else:
                print(clr+str(msg)+cNrm)
    def allways(self,msg,clr=defnodecolor):
        self.dolog(0,msg,clr)
    def err(self,msg,clr=defnodecolor):
        self.dolog(l_err,msg,clr)

    def error(self,msg,clr=defnodecolor):
        self.dolog(l_err,msg,clr)

    def warn(self,msg,clr=defnodecolor):
        self.dolog(l_warn,msg,clr)

    def info(self,msg,clr=defnodecolor):
        self.dolog(l_info,msg,clr)

    def verbose(self,msg,clr=defnodecolor):
        self.dolog(l_verbose,msg,clr)

    def deb(self,msg,clr=defnodecolor):
        self.dolog(l_debug,msg,clr)

    def debug(self,msg,clr=defnodecolor):
        self.dolog(l_debug,msg,clr)      

    def GetLogger(self):
        return getlgg()

def allways(msg,clr=defnodecolor):
    _lgg.allways(msg,clr)
def err(msg,clr=defnodecolor):
    _lgg.err(msg,clr)
def error(msg,clr=defnodecolor):
    _lgg.error(msg,clr)
def warm(msg,clr=defnodecolor):
    _lgg.warn(msg,clr)
def info(msg,clr=defnodecolor):
    _lgg.info(msg,clr)    
def verbose(msg,clr=defnodecolor):
    _lgg.verbose(msg,clr)    
def deb(msg,clr=defnodecolor):
    _lgg.deb(msg,clr)
def debug(msg,clr=defnodecolor):
    _lgg.debug(msg,clr)

def set_level(level):
    if level<l_err:
        level = l_err
    if level>l_debug:
        level = l_debug
    _lgg.verbosity = level

def get_level():
    return _lgg.verbosity

def get_level_str():
    i = _lgg.verbosity
    return levelstring[i]

_lgg = Lgger("nonname2",3)    
print("Initialized lgger - level:"+get_level_str())

