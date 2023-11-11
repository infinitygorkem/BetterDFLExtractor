import sys
import os
from pathlib import Path
from subprocess import call

ddfa_v2_path = Path(__file__).resolve().parent.joinpath('ddfa_v2')
sys.path.append(str(ddfa_v2_path))

from DFLIMG import DFLJPG
from mainscripts.Extractor import ExtractSubprocessor, FaceType, LandmarksProcessor
from mainscripts import XSegUtil
from core.leras import nn

os.chdir(str(ddfa_v2_path))
call(["python", str(ddfa_v2_path.joinpath("newextractor.py"))])