from __future__ import print_function
import numpy as np, logging, os.path as osp, os, gc
import seutils
from math import pi
import matplotlib._color_data as mcd
import matplotlib.pyplot as plt
from contextlib import contextmanager

import uptools
import trees

# ___________________________________________________
# General utils

DEFAULT_LOGGING_LEVEL = logging.INFO
def setup_logger(name='hnt'):
    if name in logging.Logger.manager.loggerDict:
        logger = logging.getLogger(name)
        logger.info('Logger %s is already defined', name)
    else:
        fmt = logging.Formatter(
            fmt = (
                '\033[33m%(levelname)7s:%(asctime)s:%(module)s:%(lineno)s\033[0m'
                + ' %(message)s'
                ),
            datefmt='%Y-%m-%d %H:%M:%S'
            )
        handler = logging.StreamHandler()
        handler.setFormatter(fmt)
        logger = logging.getLogger(name)
        logger.setLevel(DEFAULT_LOGGING_LEVEL)
        logger.addHandler(handler)
    return logger
logger = setup_logger()

def is_interactive():
    import __main__ as main
    return not hasattr(main, '__file__')

if is_interactive():
    from tqdm.notebook import tqdm
    logger.debug('Using tqdm notebook')
else:
    from tqdm import tqdm

@contextmanager
def loglevel(loglevel=logging.DEBUG):
    """
    Temporarily sets the logging level to some other level
    """
    current_level = logger.level
    try:
        logger.setLevel(loglevel)
        yield True
    finally:
        logger.setLevel(current_level)

def is_string(string):
    """
    Checks strictly whether `string` is a string
    Python 2/3 compatibility (https://stackoverflow.com/a/22679982/9209944)
    """
    try:
        basestring
    except NameError:
        basestring = str
    return isinstance(string, basestring)

def bytes_to_human_readable(num, suffix='B'):
    """
    Convert number of bytes to a human readable string
    """
    for unit in ['','k','M','G','T','P','E','Z']:
        if abs(num) < 1024.0:
            return '{0:3.1f} {1}b'.format(num, unit)
        num /= 1024.0
    return '{0:3.1f} {1}b'.format(num, 'Y')

def add_to_bytestring(bytestring, tag):
    normal_string = bytestring.decode('utf-8')
    normal_string += tag
    return normal_string.encode('utf-8')


# ___________________________________________________
# Geometry constants

Z_POS_LAYERS = [
    320.5,
    322.103, 323.047, 325.073, 326.017, 328.043, 328.987, 331.013,
    331.957, 333.983, 334.927, 336.953, 337.897, 339.923, 340.867,
    342.893, 343.837, 345.863, 346.807, 348.833, 349.777, 351.803,
    352.747, 354.773, 355.717, 357.743, 358.687, 360.713, 361.657,
    367.699, 373.149, 378.599, 384.049, 389.499, 394.949, 400.399,
    405.849, 411.299, 416.749, 422.199, 427.649, 436.199, 444.749,
    453.299, 461.849, 470.399, 478.949, 487.499, 496.049, 504.599,
    513.149
    ]
Z_NEG_LAYERS = [
    -320.5,
    -322.103, -323.047, -325.073, -326.017, -328.043, -328.987, -331.013,
    -331.957, -333.983, -334.927, -336.953, -337.897, -339.923, -340.867,
    -342.893, -343.837, -345.863, -346.807, -348.833, -349.777, -351.803,
    -352.747, -354.773, -355.717, -357.743, -358.687, -360.713, -361.657,
    -367.699, -373.149, -378.599, -384.049, -389.499, -394.949, -400.399,
    -405.849, -411.299, -416.749, -422.199, -427.649, -436.199, -444.749,
    -453.299, -461.849, -470.399, -478.949, -487.499, -496.049, -504.599,
    -513.149
    ]

HGCAL_ZMIN_POS = min(Z_POS_LAYERS)
HGCAL_ZMAX_POS = max(Z_POS_LAYERS)
HGCAL_ZMIN_NEG = min(Z_NEG_LAYERS)
HGCAL_ZMAX_NEG = max(Z_NEG_LAYERS)


# ___________________________________________________
# plotting

PDGID_COLORS = {
    0 : 'xkcd:purple', # undefined
    13 : 'xkcd:light blue', # muon
    11 : 'g', # electron
    22 : 'r', # photon
    211 : 'xkcd:orange', # pion
    }

def color_pdgid(pdgid, default_value='xkcd:gray'):
    """
    Should work for numpy arrays, will NOT work for JaggedArrays
    https://stackoverflow.com/a/16993364/9209944
    """
    if hasattr(pdgid, 'shape'):
        pdgid = np.abs(pdgid)
        # Get only the unique values for which to call the dict.get method,
        # and get the indices per unique value to reconstruct later
        u, inv = np.unique(pdgid, return_inverse=True)
        # Do dict mapping, and reconstruct to shape of pdgid
        color = np.array([PDGID_COLORS.get(x, default_value) for x in u])[inv].reshape(pdgid.shape)
    else:
        color = PDGID_COLORS.get(abs(pdgid), default_value)
    return color

_all_colors = list(mcd.XKCD_COLORS.keys())
_assigned_colors = {}
np.random.seed(44)
def color_for_id(i):
    global _assigned_colors, _all_colors
    if not i in _assigned_colors:
        i_picked_color = np.random.randint(len(_all_colors))
        _assigned_colors[i] = _all_colors.pop(i_picked_color)
        if len(_all_colors) == 0:
            logger.warning('Out of colors: resetting color wheel')
            _all_colors = list(mcd.XKCD_COLORS.keys())
    return _assigned_colors[i]

def shuffle_colors(seed=1001):
    '''
    Shuffles only currently assigned colors
    '''
    np.random.seed(seed)
    global _assigned_colors
    assigned_colors = _assigned_colors.values()
    np.random.shuffle(assigned_colors)
    for i, key in enumerate(_assigned_colors):
        _assigned_colors[key] = assigned_colors[i]


class IDColor:
    '''Returns a consistent color when given the same object'''
    def __init__(self, colors=None, seed=44):
        self.colors = list(mcd.XKCD_COLORS.keys()) if colors is None else colors
        np.random.seed(seed)
        np.random.shuffle(self.colors)
        self._original_colors = self.colors.copy()
        self.assigned_colors = {}
        
    def __call__(self, thing):
        if thing in self.assigned_colors:
            return self.assigned_colors[thing]
        else:
            color = self.colors.pop()
            self.assigned_colors[thing] = color
            if not(self.colors): self.colors = self._original_colors.copy()
            return color


def build_tree(event):
    tracksview = uptools.Bunch.from_branches(event, [k for k in event.keys() if k.decode().startswith('simtrack_')])
    hitsview = uptools.Bunch.from_branches(event, [k for k in event.keys() if k.decode().startswith('simhit_')])
    id_to_track = {} 
    tracks = []
    for i in range(len(tracksview)):
        trackview = tracksview[i]
        track = trees.Track(
            crossedBoundary    = bool(trackview.simtrack_crossedboundary),
            energy             = float(trackview.simtrack_energy),
            energyAtBoundary   = float(trackview.simtrack_boundary_energy),
            noParent           = bool(trackview.simtrack_noparent),
            parentTrackId      = int(trackview.simtrack_parenttrackid),
            pdgid              = int(trackview.simtrack_pdgid),
            trackid            = int(trackview.simtrack_trackid),
            x                  = float(trackview.simtrack_x),
            y                  = float(trackview.simtrack_y),
            z                  = float(trackview.simtrack_z),
            vertex_x           = float(trackview.simtrack_vertex_x),
            vertex_y           = float(trackview.simtrack_vertex_y),
            vertex_z           = float(trackview.simtrack_vertex_z),
            xAtBoundary        = float(trackview.simtrack_boundary_x),
            yAtBoundary        = float(trackview.simtrack_boundary_y),
            zAtBoundary        = float(trackview.simtrack_boundary_z),
            )
        if trackview.simtrack_hashits:
            hits_for_track = hitsview[hitsview.simhit_trackid == trackview.simtrack_trackid]
            track.hits = [
                trees.Hit(
                    hits_for_track.simhit_detid[i_hit],
                    hits_for_track.simhit_x[i_hit],
                    hits_for_track.simhit_y[i_hit],
                    hits_for_track.simhit_z[i_hit],
                    hits_for_track.simhit_energy[i_hit],
                    parent=track
                    ) for i_hit in range(len(hits_for_track))
                ]        
        else:
            track.hits = []
        id_to_track[track.trackid] = track
        tracks.append(track)
    # Set parents and children
    root = trees.Track(root=True)
    for i_track, track in enumerate(tracks):
        parent = id_to_track.get(track.parentTrackId, None)
        if parent is None: parent = root
        track.parent = parent
        parent.children.append(track)
    # Prune branches that have no children and no hits
    # First mark all tracks as removable:
    for track in tracks:
        track.keep = False
    # Then only keep tracks that have hits, or are the parent of a track with hits:
    for track in tracks:
        if track.nhits > 0:
            for parent in track.traverse_up():
                parent.keep = True
    # Remove all other nodes
    for track in tracks:
        if not track.keep: trees.remove(track)
    return root


def split_endcaps(root, flip=False):
    pos = trees.Track(root=True, z=0.001)
    neg = trees.Track(root=True, z=-0.001)
    for track in root.children:
        if track.z < 0.:
            neg.children.append(track)
            track.parent = neg
        else:
            pos.children.append(track)
            track.parent = pos
    if flip:
        neg = trees.flipz_tree(neg)
    return pos, neg
