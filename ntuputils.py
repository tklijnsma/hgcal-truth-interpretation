from __future__ import print_function
import uproot
import numpy as np, logging, os.path as osp, os
import seutils
from math import pi
import matplotlib.pyplot as plt
from contextlib import contextmanager

# ___________________________________________________
# General utils

DEFAULT_LOGGING_LEVEL = logging.DEBUG
def setup_logger(name='hgcalplot'):
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
    logger.info('Using tqdm notebook')
else:
    import tqdm.tqdm as tqdm

@contextmanager
def temporarily_set_loglevel(loglevel=logging.WARNING):
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
# arrays looping utils

def numentries(arrays):
    return arrays[list(arrays.keys())[0]].shape[0]
    
def iterate_events(arrays):
    """
    Iterates event by event from an arrays of events
    """
    n = numentries(arrays)
    for i in range(n):
        yield { k : v[i:i+1] for k, v in arrays.items() }

class Dataset(object):
    """
    Container for a bunch of root files with an iterate method to easily do event loops
    over an arbitrary number of files.
    Has a cache functionality to store events in memory, making relooping very fast.
    """
    
    def __init__(self, rootfiles, treename='HistoryNTupler/tree', make_cache=True, **kwargs):
        super().__init__()
        if is_string(rootfiles) and '*' in rootfiles:
            import seutils
            rootfiles = seutils.ls_wildcard(rootfiles)
        self.rootfiles = [rootfiles] if is_string(rootfiles) else rootfiles
        self.treename = treename
        self.cache = []
        if make_cache:
            self.make_cache(**kwargs)

    def __repr__(self):
        return super().__repr__().replace('Dataset', 'Dataset ({0} root files)'.format(len(self.rootfiles)))

    def iterate(self, progressbar=True, n_files=None, use_cache=True, **kwargs):
        """
        Wrapper around uproot.iterate:
        - Gets a progress bar option
        - Possibility to limit number of files
        - Can use a class cache variable
        """
        if use_cache:
            if not len(self.cache):
                raise Exception('use_cache was True but no cache was set')
            logger.info('Using cache')
            iterator = iter(self.cache)
            total = len(self.cache)
            inplace_modifier = kwargs.get('inplace_modifier', None)
        else:
            # Allow reading only the first n_files root files
            rootfiles = self.rootfiles[:]
            if n_files: rootfiles = rootfiles[:n_files]
            iterator = uproot.iterate(rootfiles, self.treename, **kwargs)
            total = len(rootfiles)
            # Function for some modifications to the arrays object that should always be made
            # (Like filling extra branches etc.)
            inplace_modifier = kwargs.get('inplace_modifier', default_arrays_modifier_hgcal)
        if progressbar:
            iterator = tqdm(iterator, total=total, desc='arrays' if use_cache else 'root files')
        for arrays in iterator:
            # If an inplace_modifier function was given, use it to modify arrays
            if inplace_modifier:
                inplace_modifier(arrays)
            yield arrays
            
    def iterate_events(self, **kwargs):
        """
        Like self.iterate(), but yields a single event per iteration
        """
        for arrays in self.iterate(**kwargs):
            for event in iterate_events(arrays):
                yield event
            
    def make_cache(self, **kwargs):
        """
        Stores result of self.iterate in a class variable for fast reuse
        """
        if not(self.cache is None): logger.info('Overwriting cache for %s', self)
        self.cache = []
        self.sizeof_cache = 0
        self.numentries_cache = 0
        branches = None
        for arrays in self.iterate(use_cache=False, **kwargs):
            self.cache.append(arrays)
            if branches is None: branches = list(arrays.keys())
            self.sizeof_cache += sum([ v.nbytes for v in arrays.values() ])
            self.numentries_cache += arrays[branches[0]].shape[0]
        logger.info(
            'Cached ~%s (%s entries, %s branches)',
            bytes_to_human_readable(self.sizeof_cache), self.numentries_cache, len(branches)
            )

    def clear_cache(self):
        self.cache = []
        
    def get_event(self, i=0,**kwargs):
        i_entry_start = 0
        kwargs['progressbar'] = False
        for arrays in self.iterate(**kwargs):
            i_entry_stop = i_entry_start + numentries(arrays) - 1
            if i > i_entry_stop:
                i_entry_start = i_entry_stop + 1
                continue
            # Cut out the one entry we're interested in in a new arrays
            return { k : v[i-i_entry_start:i-i_entry_start+1] for k, v in arrays.items() }
        else:
            raise Exception(
                'Requested entry {0} not in range; reached end of stored events at entry {1}'
                .format(i, i_entry_stop)
                )


# ___________________________________________________
# arrays modification utils

def default_arrays_modifier_hgcal(arrays):
    """
    Default inplace modification for arrays
    """
    fill_hit_track_index(arrays)
    fill_track_vertex(arrays)
    # filter_tracks_to_origin(arrays, inplace=True)

def get_hit_track_index(
    arrays,
    hit_track_id_branch=b'simhit_fineTrackId',
    track_id_branch=b'simtrack_trackId',
    ):
    """
    Translates the track_id of a hit to the index of that track in the event.
    This is a rather complex operation for jagged arrays.
    Not sure how to do out without at least one for-loop.
    """
    # Use dependency for single-ndarray index finding, this is already hard
    import numpy_indexed as npi
    import awkward
    all_hit_track_id = arrays[hit_track_id_branch]
    all_track_id = arrays[track_id_branch]
    # Run index finding per entry, and make jagged array later
    hit_track_index_values = []
    hit_track_index_starts = []
    hit_track_index_stops = []
    i_start = 0
    for i in range(numentries(arrays)):
        hit_track_id = all_hit_track_id[i]
        track_id = all_track_id[i]
        n_hits = hit_track_id.shape[0]
        # Get indices for all track ids; this is vectorized fortunately
        hit_track_index = npi.indices(track_id, hit_track_id)
        # logger.debug('event %s', i)
        # logger.debug(hit_track_id)
        # logger.debug(track_id)
        # logger.debug(hit_track_index)
        hit_track_index_values.extend(hit_track_index)
        hit_track_index_starts.append(i_start)
        hit_track_index_stops.append(i_start + n_hits)
        i_start += n_hits
    return awkward.JaggedArray(hit_track_index_starts, hit_track_index_stops, hit_track_index_values)

def fill_hit_track_index(
    arrays,
    hit_track_id_branch=b'simhit_fineTrackId',
    track_id_branch=b'simtrack_trackId',
    hit_track_index_branch=None,
    fill_hit_pdgid=True,
    track_pdgid_branch=b'simtrack_pdgid',
    hit_pdgid_branch=b'simhit_pdgid',
    ):
    """
    Like get_hit_track_index, but sets the result in a new branch of arrays (i.e. an inplace operation)
    """
    if hit_track_index_branch is None:
        hit_track_index_branch = add_to_bytestring(hit_track_id_branch, '_index')
    arrays[hit_track_index_branch] = get_hit_track_index(arrays, hit_track_id_branch, track_id_branch)
    # Translate track index to a pdgid for the hit
    if fill_hit_pdgid:
        arrays[hit_pdgid_branch] = arrays[track_pdgid_branch][arrays[hit_track_index_branch]]
        
def fill_track_vertex(arrays):
    """
    Sets variables track_vertex_<coordinate> using track_vertexIndex
    """
    vertex_index = arrays[b'simtrack_vertexIndex']
    arrays[b'simtrack_vertex_x'] = arrays[b'simvertex_x'][vertex_index]
    arrays[b'simtrack_vertex_y'] = arrays[b'simvertex_y'][vertex_index]
    arrays[b'simtrack_vertex_z'] = arrays[b'simvertex_z'][vertex_index]
    arrays[b'simtrack_noParent'] = arrays[b'simvertex_noParent'][arrays[b'simtrack_vertexIndex']]

def filter_tracks_to_origin(arrays, inplace=False):
    """
    Filters out tracks that point to the origin
    """
    return select(
        arrays,
        sel_track = (arrays[b'simtrack_x'] == 0.) & (arrays[b'simtrack_y'] == 0.) & (arrays[b'simtrack_z'] == 0.),
        invert=True,
        inplace=inplace
        )

def select(arrays, sel_track=None, sel_vertex=None, sel_hit=None, invert=False, inplace=False):
    """
    Generic function that returns arrays, but with selected tracks/vertices/hits.
    If inplace=True, modifies the input arrays in place.
    """
    selector_dict = {}
    if not(sel_track is None): selector_dict['simtrack'] = sel_track
    if not(sel_vertex is None): selector_dict['simvertex'] = sel_vertex
    if not(sel_hit is None): selector_dict['simhit'] = sel_hit
    # Loop over keys and perform selection
    arrays_copy = {}
    for bkey in arrays.keys():
        key = bkey.decode('utf-8')        
        # Little hacky, split on underscores and get first part,
        # use it as a key for the dict above
        selector_key = key.split('_')[0]
        selector = selector_dict.get(selector_key, None)
        if selector is None:
            # Just copy in this case
            if inplace:
                pass
            else:
                arrays_copy[bkey] = arrays[bkey][:]
        else:
            if invert: selector = np.logical_not(selector)
            # Select according to selector or just select all if None (do copy the vals though)
            if inplace:
                arrays[bkey] = arrays[bkey][selector]
            else:
                arrays_copy[bkey] = arrays[bkey][selector]
    return arrays if inplace else arrays_copy

def select_pos(arrays, invert=False):
    """
    Returns arrays (hits, tracks, vertices) for only the + side of the detector.
    Selects only the - side if invert=True.
    Works on JaggedArray, i.e. whole group of events
    """
    return select(
        arrays,
        sel_hit = arrays[b'simhit_z'] > 0.,
        sel_track = arrays[b'simtrack_z'] > 0.,
        sel_vertex = arrays[b'simvertex_z'] > 0.,
        invert = invert
        )

def select_neg(arrays):
    return select_pos(arrays, invert=True)


# ___________________________________________________
# plotting

PDGID_COLORS = {
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

Z_POS_LAYERS = [
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


@contextmanager
def save_plots(flag=True, dir=None):
    try:
        current_flag = PlotBase.SAVEPLOTS
        current_dir = PlotBase.PLOTDIR
        PlotBase.SAVEPLOTS = flag
        if dir: PlotBase.PLOTDIR = dir
        yield PlotBase.PLOTDIR
    finally:
        PlotBase.SAVEPLOTS = current_flag
        PlotBase.PLOTDIR = current_dir

class PlotBase(object):
    """
    Base class for plots
    """
    SAVEPLOTS = False
    PLOTDIR = '.'

    def __init__(self):
        self.ax = None

    
class Plot3DSingleEndcap(PlotBase):

    title = 'plot3d'

    @classmethod
    def plot_both_endcaps(cls, arrays, trim_tracks=True, title=None):
        from ipywidgets.widgets.interaction import show_inline_matplotlib_plots
        fig = plt.figure(figsize=(23,11))
        ax1 = fig.add_subplot(121, projection='3d')
        logger.debug('PLOTTING MINUS ENDCAP')
        cls(arrays, pos=False, ax=ax1).plot(trim_tracks=trim_tracks)
        ax2 = fig.add_subplot(122, projection='3d')
        logger.debug('PLOTTING PLUS ENDCAP')
        cls(arrays, pos=True, ax=ax2).plot(trim_tracks=trim_tracks)
        if cls.SAVEPLOTS:
            dst = osp.join(cls.PLOTDIR, (cls.title if title is None else title) + '.png')
            if not osp.isdir(osp.dirname(dst)): os.makedirs(osp.dirname(dst))
            fig.savefig(dst, bbox_inches='tight', dpi=300)
            fig.savefig(dst.replace('.png', '.pdf'), bbox_inches='tight')
            plt.close()
        else:
            show_inline_matplotlib_plots()
    
    def __init__(self, arrays, pos=True, ax=None):
        super().__init__()
        if numentries(arrays) > 1:
            logger.warning(
                '%s: Selecting first event only out of %s events',
                self.__class__.__name__, numentries(arrays)
                )
        event = { k : v[:1] for k, v in arrays.items() }
        self.event = select_pos(event) if pos else select_neg(event)
        self.ax = ax
        self.zmin = HGCAL_ZMIN_POS if pos else HGCAL_ZMIN_NEG
        self.zmax = HGCAL_ZMAX_POS if pos else HGCAL_ZMAX_NEG
    
    def get_ax(self):
        if self.ax is None:
            self.fig = plt.figure(figsize=(11,11))
            self.ax = fig.add_subplot(111, projection='3d')
        return self.ax

    def trim_track(self, x, y, z, zmin, zmax):
        """
        Takes 2-element lists x, y, and z, and trims anything outside zmin and zmax.
        """
        # Make sure first is the lowest z; Flip back in end result if so
        flip_back = False
        if z[0] > z[1]:
            x = x[::-1]
            y = y[::-1]
            z = z[::-1]
            flip_back = True
        dx = x[1] - x[0]
        dy = y[1] - y[0]
        dz = z[1] - z[0]
        # Trim potential part of track below zmin
        if z[0] < zmin:
            smin = (zmin - z[0]) / dz
            xmin = x[0] + smin * dx
            ymin = y[0] + smin * dy
        else:
            xmin = x[0]
            ymin = y[0]
            zmin = z[0]
        # Trim potential part of track above zmax
        if z[1] > zmax:
            smax = (zmax - z[0]) / dz
            xmax = x[0] + smax * dx
            ymax = y[0] + smax * dy
        else:
            xmax = x[1]
            ymax = y[1]
            zmax = z[1]
        if flip_back:
            return np.array([xmax, xmin]), np.array([ymax, ymin]), np.array([zmax, zmin])
        else:
            return np.array([xmin, xmax]), np.array([ymin, ymax]), np.array([zmin, zmax])
    
    def plot_beamline(self):
        ax = self.get_ax()
        ax.plot(
            [self.zmin, self.zmax], [0., 0.], [0., 0.],
            c = 'xkcd:magenta',
            linewidth = 0.2
            )
    
    def plot_hits_per_pdgid(self):
        ax = self.get_ax()
        hit_x = self.event[b'simhit_x'][0]
        hit_y = self.event[b'simhit_y'][0]
        hit_z = self.event[b'simhit_z'][0]
        hit_energy = self.event[b'simhit_energy'][0]
        hit_pdgid = self.event[b'simhit_pdgid'][0]
        unique_pdgids = np.unique(np.abs(hit_pdgid))
        for pdgid in unique_pdgids:
            select_hits = (np.abs(hit_pdgid) == pdgid) & (hit_z <= self.zmax) & (hit_z >= self.zmin)
            ax.scatter(
                hit_z[select_hits],
                hit_x[select_hits],
                hit_y[select_hits],
                c = color_pdgid(int(pdgid)),
                s = 10000. * hit_energy[select_hits],
                label = '{0:0d}'.format(int(pdgid))
                )

    def compose_line_segments(self, x_in, x_out, y_in, y_out, z_in, z_out):
        # Compose nodes list: should be [in, out, None, in, out, None, ...]
        nones = np.array([None for i in range(x_in.shape[0])])
        x = np.stack((x_in, x_out, nones)).T.flatten()
        y = np.stack((y_in, y_out, nones)).T.flatten()
        z = np.stack((z_in, z_out, nones)).T.flatten()
        return list(x), list(y), list(z)

    def plot_tracks_per_pdgid(self, trim=True, indicate_no_parent=True):
        ax = self.get_ax()
        x_in = self.event[b'simtrack_vertex_x'][0]
        y_in = self.event[b'simtrack_vertex_y'][0]
        z_in = self.event[b'simtrack_vertex_z'][0]
        x_out = self.event[b'simtrack_x'][0]
        y_out = self.event[b'simtrack_y'][0]
        z_out = self.event[b'simtrack_z'][0]
        no_parent = self.event[b'simtrack_noParent'][0]
        x = np.stack((x_in, x_out)).T
        y = np.stack((y_in, y_out)).T
        z = np.stack((z_in, z_out)).T
        track_pdgid = self.event[b'simtrack_pdgid'][0]
        # Store coordinates where to draw a "no parent indicator" (a cross), and draw in 1 go later
        no_parent_indicator_coordinates = []
        for i in range(x.shape[0]):
            this_color = color_pdgid(int(track_pdgid[i]))
            this_x, this_y, this_z = x[i], y[i], z[i]
            # First check if at least a part of the track is in this endcap
            if all(this_z < self.zmin) or all(this_z > self.zmax):
                logger.debug('Track z = %s is outside this endcap (%s - %s)', this_z, self.zmin, self.zmax)
                continue
            # If partly outside, trim the track to prevent weird lines in the plot
            if trim:
                this_x, this_y, this_z = self.trim_track(x[i], y[i], z[i], self.zmin, self.zmax)
                logger.debug(
                    'Trimmed track (pdgid %s):\n'
                    '    z: %s --> %s\n'
                    '    x: %s --> %s\n'
                    '    y: %s --> %s',
                    int(track_pdgid[i]), z[i], this_z, x[i], this_x, y[i], this_y,
                    )
            ax.plot(
                this_z, this_x, this_y,
                c = this_color,
                linewidth = 0.5 if trim != 'both' else 1.5, # Make thicker if drawing both
                )
            # Save in-coordinates if no parent
            if no_parent[i]:
                no_parent_indicator_coordinates.append([this_x[0], this_y[0], this_z[0]])
            # Draw untrimmed track if asked
            if trim == 'both':
                ax.plot(
                    z[i], x[i], y[i],
                    c = this_color,
                    linewidth = 0.5
                    )
        # Draw no-parent indicators if asked
        if indicate_no_parent:
            logger.debug(
                'Found %s tracks without parents; plotting points at: %s',
                len(no_parent_indicator_coordinates), no_parent_indicator_coordinates
                )
            no_parent_indicator_coordinates = np.array(no_parent_indicator_coordinates).T
            x, y, z = no_parent_indicator_coordinates[0], no_parent_indicator_coordinates[1], no_parent_indicator_coordinates[2]
            ax.scatter(
                z, x, y,
                c = 'r',
                marker = 'x',
                s = 100.,
                label = 'Tracks w/o parent'
                )


    def plot(self, trim_tracks=True):
        ax = self.get_ax()
        
        ax.set_xlim(self.zmin, self.zmax)
        ax.set_xlabel('z')
        # xmin = -150
        # xmax = 150
        # ax.set_ylim(xmin, xmax)
        # ax.set_zlim(xmin, xmax)
        ax.set_ylabel('x')
        ax.set_zlabel('y')
        
        self.plot_beamline()
        self.plot_hits_per_pdgid()
        self.plot_tracks_per_pdgid(trim=trim_tracks)
        ax.legend()
