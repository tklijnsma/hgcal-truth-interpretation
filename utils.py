# Some HGCAL things

layers = list(range(1, 51))
z_pos_layers = [
    322.103, 323.047, 325.073, 326.017, 328.043, 328.987, 331.013,
    331.957, 333.983, 334.927, 336.953, 337.897, 339.923, 340.867,
    342.893, 343.837, 345.863, 346.807, 348.833, 349.777, 351.803,
    352.747, 354.773, 355.717, 357.743, 358.687, 360.713, 361.657,
    367.699, 373.149, 378.599, 384.049, 389.499, 394.949, 400.399,
    405.849, 411.299, 416.749, 422.199, 427.649, 436.199, 444.749,
    453.299, 461.849, 470.399, 478.949, 487.499, 496.049, 504.599,
    513.149
    ]
z_neg_layers = [
    -322.103, -323.047, -325.073, -326.017, -328.043, -328.987, -331.013,
    -331.957, -333.983, -334.927, -336.953, -337.897, -339.923, -340.867,
    -342.893, -343.837, -345.863, -346.807, -348.833, -349.777, -351.803,
    -352.747, -354.773, -355.717, -357.743, -358.687, -360.713, -361.657,
    -367.699, -373.149, -378.599, -384.049, -389.499, -394.949, -400.399,
    -405.849, -411.299, -416.749, -422.199, -427.649, -436.199, -444.749,
    -453.299, -461.849, -470.399, -478.949, -487.499, -496.049, -504.599,
    -513.149
    ]

hgcal_zmin_pos = min(z_pos_layers)
hgcal_zmax_pos = max(z_pos_layers)
hgcal_zmin_neg = min(z_neg_layers)
hgcal_zmax_neg = max(z_neg_layers)

def in_hgcal(z):
    """
    Determines whether z is in hgcal
    """
    return in_hgcal_pos(z) or in_hgcal_neg(z)

def in_hgcal_pos(z):
    return z >= hgcal_zmin_pos and z <= hgcal_zmax_pos

def in_hgcal_neg(z):
    return z >= hgcal_zmin_neg and z <= hgcal_zmax_neg

def get_z_for_layer(layer, do_endcap='+'):
    if not layer in layers:
        raise ValueError(
            'Layer {0} is not registered'.format(layer)
            )
    index = layers.index(layer)
    if do_endcap == '+':
        return z_pos_layers[index]
    else:
        return z_neg_layers[index]


# ___________________________________________________
# Some IO for the ntuple
import uproot
import numpy as np
import seutils
from math import pi

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

def get_multiple_indices(subset_values, all_values):
    '''From https://stackoverflow.com/a/32191125/9209944'''
    sorter = np.argsort(all_values)
    return sorter[np.searchsorted(all_values, subset_values, sorter=sorter)]

def get_ntup(rootfile):
    fin = uproot.open(rootfile)
    tree = fin[b'HistoryNTupler'][b'tree']
    return (
        tree.arrays(b'simhit_*'),
        tree.arrays(b'simtrack_*'),
        tree.arrays(b'simvertex_*')
        )

def get_flat_event_iterator(hits, tracks, vertices, filter=None):
    if not(filter is None):
        hits = { key : value[filter] for key, value in hits.items()}
        tracks = { key : value[filter] for key, value in tracks.items()}
        vertices = { key : value[filter] for key, value in vertices.items()}
    n = hits[b'simhit_detid'].shape[0] # Can use any branch
    for i in range(n):
        # Select only this event
        hits_thisevt = { key : value[i] for key, value in hits.items()}
        tracks_thisevt = { key : value[i] for key, value in tracks.items()}
        vertices_thisevt = { key : value[i] for key, value in vertices.items()}
        event = FlatEvent(hits_thisevt, tracks_thisevt, vertices_thisevt)
        event.i = i
        yield event

def get_flat_event_iterator_rootfiles(rootfiles, skip=None):
    """
    Also accepts just a string
    """
    if is_string(rootfiles): rootfiles = [rootfiles]
    for arrays in uproot.iterate(rootfiles, b'HistoryNTupler/tree'):
        n_events = arrays[b'simhit_detid'].shape[0] # Can use any branch
        for i in range(n_events):
            if i < skip: continue
            arrays_thisevt = { key : value[i] for key, value in arrays.items()}
            yield FlatEvent(arrays_thisevt, arrays_thisevt, arrays_thisevt)

def load_event(rootfiles, i=0):
    event = next(get_flat_event_iterator_rootfiles(rootfiles, skip=i))
    return event

class FlatEvent:
    def __init__(self, dhits=None, dtracks=None, dvertices=None):
        if not(dhits is None):
            self.read_dicts(dhits, dtracks, dvertices)
        
    def read_dicts(self, dhits, dtracks, dvertices):
        self.hit_x = dhits[b'simhit_x'].flatten()
        self.hit_y = dhits[b'simhit_y'].flatten()
        self.hit_z = dhits[b'simhit_z'].flatten()
        self.hit_energy = dhits[b'simhit_energy'].flatten()
        self.hit_trackId = dhits[b'simhit_fineTrackId'].flatten()
        self.hit_parentTrackId = dhits[b'simhit_trackId'].flatten()

        self.track_x = dtracks[b'simtrack_x'].flatten()
        self.track_y = dtracks[b'simtrack_y'].flatten()
        self.track_z = dtracks[b'simtrack_z'].flatten()
        self.track_trackId = dtracks[b'simtrack_trackId'].flatten()
        self.track_vertexIndex = dtracks[b'simtrack_vertexIndex'].flatten()
        self.track_pdgid = dtracks[b'simtrack_pdgid'].flatten()
        self.track_energy = dtracks[b'simtrack_momentum'].E.flatten()

        self.vertex_x = dvertices[b'simvertex_x']
        self.vertex_y = dvertices[b'simvertex_y']
        self.vertex_z = dvertices[b'simvertex_z']
        self.vertex_id = dvertices[b'simvertex_id']
        self.vertex_noParent = dvertices[b'simvertex_noParent']
        self.vertex_parentTrackId = dvertices[b'simvertex_parentTrackId']

        # Find parent track per hit, and propegate pdgid of the track to the hit
        # Not sure how to do this nice and columnary
        self.hit_pdgid = np.zeros(self.hit_z.shape)
        for i in range(self.hit_z.shape[0]):
            self.hit_pdgid[i] = self.track_pdgid[self.track_trackId == self.hit_trackId[i]]

        self.track_vertex_x = self.vertex_x[self.track_vertexIndex]
        self.track_vertex_y = self.vertex_y[self.track_vertexIndex]
        self.track_vertex_z = self.vertex_z[self.track_vertexIndex]
    
        assert self.track_vertex_x.shape == self.track_x.shape
    
    def subselection(self, select_hits, select_tracks, select_vertices):
        new = FlatEvent()
        new.hit_x = self.hit_x[select_hits]
        new.hit_y = self.hit_y[select_hits]
        new.hit_z = self.hit_z[select_hits]
        new.hit_energy = self.hit_energy[select_hits]
        new.hit_trackId = self.hit_trackId[select_hits]
        new.hit_pdgid = self.hit_pdgid[select_hits]
        new.track_x = self.track_x[select_tracks]
        new.track_y = self.track_y[select_tracks]
        new.track_z = self.track_z[select_tracks]
        new.track_trackId = self.track_trackId[select_tracks]
        new.track_vertexIndex = self.track_vertexIndex[select_tracks]
        new.track_pdgid = self.track_pdgid[select_tracks]
        new.track_energy = self.track_energy[select_tracks]
        new.vertex_x = self.vertex_x[select_vertices]
        new.vertex_y = self.vertex_y[select_vertices]
        new.vertex_z = self.vertex_z[select_vertices]
        new.vertex_noParent = self.vertex_noParent[select_vertices]
        new.vertex_parentTrackId = self.vertex_parentTrackId[select_vertices]
        new.track_vertex_x = self.track_vertex_x[select_tracks]
        new.track_vertex_y = self.track_vertex_y[select_tracks]
        new.track_vertex_z = self.track_vertex_z[select_tracks]
        if hasattr(self, 'i'): new.i = self.i
        return new
    
    def select_pos(self):
        return self.subselection(
            (self.hit_z >= 0.), (self.track_z >= 0.), (self.vertex_z >= 0.)
            )

    def select_neg(self):
        return self.subselection(
            (self.hit_z <= 0.), (self.track_z <= 0.), (self.vertex_z <= 0.)
            )

    def select_muon(self):
        return self.subselection(
            np.abs(self.hit_pdgid) == 13,
            np.abs(self.track_pdgid) == 13,
            [],
            )
    
    def select_not_muon(self):
        return self.subselection(
            np.abs(self.hit_pdgid) != 13,
            np.abs(self.track_pdgid) != 13,
            [],
            )
