# Some HGCAL things

layers = range(1, 28)
z_pos_layers = [
    322.10275269, 323.04727173, 325.07275391, 326.01730347, 328.04275513,
    328.98727417, 331.01272583, 331.95724487, 333.98275757, 334.92724609,
    336.95275879, 337.89724731, 339.92276001, 340.86727905, 342.89273071,
    343.83724976, 345.86276245, 346.80725098, 348.83276367, 349.7772522,
    351.80276489, 352.7472229,  354.77279663, 355.71725464, 357.74276733,
    358.68725586, 360.71276855, 361.65725708
    ]
z_neg_layers = [
    -322.10275269, -323.04727173, -325.07275391, -326.01730347, -328.04275513,
    -328.98727417, -331.01272583, -331.95724487, -333.98275757, -334.92724609,
    -336.95275879, -337.89724731, -339.92276001, -340.86721802, -342.89279175,
    -343.83724976, -345.86276245, -346.80725098, -348.83276367, -349.7772522,
    -351.80276489, -352.74728394, -354.7727356,  -355.71725464, -357.74276733,
    -358.68725586, -360.71276855, -361.65725708,
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

def get_flat_event_iterator_rootfiles(rootfiles):
    for arrays in uproot.iterate(rootfiles, b'HistoryNTupler/tree'):
        n_events = arrays[b'simhit_detid'].shape[0] # Can use any branch
        for i in range(n_events):
            arrays_thisevt = { key : value[i] for key, value in arrays.items()}
            yield FlatEvent(arrays_thisevt, arrays_thisevt, arrays_thisevt)


class FlatEvent:
    def __init__(self, dhits=None, dtracks=None, dvertices=None):
        if not(dhits is None):
            self.read_dicts(dhits, dtracks, dvertices)
        
    def read_dicts(self, dhits, dtracks, dvertices):
        self.hit_x = dhits[b'simhit_x'].flatten()
        self.hit_y = dhits[b'simhit_y'].flatten()
        self.hit_z = dhits[b'simhit_z'].flatten()
        self.hit_energy = dhits[b'simhit_energy'].flatten()
        self.hit_trackId = dhits[b'simhit_trackId'].flatten()

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
    