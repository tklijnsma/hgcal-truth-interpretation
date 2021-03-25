import uptools, trees

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


def build_tree_newntup(event):
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
