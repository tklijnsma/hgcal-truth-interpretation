from math import pi
import matplotlib.pyplot as plt
import numpy as np, logging, os.path as osp, os
import ntuputils

def traverse(node, yield_depth=False, depth=0):
    yield (node, depth) if yield_depth else node
    for child in node.children:
        yield from traverse(child, yield_depth, depth+1)

def traverse_postorder(node, yield_depth=False, depth=0):
    for child in node.children:
        yield from traverse_postorder(child, yield_depth, depth+1)
    yield (node, depth) if yield_depth else node

def traverse_up(node):
    while True:
        yield node
        if node.parent is None:
            break
        node = node.parent

def remove(node):
    if node.parent: node.parent.children.remove(node)

def get_by_id(root, trackid):
    for node in traverse(root):
        if node.trackid == trackid:
            return node
    raise LookupError 
        
def print_tree(root):
    short_repr = lambda track: (
        '{} E={:.2f} pdg={} {} {}'
        .format(
            track.trackid, track.energy, track.pdgid,
            'nhits={}'.format(track.nhits) if track.nhits>0 else '',
            'X' if track.crossedBoundary else ''
            )
        )
    for node, depth in traverse(root, yield_depth=True):
        print('__'*depth + short_repr(node))

def deltar(eta1, phi1, eta2, phi2):
    dphi = phi1 - phi2
    # Some normalization of phi - first substract whole 2*pi's,
    # then flip to -pi < phi < pi regime
    if isinstance(dphi, float):
        if dphi > 2.*pi: dphi -= 2.*pi
        if dphi < -2.*pi: dphi += 2.*pi
        if dphi > pi: dphi -= 2.*pi
        if dphi < -pi: dphi += 2.*pi
    else:
        dphi[dphi > 2.*pi] -= 2.*pi
        dphi[dphi < -2.*pi] += 2.*pi
        dphi[dphi > pi] -= 2.*pi
        dphi[dphi < -pi] += 2.*pi
    dr = np.sqrt( dphi**2 + (eta1-eta2)**2 )
    return dr

def deltar_tracks(t1, t2):
    p1, p2 = t1.momentum, t2.momentum
    return deltar(p1.eta, p1.phi, p2.eta, p2.phi)

def hitcentroid(track):
    if track.nhits == 0: return None
    if track.nhits == 1: return np.array([track.hits[0].x, track.hits[0].y, track.hits[0].z]), 0.
    e_total = sum([ hit.energy for hit in track.hits ])
    centroid = np.array(
        [ hit.energy/e_total * np.array([hit.x, hit.y, hit.z]) for hit in track.hits ]
        ).sum(axis=0)
    # Check if point on the track at same norm of centroid is approximately the same
    # print(centroid)
    # origin = np.array([track.vertex_x, track.vertex_y, track.vertex_z])
    # pos = np.array([track.x, track.y, track.z])
    # d = pos - origin
    # d = d / np.linalg.norm(d) * np.linalg.norm(centroid-origin)
    # print(d)
    variance = np.sqrt(np.array(
        [ (hit.energy/e_total * np.linalg.norm(np.array([hit.x, hit.y, hit.z])-centroid))**2 for hit in track.hits ]
        ).sum(axis=0))
    return centroid, variance

def copy_tree(root):
    import copy
    return copy.deepcopy(root)

class Track(object):
    def __init__(self, parent=None, children=None, **kwargs):
        self.parent = parent
        self.children = [] if children is None else children
        self.__dict__.update(kwargs)
        self.nhits = 0
        self.hits = []
        self.centroid = None
        self.secondmoment = None
        
    def __deepcopy__(self, memo):
        import copy
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if k == 'hits':
                # Only do a shallow copy of the hits list (We're not modifying hits anyway)
                setattr(result, k, copy.copy(v))
            else:
                setattr(result, k, copy.deepcopy(v, memo))
        return result

    def traverse(self, *args, **kwargs):
        yield from traverse(self, *args, **kwargs)

    def traverse_up(self, *args, **kwargs):
        yield from traverse_up(self, *args, **kwargs)

    def traverse_postorder(self, *args, **kwargs):
        yield from traverse_postorder(self, *args, **kwargs)
            
    def get_by_id(self, trackid):
        return get_by_id(self, trackid)
    
    def print(self):
        print_tree(self)

    def deltar(self, other):
        return deltar_tracks(self, other)

    def get(self, key, ndec=2):
        if not key in self.__dict__:
            return '?'
        val = self.__dict__[key]
        if isinstance(val, float):
            return '{:.{ndec}f}'.format(val, ndec=ndec)
        return val        
            
    def __repr__(self):
        return super().__repr__().replace(
            'object',
            '{} E={} ({},{},{}) pdgid={}'
            .format(
                self.get('trackid'), self.get('energy'),
                self.get('vertex_x',3), self.get('vertex_y',3), self.get('vertex_z',3),
                self.get('pdgid')
                )
            )
    
    def get_centroid(self):
        if self.centroid is None: self.centroid, self.secondmoment = hitcentroid(self)
        return self.centroid

    def get_secondmoment(self):
        if self.centroid is None: self.centroid, self.secondmoment = hitcentroid(self)
        return self.secondmoment


class Hit(object):
    def __init__(self, detid, x, y, z, energy, parent):
        self.detid, self.x, self.y, self.z, self.energy, self.parent = detid, x, y, z, energy, parent
    
    def __repr__(self):
        return super().__repr__().replace(
            'object',
            '{} E={:.2e} ({:.2f},{:.2f},{:.2f}) parent={}'
            .format(
                self.detid, self.energy, self.x, self.y, self.z, self.parent.trackid
                )
            )    
    
def build_tree(event, include_hits=True):
    # First create node objects for all tracks
    n_tracks = event[b'simtrack_x'].counts.sum()
    # Dict and list to be deleted at the end of this function
    id_to_track = {} 
    tracks = []
    for i_track in range(n_tracks):
        track = Track(
            crossedBoundary    = bool(event[b'simtrack_crossedBoundary'][0][i_track]),
            idAtBoundary       = int(event[b'simtrack_idAtBoundary'][0][i_track]),
            momentum           = event[b'simtrack_momentum'][0][i_track],
            energy             = event[b'simtrack_momentum'][0][i_track].E,
            momentumAtBoundary = event[b'simtrack_momentumAtBoundary'][0][i_track],
            energyAtBoundary   = event[b'simtrack_momentumAtBoundary'][0][i_track].E,
            noParent           = bool(event[b'simtrack_noParent'][0][i_track]),
            parentTrackId      = int(event[b'simtrack_parentTrackId'][0][i_track]),
            pdgid              = int(event[b'simtrack_pdgid'][0][i_track]),
            trackid            = int(event[b'simtrack_trackId'][0][i_track]),
            vertexIndex        = int(event[b'simtrack_vertexIndex'][0][i_track]),
            vertex_x           = float(event[b'simtrack_vertex_x'][0][i_track]),
            vertex_y           = float(event[b'simtrack_vertex_y'][0][i_track]),
            vertex_z           = float(event[b'simtrack_vertex_z'][0][i_track]),
            x                  = float(event[b'simtrack_x'][0][i_track]),
            xAtBoundary        = float(event[b'simtrack_xAtBoundary'][0][i_track]),
            y                  = float(event[b'simtrack_y'][0][i_track]),
            yAtBoundary        = float(event[b'simtrack_yAtBoundary'][0][i_track]),
            z                  = float(event[b'simtrack_z'][0][i_track]),
            zAtBoundary        = float(event[b'simtrack_zAtBoundary'][0][i_track]),
            )
        id_to_track[track.trackid] = track
        tracks.append(track)
    # Add hit information
    trackids_with_hits, counts = np.unique(event[b'simhit_fineTrackId'][0].flatten(), return_counts=True)
    for track in tracks:
        if track.trackid in trackids_with_hits:
            track.nhits = counts[trackids_with_hits == track.trackid][0]
        # Also store all actual hits
        if include_hits:
            select_hits = event[b'simhit_fineTrackId'][0] == track.trackid
            xhits = event[b'simhit_x'][0][select_hits]
            yhits = event[b'simhit_y'][0][select_hits]
            zhits = event[b'simhit_z'][0][select_hits]
            energyhits = event[b'simhit_energy'][0][select_hits]
            detidhits = event[b'simhit_detid'][0][select_hits]
            nhits = select_hits.sum()
            for ihit in range(nhits):
                hit = Hit(detidhits[ihit], xhits[ihit], yhits[ihit], zhits[ihit], energyhits[ihit], parent=track)
                track.hits.append(hit)        
    # Set parents and children
    for i_track, track in enumerate(tracks):
        parent_track_id = event[b'simtrack_parentTrackId'][0][i_track]
        parent = id_to_track.get(parent_track_id, None)
        track.parent = parent
        if parent: parent.children.append(track)
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
        if not track.keep: remove(track)
    # Find roots
    roots = []
    for track in tracks:
        if track.parent is None:
            roots.append(track)
            ntuputils.logger.info('Adding %s as a root', track)
    return roots

def trim_tree(root, inplace=False):
    if not inplace: root = copy_tree(root) # Keep original tree intact
    for node in list(traverse(root)): # No generator, since children are modified in loop
        # If node is not a root, has children, but has no hits, trim it
        if not(node.parent is None) and node.children and node.nhits == 0:
            node.parent.children.remove(node)
            node.parent.children.extend(node.children)
            for child in node.children:
                child.parent = node.parent
    return root

def make_graph(node):
    import networkx as nx
    G = nx.Graph()

    nodes_ids = []
    nodes_infos = []
    edges = []
    labels = {}
    
    for track in traverse(node):
        nodes_ids.append(track.trackid)
        nodes_infos.append(
            {'pdgid': track.pdgid, 'energy' : track.energy}
            )
        for child in track.children:
            edges.append([track.trackid, child.trackid])
            
        label = '{} E={:.1f}'.format(track.trackid, track.energy)
        if track.nhits > 0: label += ' nhits={}'.format(track.nhits)
        if track.crossedBoundary: label += ' X'
        labels[track.trackid] = label

    G.add_nodes_from(
        np.stack((
            np.array(nodes_ids),
            np.array(nodes_infos, dtype=object)
            )).T
        )    
    G.add_edges_from(np.array(edges))
    G.mylabels = labels
    return G


# ____________________________________________________
# Plotting

def plot_graph(node, *args, **kwargs):
    G = make_graph(node)
    labels = G.mylabels if kwargs.pop('labels', True) else None
    ntuputils.plot_graph(G, *args, labels=labels, **kwargs)


def plot_node(node, ax=None, labels=True, plot_hits=True):
    """
    Plots tracks and hits. The negative endcap is flipped.
    """
    if not ax:
        fig = plt.figure(figsize=(24,24))
        ax = fig.add_subplot(111 , projection='3d')

    for track in node.traverse():
        color = ntuputils.color_pdgid(track.pdgid)            
        x_in, y_in, z_in = track.vertex_x, track.vertex_y, track.vertex_z
        x_out, y_out, z_out = track.x, track.y, track.z

        if track.crossedBoundary:
            x_bound, y_bound, z_bound = track.xAtBoundary, track.yAtBoundary, track.zAtBoundary
            x,y,z = [ x_in, x_bound, x_out ], [ y_in, y_bound, y_out ], [ z_in, z_bound, z_out ]
            ax.scatter(z_bound, x_bound, y_bound, c=color, marker='x', s=35.)
        else:
            x,y,z = [ x_in, x_out ], [ y_in, y_out ], [ z_in, z_out ]

        if plot_hits and track.nhits > 0:
            positions = np.array([ np.array((hit.x, hit.y, hit.z)) for hit in track.hits ])
            sizes = 10000. * np.array([ hit.energy for hit in track.hits ])
            ax.scatter(positions[:,2], positions[:,0], positions[:,1], s=sizes, c=color)
            
        ax.plot(
            z, x, y, c=color,
            label = '{} ({}, E={:.2f})'.format(track.trackid, track.pdgid, track.energy),
            linewidth = 0.5
            )

        if labels:
            ax.text(
                z[-2], x[-2], y[-2],
                r'$\mathbf{{{}}}_{{{},\,E={:.1f}}}$'.format(track.trackid, track.pdgid, track.energy),
                color=color,
                fontsize=14,
                horizontalalignment='left' if z[-1] < 0. else 'right'
                )

    pos_endcap = node.z > 0.        
    if pos_endcap:
        zmin = 0.
        zmax = ntuputils.HGCAL_ZMAX_POS
    else:
        zmin = ntuputils.HGCAL_ZMIN_NEG
        zmax = 0.

    max_xy_dim = 50.
    ax.set_xlim(zmin, zmax)
    ax.set_xlabel('z')
    ax.set_ylabel('x')
    ax.set_zlabel('y')
    ax.set_ylim(-max_xy_dim, max_xy_dim)
    ax.set_zlim(-max_xy_dim, max_xy_dim)
    return ax


def plot_node_rotated(
        node,
        scale_large_norm=True,
        ax=None,
        labels=True,
        plot_hits=True,
        plot_clusters=False,
        color_by_pdgid=True,
        zmin=None,
        zmax=None
        ):
    """
    Puts the given track on the z-axis (the part from vertex to boundary
    crossing, or position if the track does not cross a boundary).
    This makes it possible to zoom in much more on the specific track.
    The plotted x, y and z axes are in a rotated coordinate system.
    """
    if not ax:
        fig = plt.figure(figsize=(24,24))
        ax = fig.add_subplot(111 , projection='3d')
    
    flipz = -1. if node.z < 0. else 1.
    
    origin = np.array([node.vertex_x, node.vertex_y, flipz*node.vertex_z])
    rotate_position = (
        np.array(
            [node.xAtBoundary, node.yAtBoundary, flipz*node.zAtBoundary]
            if node.crossedBoundary else [node.x, node.y, flipz*node.z]
            )
        - origin
        )
    rotate_position_norm = np.linalg.norm(rotate_position)
    ntuputils.logger.debug('origin = {}, rotate_position={}, rotate_position_norm={}'.format(origin, rotate_position, rotate_position_norm))
    
    from numpy import sin, cos, arctan2, arcsin
    dx = arctan2(rotate_position[1],rotate_position[2])
    dy = -arcsin(rotate_position[0] / np.linalg.norm(rotate_position))
    ntuputils.logger.debug('Rotation angle x-axis: %s', dx)
    ntuputils.logger.debug('Rotation angle y-axis: %s', dy)
    Rx = np.array([
        [1., 0., 0.],
        [0., cos(dx), -sin(dx)],
        [0., sin(dx), cos(dx)],
        ])
    Ry = np.array([
        [cos(dy), 0., sin(dy)],
        [0., 1., 0.],
        [-sin(dy), 0., cos(dy)],
        ])
    R = Rx.dot(Ry)
    ntuputils.logger.debug('Rotation matrix:\n%s', R)
    
    rotate = lambda v: v.dot(Rx.T).dot(Ry.T)
    
    max_perpendicular_dim = 0.
    max_longitudinal_dim = 0.
    
    for i_track, track in enumerate(traverse(node)):
        c = ntuputils.color_pdgid(track.pdgid) if color_by_pdgid else ntuputils.color_for_id(track.trackid)

        vertex = np.array([track.vertex_x, track.vertex_y, flipz*track.vertex_z]) - origin
        vertex_norm = np.linalg.norm(vertex)
        vertex /= vertex_norm if vertex_norm > 0. else 1.
        
        position = np.array([track.x, track.y, flipz*track.z]) - origin
        position_norm = np.linalg.norm(position)
        position /= position_norm if position_norm > 0. else 1.

        if plot_clusters:
            positions = np.array([ np.array((hit.x, hit.y, flipz*hit.z)) - origin for hit in track.hits() ])
            if positions.shape[0] > 0: 
                positions = rotate(positions)
                sizes = 10000. * np.array([ hit.energy for hit in track.hits() ])
                ax.scatter(positions[:,2], positions[:,0], positions[:,1], s=sizes, c=c)
        elif plot_hits and track.nhits > 0:
            positions = np.array([ np.array((hit.x, hit.y, flipz*hit.z)) - origin for hit in track.hits ])
            positions = rotate(positions)
            sizes = 10000. * np.array([ hit.energy for hit in track.hits ])
            ax.scatter(positions[:,2], positions[:,0], positions[:,1], s=sizes, c=c)
        
        if track.crossedBoundary:
            atbound = np.array([track.xAtBoundary, track.yAtBoundary, flipz*track.zAtBoundary]) - origin
            atbound = rotate(atbound)
            ax.scatter(atbound[2], atbound[0], atbound[1], c=c, marker='x', s=35.)

        ntuputils.logger.debug('\ntrack %s %s', i_track, track.trackid)
        ntuputils.logger.debug('  Unrotated: %s %s', vertex, position)
        vertex = rotate(vertex)    
        position = rotate(position)
        ntuputils.logger.debug('  Rotated:   %s %s', vertex, position)
            
        vertex *= vertex_norm
        position *= position_norm
        
        # Painful: Matplotlib rolls over the line to the other side if the value is too large
        # No idea why, can't find the issue online, but it's clearly a bug.
        # Scale down the track to prevent this
        
        if scale_large_norm and position_norm > 1000.:
            ntuputils.logger.debug('  Large norm {}: must scale down'.format(position_norm))
            # Get displacement vector from vertex to position
            d = position - vertex
            # Scale it down
            d = d / np.linalg.norm(d) * 500.
            position = vertex + d
        ntuputils.logger.debug('  Final positions: %s %s', vertex, position)
    
        if track.crossedBoundary:
            x,y,z = [vertex[0], atbound[0], position[0]], [vertex[1], atbound[1], position[1]], [vertex[2], atbound[2], position[2]]
        else:
            x,y,z = [vertex[0], position[0]], [vertex[1], position[1]], [vertex[2], position[2]]

        ax.plot(z, x, y, linewidth=1, c=c)

        if labels:
            ax.text(
                z[-1], x[-1], y[-1],
                r'$\mathbf{{{}}}_{{{},\,E={:.1f}}}$'.format(track.trackid, track.pdgid, track.energy),
                color=c,
                fontsize=14,
                horizontalalignment='left' if z[-1] < 0. else 'right'
                )

        # Save the maximum found perpendicular and longitudinal dimensions for plotting
        this_max_perpendicular_dim = np.max(np.abs((vertex[:2], position[:2])))
        max_perpendicular_dim = max(this_max_perpendicular_dim, max_perpendicular_dim)
        this_max_longitudinal_dim = np.abs(position[2])
        max_longitudinal_dim = max(this_max_longitudinal_dim, max_longitudinal_dim)
        
        
    max_perpendicular_dim *= 0.7
    max_perpendicular_dim = max(10., max_perpendicular_dim) # Ensure some minimum dimension
    
    if zmin is None: zmin = 0.
    if zmax is None: zmax = min(ntuputils.HGCAL_ZMAX_POS, max_longitudinal_dim)

    ax.set_xlim(zmin, zmax)
    ax.set_xlabel('z')
    ax.set_ylabel('x')
    ax.set_zlabel('y')
    ax.set_ylim(-max_perpendicular_dim, max_perpendicular_dim)
    ax.set_zlim(-max_perpendicular_dim, max_perpendicular_dim)

    ax.text2D(
        0.1, 0.85,
        'Track {} E={:.2f} n_tracks={}'
        .format(
            node.trackid, node.energy, len(list(traverse(node)))
            ),
        color=ntuputils.color_pdgid(node.pdgid) if color_by_pdgid else ntuputils.color_for_id(node.trackid),
        fontsize=14,
        horizontalalignment='left',
        verticalalignment='top',
        transform=ax.transAxes
        )


# ____________________________________________________
# Clustering algorithms


class Cluster(object):
    """
    Much like a track object but has other Clusters as children.
    The same tree algo's should work on this object.
    Attribute calls will be tried on the track object too, so you can 
    do Cluster.trackid or Cluster.pt.
    
    To get all hits in a cluster, use Cluster.hits(), which will loop
    over all tracks in the cluster and yield the hits accordingly
    """
    
    def __init__(self, track, parent=None):
        self.track = track
        self.parent = parent
        self.tracks = []
        self.children = []
        
    def __getattr__(self, key):
        try:
            super().__getattr__(key)
        except AttributeError:
            return getattr(self.track, key)
        
    def hits(self):
        for track in self.tracks:
            for hit in track.hits:
                yield hit

def print_clustering(node):
    """
    Prints a shortened representatino of tracks along with a clear mark where a
    new cluster begins
    """
    def shortrepr(track):
        if track.parent:
            efrac = track.energy / track.parent.energy
            dr = deltar_tracks(track.parent, track)
            info = 'dr={:.2f} efrac={:.2f} e={:.2f}'.format(dr, efrac, track.energy)
        else:
            info = 'e={:.2f}'.format(track.energy)
        return (
            '{} {}'.format(track.trackid, info)
            + (' CLUSTER' if track.merge is False else '')
            )
    for track, depth in traverse(node, yield_depth=True):
        print('__'*depth + shortrepr(track))
                
def traverse_unmergeable_daughters(node, parent=None):
    """
    Traverses only unmergeable nodes. `parent` refers to the first unmergeable parent (not the actual parent)
    """
    if node.merge is False:
        # If the node is not merged, it is a cluster
        yield node, parent
        parent = node
    for child in node.children:
        yield from traverse_unmergeable_daughters(child, parent=parent)
        
def traverse_mergeable_daughters(node, except_first=True):
    """
    Traverses only mergeable daughters of a node; daughter nodes that are not mergeable are skipped.
    The node itself should always be yielded, hence the except_first argument
    """
    if except_first or node.merge is True:
        yield node
        for child in node.children:
            yield from traverse_mergeable_daughters(child, except_first=False)

def clusterize(root):
    """
    Builds a tree of Cluster objects, and returns root clusters
    """
    root_cluster = None
    # Temporary map of nodes to clusters
    node_to_cluster = {}
    for node, parent in traverse_unmergeable_daughters(root):
        cluster = Cluster(node, parent=parent)
        node_to_cluster[node] = cluster
        if parent is None:
            root_cluster = cluster
        else:
            node_to_cluster[parent].children.append(cluster)
        cluster.tracks = list(traverse_mergeable_daughters(node))
    return root_cluster

def clusterize_decorator(merging_algo):
    def wrapper(*args, **kwargs):
        merging_algo(*args, **kwargs) # Apply the merging algorith
        node = args[0] # Assume the node is the first argument
        return clusterize(node) # Return the root cluster
    return wrapper
