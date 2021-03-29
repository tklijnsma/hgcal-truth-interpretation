from math import pi
import matplotlib.pyplot as plt
import numpy as np, logging, os.path as osp, os
import hgcalntuptool
logger = hgcalntuptool.logger
import numba

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

def retaphi(x, y, z):
    '''
    Transforms cartesian coordinates into eta-phi and the radial distance
    '''
    norm = np.linalg.norm
    p = np.array([x, y, z])
    pz = np.array([0, 0, z])
    theta = np.arccos(p.dot(pz) / (norm(p)*norm(pz)))
    eta = -np.log(np.tan(theta/2.))
    phi = np.arctan2(x, y)
    R = norm(p)
    return R, eta, phi

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
    '''
    Copies a whole tree (except hit information)
    '''
    import copy
    return copy.deepcopy(root)

def flipz_tree(root):
    '''
    Flips all z-axis properties of a track
    '''
    import copy
    root = copy_tree(root)
    for node in root.traverse():
        node.z *= -1.
        node.zAtBoundary *= -1.
        node.vertex_z *= -1.
        # hits are not copied with deep copy, so have to do it manually
        node.hits = copy.deepcopy(node.hits)
        for hit in node.hits:
            hit.z *= -1.
    return root


def needs_hit_displacement_quantities(method):
    def wrapper(*args, **kwargs):
        self = args[0]
        if not hasattr(self, '_'+method.__name__):
            logger.debug('Updating self.update_hit_displacement_quantities()')
            self.update_hit_displacement_quantities()
        return method(*args, **kwargs)
    return wrapper


class Track(object):

    root_track = {
        'crossedBoundary' : 0,
        'energy' : 0,
        'energyAtBoundary' : 0,
        'noParent' : 0,
        'parentTrackId' : 0,
        'pdgid' : 0,
        'trackid' : 0,
        'x' : 0,
        'y' : 0,
        'z' : 0,
        'vertex_x' : 0,
        'vertex_y' : 0,
        'vertex_z' : 0,
        'xAtBoundary' : 0,
        'yAtBoundary' : 0,
        'zAtBoundary' : 0,
        'parent' : None
        }

    def __init__(self, parent=None, children=None, **kwargs):
        self.parent = parent
        self.children = [] if children is None else children
        if kwargs.get('root', False): self.__dict__.update(self.root_track)
        self.__dict__.update(kwargs)
        self.hits = []
        self.merged_tracks = [self]
        
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

    def maxdepth(self):
        maxdepth = 0
        for node, depth in self.traverse(yield_depth=True):
            maxdepth = max(depth, maxdepth)
        return maxdepth

    @property
    def is_root(self):
        return self.parent is None

    @property
    def nhits(self):
        return len(self.hits)

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
            '{} E={} ({},{},{}) pdgid={} nhits={}'
            .format(
                self.get('trackid'), self.get('energy'),
                self.get('vertex_x',3), self.get('vertex_y',3), self.get('vertex_z',3),
                self.get('pdgid'),
                len(self.hits)
                )
            )

    def update_hit_dependent_quantities(self, displacement_quantities=True):
        self._centroid, self._secondmoment = hitcentroid(self)
        # Shower axis info (cheap to recompute)
        self._b = np.array([self.xAtBoundary, self.yAtBoundary, self.zAtBoundary])
        self._e = self.centroid
        self._axis = (self.e-self.b) / np.linalg.norm(self.e-self.b)
        if displacement_quantities:
            self.update_hit_displacement_quantities()

    @property
    def centroid(self):
        if not hasattr(self, '_centroid'): self.update_hit_dependent_quantities()
        return self._centroid

    @property
    def secondmoment(self):
        if not hasattr(self, '_secondmoment'): self.update_hit_dependent_quantities()
        return self._secondmoment

    @property
    def b(self):
        if not hasattr(self, '_b'): self.update_hit_dependent_quantities()
        return self._b

    @property
    def e(self):
        if not hasattr(self, '_e'): self.update_hit_dependent_quantities()
        return self._e

    @property
    def axis(self):
        if not hasattr(self, '_axis'): self.update_hit_dependent_quantities()
        return self._axis


    def update_hit_displacement_quantities(self):
        # Compute the displacements
        # v_to_axis = []
        d_to_axis = []
        d_along_axis = []
        energies = []
        for i, hit in enumerate(self.hits):
            hitpos = np.array([hit.x, hit.y, hit.z]) - self.b  # Shift to the begin_point
            proj_along_axis = hitpos.dot(self.axis)*self.axis
            v = hitpos - proj_along_axis  # Subtract the projection on the axis (yielding the perpendicular component)
            # v_to_axis.append(v)
            d_to_axis.append(np.linalg.norm(v))
            d_along_axis.append(np.linalg.norm(proj_along_axis))
            energies.append(hit.energy)

        # Make np arrays
        energies = np.array(energies)
        total_energy = np.sum(energies)
        d_to_axis = np.array(d_to_axis)
        d_along_axis = np.array(d_along_axis)
        # v_to_axis = np.array(v_to_axis)

        self._ds_to_axis = d_to_axis

        order = np.argsort(d_to_axis)
        self._sorted_d_to_axis = d_to_axis[order]
        self._energy_fractions_to_axis = np.cumsum(energies[order]) / total_energy

        order = np.argsort(d_along_axis)
        self._sorted_d_along_axis = d_along_axis[order]
        self._energy_fractions_along_axis = np.cumsum(energies[order]) / total_energy

        self._long_q10 = self.longitudinal_energy_containment(.1)
        self._v_q10 = self.b +  self._long_q10 * self.axis
        self._long_q90 = self.longitudinal_energy_containment(.9)
        self._v_q90 = self.b +  self._long_q90 * self.axis

    @property
    @needs_hit_displacement_quantities
    def ds_to_axis(self):
        '''
        Returns the perpendicular displacement of each hit w.r.t. the shower axis
        (Somewhat expensive to recompute)
        '''
        return self._ds_to_axis

    @property
    @needs_hit_displacement_quantities
    def energy_fractions_along_axis(self):
        return self._energy_fractions_along_axis

    @property
    @needs_hit_displacement_quantities
    def energy_fractions_to_axis(self):
        return self._energy_fractions_to_axis

    @property
    @needs_hit_displacement_quantities
    def sorted_d_to_axis(self):
        return self._sorted_d_to_axis

    @property
    @needs_hit_displacement_quantities
    def sorted_d_along_axis(self):
        return self._sorted_d_along_axis
    
    @property    
    @needs_hit_displacement_quantities
    def long_q10(self):
        return self._long_q10

    @property    
    @needs_hit_displacement_quantities
    def v_q10(self):
        return self._v_q10

    @property    
    @needs_hit_displacement_quantities
    def long_q90(self):
        return self._long_q90

    @property    
    @needs_hit_displacement_quantities
    def v_q90(self):
        return self._v_q90

    def moliere_radius(self, r):
        return self.sorted_d_to_axis[np.argmax(self.energy_fractions_to_axis > r)]

    def longitudinal_energy_containment(self, quantile):
        return self.sorted_d_along_axis[np.argmax(self.energy_fractions_along_axis > quantile)]

    def average_boundary_pos(self):
        x = 0.
        y = 0.
        z = 0.
        total_energy = 0.
        for track in self.merged_tracks:
            x += track.energyAtBoundary * track.xAtBoundary
            y += track.energyAtBoundary * track.yAtBoundary
            z += track.energyAtBoundary * track.zAtBoundary
            total_energy += track.energyAtBoundary
        x /= total_energy
        y /= total_energy
        z /= total_energy
        return x, y, z, total_energy

    def nphits(self, include_energy=True):
        tolist = lambda h: [h.x, h.y, h.z, h.energy] if include_energy else [h.x, h.y, h.z]
        mat = [ tolist(h) for h in self.hits ]
        return np.array(mat)


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
            logger.info('Adding %s as a root', track)
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

def plot_graph(node, with_labels=True, labels=None, prog='twopi', ax=None):
    import networkx as nx
    import pygraphviz
    from networkx.drawing.nx_agraph import graphviz_layout
    G = make_graph(node)
    labels = G.mylabels if labels else None
    pos = graphviz_layout(G, prog=prog, args='')
    fig = plt.figure(figsize=(8, 8))
    pdgids = np.array([d.get('pdgid', 0) for n, d in G.nodes(data=True)])
    node_color = color_pdgid(pdgids)
    energies = np.array([d.get('energy', 10.) for n, d in G.nodes(data=True)])
    normed_energies = 100. * np.log(energies + 1.) / np.max(np.log(energies + 1.))
    nx.draw(
        G, pos,
        node_color = node_color,
        node_size = normed_energies,
        alpha = 0.5,
        with_labels = with_labels,
        labels = labels,
        **({} if ax is None else {'ax':ax})
        )
    plt.axis('equal')


def plot_node(node, ax=None, labels=True, plot_hits=True, color_by_pdgid=True, scale_hitsize=True):
    """
    Plots tracks and hits. The negative endcap is flipped.
    """
    fresh_ax = False
    if not ax:
        fresh_ax = True
        fig = plt.figure(figsize=(24,24))
        ax = fig.add_subplot(111 , projection='3d')

    for track in node.traverse():
        color = hgcalntuptool.color_pdgid(track.pdgid) if color_by_pdgid else hgcalntuptool.color_for_id(track.trackid)
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
            size_option = { 's' : 10000. * np.array([ hit.energy for hit in track.hits ]) } if scale_hitsize else {}
            ax.scatter(positions[:,2], positions[:,0], positions[:,1], c=color, **size_option)

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

    if fresh_ax:
        pos_endcap = node.z > 0.        
        if pos_endcap:
            zmin = 0.
            zmax = hgcalntuptool.HGCAL_ZMAX_POS
        else:
            zmin = hgcalntuptool.HGCAL_ZMIN_NEG
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
        color_by_pdgid=True,
        zmin=None,
        zmax=None,
        ref_node=None,
        scale_hitsize=True,
        plot_shower_axis=False
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
    
    if ref_node is None: ref_node = node
    flipz = -1. if ref_node.z < 0. else 1.
    origin = np.array([ref_node.vertex_x, ref_node.vertex_y, flipz*ref_node.vertex_z])
    rotate_position = (
        np.array(
            [ref_node.xAtBoundary, ref_node.yAtBoundary, flipz*ref_node.zAtBoundary]
            if ref_node.crossedBoundary else [ref_node.x, ref_node.y, flipz*ref_node.z]
            )
        - origin
        )
    rotate_position_norm = np.linalg.norm(rotate_position)
    logger.debug('origin = {}, rotate_position={}, rotate_position_norm={}'.format(origin, rotate_position, rotate_position_norm))
    
    from numpy import sin, cos, arctan2, arcsin
    dx = arctan2(rotate_position[1],rotate_position[2])
    dy = -arcsin(rotate_position[0] / np.linalg.norm(rotate_position))
    logger.debug('Rotation angle x-axis: %s', dx)
    logger.debug('Rotation angle y-axis: %s', dy)
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
    logger.debug('Rotation matrix:\n%s', R)
    
    rotate = lambda v: v.dot(Rx.T).dot(Ry.T)
    
    max_perpendicular_dim = 0.
    max_longitudinal_dim = 0.
    
    total_nhits = 0;
    for i_track, track in enumerate(traverse(node)):
        if plot_shower_axis and track.nhits == 0: continue
        total_nhits += track.nhits
        c = hgcalntuptool.color_pdgid(track.pdgid) if color_by_pdgid else hgcalntuptool.color_for_id(track.trackid)

        vertex = np.array([track.vertex_x, track.vertex_y, flipz*track.vertex_z]) - origin
        vertex_norm = np.linalg.norm(vertex)
        vertex /= vertex_norm if vertex_norm > 0. else 1.
        
        position = np.array([track.x, track.y, flipz*track.z]) - origin
        position_norm = np.linalg.norm(position)
        position /= position_norm if position_norm > 0. else 1.

        centroid = np.copy(track.centroid)
        centroid[2] *= flipz
        rcentroid = rotate(centroid-origin)

        if plot_hits and track.nhits > 0:
            positions = np.array([ np.array((hit.x, hit.y, flipz*hit.z)) - origin for hit in track.hits ])
            positions = rotate(positions)
            size_option = { 's' : 10000. * np.array([ hit.energy for hit in track.hits ]) } if scale_hitsize else {}
            ax.scatter(positions[:,2], positions[:,0], positions[:,1], c=c, **size_option)
            ax.scatter(rcentroid[2], rcentroid[0], rcentroid[1], c=c, s=100., marker='*')
        
        if track.crossedBoundary:
            atbound = np.array([track.xAtBoundary, track.yAtBoundary, flipz*track.zAtBoundary]) - origin
            atbound = rotate(atbound)
            if not plot_shower_axis: ax.scatter(atbound[2], atbound[0], atbound[1], c=c, marker='x', s=35.)

        logger.debug('\ntrack %s %s', i_track, track.trackid)
        logger.debug('  Unrotated: %s %s', vertex, position)
        vertex = rotate(vertex)    
        position = rotate(position)
        logger.debug('  Rotated:   %s %s', vertex, position)
            
        vertex *= vertex_norm
        position *= position_norm
        
        # Painful: Matplotlib rolls over the line to the other side if the value is too large
        # No idea why, can't find the issue online, but it's clearly a bug.
        # Scale down the track to prevent this
        
        if scale_large_norm and position_norm > 1000.:
            logger.debug('  Large norm {}: must scale down'.format(position_norm))
            # Get displacement vector from vertex to position
            d = position - vertex
            # Scale it down
            d = d / np.linalg.norm(d) * 500.
            position = vertex + d
        logger.debug('  Final positions: %s %s', vertex, position)

        if plot_shower_axis:
            x,y,z = [atbound[0], rcentroid[0]], [atbound[1], rcentroid[1]], [atbound[2], rcentroid[2]]
        elif track.crossedBoundary:
            x,y,z = [vertex[0], atbound[0], position[0]], [vertex[1], atbound[1], position[1]], [vertex[2], atbound[2], position[2]]
        else:
            x,y,z = [vertex[0], position[0]], [vertex[1], position[1]], [vertex[2], position[2]]
        ax.plot(z, x, y, linewidth=1, c=c)

        if labels:
            ax.text(
                z[-1], x[-1], y[-1],
                r'$\mathbf{{{}}}_{{{},\,n={}}}$'.format(track.trackid, track.pdgid, track.nhits),
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
    if zmax is None: zmax = min(hgcalntuptool.HGCAL_ZMAX_POS, max_longitudinal_dim)

    ax.set_xlim(zmin, zmax)
    ax.set_xlabel('z')
    ax.set_ylabel('x')
    ax.set_zlabel('y')
    ax.set_ylim(-max_perpendicular_dim, max_perpendicular_dim)
    ax.set_zlim(-max_perpendicular_dim, max_perpendicular_dim)

    ax.text2D(
        0.1, 0.85,
        'Track {} E={:.2f} n_tracks={}, n_hits={}'
        .format(
            node.trackid, node.energy, len(list(traverse(node))), total_nhits
            ),
        color=hgcalntuptool.color_pdgid(node.pdgid) if color_by_pdgid else hgcalntuptool.color_for_id(node.trackid),
        fontsize=14,
        horizontalalignment='left',
        verticalalignment='top',
        transform=ax.transAxes
        )
    return ax


# ____________________________________________________
# Merging algos december

from itertools import combinations

def trim_trivial_tracks(root, inplace=False):
    """
    Skip single-child + no-hit tracks. Returns a copy of the tree (unless inplace=True)
    """
    if not inplace: root = copy_tree(root) # Keep original tree intact
    for node in list(traverse(root)): # No generator, since children are modified in loop
        # If node is not a root, has 1 child, but has no hits, trim it
        if not(node.parent is None) and len(node.children)==1 and node.nhits==0:
            node.parent.children.remove(node)
            node.parent.children.extend(node.children)
            for child in node.children:
                child.parent = node.parent
    return root

def dist(track1, track2):
    p1, v1 = track1.centroid, track1.secondmoment
    p2, v2 = track2.centroid, track2.secondmoment
    d = np.linalg.norm(p2-p1)
    return d
    # denominator = v1 + v2
    # if v1 == 0. and v2 == 0.: denominator = 1.
    # r = d / ((track1.nhits + track2.nhits)**0.5)
    # return r

def _print_dist_shortrepr(track):
    """Short representation of a track for print_dist function"""
    info = 'e={:.2f} nhits={}'.format(track.energy, track.nhits)
    return '{} {}'.format(track.trackid, info)
    
def print_dist(node, print_fn=print):
    """
    Prints a tree with distance measures between the track and its children
    """
    for track, depth in traverse(node, yield_depth=True):
        print_fn('__'*depth + _print_dist_shortrepr(track))
        if track.children:
            children_with_hits = list(filter(lambda c: c.nhits > 0, track.children))
            if track.nhits > 0: children_with_hits.append(track)
            sibling_relations = combinations(children_with_hits, 2)
            for c1, c2 in sibling_relations:
                r = dist(c1, c2)
                print_fn(
                    '__'*(depth+1) +
                    'dist between {:6} and {:6}: {:.3f}'.format(c1.trackid, c2.trackid, r)
                    )


class CachedDistFn():
    '''
    Some distance fn's are expensive - use a cache to avoid unnecessary recomputation
    '''
    def __init__(self, fn):
        self.fn = fn
        self.cache = {}

    def __call__(self, t1, t2, *args, **kwargs):
        if (t1, t2) not in self.cache:
            self.cache[(t1, t2)] = self.fn(t1, t2, *args, **kwargs)
        return self.cache[(t1, t2)]

    def remove(self, t):
        for t1, t2 in self.cache:
            if t1 == t or t2 == t:
                del self.cache[(t1, t2)]

    def remove2(self, ta, tb):
        for t1, t2 in list(self.cache.keys()):
            if ta==t1 or ta==t2 or tb==t1 or tb==t2:
                del self.cache[(t1, t2)]


def make_rotation(axis, include_inverse=False):
    '''
    Takes a 3D axis, and builds a rotation matrix such that
    R.dot(v) will rotate v to a coordinate system where the z-axis
    is aligned with the z-axis of `axis`
    '''
    # Build rotation matrix
    from numpy import sin, cos, arctan2, arcsin
    dx = arctan2(axis[1],axis[2])
    dy = -arcsin(axis[0] / np.linalg.norm(axis))
    # logger.debug('Rotation angle x-axis: %s', dx)
    # logger.debug('Rotation angle y-axis: %s', dy)
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
    # logger.debug('Rotation matrix:\n%s', R)
    rotate = lambda v: v.dot(Rx.T).dot(Ry.T)
    if include_inverse:
        inv_rotate = lambda v: v.dot(Ry).dot(Rx)
        return rotate, inv_rotate
    return rotate

def get_circle(r=1., N=30):
    '''returns N x 3 matrix that represents a circle in the xy plane (dims 0 and 1)'''
    angles = np.linspace(-pi, pi, 30)
    circle = np.stack((np.cos(angles), np.sin(angles), np.zeros_like(angles))).T * r
    return circle

from numba import float64, boolean

@numba.njit
def is_inside_numba(px, py, polyx, polyy):
    '''
    Winding number calculation to determine if a point is inside a 2D polygon
    (https://en.wikipedia.org/wiki/Nonzero-rule)
    
    point: (x, y)
    polygon: [ (x1, y1), ..., (xn, yn) ]
    '''
    wn = 0

    # Structured like x1, y1, x2, y2
    x1 = polyx[:-1]
    y1 = polyy[:-1]
    x2 = polyx[1:]
    y2 = polyy[1:]

    # print(x1, y1, x2, y2)

    # Throw away edges that are completely to the left of px
    select_valid = ~( (x1 < px) & (x2 < px) )
    x1 = x1[select_valid]
    y1 = y1[select_valid]
    x2 = x2[select_valid]
    y2 = y2[select_valid]

    # Select edges that cross the 'y-axis' (at height py)
    select_cross_y_axis = (((y1 <= py) & (y2 >= py)) | ((y1 >= py) & (y2 <= py)))
    x1 = x1[select_cross_y_axis]
    y1 = y1[select_cross_y_axis]
    x2 = x2[select_cross_y_axis]
    y2 = y2[select_cross_y_axis]

    select_vertical_edges = y1 == y2

    # Vertical edges are easy - sum up number of edges right of p
    wn += np.sum( x1[select_vertical_edges] > px )

    # Diagonal edges a little harder, have to calculate intersection
    x1 = x1[~select_vertical_edges]
    y1 = y1[~select_vertical_edges]
    x2 = x2[~select_vertical_edges]
    y2 = y2[~select_vertical_edges]
    a = (y2-y1)/(x2-x1)
    x_intersect = (py - y1)/a + x1
    wn += np.sum(x_intersect > px)

    return bool(wn % 2) # 0 if even (out), and 1 if odd (in)


def is_inside(point, polygon):
    '''
    Winding number calculation to determine if a point is inside a 2D polygon
    (https://en.wikipedia.org/wiki/Nonzero-rule)
    
    point: (x, y)
    polygon: [ (x1, y1), ..., (xn, yn) ]
    '''
    wn = 0
    px, py = point
    for (x1, y1), (x2, y2) in zip(polygon[:-1], polygon[1:]):
        if x1 < px and x2 < px: continue # Will never be a valid crossing        
        if y1 <= py <= y2 or y2 <= py <= y1: # either upward or downward crossing
            # Compute intersection x coordinate
            if x1 == x2:
                # Vertical lines need no interpolation
                x_intersect = x1
            else:
                a = (y2-y1) / (x2-x1)
                x_intersect = (py - y1)/a + x1
            if px < x_intersect: # Valid intersection (i.e. right of px)
                wn += 1
    return bool(wn % 2) # 0 if even (out), and 1 if odd (in)
    
def polygon_overlap(p1, p2, nbins=30, draw=False):
    xmin = min(np.min(p1[:,0]), np.min(p2[:,0]))
    xmax = max(np.max(p1[:,0]), np.max(p2[:,0]))
    ymin = min(np.min(p1[:,1]), np.min(p2[:,1]))
    ymax = max(np.max(p1[:,1]), np.max(p2[:,1]))
    
    x_binning = np.linspace(xmin, xmax, nbins+1)
    y_binning = np.linspace(ymin, ymax, nbins+1)
    x_centers = .5*(x_binning[:-1] + x_binning[1:])
    y_centers = .5*(y_binning[:-1] + y_binning[1:])
    
    n_inside_2 = 0
    n_inside_1_2 = 0    
    if draw: values = np.zeros((nbins, nbins))
    for ix in range(nbins):
        for iy in range(nbins):
            point = [ x_centers[ix], y_centers[iy] ]
            inside_1 = is_inside(point, p1)
            inside_2 = is_inside(point, p2)
            if inside_2:
                n_inside_2 += 1
                if inside_1:
                    n_inside_1_2 += 1
            if draw: values[ix,iy] = inside_1*2 + inside_2 # For plotting purposes
    frac_2_in_1 = n_inside_1_2 / n_inside_2 if n_inside_2 > 0 else 0.

    if draw:
        h = {
            'nbins' : nbins,
            'x_binning' : x_binning,
            'y_binning' : y_binning,
            'x_centers' : x_centers,
            'y_centers' : y_centers,
            'xmin' : xmin,
            'xmax' : xmax,
            'ymin' : ymin,
            'ymax' : ymax,
            'values' : values
            }
        return frac_2_in_1, h
    else:
        return frac_2_in_1

@numba.njit(parallel=True)
def polygon_overlap_numba(p1, p2, nbins=30):
    p1x = p1[:,0]
    p1y = p1[:,1]
    p2x = p2[:,0]
    p2y = p2[:,1]

    xmin = min(np.min(p1x), np.min(p2x))
    xmax = max(np.max(p1x), np.max(p2x))
    ymin = min(np.min(p1y), np.min(p2y))
    ymax = max(np.max(p1y), np.max(p2y))
    
    x_binning = np.linspace(xmin, xmax, nbins+1)
    y_binning = np.linspace(ymin, ymax, nbins+1)
    x_centers = .5*(x_binning[:-1] + x_binning[1:])
    y_centers = .5*(y_binning[:-1] + y_binning[1:])
    
    # Generate all points in the histogram
    x = x_centers.repeat(nbins)
    y = y_centers.repeat(nbins).reshape((-1, nbins)).T.flatten()

    n_inside_2 = 0
    n_inside_1_and_2 = 0
    for i in numba.prange(nbins**2):
        inside1 = is_inside_numba(x[i], y[i], p1x, p1y)
        inside2 = is_inside_numba(x[i], y[i], p2x, p2y)
        if inside2:
            n_inside_2 += 1
            if inside1:
                n_inside_1_and_2 += 1

    return float(n_inside_1_and_2) / n_inside_2 if n_inside_2 > 0. else 0.


def longitudinal_dist(t1, t2):
    '''
    First rotates the q10 and q90 positions.
    Then calculates delta z between two tracks:

    b1------e1
                    b2----------e2
              ----->


    b1------e1
         b2----------e2
         <----


    b1-----------------e1
         b2-----e2
         <---------------
    '''
    rotate = t1.rotate
    o = t1.b
    b1, e1 = t1.v_q10, t1.v_q90
    b2, e2 = t2.v_q10, t2.v_q90
    rb1 = rotate(b1-o)
    re1 = rotate(e1-o)
    rb2 = rotate(b2-o)
    re2 = rotate(e2-o)
    if rb2[2] < rb1[2]: rb1, re1, rb2, re2 = rb2, re2, rb1, re1
    # print(rb1[2], re1[2], rb2[2], re2[2])
    return rb2[2] - re1[2]

def overlap(t1, t2, draw=False, use_numba=True):
    if draw and use_numba:
        logger.warning('Turning off use_numba since draw is active')
        use_numba = False

    if t2.energyAtBoundary > t1.energyAtBoundary: t1, t2 = t2, t1

    for t in [t1, t2]:
        t.r = max(t.moliere_radius(.85), 1.5)
        t.rotate, t.inv_rotate = make_rotation(t.axis, include_inverse=True)
    
    # Rotate everything to a coordinate system where axis1-z is the z-axis    
    t1.raxis = t1.rotate(t1.axis) # Should be just (0, 0, 1), but compute it anyway
    t1.rb = np.zeros(3)
    t1.re = t1.rotate(t1.e-t1.b)

    t2.raxis = t1.rotate(t2.axis)
    t2.rb = t1.rotate(t2.b-t1.b)
    t2.re = t1.rotate(t2.e-t1.b)
    
    t1.rcircle = t1.inv_rotate(get_circle(t1.r)) + t1.re

    t2.rcircle = t2.inv_rotate(get_circle(t2.r)) + t2.e
    t2.rcircle = t1.rotate(t2.rcircle - t1.b)
    
    if draw:
        frac_2_in_1, h = polygon_overlap(t1.rcircle[:,:2], t2.rcircle[:,:2], draw=draw)

        # Unrotated plot
        o = t1.b
        fig = plt.figure(figsize=(24,36))
        ax1 = fig.add_subplot(321 , projection='3d')
        ax2 = fig.add_subplot(322 , projection='3d')
        ax2.view_init(30, 60)
        for ax in [ax1, ax2]:
            plot_node(t1, scale_hitsize=False, color_by_pdgid=False, ax=ax)
            plot_node(t2, scale_hitsize=False, color_by_pdgid=False, ax=ax)

            ax.set_xlim(o[2], o[2] + 90.)
            ax.set_ylim(o[0]-15., o[0]+15.)
            ax.set_zlim(o[1]-15., o[1]+15.)

            for t in [t1, t2]:
                c = hgcalntuptool.color_for_id(t.trackid)
                ax.plot([t.b[2], t.e[2]], [t.b[0], t.e[0]], [t.b[1], t.e[1]], c=c, linewidth=2., linestyle='--')
                # Also draw axis from .1 to .9 longitudinal energy containment
                b = t.b + t.longitudinal_energy_containment(.1) * t.axis
                e = t.b + t.longitudinal_energy_containment(.9) * t.axis
                ax.plot([b[2], e[2]], [b[0], e[0]], [b[1], e[1]], c=c, linewidth=5.)
                rcircle = (t.inv_rotate(get_circle(t.r)) + t.e).T
                ax.plot(rcircle[2], rcircle[0], rcircle[1], c=c)
                ax.scatter(t.centroid[2], t.centroid[0], t.centroid[1], marker='+', s=1000., c=c)

        # Rotated plot
        scale=False
        t1.rhits = t1.rotate(t1.nphits(False)-t1.b)
        t2.rhits = t1.rotate(t2.nphits(False)-t1.b)
        
        ax1 = fig.add_subplot(323 , projection='3d')
        ax2 = fig.add_subplot(324 , projection='3d')
        ax2.view_init(30, 60)
        for ax in [ax1, ax2]:
            for t, c in [ (t1, 'b'), (t2, 'r') ]:
                ax.plot([t.rb[2], t.re[2]], [t.rb[0], t.re[0]], [t.rb[1], t.re[1]], c=c)
                s = { 's' : 10000.*[h.energy for h in t.hits] } if scale else { 's' : 10. }
                ax.scatter(t.rhits[:,2], t.rhits[:,0], t.rhits[:,1], c=c, **s)
                ax.plot(t.rcircle[:,2], t.rcircle[:,0], t.rcircle[:,1], c=c)
            ax.set_xlim(0., 2.*max(np.linalg.norm(t2.e-t2.b), np.linalg.norm(t1.e-t1.b)) )
            dxy = 2.*np.linalg.norm( (t2.re-t1.re)[:2] )
            ax.set_ylim(-dxy,dxy)
            ax.set_zlim(-dxy,dxy)
    
        # Project on xy plane of rotated system
        ax = fig.add_subplot(325)
        for t, c in [ (t1, 'b'), (t2, 'r') ]:
            ax.scatter(t.rhits[:,0], t.rhits[:,1], c=c)
            ax.plot(t.rcircle[:,0], t.rcircle[:,1], c=c)
        dxy = np.linalg.norm((t2.re-t1.re)[:2]) + t1.r + t2.r
        ax.set_xlim(-dxy,dxy)
        ax.set_ylim(-dxy,dxy)
        
        # xy projection with the 2D histogram
        ax = fig.add_subplot(326)
        ax.plot(t1.rcircle[:,0], t1.rcircle[:,1], c='b')
        ax.plot(t2.rcircle[:,0], t2.rcircle[:,1], c='r')
        dxy = np.linalg.norm((t2.re-t1.re)[:2]) + t1.r + t2.r
        ax.set_xlim(h['xmin'],h['xmax'])
        ax.set_ylim(h['ymin'],h['ymax'])
        ax.hist2d(
            np.tile(h['x_centers'], (h['nbins'], 1)).T.flatten(),
            np.tile(h['y_centers'], (h['nbins'], 1)).flatten(),
            bins = [h['x_binning'], h['y_binning']],
            weights = h['values'].flatten()
            )

    elif use_numba:
        frac_2_in_1 = polygon_overlap_numba(t1.rcircle[:,:2], t2.rcircle[:,:2])
    else:
        frac_2_in_1 = polygon_overlap(t1.rcircle[:,:2], t2.rcircle[:,:2])

    longd = longitudinal_dist(t1, t2)
    return frac_2_in_1, longd


def perform_merging_for_node(
    node, use_overlap_algo=False,
    default_min_r=10., min_overlap = 0.5,
    overlap_fn=None
    ):
    """
    Looks at a track and its children, and decides which things to merge.
    `default_min_r` is the theshold up to which tracks will be merged, i.e.
    distances among tracks >default_min_r will not be merged.
    """
    logger.debug('Performing merging for leaf parent %s', node.trackid)
    # Check whether we're really in a leaf parent
    for c in node.children:
        if c.nhits == 0:
            raise Exception(
                'Merging of node {}: Child {} has zero hits!'
                .format(node.trackid, c.trackid)
                )
    # Remove all children from this node
    children = node.children
    node.children = []
    # Also allow node itself to be merged if it has hits and is not the parent
    if not(node.is_root) and node.nhits > 0: children.append(node)
    is_updated = False
    
    def merge(c1, c2, metric):
        if c2.energy > c1.energy: c1, c2 = c2, c1
        logger.debug(
            'Merging {} into {}, metric={}'
            .format(c2.trackid, c1.trackid, metric)
            )
        c1.hits.extend(c2.hits)
        c1.merged_tracks.extend(c2.merged_tracks)
        children.remove(c2)
        c1.update_hit_dependent_quantities()

    # Keep merging siblings as long as r < some_threshold, recalc r after every merge
    while True:
        current_max_overlap = (min_overlap, 0.)
        min_r = default_min_r
        to_merge = None
        for c1, c2 in combinations(children, 2):
            if use_overlap_algo:
                overlap, dz = overlap_fn(c1, c2)
                if overlap > current_max_overlap[0] and dz < 10.:
                    current_max_overlap = (overlap, dz)
                    to_merge = (c1, c2)
                    if overlap == 1.: break # It's not going to get larger anyway
            else:
                r = dist(c1, c2)
                if r < min_r:
                    min_r = r
                    to_merge = (c1, c2)
        if to_merge:
            is_updated = True
            merge(*to_merge, current_max_overlap if use_overlap_algo else min_r)
            if use_overlap_algo and hasattr(overlap_fn, 'remove2'): overlap_fn.remove2(*to_merge)
        else:
            break

    if node.is_root:
        # If the node was a root, the new merged children will just be set as an attribute
        node.children = children
        if is_updated:
            logger.debug('Updating root children to %s', [c.trackid for c in children])
        else:
            logger.debug('Node is root; no update found!')
        return is_updated
    else:
        # If the node is not a root, there must always be an update (even if it's only a flattening)
        logger.debug(
            'Merge of node {}: Adding to parent {} the following children: {}'
            .format(
                node.trackid, node.parent.trackid, [c.trackid for c in children]
                )
            )
        # If all children are merged into one cluster, but the parent didn't have hits,
        # use the parent pdgid instead
        if node.nhits == 0 and len(children) == 1 and children[0].pdgid != node.pdgid:
            logger.debug(
                'Overwriting track {} pdgid={} with pdgid={} since that is its parent'
                .format(children[0].trackid, children[0].pdgid, node.pdgid)
                )
            children[0].pdgid = node.pdgid
        node.parent.children.remove(node)
        node.parent.children.extend(children)
        return True

    
def traverse_only_leafparents(node):
    """
    Traverses only tracks whose children are exclusively leafs (i.e. 'leafparents').
    A leafparent (1) must have children; and (2) all its children must NOT have children
    """
    for child in node.children:
        yield from traverse_only_leafparents(child)
    if len(node.children) > 0:
        if all(len(child.children)==0 for child in node.children):
            yield node

def merging_algo(root, inplace=False, progress=True, **kwargs):
    """
    Merging algorithm entrypoint
    """
    if not inplace: root = copy_tree(root)
    root.parent = None # Make sure root has no parent (in case of dealing with a subtree)
    trim_trivial_tracks(root, inplace=True)
    i = 0
    maxdepth = root.maxdepth()
    if progress:
        import tqdm
        pbar = tqdm.tqdm(total=maxdepth, desc='merging iterations')
    while True:
        if progress: pbar.update()
        # logger.info('Merging iteration %s/%s', i, maxdepth)
        did_update = False
        for node in list(traverse_only_leafparents(root)):
            if perform_merging_for_node(node, **kwargs):
                did_update = True
        if not did_update:
            logger.debug('No update - breaking')
            break
        # if i > 3: root.print()
        # print_dist(root, logger.debug)
        i += 1
    if progress: pbar.close()
    return root
                 

def savefig(*args, **kwargs):
    '''
    Wrapper around plt.savefig that always adds `bbox_inches='tight'`,
    creates the output directory if it doesn't exist, and runs strftime
    formatting on the path
    '''
    from time import strftime
    args = list(args)
    args[0] = strftime(args[0])
    directory = osp.dirname(args[0])
    if not osp.isdir(directory):
        logger.info('Creating %s', directory)
        os.makedirs(directory)
    kwargs.setdefault('bbox_inches', 'tight')
    plt.savefig(*args, **kwargs)
