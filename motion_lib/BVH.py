##############################
#
# based on http://theorangeduck.com/page/deep-learning-framework-character-motion-synthesis-and-editing
#
##############################

import re
import numpy as np

try:
    from .Animation import Animation
    from . import AnimationStructure
    from .Quaternions import Quaternions
    from .root_collapse import collapse_root_skeleton
except:
    from Animation import Animation
    import AnimationStructure
    from Quaternions import Quaternions
    from root_collapse import collapse_root_skeleton

channelmap = {
    'Xrotation' : 'x',
    'Yrotation' : 'y',
    'Zrotation' : 'z'   
}

channelmap_inv = {
    'x': 'Xrotation',
    'y': 'Yrotation',
    'z': 'Zrotation',
}

ordermap = {
    'x' : 0,
    'y' : 1,
    'z' : 2,
}

def load(filename, start=None, end=None, order=None, world=True, collapse_root=True):
    """
    Reads a BVH file and constructs an animation
    
    Parameters
    ----------
    filename: str
        File to be opened
        
    start : int
        Optional Starting Frame
        
    end : int
        Optional Ending Frame
    
    order : str
        Optional Specifier for joint order.
        Given as string E.G 'xyz', 'zxy'
        
    world : bool
        If set to true euler angles are applied
        together in world space rather than local
        space

    collapse_root : bool, default True
        When False, skips redundant root joint removal and wrapper root
        collapsing. Mirrors ``motion_lib.FBX.load``.

    Returns
    -------
    
    (animation, joint_names, frametime)
        Tuple of loaded animation and joint names
    """
    
    f = open(filename, "r")

    i = 0
    active = -1
    end_site = False
    end_site_is_joint = False

    names = []
    orients = Quaternions.id(0)
    offsets = np.array([]).reshape((0,3))
    parents = np.array([], dtype=int)
    end_site_joints = np.array([], dtype=int)
    
    for line in f:
        
        if "HIERARCHY" in line: continue
        if "MOTION" in line: continue

        rmatch = re.match(r"\s*ROOT\s+(\S+)", line)
        if rmatch:
            names.append(rmatch.group(1))
            offsets    = np.append(offsets,    np.array([[0,0,0]]),   axis=0)
            orients.qs = np.append(orients.qs, np.array([[1,0,0,0]]), axis=0)
            parents    = np.append(parents, active)
            active = (len(parents)-1)
            continue

        if "{" in line: continue

        if "}" in line:
            # if end_site: end_site = False
            # else: active = parents[active]
            if not end_site or end_site_is_joint:
                active = parents[active]
            if end_site:
                end_site = False
                end_site_is_joint = False
            continue
        
        offmatch = re.match(r"\s*OFFSET\s+([\-\d\.e]+)\s+([\-\d\.e]+)\s+([\-\d\.e]+)", line)
        if offmatch:
            offsets[active] = np.array([list(map(float, offmatch.groups()))])
            if end_site and all(offsets[active]==0):
                # an end site of offset zero is not considered a joint
                names = names[:-1]
                offsets = offsets[:-1]
                orients.qs = orients.qs[:-1]
                active = parents[active]
                parents = parents[:-1]
                end_site_joints = end_site_joints[:-1]
                end_site_is_joint = False
            continue
           
        chanmatch = re.match(r"\s*CHANNELS\s+(\d+)", line)
        if chanmatch:
            channels = int(chanmatch.group(1))
            if not order:  # do NOT ask 'if order is NONE' because order may be an empty srint ('')
                channelis = 0 if channels == 3 else 3
                channelie = 3 if channels == 3 else 6
                parts = line.split()[2+channelis:2+channelie]
                if any([p not in channelmap for p in parts]):
                    continue
                print_order = "".join([channelmap[p] for p in parts])
                order = print_order[::-1] # in a bvh file, first rotation axis is printed last
            continue

        jmatch = re.match("\s*JOINT\s+(\S+)", line) #  match <white-spaces>Joint<white-spaces><non-white-spaces>, .e.g: Joint mixamorig:LeftArm
        if jmatch:
            names.append(jmatch.group(1))
            offsets    = np.append(offsets,    np.array([[0,0,0]]),   axis=0)
            orients.qs = np.append(orients.qs, np.array([[1,0,0,0]]), axis=0)
            parents    = np.append(parents, active)
            active = (len(parents)-1)
            continue
        
        if "End Site" in line:
            end_site = True
            offsets = np.append(offsets, np.array([[0, 0, 0]]), axis=0)
            orients.qs = np.append(orients.qs, np.array([[1, 0, 0, 0]]), axis=0)
            parents = np.append(parents, active)
            active = (len(parents) - 1)
            end_site_joints = np.append(end_site_joints, active)
            end_site_is_joint = True
            end_site_match = re.match(".*#\s*name\s*:\s*(\w+).*", line)
            if end_site_match:
                names.append(end_site_match.group(1))
            else:
                names.append('{}_end_site'.format(names[parents[active]]))
            continue
              
        fmatch = re.match("\s*Frames:\s+(\d+)", line)
        if fmatch:
            if start and end:
                fnum = (end - start)-1
            else:
                fnum = int(fmatch.group(1))
            jnum = len(parents)
            positions = offsets[np.newaxis].repeat(fnum, axis=0)
            rotations = np.zeros((fnum, len(orients), 3))
            continue
        
        fmatch = re.match("\s*Frame Time:\s+([\d\.]+)", line)
        if fmatch:
            frametime = float(fmatch.group(1))
            continue
        
        if (start and end) and (i < start or i >= end-1):
            i += 1
            continue
        
        dmatch = line.strip().split(' ')
        if dmatch:
            data_block = np.array(list(map(float, dmatch)))
            N = len(parents) - len(end_site_joints)
            non_end_site_joints = list( set(range(len(parents)))-set(end_site_joints) )
            fi = i - start if start else i
            if   channels == 3:
                positions[fi,0:1] = data_block[0:3]
                rotations[fi, non_end_site_joints ] = data_block[3: ].reshape(N,3)
            elif channels == 6:
                data_block = data_block.reshape(N,6)
                positions[fi,non_end_site_joints] = data_block[:,0:3]
                rotations[fi,non_end_site_joints] = data_block[:,3:6]
            elif channels == 9:
                assert False, 'need to change code to handle end_site_joints'
                positions[fi,0] = data_block[0:3]
                data_block = data_block[3:].reshape(N-1,9)
                rotations[fi,1:] = data_block[:,3:6]
                positions[fi,1:] += data_block[:,0:3] * data_block[:,6:9]
            else:
                raise Exception("Too many channels! %i" % channels)

            i += 1

    f.close()

    rotations = rotations[..., ::-1]
    quat_rotations = Quaternions.from_euler(np.radians(rotations), order=order, world=world)
    if collapse_root:
        names, parents, offsets, rotations_qs, positions, orients = collapse_root_skeleton(
            names,
            parents,
            offsets,
            quat_rotations.qs,
            positions,
            orients,
            warn_path=filename,
        )
        quat_rotations = Quaternions(rotations_qs)


    return (Animation(quat_rotations, positions, orients, offsets, parents), names, frametime)

    
_ALL_EULER_ORDERS = ('xyz', 'xzy', 'yxz', 'yzx', 'zxy', 'zyx')


def _select_gimbal_safe_order(rotations, candidates=_ALL_EULER_ORDERS):
    """Pick the uniform euler order that round-trips exactly and minimises jitter.

    A BVH rotation channel exhibits two kinds of frame-to-frame "jitter" that are
    artifacts of the euler representation, not of the (smooth) orientation:

      * gimbal-lock: when the middle rotation of the order hits +/-90 deg the two
        outer angles become a degenerate coupled pair and split arbitrarily,
        swinging huge amounts while the orientation barely moves.
      * angle wrap: an outer angle crossing +/-180 deg jumps by 360 deg.

    The wrap is benign and removed by temporal unwrapping (``np.unwrap``), which is
    orientation-exact (angles are 2*pi periodic). True gimbal swings survive
    unwrapping. We therefore score each candidate by its worst-case unwrapped
    second difference over all joints/channels and pick the smallest — i.e. the
    order whose channels are smoothest after wraps are removed.

    CRITICAL round-trip constraint: ``save`` writes euler via
    ``Quaternions.euler(order)`` and ``load`` reads it back via
    ``Quaternions.from_euler(order, world=True)``. These are exact inverses for
    *generic* rotations, but right ON an order's gimbal singularity (e.g. a joint
    whose orientation is an exact 90-deg multiple, common in canonicalized rest
    poses) the closed-form ``euler`` extraction and ``from_euler`` disagree, so the
    clip would round-trip to a *different* orientation — silently flipping a whole
    sub-chain's facing by 90 deg. The jitter score does not always catch this
    (a static pose sitting on the singularity has zero temporal jitter). We
    therefore explicitly verify the quaternion round-trip per candidate and only
    consider orders that reconstruct every joint exactly; among those we pick the
    smoothest. If no candidate round-trips (should not happen), fall back to the
    overall-smoothest order so behaviour degrades gracefully.
    """
    best_order, best_score = None, np.inf
    fallback_order, fallback_score = 'xyz', np.inf
    for order in candidates:
        e = rotations.euler(order=order)  # (F, J, 3), radians
        # Verify save->load is orientation-exact for this order (see docstring).
        q_round = Quaternions.from_euler(e, order=order, world=True)
        roundtrip_dot = np.abs(np.sum(rotations.qs * q_round.qs, axis=-1))
        roundtrip_ok = bool(roundtrip_dot.size == 0 or np.min(roundtrip_dot) > 1.0 - 1e-4)
        e_unwrapped = np.unwrap(e, axis=0)
        # second difference along time = curvature; gimbal/residual artifacts spike it
        score = float(np.max(np.abs(np.diff(e_unwrapped, axis=0, n=2)))) if e.shape[0] > 2 else 0.0
        if score < fallback_score:
            fallback_score, fallback_order = score, order
        if roundtrip_ok and score < best_score:
            best_score, best_order = score, order
    return best_order if best_order is not None else fallback_order


def save(filename, anim, names=None, frametime=1.0/30.0, order='auto', positions=False, orients=True):
    """
    Saves an Animation to file as BVH
    
    Parameters
    ----------
    filename: str
        File to be saved to
        
    anim : Animation
        Animation to save
        
    names : [str]
        List of joint names
    
    order : str
        Optional Specifier for joint rotation order, from left to right (not print order!).
        Given as string E.G 'xyz', 'zxy'. Defaults to 'auto', which picks the
        uniform order that round-trips exactly and keeps every joint farthest from
        euler gimbal-lock (see _select_gimbal_safe_order). Pass an explicit order
        only when a fixed channel order is required; note that fixed orders are not
        gimbal-safe for skeletons whose joints sit on that order's singularity.
    
    frametime : float
        Optional Animation Frame time
        
    positions : bool
        Optional specfier to save bone
        positions for each frame
        
    orients : bool
        Multiply joint orients to the rotations
        before saving.
        
    """

    if order == 'auto':
        order = _select_gimbal_safe_order(anim.rotations)
    print_order = order[::-1] # in a bvh file, rotations are printed from last to first
    if names is None:
        names = ["joint_" + str(i) for i in range(len(anim.parents))]

    children = AnimationStructure.children_list(anim.parents)
    # Always treat all joints as named JOINT entries (no End Site collapsing)
    end_sites = []

    with open(filename, 'w') as f:

        t = ""
        f.write("%sHIERARCHY\n" % t)
        f.write("%sROOT %s\n" % (t, names[0]))
        f.write("%s{\n" % t)
        t += '\t'

        f.write("%sOFFSET %f %f %f\n" % (t, anim.offsets[0,0], anim.offsets[0,1], anim.offsets[0,2]) )
        f.write("%sCHANNELS 6 Xposition Yposition Zposition %s %s %s \n" %
            (t, channelmap_inv[print_order[0]], channelmap_inv[print_order[1]], channelmap_inv[print_order[2]]))

        for child in children[0]:
            t = save_joint(f, anim, names, t, child, print_order=print_order, children=children, positions=positions)

        t = t[:-1]
        f.write("%s}\n" % t)

        f.write("MOTION\n")
        f.write("Frames: %i\n" % anim.shape[0]);
        f.write("Frame Time: %f\n" % frametime);

        # Unwrap along the time axis so an outer angle crossing +/-180 deg does not
        # leave a benign-but-ugly 360 deg jump in the channel curve. Unwrapping adds
        # integer multiples of 2*pi, so it is orientation-exact and round-trips
        # through ``load`` (from_euler is 2*pi periodic).
        rots = np.degrees(np.unwrap(anim.rotations.euler(order=order), axis=0))
        poss = anim.positions
        end_sites_set = set(end_sites)

        # Vectorize MOTION data: build a single 2D float array and use
        # np.savetxt which runs in C and releases the GIL.
        n_frames, n_joints = anim.shape
        # ``rots`` (= euler(order=order)) is indexed by ORDER POSITION:
        # rots[..., k] is the angle about axis ``order[k]`` (matching from_euler,
        # which composes axis[order[0..2]]). A BVH file prints rotations in reverse
        # application order, so the channel columns are rots[...,2], [...,1], [...,0].
        # ``load`` reverses the file columns back to order-position before from_euler,
        # so writing the reversed positions here keeps save/load consistent for ANY
        # order. (For 'xyz' this equals the previous axis-indexed mapping, so existing
        # 'xyz' exports are byte-identical.)
        p0, p1, p2 = 2, 1, 0

        # Collect all float columns in BVH joint order (skip end sites)
        all_vals = np.empty((n_frames, 0), dtype=np.float64)
        for j in range(n_joints):
            if j in end_sites_set:
                continue
            if positions or j == 0:
                cols = np.column_stack([
                    poss[:, j, 0], poss[:, j, 1], poss[:, j, 2],
                    rots[:, j, p0], rots[:, j, p1], rots[:, j, p2]
                ])
            else:
                cols = np.column_stack([
                    rots[:, j, p0], rots[:, j, p1], rots[:, j, p2]
                ])
            all_vals = np.hstack([all_vals, cols])

        # np.savetxt is C-level I/O, releases GIL during the heavy lifting
        np.savetxt(f, all_vals, fmt="%f", delimiter=" ")
    
    
def save_joint(f, anim, names, t, i, print_order, children, positions=False):
    end_site = False
    if len(children[i]) == 0:
        end_site = True

    # Always write all joints as JOINT to preserve names
    f.write("%sJOINT %s\n" % (t, names[i]))

    f.write("%s{\n" % t)
    t += '\t'

    f.write("%sOFFSET %f %f %f\n" % (t, anim.offsets[i,0], anim.offsets[i,1], anim.offsets[i,2]))

    # Always write CHANNELS for all joints (including leaf joints)
    if positions:
        f.write("%sCHANNELS 6 Xposition Yposition Zposition %s %s %s \n" % (t,
            channelmap_inv[print_order[0]], channelmap_inv[print_order[1]], channelmap_inv[print_order[2]]))
    else:
        f.write("%sCHANNELS 3 %s %s %s\n" % (t,
            channelmap_inv[print_order[0]], channelmap_inv[print_order[1]], channelmap_inv[print_order[2]]))

    for j in children[i]:
        t = save_joint(f, anim, names, t, j, print_order=print_order, children=children, positions=positions)

    t = t[:-1]
    f.write("%s}\n" % t)
    
    return t