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
except:
    from Animation import Animation
    import AnimationStructure
    from Quaternions import Quaternions

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

def load(filename, start=None, end=None, order=None, world=True):
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
    # added for handling truebones common problem of redundant root joint
    if np.isclose(offsets[1], 0).all():
        if len(parents[parents == 1]) == 0: # redundant joint #1, just remove
            offsets[1] = offsets[0]
            offsets = offsets[1:]
            quat_rotations[:, 1] = quat_rotations[:, 0]
            quat_rotations = quat_rotations[:, 1:]
            positions[:, 1] = positions[:, 0]
            positions = positions[:, 1:]
            orients = orients[1:]
            parents = parents[1:] - 1
            parents[1:][parents[1:] < 0] = 0
            names[1] = names[0]
            names = names[1:]
        elif len(parents[parents == 0]) == 1: # remove root joint by composing rots, adding pos & offs
            parent_rots = quat_rotations[:, 0]  # save before composing rotations
            offsets[1] = offsets[0] + (parent_rots[0:1] * offsets[1:2])[0]  # apply canonical rotation to static offset
            offsets=offsets[1:]
            quat_rotations[:, 1] = quat_rotations[:, 0] * quat_rotations[:, 1]
            quat_rotations = quat_rotations[:, 1: ]
            positions[:, 1] = positions[:, 0] + parent_rots * positions[:, 1]  # apply parent rotation per-frame
            positions = positions[:, 1:]
            orients = orients[1:]
            parents = parents[1:] - 1
            names = names[1:]


    return (Animation(quat_rotations, positions, orients, offsets, parents), names, frametime)

    
def save(filename, anim, names=None, frametime=1.0/24.0, order='xyz', positions=False, orients=True, all_joints_as_names=False):
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
        Given as string E.G 'xyz', 'zxy'
    
    frametime : float
        Optional Animation Frame time
        
    positions : bool
        Optional specfier to save bone
        positions for each frame
        
    orients : bool
        Multiply joint orients to the rotations
        before saving.

    all_joints_as_names : bool
        If True, leaf joints are written as JOINT with CHANNELS instead of End Site.
        This preserves all joint names in the BVH instead of losing leaf joints as unnamed End Sites.
        
    """

    print_order = order[::-1] # in a bvh file, rotations are printed from last to first
    if names is None:
        names = ["joint_" + str(i) for i in range(len(anim.parents))]

    children = AnimationStructure.children_list(anim.parents)
    if all_joints_as_names:
        # Treat all joints as named JOINT entries (no End Site collapsing)
        end_sites = []
    elif anim.shape[1] > 1:
        end_sites = [i for i,c in enumerate(children) if len(c)==0]
    else:
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
            t = save_joint(f, anim, names, t, child, print_order=print_order, children=children, positions=positions, all_joints_as_names=all_joints_as_names)

        t = t[:-1]
        f.write("%s}\n" % t)

        f.write("MOTION\n")
        f.write("Frames: %i\n" % anim.shape[0]);
        f.write("Frame Time: %f\n" % frametime);

        rots = np.degrees(anim.rotations.euler(order=order))
        poss = anim.positions
        end_sites_set = set(end_sites)

        # Vectorize MOTION data: build a single 2D float array and use
        # np.savetxt which runs in C and releases the GIL.
        n_frames, n_joints = anim.shape
        p0, p1, p2 = ordermap[print_order[0]], ordermap[print_order[1]], ordermap[print_order[2]]

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
    
    
def save_joint(f, anim, names, t, i, print_order, children, positions=False, all_joints_as_names=False):
    end_site = False
    if len(children[i]) == 0:
        end_site = True

    if not end_site:
        f.write("%sJOINT %s\n" % (t, names[i]))
    elif all_joints_as_names:
        # Write leaf joints as JOINT instead of End Site to preserve names
        f.write("%sJOINT %s\n" % (t, names[i]))
    else:
        f.write("%sEnd Site\n" % t)

    f.write("%s{\n" % t)
    t += '\t'

    f.write("%sOFFSET %f %f %f\n" % (t, anim.offsets[i,0], anim.offsets[i,1], anim.offsets[i,2]))

    if not end_site or all_joints_as_names:
        if positions:
            f.write("%sCHANNELS 6 Xposition Yposition Zposition %s %s %s \n" % (t,
                channelmap_inv[print_order[0]], channelmap_inv[print_order[1]], channelmap_inv[print_order[2]]))
        else:
            f.write("%sCHANNELS 3 %s %s %s\n" % (t,
                channelmap_inv[print_order[0]], channelmap_inv[print_order[1]], channelmap_inv[print_order[2]]))

        for j in children[i]:
            t = save_joint(f, anim, names, t, j, print_order=print_order, children=children, positions=positions, all_joints_as_names=all_joints_as_names)

    t = t[:-1]
    f.write("%s}\n" % t)
    
    return t