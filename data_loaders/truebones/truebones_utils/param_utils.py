import os
import statistics 
import numpy as np


DEFAULT_RAW_DATA_DIR = "dataset/truebones/zoo/Truebone_Z-OO"
DEFAULT_DATASET_DIR = "dataset/truebones/zoo/truebones_processed"


def get_raw_data_dir(raw_data_dir=None):
        if raw_data_dir is not None:
                return raw_data_dir
        return os.environ.get("ANYTOP_RAW_DATA_DIR", DEFAULT_RAW_DATA_DIR)


def get_dataset_dir():
        return os.environ.get("ANYTOP_DATASET_DIR", DEFAULT_DATASET_DIR)


RAW_DATA_DIR = get_raw_data_dir()
DATASET_DIR = get_dataset_dir()
MOTION_DIR = "motions"
CORRUPTED_REFERENCE_DIR = "corrupted_references"
ANIMATIONS_DIR = "animations"
BVHS_DIR = "bvhs"
FOOT_CONTACT_HEIGHT_THRESH = 0.3
FOOT_CONTACT_VEL_THRESH = 0.002
MAX_PATH_LEN = 5.
# Vertical clamp thresholds expressed as a ratio of the character's reference
# body length (measured from the processed skeleton's rest-pose joint span).
# Motion within VERTICAL_CLAMP_MIN_RATIO is left unchanged; only the excess is
# compressed into the [min, max] band.
VERTICAL_CLAMP_MIN_RATIO = 0.5
VERTICAL_CLAMP_MAX_RATIO = 1.0

COSMETICS = ["PolarBearB", "KingCobra", "Hamster", "Skunk", "Comodoa", "Hippopotamus", "Leapord", "Rhino", "Hound"]
NO_HANDS = ["Raptor", "Anaconda"]
MILLIPEDS = ["Cricket", "SpiderG" , "Scorpion", "Isopetra", "FireAnt", "Crab", "Centipede", "Roach", "Ant", "HermitCrab", "Scorpion-2", "Spider"]
SNAKES = ["Anaconda", "KingCobra"]
# Maps object_type -> joint index tuple used to compute the forward direction for
# creatures without usable limb pairs (snakes, fish).
# 2-tuple (neck, head)        -> forward = head - neck
# 3-tuple (base, neck, head)  -> forward = (head - neck) + (neck - base)
CHAIN_FORWARD_JOINTS = {
    'Anaconda': (20, 21),
    'KingCobra': (4, 8),
    'Pirrana': (9, 2, 3),   # kosi → mune → atama (tail to head)
}
FLYING = ["Bat", "Dragon", "Bird", "Buzzard", "Eagle", "Giantbee", "Parrot", "Parrot2", "Pigeon", "Pteranodon", "Tukan"]
CONNECTED_TO_GROUND = ["Bear", "Camel", "Hippopotamus", "Horse", "Pirrana", "Pteranodon", "Raptor3", "Rat", "SabreToothTiger", "Scorpion-2", "Spider", "Trex", "Tukan", "Pirrana"]
FISH = ["Pirrana"]
BIPEDS = ["Ostrich", "Flamingo", "Raptor", "Raptor2", "Raptor3", "Trex", "Chicken", "Tyranno"]
QUADROPEDS = ["Horse", "Hippopotamus", "Comodoa", "Camel", "Bear", "Buffalo", "Cat", "BrownBear", "Coyote", "Crocodile", "Elephant", "Deer", "Fox", "Gazelle", 
           "Goat", "Jaguar","Lynx", "Tricera", "Stego" , "SandMouse", "Raindeer", "Puppy", "PolarBear", "Monkey", "Mammoth", "Alligator", "Hamster", 
           "Hound", "Leapord", "Lion", "PolarBearB", "Rat", "Rhino", "SabreToothTiger", "Skunk", "Turtle"]
OBJECT_SUBSETS_DICT = {"all" : QUADROPEDS + BIPEDS + MILLIPEDS + SNAKES + FISH + FLYING,
                       "quadropeds": QUADROPEDS,
                       "flying": FLYING,
                       "bipeds": BIPEDS, 
                       "millipeds": MILLIPEDS,
                       "millipeds_snakes": MILLIPEDS + SNAKES, 
                       "quadropeds_clean": [quad for quad in QUADROPEDS if quad not in CONNECTED_TO_GROUND], 
                       "millipeds_clean": [mill for mill in MILLIPEDS if mill not in CONNECTED_TO_GROUND], 
                       "bipeds_clean": [bip for bip in BIPEDS if bip not in CONNECTED_TO_GROUND], 
                       "flying_clean": [fly for fly in FLYING if fly not in CONNECTED_TO_GROUND], 
                       "all_clean": [obj for obj in  QUADROPEDS + BIPEDS + MILLIPEDS + SNAKES + FISH + FLYING if obj not in CONNECTED_TO_GROUND] 
                       }


def parse_motion_name_keywords(raw_keywords):
        if raw_keywords is None:
                return tuple()
        if isinstance(raw_keywords, str):
                tokens = raw_keywords.replace(';', ',').split(',')
        else:
                tokens = raw_keywords
        return tuple(token.strip().lower() for token in tokens if str(token).strip())


def filter_motion_names_by_keywords(motion_names, raw_keywords):
        keywords = parse_motion_name_keywords(raw_keywords)
        if not keywords:
                return motion_names

        filtered = {
                name for name in motion_names
                if any(keyword in name.lower() for keyword in keywords)
        }
        return filtered

MAX_JOINTS=143
FPS=20
FEATS_LEN=13
SMPL_OFFSETS = np.array([[ 0.0000,  0.0000,  0.0000],
        [ 0.1031,  0.0000,  0.0000],
        [-0.1099,  0.0000,  0.0000],
        [ 0.0000,  0.1316,  0.0000],
        [ 0.0000, -0.3936,  0.0000],
        [ 0.0000, -0.3902,  0.0000],
        [ 0.0000,  0.1432,  0.0000],
        [ 0.0000, -0.4324,  0.0000],
        [ 0.0000, -0.4256,  0.0000],
        [ 0.0000,  0.0574,  0.0000],
        [ 0.0000,  0.0000,  0.1434],
        [ 0.0000,  0.0000,  0.1494],
        [ 0.0000,  0.2194,  0.0000],
        [ 0.1375,  0.0000,  0.0000],
        [-0.1434,  0.0000,  0.0000],
        [ 0.0000,  0.0000,  0.1030],
        [ 0.0000, -0.1316,  0.0000],
        [ 0.0000, -0.1230,  0.0000],
        [ 0.0000, -0.2568,  0.0000],
        [ 0.0000, -0.2631,  0.0000],
        [ 0.0000, -0.2660,  0.0000],
        [ 0.0000, -0.2699,  0.0000]])
HML_AVG_BONELEN = statistics.mean(np.linalg.norm(SMPL_OFFSETS[1:], axis=1))
