import os
import statistics 
import numpy as np


DEFAULT_RAW_DATA_DIR = "dataset/truebones/zoo/Truebone_Z-OO"
DEFAULT_DATASET_DIR = "dataset/truebones/zoo/truebones_processed"


def get_raw_data_dir():
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
COSMETICS = ["PolarBearB", "KingCobra", "Hamster", "Skunk", "Comodoa", "Hippopotamus", "Leapord", "Rhino", "Hound"]
NO_HANDS = ["Raptor", "Anaconda"]
NO_BVHS =["Jaws", "Crow", "Dog", "Dog-2"]
IGNORE_OBJECTS = NO_BVHS
MILLIPEDS = ["Cricket", "SpiderG" , "Scorpion", "Isopetra", "FireAnt", "Crab", "Centipede", "Roach", "Ant", "HermitCrab", "Scorpion-2", "Spider"]
SNAKES = ["Anaconda", "KingCobra"]
# Joint index pairs (neck, head) used to compute forward direction for limbless animals.
# Each entry maps object_type -> (neck_joint_index, head_joint_index).
SNAKE_FORWARD_CHAINS = {
    'Anaconda': (20, 21),
    'KingCobra': (4, 8),
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

FACE_JOINTS = {"Alligator": [8, 11, 17, 20] , "Crow": [18, 21, 7, 11], "Anaconda": [13, 26, 13, 26], "Ant": [9, 15, 23, 30], "Bat": [6, 15, 26, 34], 
               "Bear": [8, 2, 36, 56], "Bird": [15, 35, 6, 11], "BrownBear": [2, 7, 15, 23], "Buffalo": [6, 12, 20, 26], "Buzzard": [7, 23, 41, 47], "Camel": [9, 15, 32, 26], "Cat": [6, 12, 21, 27], 
               "Centipede": [7, 2, 41, 47], "Chicken": [5, 17, 30, 32], "Comodoa": [11, 1, 33, 43], "Coyote": [5, 11, 20, 27], "Crab": [14, 20, 51, 47], 
               "Cricket": [20, 25, 32, 36], "Crocodile": [7, 12, 21, 27], "Deer": [4, 9, 29, 35], "Dog": [8, 14, 26, 32], "Dog-2": [8, 14, 26, 32], 
               "Dragon":[10, 23, 47, 83], "Eagle": [7, 20, 35, 41], "Elephant": [6, 10, 32, 36], "FireAnt": [15, 19, 25, 29], 
               "Flamingo": [15, 22, 10, 6], "Fox": [27, 33, 15, 8], "Gazelle": [4, 10, 20, 26], "Giantbee": [11, 16, 3, 1], "Goat": [24, 19, 13, 7], "Hamster": [3, 9, 19, 25], 
               "HermitCrab": [51, 46, 8, 12], "Hippopotamus": [5, 11, 28, 34], "Horse": [10, 16, 33, 41], "Hound": [3, 9, 19, 25], "Isopetra": [48, 55, 18, 26], "Jaguar": [6, 12, 22, 28], 
               "KingCobra": [6, 7, 6, 7], "Leapord": [7, 13, 25, 31], "Lion": [6, 11, 18, 23], "Lynx": [2, 8, 18, 24], "Mammoth": [7, 11, 34, 38], 
               "Monkey": [9, 21, 56, 36], "Ostrich": [6, 16, 36, 28], "Parrot": [9, 25, 65, 43], "Parrot2": [7, 23, 42, 48], "Pigeon": [3, 4, 1, 6], "Pirrana": [19, 20, 4, 5], 
               "PolarBear": [3, 9, 19, 25], "PolarBearB": [3, 8, 17, 23], "Pteranodon": [16, 5, 40, 35], "Puppy": [5, 11, 20, 26], "Raindeer": [3, 9, 18, 24], "Raptor": [13, 19, 13, 19], 
               "Raptor2": [52, 40, 23, 13], "Raptor3": [53, 41, 24, 14], "Rat": [12, 15, 9, 6], "Rhino": [5, 11, 21, 27], "Roach": [2, 6, 29, 25], "SabreToothTiger": [7, 2, 37, 51], 
               "SandMouse": [7, 13, 30, 34], "Scorpion": [58, 29, 20, 25], "Scorpion-2": [55, 23, 48, 16], "Skunk": [10, 15, 28, 32], "Spider": [21, 27, 5, 9], "SpiderG": [13, 19, 27, 33],
               "Stego": [7, 12, 27, 21], "Trex": [38, 50, 23, 15], "Tricera": [6, 11, 24, 28], "Tukan": [4, 6, 9, 11], "Turtle": [31, 40, 12, 22], "Tyranno": [7, 20, 37, 44]}

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
