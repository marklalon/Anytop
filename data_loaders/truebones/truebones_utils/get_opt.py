from argparse import Namespace
import re
from pathlib import Path

from data_loaders.truebones.truebones_utils.param_utils import (
    MAX_JOINTS,
    FEATS_LEN,
    MAX_PATH_LEN,
    FPS,
    DEFAULT_DATASET_DIR,
)
from data_loaders.truebones.truebones_utils import dataset_tags as _dataset_tags
from data_loaders.truebones.truebones_utils.dataset_tags import dataset_tags
from data_loaders.truebones.truebones_utils.cond_schema import load_cond
from data_loaders.truebones.truebones_utils.dataset_sources import (
    COND_FILE,
    resolve_anytop_path,
    sources_from_cond,
)


def is_float(numStr):
    flag = False
    numStr = str(numStr).strip().lstrip('-').lstrip('+')
    try:
        reg = re.compile(r'^[-+]?[0-9]+\.[0-9]+$')
        res = reg.match(str(numStr))
        if res:
            flag = True
    except Exception as ex:
        print("is_float() - error: " + str(ex))
    return flag


def is_number(numStr):
    flag = False
    numStr = str(numStr).strip().lstrip('-').lstrip('+')    # 去除正数(+)、负数(-)符号
    if str(numStr).isdigit():
        flag = True
    return flag


DEFAULT_COND_PATH = str(Path(DEFAULT_DATASET_DIR) / COND_FILE)


def get_opt(device, cond_path=None, cond_dict=None):
    """Build the run options from one ``cond.npy``.

    ``cond.npy`` is the single entry point: it names the species, and (through
    each entry's ``dataset_namespace`` / ``dataset_root``) the dataset
    directories their clips live in.  There is no single ``data_root`` any more
    -- ``opt.sources`` drives every enumeration, and a single-dataset run is
    simply the ``len(sources) == 1`` case.

    Configuring ``dataset_tags`` happens here because ``opt.subsets_dict`` is
    read from it immediately: sources whose directories are present are read
    from their ``species_tags.jsonl`` sidecars, and otherwise the tags come from
    the cond's own baked copies (the inference contract, where no dataset
    directory need exist).
    """
    cond_path = str(resolve_anytop_path(cond_path or DEFAULT_COND_PATH))
    if cond_dict is None:
        cond_dict = load_cond(cond_path)

    sources = sources_from_cond(cond_dict, cond_path)
    if all(Path(source.root).is_dir() for source in sources):
        _dataset_tags.configure(sources=sources)
    else:
        _dataset_tags.configure_from_cond(cond_dict)

    opt = Namespace()
    opt.cond_file = cond_path
    opt.sources = sources
    opt.max_joints = MAX_JOINTS
    opt.feature_len = FEATS_LEN
    opt.is_continue = False
    opt.device = device
    opt.max_path_len = MAX_PATH_LEN
    opt.fps = FPS
    opt.subsets_dict = dataset_tags().object_subsets
    return opt
