from argparse import Namespace
import re
from pathlib import Path

from data_loaders.truebones.truebones_utils.param_utils import (
    MAX_JOINTS,
    FEATS_LEN,
    FPS,
    DEFAULT_DATASET_DIR,
)
from data_loaders.truebones.truebones_utils import dataset_tags as _dataset_tags
from data_loaders.truebones.truebones_utils.dataset_tags import dataset_tags, SPECIES_TAGS_FILE
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


def get_opt(device, cond_path=None, cond_dict=None, *, inference=False):
    """Build the run options from one ``cond.npy``.

    ``cond.npy`` is the single entry point: it names the species, and (through
    each entry's ``dataset_namespace`` / ``dataset_root``) the dataset
    directories their clips live in.  There is no single ``data_root`` any more
    -- ``opt.sources`` drives every enumeration, and a single-dataset run is
    simply the ``len(sources) == 1`` case.

    Configuring ``dataset_tags`` happens here because ``opt.subsets_dict`` is
    read from it immediately, and ``inference`` picks which of the two contracts
    supplies it:

    * ``inference=False`` (training, preprocessing, dataset tools) -- the
      dataset directories are the live source of truth and their
      ``species_tags.jsonl`` is read.  A dataset dir that carries no sidecar
      falls back to the cond's baked tags; there is no fallback to the *default*
      dataset's tags, so a missing sidecar never borrows from another species.
    * ``inference=True`` (generation) -- the cond's own baked tags, always, and
      the dataset directories are not touched at all, not even to ask whether
      they exist.  A checkpoint's cond.npy is a snapshot of what the weights
      were trained against; the dataset it came from is re-preprocessed and
      re-tagged between runs, so reading it back here would silently describe
      the checkpoint's species with tags it never saw.  This is what makes a
      checkpoint directory self-contained: model + args.json + cond.npy and
      nothing else.
    """
    cond_path = str(resolve_anytop_path(cond_path or DEFAULT_COND_PATH))
    if cond_dict is None:
        cond_dict = load_cond(cond_path)

    # probe_filesystem=False under inference: a stored ``dataset_root`` is
    # normally the portable Anytop-relative form, and resolving one the ordinary
    # way stats the dataset directory to choose between the cwd and the Anytop
    # root. Lexical resolution both keeps the contract and makes opt.sources a
    # function of the cond alone rather than of where generation was launched.
    sources = sources_from_cond(cond_dict, cond_path, probe_filesystem=not inference)
    if inference:
        # No ``.is_file()`` probe either: the point is that generation runs with
        # the dataset directories absent, renamed or stale, so nothing here may
        # depend on what is at ``source.root``. ``opt.sources`` keeps the roots
        # the cond names for diagnostics only.
        _dataset_tags.configure_from_cond(cond_dict)
    else:
        sidecars_present = all(
            (Path(source.root) / SPECIES_TAGS_FILE).is_file() for source in sources
        )
        if sidecars_present:
            _dataset_tags.configure(sources=sources)
        else:
            _dataset_tags.configure_from_cond(cond_dict)

    opt = Namespace()
    opt.inference = bool(inference)
    opt.cond_file = cond_path
    opt.sources = sources
    opt.max_joints = MAX_JOINTS
    opt.feature_len = FEATS_LEN
    opt.is_continue = False
    opt.device = device
    # No opt.max_path_len: nothing ever read it. The model sized its hop table
    # from its own hardcoded default, so raising MAX_PATH_LEN here would have
    # emitted out-of-range indices and died in a device-side gather. Both sides
    # now derive from topology_relations instead.
    opt.fps = FPS
    opt.subsets_dict = dataset_tags().object_subsets
    return opt
