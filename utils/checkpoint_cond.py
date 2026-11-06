"""Save the exact training condition file with a checkpoint's inference assets."""
import hashlib
from pathlib import Path


def _file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def snapshot_training_cond(source, save_dir, expected_hash):
    """Refresh cond.npy only from data matching the accepted training contract.

    The cond file is large (tens of MB), so it is hashed and copied in a single
    pass into a staging file; a mismatch leaves the existing snapshot untouched.
    """
    source = Path(source)
    destination = Path(save_dir) / 'cond.npy'
    if source.resolve() == destination.resolve():
        if _file_sha256(source) != expected_hash:
            raise ValueError('Training cond changed before checkpoint snapshot')
        return destination
    staging = destination.with_name(f'{destination.name}.tmp')
    digest = hashlib.sha256()
    try:
        with open(source, 'rb') as src, open(staging, 'wb') as dst:
            for block in iter(lambda: src.read(1024 * 1024), b''):
                digest.update(block)
                dst.write(block)
        if digest.hexdigest() != expected_hash:
            raise ValueError('Training cond changed before checkpoint snapshot')
        staging.replace(destination)
    finally:
        staging.unlink(missing_ok=True)
    if _file_sha256(destination) != expected_hash:
        raise ValueError('Saved cond snapshot hash differs from training')
    return destination
