"""The ``joint_parts.jsonl`` a generation run writes beside its motions.

Generated ``.npy`` files are bare feature arrays; what each joint is and
whether it stands on the ground travels next to them, in the same sidecar
format a dataset keeps (``joint_parts.load_joint_parts`` reads it). One row
per skeleton, shared by every motion of it in the directory:

* a skeleton whose cond has no baked annotation (``process_new_skeleton``)
  gets the auxiliary head's prediction (``source: model``); readers of its
  motions take the parts from this row;
* a dataset species keeps its baked annotation in cond, which readers use; its
  row (``source: annotation``) is written only when the checkpoint has the
  head, as a diagnostic of how far the head's ``part_prob`` / ``contact_prob``
  are from the annotation.
"""

from __future__ import annotations

import os

import torch

from data_loaders.truebones.truebones_utils.joint_parts import (
    JOINT_CONTACT_KEY,
    JOINT_PARTS_FILE,
    JOINT_PARTS_KEY,
    has_joint_parts,
    joint_parts_row,
    species_of,
    write_output_joint_parts,
)
from model.anytop import AnyTop

# Contact probability at or above which a predicted joint is a contact.
CONTACT_THRESHOLD = 0.5


def _unwrapped(model):
    while not isinstance(model, AnyTop):
        model = getattr(model, '_orig_mod', None) or getattr(model, 'module', None) or getattr(model, 'model')
    return model


@torch.no_grad()
def predict_joint_parts(model, sample, model_kwargs):
    """``(part_prob [B, J, C], contact_prob [B, J])`` for the finished samples.

    One forward of the conditional model on each clean sample at t = 0, the
    head reading the motion the run actually produced. ``None`` when the
    checkpoint has no part head.
    """
    anytop = _unwrapped(model)
    if anytop.joint_part_head is None:
        return None
    timesteps = torch.zeros(sample.shape[0], dtype=torch.long, device=sample.device)
    _, aux = anytop(sample, timesteps, y=model_kwargs['y'], return_aux=True)
    part_prob = torch.softmax(aux['part_logits'].float(), dim=-1)
    contact_prob = torch.sigmoid(aux['contact_logit'].float())
    return part_prob.cpu().numpy(), contact_prob.cpu().numpy()


def write_generation_joint_parts(model, sample, model_kwargs, species_keys, cond_dict, out_path):
    """Merge one row per skeleton of this batch into ``<out_path>/joint_parts.jsonl``.

    ``species_keys[i]`` is the cond key of ``sample[i]``. A skeleton's
    probabilities are averaged over all of its samples in the batch: the parts
    are the skeleton's, not one motion's.
    """
    prediction = predict_joint_parts(model, sample, model_kwargs)
    n_joints = [int(count) for count in torch.as_tensor(model_kwargs['y']['n_joints']).reshape(-1).tolist()]
    rows = []
    for key in dict.fromkeys(species_keys):
        entry = cond_dict[key]
        annotated = has_joint_parts(entry)
        if prediction is None:
            if not annotated:
                print(f'  [WARN] {key}: no baked annotation in cond and no part head to predict '
                      f'one; {JOINT_PARTS_FILE} gets no row for it.')
            continue
        indices = [i for i, k in enumerate(species_keys) if k == key]
        joint_count = n_joints[indices[0]]
        names = list(entry['joints_names'])
        if joint_count != len(names):
            raise ValueError(f'{key}: the sample has {joint_count} joints, its cond entry {len(names)}.')
        part_prob = prediction[0][indices, :joint_count].mean(axis=0)
        contact_prob = prediction[1][indices, :joint_count].mean(axis=0)
        if annotated:
            part_ids, contact = entry[JOINT_PARTS_KEY], entry[JOINT_CONTACT_KEY]
            source = src = 'annotation'
        else:
            part_ids, contact = part_prob.argmax(axis=-1), contact_prob >= CONTACT_THRESHOLD
            source = src = 'model'
        rows.append(joint_parts_row(
            species_of(entry), names, entry['parents'], part_ids=part_ids, contact=contact,
            source=source, src=src, part_prob=part_prob, contact_prob=contact_prob,
        ))
    for warning in write_output_joint_parts(out_path, rows):
        print(f'  [WARN] {warning}')
    return os.path.join(out_path, JOINT_PARTS_FILE)
