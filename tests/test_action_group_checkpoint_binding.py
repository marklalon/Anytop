"""The action group is a property of the weights, not of the request.

Training requires ``--action_group`` (exactly one of the three groups) and
records it in the checkpoint's args.json; generation has no such flag and reads
the group back from there, because the group also fixes the multi-hot mask the
model was trained with.
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from argparse import ArgumentParser
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from data_loaders.truebones.truebones_utils.motion_labels import (  # noqa: E402
    ACTION_GROUPS as LABEL_ACTION_GROUPS,
)
from utils import parser_util  # noqa: E402
from utils.parser_util import add_data_options, generate_args  # noqa: E402


def _checkpoint_dir(tmp_dir, **args_json):
    """A minimal save_dir: args.json next to a (never opened) model####.pt."""
    save_dir = Path(tmp_dir)
    (save_dir / 'args.json').write_text(json.dumps(args_json), encoding='utf-8')
    return str(save_dir / 'model000100.pt')


def _generate_args(model_path, *extra):
    return generate_args(argv=['--model_path', model_path, *extra])


class ActionGroupIsBoundToTheCheckpoint(unittest.TestCase):

    def test_parser_groups_match_the_dataset_definition(self):
        # parser_util keeps its own copy to stay import-light; it must not drift.
        self.assertEqual(tuple(parser_util.ACTION_GROUPS), tuple(LABEL_ACTION_GROUPS))

    def test_generation_takes_the_group_from_the_checkpoint(self):
        for group in parser_util.ACTION_GROUPS:
            with self.subTest(group=group):
                with tempfile.TemporaryDirectory() as tmp:
                    model_path = _checkpoint_dir(tmp, action_group=group)
                    self.assertEqual(_generate_args(model_path).action_group, group)

    def test_generation_rejects_an_action_group_flag(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_path = _checkpoint_dir(tmp, action_group='locomotion')
            with self.assertRaises(SystemExit):
                _generate_args(model_path, '--action_group', 'stationary')
            # Not even the checkpoint's own group may be restated: the flag is gone.
            with self.assertRaises(SystemExit):
                _generate_args(model_path, '--action_group', 'locomotion')

    def test_checkpoint_predating_the_flag_generates_without_a_group(self):
        # Unconditional generation still works; sample/generate.py is what refuses
        # --action_label for such a checkpoint.
        for recorded in ({}, {'action_group': ''}, {'action_group': 'all'}):
            with self.subTest(recorded=recorded):
                with tempfile.TemporaryDirectory() as tmp:
                    model_path = _checkpoint_dir(tmp, **recorded)
                    self.assertEqual(_generate_args(model_path).action_group, '')

    def test_other_dataset_args_still_come_from_the_checkpoint(self):
        # Guards the mechanism the binding rides on: the rest of the dataset group
        # is still overwritten by the training args.
        with tempfile.TemporaryDirectory() as tmp:
            model_path = _checkpoint_dir(
                tmp, action_group='locomotion', objects_subset='quadruped')
            args = _generate_args(model_path, '--objects_subset', 'winged')
            self.assertEqual(args.objects_subset, 'quadruped')


class TrainingMustNameExactlyOneGroup(unittest.TestCase):

    @staticmethod
    def _training_parser():
        parser = ArgumentParser()
        add_data_options(parser, training=True)
        return parser

    def test_each_group_is_accepted(self):
        for group in parser_util.ACTION_GROUPS:
            with self.subTest(group=group):
                args = self._training_parser().parse_args(['--action_group', group])
                self.assertEqual(args.action_group, group)

    def test_the_flag_is_required(self):
        with self.assertRaises(SystemExit):
            self._training_parser().parse_args([])

    def test_all_and_empty_and_lists_are_refused(self):
        for value in ('all', '', 'locomotion,stationary', 'Locomotion'):
            with self.subTest(value=value):
                with self.assertRaises(SystemExit):
                    self._training_parser().parse_args(['--action_group', value])


class ResumeCannotRewriteTheRecordedGroup(unittest.TestCase):

    def _assert_resume(self, recorded, requested, resume_checkpoint='model000100.pt'):
        from argparse import Namespace
        from train.train_anytop import assert_resume_keeps_action_group
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / 'args.json').write_text(
                json.dumps({'action_group': recorded}), encoding='utf-8')
            args = Namespace(
                resume_checkpoint=resume_checkpoint, action_group=requested)
            assert_resume_keeps_action_group(args, tmp)

    def test_resuming_the_same_group_is_allowed(self):
        self._assert_resume('locomotion', 'locomotion')

    def test_switching_the_group_on_resume_is_refused(self):
        with self.assertRaises(SystemExit):
            self._assert_resume('locomotion', 'stationary')

    def test_resuming_a_group_less_run_is_refused(self):
        # Nothing to continue it as -- 'all' is no longer a group.
        for recorded in ('', 'all'):
            with self.subTest(recorded=recorded):
                with self.assertRaises(SystemExit):
                    self._assert_resume(recorded, 'locomotion')

    def test_fresh_run_is_not_checked(self):
        # No resume: the existing args.json is about to be replaced wholesale.
        self._assert_resume('locomotion', 'stationary', resume_checkpoint='')


if __name__ == '__main__':
    unittest.main()
