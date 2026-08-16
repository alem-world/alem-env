"""Tests for the image handling in HistoryPromptBuilder.

Run with:
    python -m unittest baselines.llm.test_prompt_builder
"""

import unittest

from PIL import Image

from baselines.llm.eval_utils.prompt_builder import HistoryPromptBuilder


def _obs(image=None):
    return {
        "text": {"long_term_context": "long", "short_term_context": "short"},
        "image": image,
    }


class TestImageSize(unittest.TestCase):
    """image_size resizes the model's copy of the frame and nothing else."""

    RENDER_SIZE = (352, 480)

    def _frame(self):
        return Image.new("RGB", self.RENDER_SIZE, "green")

    def _attachments(self, builder):
        return [m.attachment for m in builder.get_prompt() if m.attachment is not None]

    def test_resizes_attachment(self):
        builder = HistoryPromptBuilder(max_image_history=1, image_size=(64, 64))
        builder.update_observation(_obs(self._frame()))
        attachments = self._attachments(builder)
        self.assertEqual(len(attachments), 1)
        self.assertEqual(attachments[0].size, (64, 64))

    def test_leaves_source_image_untouched(self):
        # The evaluator renders GIFs and HTML from this same object, so resizing
        # in place would silently degrade every saved artifact.
        obs = _obs(self._frame())
        builder = HistoryPromptBuilder(max_image_history=1, image_size=(64, 64))
        builder.update_observation(obs)
        self.assertEqual(obs["image"].size, self.RENDER_SIZE)

    def test_none_sends_render_as_is(self):
        builder = HistoryPromptBuilder(max_image_history=1, image_size=None)
        builder.update_observation(_obs(self._frame()))
        self.assertEqual(self._attachments(builder)[0].size, self.RENDER_SIZE)

    def test_accepts_list_from_config(self):
        # Hydra hands over a ListConfig/list, not a tuple.
        builder = HistoryPromptBuilder(max_image_history=1, image_size=[64, 64])
        builder.update_observation(_obs(self._frame()))
        self.assertEqual(self._attachments(builder)[0].size, (64, 64))

    def test_text_only_observation_is_unaffected(self):
        builder = HistoryPromptBuilder(max_image_history=1, image_size=(64, 64))
        builder.update_observation(_obs(image=None))
        self.assertEqual(self._attachments(builder), [])


if __name__ == "__main__":
    unittest.main()
