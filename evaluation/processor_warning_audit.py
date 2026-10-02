"""CPU-only prompt-rendering parity check for the SmolVLM processor warning."""

from __future__ import annotations

import argparse
import hashlib
import logging
from pathlib import Path

import torch
import transformers
from transformers import AutoProcessor
from transformers.models.smolvlm import processing_smolvlm
from transformers import processing_utils

from evaluation.common import ROOT, file_hash, read_json, write_json


class WarningCapture(logging.Handler):
    def __init__(self):
        super().__init__()
        self.count = 0

    def emit(self, record):
        if 'have to be in `processor_kwargs` dict' in record.getMessage():
            self.count += 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve previous diagnostic evidence')
    config = read_json(args.checkpoint / 'config.json')
    mapping = read_json(ROOT / 'studies/recovery/diagnostics/task_instruction_mapping.json')
    pins = read_json(ROOT / 'evaluation/sources.json')
    model = pins['models']['smolvlm']
    processor = AutoProcessor.from_pretrained(model['repo'], revision=model['revision'], local_files_only=True)
    video_kwargs = {'num_frames': processor.video_processor.num_frames, 'fps': processor.video_processor.fps}
    action_prompt = ''.join(config['special_action_token'].format(i) * config['num_action_tokens_per_timestep']
                            for i in range(3))
    embodied = config['embodied_action_token'] * config['num_embodied_action_tokens_per_instruction']
    capture = WarningCapture()
    logger = logging.getLogger('transformers.processing_utils')
    previous_handlers, previous_propagate = logger.handlers[:], logger.propagate
    logger.handlers, logger.propagate = [capture], False
    rows = []
    try:
        for task in mapping['mapping']:
            prompt = config['prompt_template'].format(instruction=task['instruction'],
                                                      actions=action_prompt, e_actions=embodied)
            content = [{'type': 'image', 'image': torch.zeros(3, 224, 224)} for _ in range(2)]
            content.append({'type': 'text', 'text': prompt})
            messages = [{'role': 'user', 'content': content}]
            before = capture.count
            original = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            original_warnings = capture.count - before
            before = capture.count
            explicit = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                      processor_kwargs=dict(video_kwargs))
            explicit_warnings = capture.count - before
            if original != explicit or original_warnings != 1 or explicit_warnings:
                raise ValueError('Unexpected rendering or warning behavior')
            rows.append({'dataset_task_id': task['dataset_task_id'], 'libero_task_id': task['libero_task_id'],
                         'rendered_prompt_sha256': hashlib.sha256(original.encode()).hexdigest(),
                         'rendering_identical': True, 'original_warnings': original_warnings,
                         'explicit_processor_kwargs_warnings': explicit_warnings})
    finally:
        logger.handlers, logger.propagate = previous_handlers, previous_propagate
    write_json(args.output, {'purpose': 'cpu_chat_template_warning_parity', 'status': 'passed',
               'transformers_version': transformers.__version__, 'processor_model': model,
               'processor_class': type(processor).__name__,
               'library_source_sha256': {'processing_smolvlm.py': file_hash(processing_smolvlm.__file__),
                                         'processing_utils.py': file_hash(processing_utils.__file__)},
               'checkpoint_config_sha256': file_hash(args.checkpoint / 'config.json'),
               'instruction_mapping_sha256': file_hash(ROOT / 'studies/recovery/diagnostics/task_instruction_mapping.json'),
               'audit_sha256': file_hash(__file__), 'template_tokenize': False,
               'explicit_video_defaults': video_kwargs, 'tasks': rows,
               'finding': 'SmolVLMProcessor injects default video kwargs into **kwargs when processor_kwargs is empty, '
               'even for string-only image prompts. ProcessorMixin logs the warning. Explicit identical defaults '
               'produce the exact same rendered strings without this warning. No image processing runs in this check.',
               'limitations': 'Prompt rendering only; no model load, image-transform equivalence, training, '
               'rollout, speedup or robot-success claim. Current frozen study source remains unchanged.'})
    print({'tasks': len(rows), 'rendered_strings_equal': True,
           'original_warnings': sum(r['original_warnings'] for r in rows), 'explicit_warnings': 0})


if __name__ == '__main__':
    main()
