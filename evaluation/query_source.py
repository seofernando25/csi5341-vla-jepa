"""Prepare a separate persistent query-adaptation source; never trains or edits RSI."""

from __future__ import annotations

import argparse
import shutil
from datetime import UTC, datetime
from pathlib import Path

from evaluation.common import ROOT, file_hash, read_json, write_json


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError('Registered source anchor changed')
    return text.replace(old, new)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--architecture-source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Never overwrite a scientific source snapshot')
    base = args.architecture_source.resolve()
    manifest = {str(p.relative_to(base)): file_hash(p) for p in sorted((base / 'src').rglob('*.py'))}
    expected = read_json(ROOT / 'studies/recovery/diagnostics/image_pipeline_amendment.json')['source_manifest']
    if manifest != expected:
        raise ValueError('Use the registered corrected-RGB n0008 parent source')
    package = Path('src/lerobot_policy_vla_jepa_smolvlm')
    config = (base / package / 'configuration_vla_jepa_smolvlm.py').read_text()
    interface = (base / package / 'smolvlm_interface.py').read_text()
    config = replace_once(config, '    unfreeze_last_n: int = 0',
        '    query_token_adaptation: str = "none"  # none | input_residual\n'
        '    smol_gradient_checkpointing: bool = False\n'
        '    unfreeze_last_n: int = 0')
    config = replace_once(config, '        if self.conditioning_dim <= 0:',
        '        if self.query_token_adaptation not in {"none", "input_residual"}:\n'
        '            raise ValueError("Unknown query-token adaptation")\n'
        '        if self.conditioning_dim <= 0:')
    interface = replace_once(interface, 'from transformers import AutoImageProcessor, AutoProcessor, SmolVLMForConditionalGeneration',
        'from transformers import AutoConfig, AutoImageProcessor, AutoProcessor, SmolVLMForConditionalGeneration')
    interface = replace_once(interface, 'from .configuration_vla_jepa_smolvlm import VLAJEPASmolVLMConfig',
        'from .configuration_vla_jepa_smolvlm import VLAJEPASmolVLMConfig\n'
        'from .query_adapter import QueryTokenResidual')
    interface = replace_once(interface, '        self._configure_trainability()\n',
        '        self.query_adapter = None\n'
        '        if config.smol_gradient_checkpointing:\n'
        '            self.model.gradient_checkpointing_enable(\n'
        '                gradient_checkpointing_kwargs={"use_reentrant": False}\n'
        '            )\n'
        '        self._configure_trainability()\n')
    interface = replace_once(interface, '    def _adapt_penultimate_output(self, _module, _inputs, output):',
        '    def _adapt_query_embeddings(self, _module, inputs, output):\n'
        '        return self.query_adapter(inputs[0], output)\n\n'
        '    def _adapt_penultimate_output(self, _module, _inputs, output):')
    interface = replace_once(interface, '            self.decoder_adapter.requires_grad_(True)',
        '            self.decoder_adapter.requires_grad_(True)\n'
        '        if self.query_adapter is not None:\n'
        '            self.query_adapter.requires_grad_(True)')
    interface = replace_once(interface, '        return action_tokens, action_token_ids, embodied_id',
        '        if self.config.query_token_adaptation == "input_residual" and self.query_adapter is None:\n'
        '            tubelet = (AutoConfig.from_pretrained(self.config.jepa_encoder_name).tubelet_size\n'
        '                       if self.config.enable_world_model else self.config.jepa_tubelet_size)\n'
        '            steps = self.config.num_video_frames // tubelet - 1\n'
        '            if not 0 <= steps <= len(action_token_ids):\n'
        '                raise ValueError("Query vocabulary does not cover the configured temporal positions")\n'
        '            ids = action_token_ids[:steps] + [embodied_id]\n'
        '            self.query_adapter = QueryTokenResidual(ids, self.native_hidden_size).to(self.model.device)\n'
        '            self._query_adapter_hook = self.model.get_input_embeddings().register_forward_hook(\n'
        '                self._adapt_query_embeddings\n'
        '            )\n'
        '        return action_tokens, action_token_ids, embodied_id')
    for name in manifest:
        target = args.output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(base / name, target)
    (args.output / package / 'configuration_vla_jepa_smolvlm.py').write_text(config)
    (args.output / package / 'smolvlm_interface.py').write_text(interface)
    shutil.copyfile(ROOT / 'evaluation/query_adapter.py', args.output / package / 'query_adapter.py')
    result = {str(p.relative_to(args.output)): file_hash(p) for p in sorted((args.output / 'src').rglob('*.py'))}
    write_json(args.output / 'amendment.json', {
        'created_at': datetime.now(UTC).isoformat(), 'parent': 'registered corrected-RGB n0008 source',
        'scope': 'Separate query-adaptation preparation; no optimizer updates or cloud launch',
        'change': 'Optional zero input-query residual and nonreentrant decoder checkpointing.',
        'default': 'Both options disabled; legacy state layout and computation retained.',
        'preserved': 'All parent tensors, RNG initialization, data/split, image processing, action/world architecture, native losses and prediction semantics.',
        'base_source_manifest': manifest, 'source_manifest': result,
        'changed_files': [name for name in result if result[name] != manifest.get(name)]})
    print('Prepared separate query source; no training launched.')


if __name__ == '__main__':
    main()
