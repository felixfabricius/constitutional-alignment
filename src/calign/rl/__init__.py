"""Phase 3 GRPO stack (chunk 7): prompts, mixed dataset, rewards R1/R2, local citation judge, trainer, monitor.

Import-light by design: only `train_grpo` imports TRL/torch, so prompts, dataset, rewards and the monitor run (and are
unit-tested) without the `rl` dependency group.
"""
