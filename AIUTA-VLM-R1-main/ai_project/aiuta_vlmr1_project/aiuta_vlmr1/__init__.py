"""
aiuta_vlmr1 -- Uncertainty-aware VLM-R1 + Knowledge Graph integration.

**Primary offline benchmark:** IDKVQA (``evaluation.idkvqa_eval``) for Yes/No/IDK quality,
calibration, abstention, and KG ablations.

**Auxiliary:** CoIN offline static integration via ``pipeline.episode_runner`` (smoke tests only;
not equivalent to official online CoIN / Habitat).

**Future:** Online CoIN / Habitat / VLFM is out of scope for the current codebase paths.

References:
  - VLM-R1: https://github.com/om-ai-lab/VLM-R1
  - CoIN/AIUTA: https://github.com/intelligolabs/CoIN
"""

__version__ = "0.1.0"
