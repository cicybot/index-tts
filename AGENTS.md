# IndexTTS2 Agent Guidelines

## Overview
IndexTTS2 is a PyTorch-based text-to-speech system that uses autoregressive models for zero-shot voice cloning with emotional expression control. This codebase follows Python ML engineering best practices with uv for dependency management.

## Build & Test Commands

### Environment Setup
```bash
# Install dependencies with all extras (recommended)
uv sync --all-extras

# Install with specific extras
uv sync --extra webui      # For web interface
uv sync --extra deepspeed  # For acceleration
uv sync --extra api        # For FastAPI server
```

### Running Scripts
All Python scripts must be executed through uv:
```bash
# Run any Python script
uv run python_script.py

# Run web UI
uv run webui.py

# Run FastAPI server
uv run api.py

# Run GPU check utility
uv run tools/gpu_check.py

# Run inference
PYTHONPATH="$PYTHONPATH:." uv run indextts/infer_v2.py
```

### Testing
Tests are located in the `tests/` directory and run as standalone Python scripts:

```bash
# Run regression tests
uv run tests/regression_test.py

# Run padding tests
uv run tests/padding_test.py

# Run a specific test file
PYTHONPATH="$PYTHONPATH:." uv run tests/specific_test.py
```

**Note**: There is no centralized test runner (like pytest). Each test file is a standalone script that can be executed directly.

### Model Management
```bash
# Download models via HuggingFace
uv tool install "huggingface-hub[cli,hf_xet]"
hf download IndexTeam/IndexTTS-2 --local-dir=checkpoints

# Download models via ModelScope
uv tool install "modelscope"
modelscope download --model IndexTeam/IndexTTS-2 --local_dir checkpoints
```

## Code Style Guidelines

### Imports
- Use absolute imports within the `indextts` package
- Group imports: standard library, third-party, local modules
- Use `from typing import` for type hints
- Suppress warnings at module level when necessary

```python
import os
import warnings
from typing import List, Union, Optional

import torch
import torch.nn as nn
from omegaconf import OmegaConf

from indextts.utils.front import TextNormalizer
```

### Naming Conventions
- **Classes**: PascalCase (`IndexTTS2`, `TextNormalizer`, `UnifiedVoice`)
- **Functions/Methods**: snake_case (`load_checkpoint`, `inference_speech`)
- **Variables**: snake_case (`model_dir`, `use_fp16`)
- **Constants**: UPPER_CASE (`STOP_MEL_TOKEN`)
- **Modules**: snake_case (`feature_extractors.py`)

### Type Hints
Use comprehensive type hints:
```python
from typing import List, Union, Optional, overload

def process_text(self, text: str, enable_glossary: bool = False) -> List[str]:
    pass

@overload
def tokenize(self, text: str) -> List[int]:
    pass
```

### Documentation
- Use Google-style docstrings with Args/Returns sections
- Document complex parameters and behavior
- Include type information in docstrings

```python
def __init__(
    self, cfg_path: str = "checkpoints/config.yaml",
    model_dir: str = "checkpoints", use_fp16: bool = False
):
    """
    Initialize IndexTTS2 model.

    Args:
        cfg_path: Path to the config file
        model_dir: Path to the model directory
        use_fp16: Whether to use half-precision
    """
```

### Error Handling
- Use try/except for optional dependencies
- Provide fallback behavior when possible
- Log informative error messages

```python
try:
    import deepspeed
except (ImportError, OSError, CalledProcessError) as e:
    use_deepspeed = False
    print(f">> Failed to load DeepSpeed. Falling back to normal inference. Error: {e}")
```

### Code Structure
- Use class-based design with clear separation of concerns
- Initialize expensive resources in `__init__`
- Use properties for computed attributes
- Follow single responsibility principle

### Performance Considerations
- Use `@torch.no_grad()` for inference
- Cache expensive computations with `@lru_cache`
- Use appropriate data types (fp16 when possible)
- Batch operations when feasible

### File Encoding
Include UTF-8 encoding declaration for files with non-ASCII characters:
```python
# -*- coding: utf-8 -*-
```

### Configuration
- Use OmegaConf for configuration management
- Store config files in YAML format
- Validate configuration parameters

### Environment Variables
Set environment variables early in the module:
```python
import os
os.environ['HF_HUB_CACHE'] = './checkpoints/hf_cache'
```

### Warnings
Suppress common warnings that don't affect functionality:
```python
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
```

## Development Workflow

### Adding New Features
1. Follow existing code patterns and naming conventions
2. Add comprehensive type hints
3. Include docstrings for public APIs
4. Test with existing test cases
5. Update documentation if needed

### Code Review Checklist
- [ ] Type hints are complete and accurate
- [ ] Docstrings follow Google style
- [ ] Naming conventions are followed
- [ ] Error handling is appropriate
- [ ] Performance considerations addressed
- [ ] Tests pass and new tests added if needed

### Common Patterns
- Use `torch.nn.Module` for neural network components
- Use `torch.device` for device management
- Use `torch.dtype` for precision management
- Use `Path` objects for file path handling
- Use context managers for resource management

## Security Considerations
- Validate input parameters to prevent injection attacks
- Use safe file path handling
- Avoid executing arbitrary code from untrusted sources
- Follow PyTorch security best practices for model loading

## Performance Optimization
- Use `torch.compile` when available
- Leverage CUDA kernels for GPU operations
- Use DeepSpeed for large model inference
- Profile code with PyTorch profiler when optimizing

## Contributing
- Follow the established code style and patterns
- Test changes thoroughly
- Update documentation for API changes
- Consider backward compatibility</content>
<parameter name="filePath">/Users/data/python/index-tts/AGENTS.md