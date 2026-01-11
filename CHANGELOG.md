# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0] - 2026-01-11

### Added
- Modern TensorFlow 2.x/Keras support (replacing legacy Keras)
- Multiple model architectures: RotateNet (improved), U-Net, ResNet
- CLI interface with subcommands for train, predict, evaluate, export
- JSON-based configuration system with dataclasses
- `requirements.txt` for dependency management
- `setup.py` for proper package installation
- Type hints throughout the codebase
- Batch normalization and dropout support
- Mixed precision training support
- Early stopping and learning rate scheduling
- TensorBoard integration for training visualization
- Model checkpointing with best model selection
- CSV logging for training history
- Modern README with comprehensive documentation
- Configuration validation
- Support for multiple optimizers
- Advanced metrics (precision, recall, AUC)
- Model export capabilities (planned)

### Changed
- Modernized model architecture with flexible design
- Improved code organization and modularity
- Updated documentation with modern examples
- Enhanced error handling and validation
- Better separation of concerns (config, model, training, inference)
- Improved naming conventions and code style

### Improved
- Training pipeline with modern callbacks
- Configuration management system
- User experience with CLI tools
- Documentation and examples
- Code maintainability and readability

### Technical Debt Addressed
- Removed hardcoded global variables in favor of configuration
- Separated model definition from training logic
- Added proper Python package structure
- Improved import organization
- Better path handling (preparing for pathlib migration)

### Future Plans
- Add comprehensive test suite
- Create Jupyter notebook examples
- Add Gradio/Streamlit web UI
- Implement data loading pipeline
- Add ONNX/TensorFlow Lite export
- Docker containerization
- Cloud deployment guides
- Add attention mechanisms
- Implement transfer learning

## [1.0.0] - 2018

### Initial Release
- Original RotateNet architecture for lineament detection
- Training and inference scripts
- TKinter GUI applet
- Support for 8-layer geophysical data
- DBSCAN clustering for post-processing
- Line and curve fitting algorithms
- Visualization tools
- Support for rotation-based data augmentation
- Probability map generation
- Basic evaluation metrics

---

### Version Numbering

- **Major version**: Significant architectural changes or API breaking changes
- **Minor version**: New features, non-breaking changes
- **Patch version**: Bug fixes, minor improvements

### Links

- [Original Thesis](http://hdl.handle.net/2429/68438)
- [GitHub Repository](https://github.com/RichardScottOZ/LineamentLearning)
