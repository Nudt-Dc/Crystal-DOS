# Environment provenance

The original training environment used Python 3.11 and PyTorch 2.12.1+cu126. `requirements.txt` records the core versions from its saved package inventory; it uses the PyPI torch version without a CUDA wheel suffix. Select the appropriate official PyTorch wheel source for your device.

Additional preprocessing versions come from the recorded preprocessing and training environments. The SOAP environment recorded NumPy 2.5.3 and SciPy 1.18.1, whereas the training environment recorded NumPy 2.4.6 and SciPy 1.17.1. `requirements-preprocessing.txt` retains the training-compatible SciPy pin while adding the recorded structure/SOAP libraries. This combined requirements selection is a dependency specification, not a newly verified full environment lock or a guarantee of numerical identity between platforms.

GPU driver, CUDA kernel and package versions can affect numerical reproducibility. The included source fixes random seeds and deterministic training settings, and uses stored split/control artifacts. Exact paper reproduction also requires the source snapshots and control artifacts named by the original run signatures.

No server configuration, credentials, private endpoint, or machine-specific package URL is included.
