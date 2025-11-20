# convert_onnx_for_github

This folder contains scripts and utilities to convert trained models to ONNX/QONNX formats and run few-shot evaluation for the PEFSL demo.

Key scripts:
- `run_convert_qonnx.sh` — convert a trained model to ONNX/QONNX using `model_to_qonnx.py`.
- `run_few_shot_evaluation.sh` — run few-shot evaluation using `few_shot_evaluation.py`.

## Requirements
- Linux
- Python 3.8+
- PyTorch, ONNX, qonnx/finn/brevitas deps used by conversion scripts

Install typical Python deps (example):
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt  # if provided
```

## Convert a model to ONNX / QONNX
Use the provided script:
```bash
./run_convert_qonnx.sh
```
The script calls `model_to_qonnx.py`. Example direct invocation:
```bash
python3 model_to_qonnx.py \
  --input-resolution 32 \
  --save-name "my_model" \
  --model-type "brain_resnet9_fm16_strided" \
  --model-specification "pt/resnet9_strided_16fmaps.pt"
```
Default output folders in this repository:
- `onnx/` — exported ONNX files
- `qonnx/` — QONNX files (if generated)
- `pt/` — PyTorch weights (reference)

Adjust arguments to match your model type and checkpoint path. Check `model_to_qonnx.py` for available options.

## Few-shot evaluation
Run the evaluation script with the provided shell wrapper:
```bash
./run_few_shot_evaluation.sh
```
Or call the Python entry directly:
```bash
python3 few_shot_evaluation.py --dataset-path /path/to/dataset --batch-size 16 --shots 5
```
See `few_shot_evaluation.py` for full argument list and configuration.

## Convert ONNX to Tensil (optional)
To compile a Tensil model from ONNX, use:
```bash
python3 onnx_to_tensil.py --onnx-path onnx/your_model.onnx --arch-path path/to/arch.json --output-dir tensil_output
```
Docker and Tensil docker image are required for the compilation step.

## Typical workflow
1. Place trained PyTorch checkpoint in `pt/` or point `--model-specification` to its path.
2. Run `./run_convert_qonnx.sh` to export ONNX/QONNX.
3. Optionally compile to Tensil with `onnx_to_tensil.py`.
4. Run few-shot evaluation with `./run_few_shot_evaluation.sh` or `few_shot_evaluation.py`.

## Notes
- See `model_to_qonnx.py` and `few_shot_evaluation.py` for detailed options and supported model types.
- This folder is part of the PEFSL project. See the parent repository README for full demo, Vivado project, and PYNQ instructions.

## License
See LICENSE in this folder.