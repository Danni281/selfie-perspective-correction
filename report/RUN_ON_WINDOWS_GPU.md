# Running the ML depth comparison on the Windows + 5070 Ti box

Three things to do on the Windows machine in order. End to end ~15 minutes
of human time.

## 1. Clone and set up the environment (one time)

```powershell
# in PowerShell, somewhere convenient
git clone https://github.com/Danni281/selfie-perspective-correction.git
cd selfie-perspective-correction

# install Python 3.12 if you do not have it
#   https://www.python.org/downloads/release/python-3120/
#   tick "Add python.exe to PATH" in the installer

py -3.12 -m venv .venv
.venv\Scripts\activate

# project dependencies
pip install -r requirements.txt

# PyTorch with CUDA 12.6 wheels (Blackwell-compatible, works on the 5070 Ti)
pip install --index-url https://download.pytorch.org/whl/cu126 torch torchvision

# Depth Anything V2 wrapper deps
pip install transformers pillow accelerate
```

Sanity check that CUDA sees the GPU.

```powershell
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# expect:  True NVIDIA GeForce RTX 5070 Ti
```

If `torch.cuda.is_available()` is `False`, update the NVIDIA driver to a
576+ build and reboot. CUDA 12.6 runtime ships inside the wheel; you do not
need a CUDA toolkit install.

## 2. Pull the CMDP images down

The annotations are in the repo. The image archives are too large for git,
download them once.

```powershell
mkdir data\cmdp
cd data\cmdp

# CMDP annotations
curl -L -o CMDP-ANNO.zip "https://data.caltech.edu/api/records/n5vnm-mqr05/files/CMDP-ANNO.zip/content"
# the API returns a redirect URL as text; fetch the real file from there
$url = Get-Content CMDP-ANNO.zip
Remove-Item CMDP-ANNO.zip
curl -L -o CMDP-ANNO.zip $url
Expand-Archive CMDP-ANNO.zip -DestinationPath . -Force

# CMDP_1.zip (~330MB)
curl -L -o CMDP_1.zip "https://data.caltech.edu/api/records/n5vnm-mqr05/files/CMDP_1.zip/content"
$url = Get-Content CMDP_1.zip
Remove-Item CMDP_1.zip
curl -L -o CMDP_1.zip $url
mkdir images
Expand-Archive CMDP_1.zip -DestinationPath images -Force

# CMDP_2.zip (~280MB)
curl -L -o CMDP_2.zip "https://data.caltech.edu/api/records/n5vnm-mqr05/files/CMDP_2.zip/content"
$url = Get-Content CMDP_2.zip
Remove-Item CMDP_2.zip
curl -L -o CMDP_2.zip $url
Expand-Archive CMDP_2.zip -DestinationPath images -Force

cd ..\..
```

## 3. Run the experiments

These three commands produce the figures and numbers we need for the report.
Approx wall time on a 5070 Ti is in brackets.

### A. Single-image qualitative comparison (~3 s, sanity check)

```powershell
python src\ml_compare.py image captures\20260525-151052_raw.png
# saves results\ml_compare_20260525-151052_raw.png
# prints raw / MediaPipe-corrected / ML-corrected nose_w_over_face_w
```

### B. CMDP one-subject grid for the report (~20 s)

```powershell
python src\ml_compare.py cmdp-grid --subject 0
# saves results\ml_compare_cmdp_subject0.png
# rows = 7 distances, cols = raw / MP / ML / depth viz
```

Pick another subject if subject 0 is not photogenic enough,
`--subject 5`, `--subject 12`, etc.

### C. CMDP full quantitative evaluation, MediaPipe vs ML, 51 subjects (~3-5 min)

This is the headline ML number for the report.

```powershell
python src\cmdp_eval.py --alpha 2.0 --ml-depth --out-prefix cmdp_eval_ml
# writes
#   results\cmdp_eval_ml.png      grouped bar chart
#   results\cmdp_eval_ml.json     mean/std per distance per method
#   results\cmdp_eval_ml_raw.csv  per-image raw rows
#   results\cmdp_eval_ml_corr.csv per-image MediaPipe-depth rows
```

The plot shows two bars per distance, "MediaPipe depth" vs "ML depth", with
the % of the raw-to-GT gap closed by each. That figure (plus the existing
`results\cmdp_eval.png`) goes into Section 5 of the report as the ML
comparison.

### D. Optional. Same clip you recorded with both depth sources (~30 s)

```powershell
# if you want the apples-to-apples temporal comparison too
# (it does not need to be the same clip you used for Phase 6; any 10s clip works)
python src\process_clip.py run clips\raw_*.mp4 --mode dense --alpha 2.0 --smooth none --out ml_baseline
# then add a --depth-source flag run if we wire it up
```

## 4. Push results back so I can use them on the Mac

```powershell
git add results\ml_compare_*.png results\cmdp_eval_ml.*
git commit -m "ML depth comparison results from the 5070 Ti box"
git push
```

Then on the Mac I can pull and integrate the figures into the report.

## Notes

- The Mac side does not have torch installed, so the live `--depth-source ml`
  mode is GPU-box only. The repo defaults still use MediaPipe depth so the
  Mac live preview keeps working.
- Default model is `Depth-Anything-V2-Small` (25M params). Try `--model-size base`
  if you want a higher-quality depth field. `large` is overkill for 720p faces.
- If CUDA OOMs on a CMDP image, the model probably just got handed a 5k-wide
  scan. Resize to 1024 wide first.
