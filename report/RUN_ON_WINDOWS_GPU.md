# Running the ML depth comparison on the Windows + 5070 Ti box

Three things to do on the Windows machine in order. End to end ~15 minutes
of human time.

## 1. Clone and set up the environment (one time)

```powershell
# in PowerShell, somewhere convenient
git clone https://github.com/Danni281/selfie-perspective-correction.git
cd selfie-perspective-correction

# install Python 3.12 if you do not have it (winget is easiest):
#   winget install --id Python.Python.3.12 --scope user
#   - or download https://www.python.org/downloads/release/python-3120/
#     and tick "Add python.exe to PATH" in the installer
# winget also installs the `py` launcher; reopen the shell so PATH updates.

py -3.12 -m venv .venv
.venv\Scripts\activate

# project dependencies
pip install -r requirements.txt

# PyTorch with CUDA 12.8 wheels. The 5070 Ti (incl. the Laptop GPU) is
# Blackwell = compute capability sm_120. The cu126 wheels are built only for
# sm_50..sm_90, so torch.cuda.is_available() returns True but the first real
# GPU op dies with "no kernel image is available for execution on the device".
# The cu128 wheels include sm_120 (cu128 currently resolves to torch 2.11).
pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision

# Depth Anything V2 wrapper deps
pip install transformers pillow accelerate
```

Sanity check that CUDA can actually run a kernel. Do NOT rely on
`torch.cuda.is_available()` alone — it returns True even when the wheel has no
sm_120 kernels for this GPU. Force a real op:

```powershell
python -c "import torch; print(torch.cuda.get_arch_list()); print(torch.cuda.get_device_name(0)); x=torch.randn(2000,2000,device='cuda'); print('gpu ok', float((x@x).sum()))"
# expect arch_list to include 'sm_120' and a number printed after 'gpu ok'.
```

If the matmul raises "no kernel image is available", the wheel lacks sm_120
(you installed a cu126 build); reinstall from the cu128 index above. If
instead `is_available()` is False, update the NVIDIA driver (576+; this box
runs 592.02 / CUDA 13.1) and reboot. The CUDA runtime ships inside the wheel;
no CUDA toolkit install is needed.

## 2. Pull the CMDP images down

The annotations are already committed in the repo (`data\cmdp\CMDP-ANNO`), so
you only need the two image archives. They are too large for git.

The Caltech Data `/content` endpoint 302-redirects to a short-lived (60 s)
presigned S3 URL. `curl.exe -L` follows that redirect and downloads the real
zip in one shot, so the old "read the redirect as text" two-step is wrong and
unnecessary (with -L, curl already fetches the actual file). Use plain curl:

```powershell
cd data\cmdp

# CMDP_1.zip (~315 MB), CMDP_2.zip (~269 MB)
curl.exe -L --fail -o CMDP_1.zip "https://data.caltech.edu/api/records/n5vnm-mqr05/files/CMDP_1.zip/content"
curl.exe -L --fail -o CMDP_2.zip "https://data.caltech.edu/api/records/n5vnm-mqr05/files/CMDP_2.zip/content"

# each zip's top-level folder is CMDP_1\ or CMDP_2\, so extract into images\
Expand-Archive CMDP_1.zip -DestinationPath images -Force
Expand-Archive CMDP_2.zip -DestinationPath images -Force

cd ..\..
```

You should now have `data\cmdp\images\CMDP_1\<subject>\...jpg` and
`...\CMDP_2\...`, 51 subjects total. Expand-Archive also creates a harmless
`images\__MACOSX` folder; ignore it.

## 3. Run the experiments

These three commands produce the figures and numbers we need for the report.
Approx wall time on a 5070 Ti is in brackets.

### A. Single-image qualitative comparison (~3 s, sanity check)

`captures\` is gitignored, so the original selfie snapshot is absent from a
fresh clone. Point this at any face image; a close-range (2 ft) CMDP image is
a good stand-in once section 2 is done:

```powershell
python src\ml_compare.py image data\cmdp\images\CMDP_1\1_K36K\K36K-060208_2.jpg
# saves results\ml_compare_K36K-060208_2.png
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
