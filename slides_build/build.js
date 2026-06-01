// Demo deck — Real-Time Perspective Correction for Selfie Video
// Plain, modest version: minimal decoration, hedged claims, no face example.
const pptxgen = require("pptxgenjs");
const p = new pptxgen();
p.layout = "LAYOUT_WIDE";            // 13.3 x 7.5
p.author = "Daoyuan Chi";
p.title = "Real-Time Perspective Correction for Selfie Video";

// minimal palette
const INK   = "1A1A1A";   // body text
const NAVY  = "1F3A5F";   // headings / title bg
const ACCENT = "2E7D9A";  // one restrained accent
const MUTE  = "6B7785";   // captions
const WHITE = "FFFFFF";
const HF = "Georgia";     // headers
const BF = "Calibri";     // body
const W = 13.3, H = 7.5;

function footer(s, n) {
  s.addText("Real-Time Perspective Correction  ·  Daoyuan Chi", {
    x: 0.7, y: H - 0.45, w: 9, h: 0.3, fontFace: BF, fontSize: 9,
    color: MUTE, align: "left", margin: 0 });
  s.addText(String(n), {
    x: W - 1.2, y: H - 0.45, w: 0.6, h: 0.3, fontFace: BF, fontSize: 9,
    color: MUTE, align: "right", margin: 0 });
}
function title(s, t) {
  s.addText(t, { x: 0.7, y: 0.5, w: 12, h: 0.7, fontFace: HF, fontSize: 28,
    bold: true, color: NAVY, margin: 0 });
}

// =====================================================================
// SLIDE 1 — Title
// =====================================================================
let s = p.addSlide();
s.background = { color: NAVY };
s.addText("Real-Time Perspective Correction for Selfie Video", {
  x: 1, y: 2.4, w: 11.3, h: 1.6, fontFace: HF, fontSize: 40, bold: true,
  color: WHITE, lineSpacingMultiple: 1.05, margin: 0 });
s.addText("A real-time, depth-based approach to reducing selfie distortion in video", {
  x: 1, y: 4.1, w: 11, h: 0.5, fontFace: BF, fontSize: 17, color: "C7D4E2", margin: 0 });
s.addText("Daoyuan Chi   ·   Stanford University", {
  x: 1, y: 5.4, w: 11, h: 0.4, fontFace: BF, fontSize: 14, color: "C7D4E2", margin: 0 });

// =====================================================================
// SLIDE 2 — Problem & motivation
// =====================================================================
s = p.addSlide(); s.background = { color: WHITE };
title(s, "The problem");
s.addText([
  { text: "Front-facing cameras are held close to the face, around 30–50 cm. At that distance, perspective magnifies the features nearest the lens — the nose and lips — relative to the ears and jaw. This is the familiar “selfie” look.", options: { breakLine: true, paraSpaceAfter: 14 } },
  { text: "Prior work addresses this, but mostly for single still images and offline:", options: { breakLine: true, paraSpaceAfter: 6 } },
  { text: "fitting a 3D head model to one photo (Fried 2016, Shih 2019)", options: { bullet: true, breakLine: true, color: MUTE } },
  { text: "deep networks that take seconds per image (Zhao 2019, DisCO 2024)", options: { bullet: true, breakLine: true, color: MUTE } },
], { x: 0.7, y: 1.6, w: 11.8, h: 2.6, fontFace: BF, fontSize: 17, color: INK, valign: "top" });
s.addText([
  { text: "This project: ", options: { bold: true, color: NAVY } },
  { text: "explore whether a similar correction can run in real time on live video, where the main added challenges are speed and frame-to-frame stability.", options: { color: INK } },
], { x: 0.7, y: 4.6, w: 11.8, h: 1.0, fontFace: BF, fontSize: 17, valign: "top" });
footer(s, 2);

// =====================================================================
// SLIDE 3 — Method
// =====================================================================
s = p.addSlide(); s.background = { color: WHITE };
title(s, "Method: a four-stage per-frame pipeline");
const steps = [
  ["1.  Detect", "MediaPipe Face Mesh gives 478 facial landmarks per frame, each with a depth (z) value."],
  ["2.  Measure", "Compute simple landmark ratios (mainly nose-width / face-width) as a distortion estimate."],
  ["3.  Correct", "Interpolate the sparse landmark depths into a dense per-pixel depth map, then apply a pinhole re-projection per pixel (one cv2.remap). A smooth depth field gives a smooth warp."],
  ["4.  Stabilize", "Smooth the landmarks over time (EMA / Kalman / 1-Euro) to reduce flicker."],
];
let yy = 1.65;
steps.forEach((st) => {
  s.addText(st[0], { x: 0.7, y: yy, w: 2.4, h: 0.6, fontFace: HF, fontSize: 17, bold: true, color: ACCENT, margin: 0, valign: "top" });
  s.addText(st[1], { x: 3.2, y: yy, w: 9.3, h: 1.1, fontFace: BF, fontSize: 15.5, color: INK, margin: 0, valign: "top" });
  yy += st[1].length > 90 ? 1.25 : 0.95;
});
s.addText("Key idea: use MediaPipe’s depth as a signal to interpolate, rather than moving landmark points directly — which avoids the seams of sparse-landmark warps.", {
  x: 0.7, y: 6.2, w: 11.8, h: 0.7, fontFace: BF, fontSize: 14, italic: true, color: MUTE, margin: 0, valign: "top" });
footer(s, 3);

// =====================================================================
// SLIDE 4 — Result 1: temporal stability
// =====================================================================
s = p.addSlide(); s.background = { color: WHITE };
title(s, "Result 1: temporal stability");
s.addChart(p.charts.BAR, [{
  name: "jitter", labels: ["None", "EMA", "1-Euro", "Kalman"],
  values: [4.07, 3.28, 3.40, 3.69],
}], {
  x: 0.7, y: 1.7, w: 6.6, h: 4.6, barDir: "col",
  chartColors: [MUTE, ACCENT, ACCENT, ACCENT],
  catAxisLabelColor: INK, catAxisLabelFontSize: 12, catAxisLabelFontFace: BF,
  valAxisLabelColor: MUTE, valAxisMinVal: 0, valAxisMaxVal: 5,
  valGridLine: { color: "E2E8F0", size: 0.5 }, catGridLine: { style: "none" },
  showValue: true, dataLabelPosition: "outEnd", dataLabelColor: INK,
  dataLabelFontFace: BF, dataLabelFontSize: 12, dataLabelFontBold: true,
  dataLabelFormatCode: "0.00", showLegend: false, showTitle: false });
s.addText("mean frame-to-frame landmark jitter (px) — lower is steadier", {
  x: 0.7, y: 6.35, w: 6.6, h: 0.3, fontFace: BF, fontSize: 11, italic: true, color: MUTE, align: "center", margin: 0 });
s.addText([
  { text: "Running the corrector per frame inherits the detector’s jitter, which causes flicker.", options: { breakLine: true, paraSpaceAfter: 12 } },
  { text: "On a 10 s / 300-frame clip, all three smoothers reduce jitter relative to the per-frame baseline:", options: { breakLine: true, paraSpaceAfter: 8 } },
  { text: "EMA ≈ 19% lower", options: { bullet: true, breakLine: true } },
  { text: "1-Euro ≈ 16% lower, with better response during motion", options: { bullet: true, breakLine: true } },
  { text: "Kalman ≈ 9% lower", options: { bullet: true } },
], { x: 7.7, y: 1.9, w: 5.0, h: 4.3, fontFace: BF, fontSize: 15, color: INK, valign: "top" });
footer(s, 4);

// =====================================================================
// SLIDE 5 — Result 2: CMDP accuracy
// =====================================================================
s = p.addSlide(); s.background = { color: WHITE };
title(s, "Result 2: accuracy on a ground-truth benchmark");
s.addImage({ path: "cmdp_clean.png", x: 0.7, y: 1.7, w: 7.6, h: 3.9,
  sizing: { type: "contain", w: 7.6, h: 3.9 } });
s.addText("Caltech Multi-Distance Portraits: 51 subjects, each shot at 2–16 ft. The 16 ft image is the near-undistorted reference.", {
  x: 0.7, y: 5.7, w: 7.6, h: 0.7, fontFace: BF, fontSize: 12, italic: true, color: MUTE, margin: 0, valign: "top" });
s.addText([
  { text: "The raw nose ratio rises steadily as the camera gets closer, so the metric does capture perspective distortion.", options: { breakLine: true, paraSpaceAfter: 12 } },
  { text: "Across 347 images, the correction moves the ratio toward the ground-truth line at every distance, closing roughly 38–93% of the gap at typical selfie distances.", options: { breakLine: true, paraSpaceAfter: 12 } },
  { text: "At a fixed strength it can over-correct at far distances — which motivates the per-user calibration below.", options: { color: MUTE } },
], { x: 8.6, y: 1.9, w: 4.1, h: 4.3, fontFace: BF, fontSize: 14.5, color: INK, valign: "top" });
footer(s, 5);

// =====================================================================
// SLIDE 6 — Observations / what I learned
// =====================================================================
s = p.addSlide(); s.background = { color: WHITE };
title(s, "Two observations");
s.addText("Per-user calibration", { x: 0.7, y: 1.7, w: 11, h: 0.45, fontFace: HF, fontSize: 18, bold: true, color: NAVY, margin: 0 });
s.addText("Face proportions vary between people, so a single population baseline can mislabel a naturally wider face as distorted. Capturing each user’s own neutral once lets the system correct only when the face is genuinely close, and do little otherwise.", {
  x: 0.7, y: 2.2, w: 11.8, h: 1.1, fontFace: BF, fontSize: 15.5, color: INK, margin: 0, valign: "top" });
s.addText("Depth quality appears to be the main limit", { x: 0.7, y: 3.7, w: 11, h: 0.45, fontFace: HF, fontSize: 18, bold: true, color: NAVY, margin: 0 });
s.addText("Substituting a learned depth model (Depth Anything V2) for MediaPipe’s depth, with the same warp, closed noticeably more of the gap at close range. This suggests the limiting factor is the depth signal rather than the warp itself.", {
  x: 0.7, y: 4.2, w: 11.8, h: 1.1, fontFace: BF, fontSize: 15.5, color: INK, margin: 0, valign: "top" });
s.addText("Limitations: the distortion metric is person-dependent, the effect is subtle at normal distance, and a single-ratio 2D warp cannot fully reproduce the 3D shape.", {
  x: 0.7, y: 5.7, w: 11.8, h: 0.8, fontFace: BF, fontSize: 14, italic: true, color: MUTE, margin: 0, valign: "top" });
footer(s, 6);

// =====================================================================
// SLIDE 7 — Summary
// =====================================================================
s = p.addSlide(); s.background = { color: NAVY };
s.addText("Summary", { x: 1, y: 0.9, w: 11, h: 0.7, fontFace: HF, fontSize: 28, bold: true, color: WHITE, margin: 0 });
s.addText([
  { text: "A real-time, depth-based perspective corrector for selfie video, running at roughly 23–30 FPS on a laptop.", options: { bullet: true, breakLine: true, paraSpaceAfter: 14 } },
  { text: "A dense depth-driven warp that reuses MediaPipe’s landmark depths and avoids sparse-warp artifacts.", options: { bullet: true, breakLine: true, paraSpaceAfter: 14 } },
  { text: "Evaluated on a 51-subject ground-truth benchmark, with three temporal smoothers compared.", options: { bullet: true, breakLine: true, paraSpaceAfter: 14 } },
  { text: "Per-user calibration so the correction is adaptive rather than fixed.", options: { bullet: true } },
], { x: 1, y: 2.0, w: 11.3, h: 3.4, fontFace: BF, fontSize: 17, color: "E6EDF4", valign: "top" });
s.addText("Future work: a faster learned-depth model in the live loop, automatic calibration, and a richer distortion metric.", {
  x: 1, y: 5.5, w: 11.3, h: 0.6, fontFace: BF, fontSize: 14, italic: true, color: "AFC0D4", margin: 0, valign: "top" });
s.addText("github.com/Danni281/selfie-perspective-correction   ·   Thank you", {
  x: 1, y: 6.4, w: 11.3, h: 0.4, fontFace: BF, fontSize: 13, color: "AFC0D4", margin: 0 });

p.writeFile({ fileName: "../report/demo_slides.pptx" }).then(f => console.log("wrote", f));
