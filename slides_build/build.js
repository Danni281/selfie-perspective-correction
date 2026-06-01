// Demo deck — Real-Time Perspective Correction for Selfie Video
const pptxgen = require("pptxgenjs");
const p = new pptxgen();
p.layout = "LAYOUT_WIDE";            // 13.3 x 7.5
p.author = "Daoyuan Chi";
p.title = "Real-Time Perspective Correction for Selfie Video";

// ---- palette: "Ocean Gradient" (deep blue dominant, teal support, mint accent) ----
const NAVY = "10243E";   // deep background
const BLUE = "1C5D8C";   // primary
const TEAL = "1C7293";   // secondary
const MINT = "37C5A3";   // accent
const ICE  = "CFE3F2";   // light text on dark
const CREAM = "F4F7FA";  // light slide bg
const INK  = "13222F";   // dark text on light
const MUTE = "5B7184";   // muted captions

const HF = "Georgia";    // headers
const BF = "Calibri";    // body
const W = 13.3, H = 7.5;

// helper: footer tag on light slides
function footer(s, n) {
  s.addText("Real-Time Perspective Correction  ·  Daoyuan Chi", {
    x: 0.6, y: H - 0.45, w: 8, h: 0.3, fontFace: BF, fontSize: 9,
    color: MUTE, align: "left", margin: 0,
  });
  s.addText(String(n), {
    x: W - 1.1, y: H - 0.45, w: 0.5, h: 0.3, fontFace: BF, fontSize: 9,
    color: MUTE, align: "right", margin: 0,
  });
}
// helper: section eyebrow + title block on light slides
function head(s, eyebrow, title) {
  s.addText(eyebrow.toUpperCase(), {
    x: 0.6, y: 0.45, w: 11, h: 0.3, fontFace: BF, fontSize: 12, bold: true,
    color: TEAL, charSpacing: 3, margin: 0,
  });
  s.addText(title, {
    x: 0.6, y: 0.78, w: 12.1, h: 0.85, fontFace: HF, fontSize: 30, bold: true,
    color: INK, margin: 0,
  });
}

// =====================================================================
// SLIDE 1 — Title (dark)
// =====================================================================
let s = p.addSlide();
s.background = { color: NAVY };
// accent block motif top-left
s.addShape(p.shapes.RECTANGLE, { x: 0, y: 0, w: 0.28, h: H, fill: { color: MINT } });
s.addText("REAL-TIME COMPUTER VISION", {
  x: 0.9, y: 1.5, w: 11, h: 0.4, fontFace: BF, fontSize: 14, bold: true,
  color: MINT, charSpacing: 4, margin: 0,
});
s.addText("Real-Time Perspective\nCorrection for Selfie Video", {
  x: 0.9, y: 2.0, w: 11.5, h: 2.2, fontFace: HF, fontSize: 46, bold: true,
  color: "FFFFFF", lineSpacingMultiple: 1.02, margin: 0,
});
s.addText([
  { text: "Adaptive, depth-driven, temporally stable — running live at 30 FPS", options: { breakLine: true } },
], { x: 0.92, y: 4.5, w: 11, h: 0.5, fontFace: BF, fontSize: 18, color: ICE, margin: 0 });
s.addText([
  { text: "Daoyuan Chi", options: { bold: true, color: "FFFFFF" } },
  { text: "   ·   Stanford University", options: { color: ICE } },
], { x: 0.92, y: 5.7, w: 11, h: 0.4, fontFace: BF, fontSize: 15, margin: 0 });

// =====================================================================
// SLIDE 2 — Problem (light, two column: text + the proof teaser)
// =====================================================================
s = p.addSlide();
s.background = { color: CREAM };
head(s, "The problem", "Selfies distort your face — and prior fixes are offline");
// left column text
s.addText([
  { text: "Front cameras sit 30–50 cm away. Perspective makes near features (nose, lips, forehead) loom larger than far ones (ears, jaw).", options: { breakLine: true, paraSpaceAfter: 12 } },
  { text: "Prior work corrects this per still image, offline:", options: { breakLine: true, paraSpaceAfter: 6, bold: true, color: INK } },
  { text: "Fried 2016, Shih 2019 — fit a 3D model to one photo", options: { bullet: true, breakLine: true, color: MUTE } },
  { text: "Zhao 2019, DisCO 2024 — deep nets, seconds per image", options: { bullet: true, breakLine: true, color: MUTE } },
], { x: 0.6, y: 1.9, w: 5.7, h: 3.0, fontFace: BF, fontSize: 16, color: INK, valign: "top" });
// the gap callout
s.addShape(p.shapes.RECTANGLE, { x: 0.6, y: 5.25, w: 5.7, h: 1.5, fill: { color: NAVY } });
s.addShape(p.shapes.RECTANGLE, { x: 0.6, y: 5.25, w: 0.12, h: 1.5, fill: { color: MINT } });
s.addText("Our gap", {
  x: 0.85, y: 5.42, w: 5, h: 0.3, fontFace: BF, fontSize: 12, bold: true,
  color: MINT, charSpacing: 2, margin: 0,
});
s.addText("No real-time, temporally-stable corrector for live selfie video.", {
  x: 0.85, y: 5.72, w: 5.2, h: 0.9, fontFace: BF, fontSize: 16, bold: true,
  color: "FFFFFF", margin: 0, valign: "top",
});
// right: proof image teaser (raw vs corrected vs truth)
s.addImage({ path: "proof_row.png", x: 6.7, y: 2.0, w: 6.0, h: 2.0 });
s.addText("Close-up (left) → our correction (middle) → the same person at 16 ft, this subject’s own ground truth (right).", {
  x: 6.7, y: 4.15, w: 6.0, h: 0.7, fontFace: BF, fontSize: 12, italic: true, color: MUTE, margin: 0,
});
// nose-ratio readout as a compact card so the mono line never wraps
s.addShape(p.shapes.RECTANGLE, { x: 6.7, y: 4.95, w: 6.0, h: 1.05, fill: { color: "FFFFFF" },
  line: { color: "DCE6EF", width: 1 } });
s.addShape(p.shapes.RECTANGLE, { x: 6.7, y: 4.95, w: 0.12, h: 1.05, fill: { color: MINT } });
s.addText("nose-width / face-width", {
  x: 6.95, y: 5.1, w: 5.6, h: 0.3, fontFace: BF, fontSize: 12, bold: true, color: MUTE, margin: 0 });
s.addText([
  { text: "0.296", options: { color: MUTE } },
  { text: "  →  ", options: { color: INK } },
  { text: "0.286", options: { bold: true, color: TEAL } },
  { text: "      subject GT 0.287", options: { color: MUTE } },
], { x: 6.95, y: 5.42, w: 5.6, h: 0.45, fontFace: "Consolas", fontSize: 16, bold: true, margin: 0 });
footer(s, 2);

// =====================================================================
// SLIDE 3 — Method (light, pipeline flow)
// =====================================================================
s = p.addSlide();
s.background = { color: CREAM };
head(s, "Method", "A measure-then-correct pipeline, per frame");
const steps = [
  ["1", "Detect", "MediaPipe Face Mesh\n478 (x, y, z) landmarks"],
  ["2", "Measure", "Distortion ratios\n+ per-user calibration"],
  ["3", "Correct", "Dense depth-driven\nperspective re-projection"],
  ["4", "Stabilize", "Temporal smoothing\nacross frames"],
];
const bx = 0.6, bw = 2.85, gap = 0.32, by = 2.1, bh = 2.5;
steps.forEach((st, i) => {
  const x = bx + i * (bw + gap);
  s.addShape(p.shapes.RECTANGLE, { x, y: by, w: bw, h: bh, fill: { color: "FFFFFF" },
    line: { color: "DCE6EF", width: 1 },
    shadow: { type: "outer", color: "000000", blur: 7, offset: 2, angle: 135, opacity: 0.10 } });
  s.addShape(p.shapes.RECTANGLE, { x, y: by, w: bw, h: 0.12, fill: { color: TEAL } });
  s.addShape(p.shapes.OVAL, { x: x + 0.28, y: by + 0.38, w: 0.7, h: 0.7, fill: { color: NAVY } });
  s.addText(st[0], { x: x + 0.28, y: by + 0.38, w: 0.7, h: 0.7, fontFace: HF, fontSize: 24, bold: true,
    color: MINT, align: "center", valign: "middle", margin: 0 });
  s.addText(st[1], { x: x + 0.2, y: by + 1.25, w: bw - 0.4, h: 0.4, fontFace: HF, fontSize: 19, bold: true,
    color: INK, margin: 0 });
  s.addText(st[2], { x: x + 0.2, y: by + 1.68, w: bw - 0.4, h: 0.75, fontFace: BF, fontSize: 13,
    color: MUTE, margin: 0, valign: "top" });
  if (i < 3) s.addText("›", { x: x + bw - 0.02, y: by + 0.7, w: gap, h: 1, fontFace: BF, fontSize: 30,
    bold: true, color: TEAL, align: "center", valign: "middle", margin: 0 });
});
// key idea band
s.addShape(p.shapes.RECTANGLE, { x: 0.6, y: 5.1, w: 12.1, h: 1.45, fill: { color: NAVY } });
s.addShape(p.shapes.RECTANGLE, { x: 0.6, y: 5.1, w: 0.12, h: 1.45, fill: { color: MINT } });
s.addText("Key idea", { x: 0.85, y: 5.28, w: 3, h: 0.3, fontFace: BF, fontSize: 12, bold: true,
  color: MINT, charSpacing: 2, margin: 0 });
s.addText([
  { text: "We treat MediaPipe’s z as a depth signal, interpolate it to a dense per-pixel map, then apply the true ", options: {} },
  { text: "pinhole re-projection per pixel", options: { bold: true, color: MINT } },
  { text: " — a smooth depth field gives a smooth warp, with no sparse-landmark artifacts.", options: {} },
], { x: 0.85, y: 5.62, w: 11.6, h: 0.85, fontFace: BF, fontSize: 15, color: "FFFFFF", margin: 0, valign: "top" });
footer(s, 3);

// =====================================================================
// SLIDE 4 — Result A: temporal stability (light, chart + numbers)
// =====================================================================
s = p.addSlide();
s.background = { color: CREAM };
head(s, "Result · temporal stability", "Smoothing cuts frame-to-frame jitter");
s.addChart(p.charts.BAR, [{
  name: "jitter", labels: ["None\n(baseline)", "EMA", "1-Euro", "Kalman"],
  values: [4.07, 3.28, 3.40, 3.69],
}], {
  x: 0.6, y: 1.95, w: 6.6, h: 4.6, barDir: "col",
  chartColors: [MUTE, MINT, TEAL, BLUE],
  chartArea: { fill: { color: "FFFFFF" } },
  catAxisLabelColor: INK, catAxisLabelFontSize: 12, catAxisLabelFontFace: BF,
  valAxisLabelColor: MUTE, valAxisHidden: false, valAxisMinVal: 0, valAxisMaxVal: 5,
  valGridLine: { color: "E2E8F0", size: 0.5 }, catGridLine: { style: "none" },
  showValue: true, dataLabelPosition: "outEnd", dataLabelColor: INK,
  dataLabelFontFace: BF, dataLabelFontSize: 13, dataLabelFontBold: true,
  dataLabelFormatCode: "0.00",
  showLegend: false, showTitle: false,
});
s.addText("mean landmark jitter (px) — lower is steadier", {
  x: 0.6, y: 6.45, w: 6.6, h: 0.3, fontFace: BF, fontSize: 11, italic: true, color: MUTE, align: "center", margin: 0 });
// right: stat callouts
const stats = [
  ["−19%", "EMA jitter vs baseline", MINT],
  ["−17%", "1-Euro — best lag/jitter balance", TEAL],
  ["10 s / 300 frames", "talking + head-turn clip", BLUE],
];
stats.forEach((st, i) => {
  const y = 2.0 + i * 1.45;
  s.addShape(p.shapes.RECTANGLE, { x: 7.6, y, w: 5.1, h: 1.25, fill: { color: "FFFFFF" },
    line: { color: "DCE6EF", width: 1 },
    shadow: { type: "outer", color: "000000", blur: 6, offset: 2, angle: 135, opacity: 0.08 } });
  s.addShape(p.shapes.RECTANGLE, { x: 7.6, y, w: 0.12, h: 1.25, fill: { color: st[2] } });
  s.addText(st[0], { x: 7.85, y: y + 0.12, w: 4.7, h: 0.7, fontFace: HF, fontSize: 30, bold: true, color: INK, margin: 0 });
  s.addText(st[1], { x: 7.87, y: y + 0.82, w: 4.7, h: 0.35, fontFace: BF, fontSize: 13, color: MUTE, margin: 0 });
});
s.addText("EMA, Kalman, and the 1-Euro filter (Casiez 2012) all reduce jitter; 1-Euro trades a little for better motion response, as expected.", {
  x: 7.6, y: 6.3, w: 5.1, h: 0.7, fontFace: BF, fontSize: 11.5, italic: true, color: MUTE, margin: 0, valign: "top" });
footer(s, 4);

// =====================================================================
// SLIDE 5 — Result B: CMDP ground-truth eval (light, image + numbers)
// =====================================================================
s = p.addSlide();
s.background = { color: CREAM };
head(s, "Result · accuracy on ground truth", "51 subjects × 7 distances (Caltech CMDP)");
s.addImage({ path: "cmdp_clean.png", x: 0.6, y: 1.9, w: 7.7, h: 3.9,
  sizing: { type: "contain", w: 7.7, h: 3.9 } });
s.addText("Caltech Multi-Distance Portraits — each subject shot at 2–16 ft; the 16 ft frame is the distortion-free ground truth. Our correction (teal) pulls the distorted curve toward it at every distance.", {
  x: 0.6, y: 5.85, w: 7.7, h: 0.7, fontFace: BF, fontSize: 12.5, italic: true, color: MUTE, margin: 0, valign: "top" });
// right callouts
const c2 = [
  ["93%", "of the distortion gap closed at 6 ft", MINT],
  ["347", "real images measured against GT", TEAL],
  ["Overshoot at 12 ft", "→ motivates per-user calibration", BLUE],
];
c2.forEach((st, i) => {
  const y = 2.0 + i * 1.5;
  s.addShape(p.shapes.RECTANGLE, { x: 8.7, y, w: 4.0, h: 1.3, fill: { color: "FFFFFF" },
    line: { color: "DCE6EF", width: 1 },
    shadow: { type: "outer", color: "000000", blur: 6, offset: 2, angle: 135, opacity: 0.08 } });
  s.addShape(p.shapes.RECTANGLE, { x: 8.7, y, w: 0.12, h: 1.3, fill: { color: st[2] } });
  s.addText(st[0], { x: 8.95, y: y + 0.14, w: 3.6, h: 0.62, fontFace: HF, fontSize: 25, bold: true, color: INK, margin: 0 });
  s.addText(st[1], { x: 8.97, y: y + 0.78, w: 3.6, h: 0.45, fontFace: BF, fontSize: 12.5, color: MUTE, margin: 0, valign: "top" });
});
footer(s, 5);

// =====================================================================
// SLIDE 6 — Calibration insight + ML extension (light, two cards)
// =====================================================================
s = p.addSlide();
s.background = { color: CREAM };
head(s, "What we learned", "Two findings that strengthen the method");
// card 1: person-dependence -> calibration
s.addShape(p.shapes.RECTANGLE, { x: 0.6, y: 1.95, w: 5.9, h: 4.4, fill: { color: "FFFFFF" },
  line: { color: "DCE6EF", width: 1 },
  shadow: { type: "outer", color: "000000", blur: 7, offset: 2, angle: 135, opacity: 0.10 } });
s.addShape(p.shapes.RECTANGLE, { x: 0.6, y: 1.95, w: 5.9, h: 0.12, fill: { color: MINT } });
s.addText("Per-user calibration", { x: 0.9, y: 2.25, w: 5.3, h: 0.45, fontFace: HF, fontSize: 21, bold: true, color: INK, margin: 0 });
s.addText([
  { text: "Face proportions vary by person, so a single population baseline mislabels a naturally-wider face as “distorted.”", options: { breakLine: true, paraSpaceAfter: 12 } },
  { text: "Fix: capture each user’s own neutral once (press ", options: {} },
  { text: "k", options: { bold: true, color: TEAL } },
  { text: "). Correction then engages only when you are genuinely close to the camera, and is a clean no-op otherwise.", options: { breakLine: true, paraSpaceAfter: 12 } },
  { text: "This is the difference between a fixed filter and a measure-then-correct system.", options: { italic: true, color: MUTE } },
], { x: 0.9, y: 2.85, w: 5.3, h: 2.7, fontFace: BF, fontSize: 15, color: INK, valign: "top" });
s.addShape(p.shapes.RECTANGLE, { x: 0.9, y: 5.7, w: 5.3, h: 0.5, fill: { color: "EAF6F1" } });
s.addText("undistorted face  →  α = 1.00  (untouched)", {
  x: 1.0, y: 5.78, w: 5.1, h: 0.35, fontFace: "Consolas", fontSize: 13, bold: true, color: TEAL, margin: 0, valign: "middle" });
// card 2: ML depth
s.addShape(p.shapes.RECTANGLE, { x: 6.8, y: 1.95, w: 5.9, h: 4.4, fill: { color: "FFFFFF" },
  line: { color: "DCE6EF", width: 1 },
  shadow: { type: "outer", color: "000000", blur: 7, offset: 2, angle: 135, opacity: 0.10 } });
s.addShape(p.shapes.RECTANGLE, { x: 6.8, y: 1.95, w: 5.9, h: 0.12, fill: { color: TEAL } });
s.addText("Learned depth swaps in cleanly", { x: 7.1, y: 2.25, w: 5.3, h: 0.45, fontFace: HF, fontSize: 21, bold: true, color: INK, margin: 0 });
s.addText([
  { text: "Same warp, better depth: we drop in ", options: {} },
  { text: "Depth Anything V2", options: { bold: true, color: TEAL } },
  { text: " (NeurIPS 2024) in place of MediaPipe’s z, via one depth-override hook.", options: { breakLine: true, paraSpaceAfter: 12 } },
  { text: "On the 51-subject CMDP benchmark it closes much more of the gap at close range, confirming the pipeline is depth-limited, not warp-limited.", options: { breakLine: true, paraSpaceAfter: 12 } },
  { text: "Future work: run ML depth inside the live loop.", options: { italic: true, color: MUTE } },
], { x: 7.1, y: 2.85, w: 5.3, h: 2.7, fontFace: BF, fontSize: 15, color: INK, valign: "top" });
s.addShape(p.shapes.RECTANGLE, { x: 7.1, y: 5.7, w: 5.3, h: 0.5, fill: { color: "E6F0F5" } });
s.addText([
  { text: "gap closed at 4 ft:   ", options: { color: MUTE } },
  { text: "MediaPipe 33%", options: { bold: true, color: MUTE } },
  { text: "  →  ", options: { color: INK } },
  { text: "ML 78%", options: { bold: true, color: TEAL } },
], { x: 7.2, y: 5.78, w: 5.1, h: 0.35, fontFace: "Consolas", fontSize: 13, margin: 0, valign: "middle" });
footer(s, 6);

// =====================================================================
// SLIDE 7 — Contributions + future work (dark close)
// =====================================================================
s = p.addSlide();
s.background = { color: NAVY };
s.addShape(p.shapes.RECTANGLE, { x: 0, y: 0, w: 0.28, h: H, fill: { color: MINT } });
s.addText("CONTRIBUTIONS", {
  x: 0.9, y: 0.7, w: 11, h: 0.4, fontFace: BF, fontSize: 13, bold: true, color: MINT, charSpacing: 4, margin: 0 });
s.addText("What’s new here", {
  x: 0.9, y: 1.1, w: 11.5, h: 0.8, fontFace: HF, fontSize: 34, bold: true, color: "FFFFFF", margin: 0 });
const contribs = [
  ["First real-time, temporally-stable selfie corrector", "30 FPS live; prior work is offline per-image"],
  ["Dense depth-driven re-projection", "MediaPipe z → per-pixel pinhole warp, artifact-free"],
  ["Validated on ground truth", "51-subject CMDP benchmark, not just demos"],
  ["Adaptive, per-user calibration", "corrects only real distortion; no-op otherwise"],
];
contribs.forEach((c, i) => {
  const y = 2.3 + i * 1.0;
  s.addShape(p.shapes.OVAL, { x: 0.95, y: y + 0.06, w: 0.34, h: 0.34, fill: { color: MINT } });
  s.addText("✓", { x: 0.95, y: y + 0.06, w: 0.34, h: 0.34, fontFace: BF, fontSize: 15, bold: true, color: NAVY, align: "center", valign: "middle", margin: 0 });
  s.addText([
    { text: c[0], options: { bold: true, color: "FFFFFF", breakLine: true } },
    { text: c[1], options: { color: ICE } },
  ], { x: 1.5, y: y - 0.06, w: 8.2, h: 0.78, fontFace: BF, fontSize: 15.5, margin: 0, valign: "middle", lineSpacingMultiple: 1.0 });
});
// future work strip
s.addShape(p.shapes.RECTANGLE, { x: 10.15, y: 2.25, w: 2.55, h: 3.95, fill: { color: TEAL } });
s.addText("NEXT", { x: 10.4, y: 2.5, w: 2.2, h: 0.3, fontFace: BF, fontSize: 13, bold: true, color: NAVY, charSpacing: 3, margin: 0 });
s.addText([
  { text: "ML depth in the live loop", options: { bullet: true, breakLine: true, paraSpaceAfter: 14 } },
  { text: "Broader subject study", options: { bullet: true, breakLine: true, paraSpaceAfter: 14 } },
  { text: "Auto-calibrate on the first frames", options: { bullet: true } },
], { x: 10.4, y: 3.05, w: 2.1, h: 2.9, fontFace: BF, fontSize: 14, color: "FFFFFF", valign: "top" });
s.addText("github.com/Danni281/selfie-perspective-correction", {
  x: 0.9, y: 6.5, w: 11, h: 0.4, fontFace: "Consolas", fontSize: 14, color: MINT, margin: 0 });

p.writeFile({ fileName: "../report/demo_slides.pptx" }).then(f => console.log("wrote", f));
