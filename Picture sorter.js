#!/usr/bin/env node
/*
   select-best-frames.js
   
   Problem it solves:
     A 30-second clip exported at ~30fps gives ~900-1000 individual frame
     images. Most of those are near-duplicates or blurry (motion blur,
     autofocus hunting, camera shake). This script scores every frame and
     automatically keeps only the sharpest, well-exposed, non-duplicate ones -
     so you go from ~1000 images down to a shortlist you can actually look at.
 
   How it decides what's "best":
     1. SHARPNESS  - converts each frame to grayscale and runs a Laplacian
        edge-detection kernel over it. A blurry photo has soft edges (low
        variance in the Laplacian result); a sharp, in-focus photo has a lot
        of high-contrast edges (high variance). This is the same core idea
        professional tools (and OpenCV's `cv2.Laplacian().var()`) use.
     2. EXPOSURE    - average brightness must fall in a sane range (rejects
        frames that are blacked out, e.g. a hand over the lens, or blown-out
        by direct sun).
     3. GROUPING    - frames are processed in the order given (e.g. by
        filename, frame_0001.jpg, frame_0002.jpg...) and split into
        consecutive groups ("windows"). Only the single sharpest, best-exposed
        frame from each window is kept as a candidate. This spreads picks out
        across the whole clip instead of clustering on one lucky moment.
     4. DE-DUPLICATION - a perceptual hash (average hash / aHash) is computed
        per candidate. If two candidates look near-identical (e.g. the
        subject barely moved between windows), the weaker one is dropped.
 
   Requirements:
     Node.js 16+
     npm install sharp
 
   Usage:
     node select-best-frames.js --input ./frames --output ./best-frames
 
   Common options:
     --input <dir>            Folder of extracted frame images (required)
     --output <dir>           Folder to copy the winning frames into (required)
     --group-size <n>         How many consecutive frames form one window
                               (default 20 -- for 30fps video, ~1 pick every
                               2/3 of a second; lower = more picks kept)
     --min-brightness <0-255> Reject frames darker than this (default 25)
     --max-brightness <0-255> Reject frames brighter than this (default 230)
     --dedupe-distance <n>    Hamming distance below which two candidates are
                               considered duplicates (default 6, out of 64)
     --report <path>          Where to write the JSON scoring report
                               (default <output>/report.json)
 
   Example for your case (30s clip, ~1000 frames, want the best ~40-60 shots
   spread evenly through the walk):
     node select-best-frames.js --input ./frames --output ./best-frames --group-size 20
    */

const fs = require("fs");
const path = require("path");
const sharp = require("sharp");

// 
// CLI argument parsing
// 
function parseArgs(argv) {
  const args = {
    input: null,
    output: null,
    groupSize: 20,
    minBrightness: 25,
    maxBrightness: 230,
    dedupeDistance: 6,
    report: null,
  };
  for (let i = 2; i < argv.length; i++) {
    const key = argv[i];
    const val = argv[i + 1];
    switch (key) {
      case "--input": args.input = val; i++; break;
      case "--output": args.output = val; i++; break;
      case "--group-size": args.groupSize = parseInt(val, 10); i++; break;
      case "--min-brightness": args.minBrightness = parseFloat(val); i++; break;
      case "--max-brightness": args.maxBrightness = parseFloat(val); i++; break;
      case "--dedupe-distance": args.dedupeDistance = parseInt(val, 10); i++; break;
      case "--report": args.report = val; i++; break;
      default:
        console.warn(`Unknown argument: ${key}`);
    }
  }
  if (!args.input || !args.output) {
    console.error(
      "Usage: node select-best-frames.js --input <dir> --output <dir> [options]\n" +
      "Run with no arguments to see full option list in the file header."
    );
    process.exit(1);
  }
  return args;
}

//
//  Natural sort so frame_2.jpg comes before frame_10.jpg
//
function naturalCompare(a, b) {
  const chunk = (s) => s.match(/(\d+|\D+)/g) || [];
  const ac = chunk(a), bc = chunk(b);
  for (let i =0; i < Math.max(ac.length, bc.length); i++) {
     const x = ac[i] || "", y = bc[i] || "";
     const xn = parseInt(x, 10), yn = parseInt(y, 10);
     if (!isNaN(xn) && !isNaN(yn)) {
      if (xn !== yn) return xn - yn;
     } else if (x !== y) {
      return x < y ? -1 : 1;
     }
  }
  return 0;
}

//
// Analyze one image: sharpness score, brightness, and a perceptual hash
//
const ANALYZE_WITH = 320; // downscale before scoring - much faster, sharpness
                           // ranking is unaffected by working at this size
const HASH_SIZE = 8;      // 8x8 = 64-bit perceptual hash

async function analyzeImage(filePath) {
  // Grayscale, downscaled raw pixel buffer for sharpness = brightness
  const { data, info } = await sharp(filePath)
  .resize({ width: ANALYZE_WITH, withoutEnlargement: true})
  .grayscale()
  .raw()
  .toBuffer({ resolveWithObject: true });

  const { width, height } = info;
  const sharpness = laplacianVariance(data, width, height);
  const brightness = averageBrightness(data);

  // Small thumbnail for the perceptual hash
  const { data: hashData } = await sharp(filePath)
  .resize(HASH_SIZE, HASH_SIZE, { fit: "fill" })
  .grayscale()
  .raw()
  .toBuffer({ resolveWithObject: true });
  const hash = averageHash(hashData);

  return { sharpness, brightness, hash };
}
  
// Laplacian kernel: 0 1 0
//                   1 -4 1
//                   0 1 0
// High variance in the result = lots of sharp edges = in-focus image.
function laplacianVariance(pixels, width, height) {
  const laplacian = new Float32Array(width * height);
  for (let y = 1; y < height - 1; y++) {
    for (let x = 1; x < width - 1; x++) {
      const idx = y * width + x;
      const value =
        pixels[idx - width] +
        pixels[idx + width] +
        pixels[idx - 1] +
        pixels[idx + 1] -
        4 * pixels[idx];
      laplacian[idx] = value;
    }
  }
  let sum = 0, sumSq = 0, n = 0;
  for (let i = 0; i < laplacian.length; i++) {
    sum += laplacian[i];
    sumSq += laplacian[i] * laplacian[i];
    n++;
  }
  const mean = sum / n;
  return sumSq / n - mean * mean; // variance
}

function averageBrightness(pixels) {
  let sum = 0;
  for (let i = 0; i < pixels.length; i++) sum += pixels[i];
  return sum / pixels.length;
}

// 64-bit average hash; 1 bit per pixel, set if brighter than the mean
function averageHash(pixels) {
  let sum = 0;
  for (let i = 0; i < pixels.length; i++) sum += pixels[i];
  const mean = sum / pixels.length;
  let hash = 0n;
  for (let i = 0; i < pixels.length; i++) {
    hash = (hash << 1n) | (pixels[i] > mean ? 1n : 0n);
  }
  return hash;
}

function hammingDistance(a, b) {
  let x = a ^ b;
  let dist = 0;
  while (x > 0n) {
    dist += Number(x & 1n);
    x >>= 1n;
  }
  return dist;
}

//
// Main pipeline
//
async function main() {
  const args = parseArgs(process.argv);

  const imageExtensions = new Set([".jpg", ".jpeg", ".png", ".webp"]);
  const files = fs
    .readdirSync(args.input)
    .filter((f) => imageExtensions.has(path.extname(f).toLowerCase()))
    .sort(naturalCompare);

  if (files.length === 0) {
    console.error(`No images found in ${args.input}`);
    process.exit(1);
  }

  console.log(`Found ${files.length} frames. Analyzing...`);

  const analyzed = [];
  for (let i = 0; i < files.length; i++) {
    const filePath = path.join(args.input, files[i]);
    try {
      const stats = await analyzeImage(filePath);
      analyzed.push({ file: files[i], ...stats });
    } catch (err) {
      console.warn(`Skipping ${files[i]} (couldn't read): ${err.message}`);
    }
    if ((i + 1) % 100 === 0 || i === files.length - 1) {
      process.stdout.write(`\r  scored ${i + 1}/${files.length}`);
    }
  }

  console.log();

  // --- Step 1: pick the best frame from each consecutive window -----------
  const candidates = [];
  for (let i = 0; i < analyzed.length; i += args.groupSize) {
    const window = analyzed.slice(i, i + args.groupSize);
    const wellExposed = window.filter(
      (f) => f.brightness >= args.minBrightness && f.brightness <= args.maxBrightness
    );
    const pool = wellExposed.length > 0 ? wellExposed : window;
    const best = pool.reduce((a, b) => (b.sharpness > a.sharpness ? b : a));
    candidates.push(best);
  }

  // --- Step 2: drop near-duplicates among the candidates -------------------
  const kept = [];
  for (const candidate of candidates) {
    const dupIndex = kept.findIndex(
      (k) => hammingDistance(k.hash, candidate.hash) <= args.dedupeDistance
    );
    if (dupIndex === -1) {
      kept.push(candidate);
    } else if (candidate.sharpness > kept[dupIndex].sharpness) {
      kept[dupIndex] = candidate;
    }
  }

  // --- Step 3: copy winners to the output folder ----------------------------
  fs.mkdirSync(args.output, { recursive: true });
  for (const winner of kept) {
    fs.copyFileSync(
      path.join(args.input, winner.file),
      path.join(args.output, winner.file)
    );
  }

  // --- Step 4: write a JSON report ------------------------------------------
  const reportPath = args.report || path.join(args.output, "report.json");
  const report = {
    totalFramesScanned: files.length,
    windowsEvaluated: candidates.length,
    framesKeptAfterDedupe: kept.length,
    settings: {
      groupSize: args.groupSize,
      minBrightness: args.minBrightness,
      maxBrightness: args.maxBrightness,
      dedupeDistance: args.dedupeDistance,
    },
    selected: kept
      .sort((a, b) => naturalCompare(a.file, b.file))
      .map((f) => ({
        file: f.file,
        sharpness: Math.round(f.sharpness * 100) / 100,
        brightness: Math.round(f.brightness * 100) / 100,
      })),
  };
  fs.writeFileSync(reportPath, JSON.stringify(report, null, 2));

  console.log(`\nDone.`);
  console.log(`  Scanned:       ${files.length} frames`);
  console.log(`  Windows:       ${candidates.length} (one best pick per window)`);
  console.log(`  Kept:          ${kept.length} after removing near-duplicates`);
  console.log(`  Copied to:     ${args.output}`);
  console.log(`  Report:        ${reportPath}`);
}

main().catch((err) => {
  console.error("Fatal error:", err);
  process.exit(1);
});
