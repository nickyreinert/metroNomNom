// Estimates tempo from mono PCM: RMS envelope -> rectified rise (onset strength)
// -> autocorrelation over the beat-period range -> log-Gaussian prior around priorBpm.
// Returns { bpm, confidence, pulsiness } or null if the input is too short or silent.
export function estimateTempo(samples, sampleRate, { minBpm = 50, maxBpm = 200, priorBpm = 120 } = {}) {
  const hop = Math.round(sampleRate * 0.005);
  const frameSec = hop / sampleRate;
  const n = Math.floor(samples.length / hop);

  const env = new Float32Array(n);
  for (let f = 0; f < n; f++) {
    let sum = 0;
    for (let i = f * hop, end = i + hop; i < end; i++) sum += samples[i] * samples[i];
    env[f] = Math.sqrt(sum / hop);
  }

  const onset = new Float32Array(n);
  let mean = 0;
  let envMean = 0;
  for (let f = 1; f < n; f++) {
    onset[f] = Math.max(0, env[f] - env[f - 1]);
    mean += onset[f];
    envMean += env[f];
  }
  mean /= n;
  envMean /= n;
  for (let f = 0; f < n; f++) onset[f] -= mean;

  const minLag = Math.floor(60 / maxBpm / frameSec);
  const maxLag = Math.ceil(60 / minBpm / frameSec);
  if (n < maxLag * 2) return null;

  let ac0 = 0;
  for (let f = 0; f < n; f++) ac0 += onset[f] * onset[f];
  if (ac0 === 0) return null;

  // r[lag]: autocorrelation per overlapping sample, relative to the zero-lag value.
  const r = new Float32Array(maxLag + 2);
  for (let lag = minLag - 1; lag <= maxLag + 1; lag++) {
    let sum = 0;
    for (let f = 0; f + lag < n; f++) sum += onset[f] * onset[f + lag];
    r[lag] = (sum / (n - lag)) / (ac0 / n);
  }

  let best = minLag;
  let bestScore = -Infinity;
  for (let lag = minLag; lag <= maxLag; lag++) {
    const bpm = 60 / (lag * frameSec);
    const octaves = Math.log2(bpm / priorBpm);
    const score = r[lag] * Math.exp(-0.5 * octaves * octaves);
    if (score > bestScore) {
      bestScore = score;
      best = lag;
    }
  }

  // Parabolic interpolation for sub-frame lag resolution.
  const denom = r[best - 1] - 2 * r[best] + r[best + 1];
  const delta = denom < 0
    ? Math.max(-1, Math.min(1, (0.5 * (r[best - 1] - r[best + 1])) / denom))
    : 0;

  // pulsiness: mean envelope rise relative to mean level; near zero for steady tones.
  return { bpm: 60 / ((best + delta) * frameSec), confidence: r[best], pulsiness: mean / envMean };
}
