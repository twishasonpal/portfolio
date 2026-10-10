/**
 * nn_inference.js - Browser Neural Network Inference Engine
 * Executes pure client-side Transformer language model inference directly
 * in the visitor's browser without any backend server or external API.
 */

window.BrowserTransformerLM = (function() {
  'use strict';

  let config = null;
  let tokenizer = null;
  let weights = null;
  let isLoaded = false;
  let loadStats = {
    downloadTimeMs: 0,
    initTimeMs: 0,
    paramCount: 0,
    weightSizeKB: 0
  };

  /**
   * Helper Math Operations on Float32Array / Arrays
   */
  function gelu(x) {
    // 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    return 0.5 * x * (1.0 + Math.tanh(Math.sqrt(2.0 / Math.PI) * (x + 0.044715 * Math.pow(x, 3))));
  }

  function softmax(arr) {
    let max = -Infinity;
    for (let i = 0; i < arr.length; i++) {
      if (arr[i] > max) max = arr[i];
    }
    let sum = 0;
    const exp = new Float32Array(arr.length);
    for (let i = 0; i < arr.length; i++) {
      exp[i] = Math.exp(arr[i] - max);
      sum += exp[i];
    }
    for (let i = 0; i < arr.length; i++) {
      exp[i] /= (sum || 1);
    }
    return exp;
  }

  function layerNorm(vec, weight, bias, eps = 1e-5) {
    const d = vec.length;
    let mean = 0;
    for (let i = 0; i < d; i++) mean += vec[i];
    mean /= d;

    let variance = 0;
    for (let i = 0; i < d; i++) {
      const diff = vec[i] - mean;
      variance += diff * diff;
    }
    variance /= d;
    const invStd = 1.0 / Math.sqrt(variance + eps);

    const out = new Float32Array(d);
    for (let i = 0; i < d; i++) {
      out[i] = ((vec[i] - mean) * invStd) * (weight ? weight[i] : 1.0) + (bias ? bias[i] : 0.0);
    }
    return out;
  }

  function matVecMul(matrix, vec, bias) {
    // matrix: [rows][cols], vec: [cols] -> out: [rows]
    const rows = matrix.length;
    const cols = vec.length;
    const out = new Float32Array(rows);
    for (let r = 0; r < rows; r++) {
      let sum = bias ? bias[r] : 0;
      const rowArr = matrix[r];
      for (let c = 0; c < cols; c++) {
        sum += rowArr[c] * vec[c];
      }
      out[r] = sum;
    }
    return out;
  }

  /**
   * Positional Encoding generator
   */
  function getPosEncoding(pos, d_model) {
    const pe = new Float32Array(d_model);
    for (let i = 0; i < d_model; i += 2) {
      const divTerm = Math.exp(i * (-Math.log(10000.0) / d_model));
      pe[i] = Math.sin(pos * divTerm);
      if (i + 1 < d_model) {
        pe[i + 1] = Math.cos(pos * divTerm);
      }
    }
    return pe;
  }

  /**
   * Load Model, Tokenizer, and Weights from model/ directory
   */
  async function loadModel(modelBaseUrl = 'model/') {
    if (isLoaded) return true;

    const startDownload = performance.now();
    try {
      const [configRes, tokenizerRes, weightsRes] = await Promise.all([
        fetch(`${modelBaseUrl}config.json`),
        fetch(`${modelBaseUrl}tokenizer.json`),
        fetch(`${modelBaseUrl}weights.json`)
      ]);

      if (!configRes.ok || !tokenizerRes.ok || !weightsRes.ok) {
        throw new Error('Could not fetch one or more model files');
      }

      config = await configRes.json();
      tokenizer = await tokenizerRes.json();
      weights = await weightsRes.json();

      loadStats.downloadTimeMs = Math.round(performance.now() - startDownload);
      const startInit = performance.now();

      // Invert tokenizer vocab for decoding
      tokenizer.id2token = {};
      for (const [token, id] of Object.entries(tokenizer.vocab)) {
        tokenizer.id2token[id] = token;
      }

      // Count parameters
      let totalParams = 0;
      for (const tensor of Object.values(weights)) {
        if (Array.isArray(tensor)) {
          if (Array.isArray(tensor[0])) {
            totalParams += tensor.length * tensor[0].length;
          } else {
            totalParams += tensor.length;
          }
        }
      }
      loadStats.paramCount = totalParams;
      loadStats.initTimeMs = Math.round(performance.now() - startInit);
      isLoaded = true;

      console.log(`[BrowserTransformerLM] Model loaded in ${loadStats.downloadTimeMs + loadStats.initTimeMs}ms (${totalParams.toLocaleString()} parameters)`);
      return true;
    } catch (err) {
      console.warn('[BrowserTransformerLM] Could not load exported model files:', err.message);
      return false;
    }
  }

  /**
   * Tokenizer Encoding & Decoding
   */
  function encode(text, addSpecial = true) {
    if (!tokenizer) return [];
    const unkId = tokenizer.vocab["<unk>"] || 1;
    const bosId = tokenizer.vocab["<bos>"] || 2;
    const eosId = tokenizer.vocab["<eos>"] || 3;

    const ids = [];
    if (addSpecial) ids.push(bosId);
    for (const char of text) {
      ids.push(tokenizer.vocab[char] !== undefined ? tokenizer.vocab[char] : unkId);
    }
    if (addSpecial) ids.push(eosId);
    return ids;
  }

  function decode(ids, skipSpecial = true) {
    if (!tokenizer) return "";
    const specialSet = new Set(tokenizer.special_tokens || ["<pad>", "<unk>", "<bos>", "<eos>"]);
    let out = "";
    for (const id of ids) {
      const tok = tokenizer.id2token[id] || "";
      if (skipSpecial && specialSet.has(tok)) continue;
      out += tok;
    }
    return out;
  }

  /**
   * Forward Pass for single token sequence: idx -> logits for last token
   */
  function forward(tokenIds) {
    const seqLen = tokenIds.length;
    const dModel = config.d_model;
    const tokEmb = weights["tok_emb.weight"]; // [vocab_size][d_model]

    // 1. Embedding + Positional Encoding for all positions
    const hiddenStates = [];
    for (let t = 0; t < seqLen; t++) {
      const tid = tokenIds[t];
      const emb = tokEmb[tid] || new Float32Array(dModel);
      const pe = getPosEncoding(t, dModel);
      const h = new Float32Array(dModel);
      const scale = Math.sqrt(dModel);
      for (let i = 0; i < dModel; i++) {
        h[i] = emb[i] * scale + pe[i];
      }
      hiddenStates.push(h);
    }

    // 2. Transformer Blocks
    const numLayers = config.n_layers || 2;
    const numHeads = config.n_heads || 2;
    const headDim = dModel / numHeads;

    for (let l = 0; l < numLayers; l++) {
      const prefix = `blocks.${l}.`;
      const ln1_w = weights[`${prefix}ln1.weight`];
      const ln1_b = weights[`${prefix}ln1.bias`];
      const q_w = weights[`${prefix}attn.q_proj.weight`];
      const q_b = weights[`${prefix}attn.q_proj.bias`];
      const k_w = weights[`${prefix}attn.k_proj.weight`];
      const k_b = weights[`${prefix}attn.k_proj.bias`];
      const v_w = weights[`${prefix}attn.v_proj.weight`];
      const v_b = weights[`${prefix}attn.v_proj.bias`];
      const out_w = weights[`${prefix}attn.out_proj.weight`];
      const out_b = weights[`${prefix}attn.out_proj.bias`];

      const ln2_w = weights[`${prefix}ln2.weight`];
      const ln2_b = weights[`${prefix}ln2.bias`];
      const fc1_w = weights[`${prefix}ffn.fc1.weight`];
      const fc1_b = weights[`${prefix}ffn.fc1.bias`];
      const fc2_w = weights[`${prefix}ffn.fc2.weight`];
      const fc2_b = weights[`${prefix}ffn.fc2.bias`];

      // Multi-Head Attention with Causal Mask
      const normH = hiddenStates.map(h => layerNorm(h, ln1_w, ln1_b));
      const Q = normH.map(h => matVecMul(q_w, h, q_b));
      const K = normH.map(h => matVecMul(k_w, h, k_b));
      const V = normH.map(h => matVecMul(v_w, h, v_b));

      const attnOuts = [];
      for (let t = 0; t < seqLen; t++) {
        // Compute attention for position t over all positions <= t
        const headOutputs = new Float32Array(dModel);
        
        for (let head = 0; head < numHeads; head++) {
          const hOffset = head * headDim;
          const scores = new Float32Array(t + 1);

          for (let s = 0; s <= t; s++) {
            let dot = 0;
            for (let d = 0; d < headDim; d++) {
              dot += Q[t][hOffset + d] * K[s][hOffset + d];
            }
            scores[s] = dot / Math.sqrt(headDim);
          }

          const probs = softmax(scores);

          // Weighted sum of V
          for (let d = 0; d < headDim; d++) {
            let sum = 0;
            for (let s = 0; s <= t; s++) {
              sum += probs[s] * V[s][hOffset + d];
            }
            headOutputs[hOffset + d] = sum;
          }
        }

        const projAttn = matVecMul(out_w, headOutputs, out_b);
        const resAttn = new Float32Array(dModel);
        for (let i = 0; i < dModel; i++) resAttn[i] = hiddenStates[t][i] + projAttn[i];
        attnOuts.push(resAttn);
      }

      // Feed-Forward Network
      for (let t = 0; t < seqLen; t++) {
        const normH2 = layerNorm(attnOuts[t], ln2_w, ln2_b);
        const fc1 = matVecMul(fc1_w, normH2, fc1_b);
        for (let i = 0; i < fc1.length; i++) fc1[i] = gelu(fc1[i]);
        const fc2 = matVecMul(fc2_w, fc1, fc2_b);

        const resFFN = new Float32Array(dModel);
        for (let i = 0; i < dModel; i++) resFFN[i] = attnOuts[t][i] + fc2[i];
        hiddenStates[t] = resFFN;
      }
    }

    // 3. Final LayerNorm for last position
    const lastHidden = layerNorm(hiddenStates[seqLen - 1], weights["ln_f.weight"], weights["ln_f.bias"]);

    // 4. LM Head projection (tied weights with tok_emb or explicit lm_head.weight)
    const lmHeadWeight = weights["lm_head.weight"] || weights["tok_emb.weight"];
    const logits = matVecMul(lmHeadWeight, lastHidden, null);
    return logits;
  }

  /**
   * Autoregressive Text Generation
   */
  async function generate(promptText, maxNewTokens = 35, temperature = 0.7) {
    if (!isLoaded) {
      const ok = await loadModel();
      if (!ok) return null;
    }

    const t0 = performance.now();
    const tokenIds = encode(promptText, false);
    const eosId = tokenizer.vocab["<eos>"] || 3;
    const maxSeqLen = config.max_seq_len || 64;

    const generatedIds = [...tokenIds];

    for (let step = 0; step < maxNewTokens; step++) {
      const windowTokens = generatedIds.slice(-maxSeqLen);
      const logits = forward(windowTokens);

      // Temperature scaling
      const scaledLogits = new Float32Array(logits.length);
      for (let i = 0; i < logits.length; i++) {
        scaledLogits[i] = logits[i] / Math.max(0.01, temperature);
      }

      const probs = softmax(scaledLogits);

      // Sample next token (multinomial / top-p or greedy if temp very low)
      let nextId = 0;
      if (temperature < 0.2) {
        let maxP = -1;
        for (let i = 0; i < probs.length; i++) {
          if (probs[i] > maxP) { maxP = probs[i]; nextId = i; }
        }
      } else {
        const rand = Math.random();
        let cum = 0;
        for (let i = 0; i < probs.length; i++) {
          cum += probs[i];
          if (rand <= cum) { nextId = i; break; }
        }
      }

      if (nextId === eosId) break;
      generatedIds.push(nextId);
    }

    const elapsedMs = Math.round(performance.now() - t0);
    const outputText = decode(generatedIds.slice(tokenIds.length));

    return {
      text: outputText.trim(),
      tokensGenerated: generatedIds.length - tokenIds.length,
      elapsedMs: elapsedMs
    };
  }

  return {
    loadModel: loadModel,
    isLoaded: () => isLoaded,
    getStats: () => loadStats,
    encode: encode,
    decode: decode,
    generate: generate
  };
})();
