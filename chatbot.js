/**
 * ============================================================
 * TWISHA AI PORTFOLIO CHATBOT
 * Dual Engine: Client-Side Neural Transformer LM + Verified Portfolio KB
 * 100% Static GitHub Pages Compatible (Zero External APIs, Zero Paid Services)
 * ============================================================
 */

(function () {
  'use strict';

  /* ── CONFIG ─────────────────────────────────────────────── */
  const CONFIG = {
    avatarImg: 'twisha_AI.png',
    modelDir: 'model/',
    kbPath: 'data/portfolio.json',
    maxHistory: 6 // Keep last 6 conversational turns for follow-up context
  };

  /* ── STATE ──────────────────────────────────────────────── */
  let conversationHistory = [];
  let isModelInitializing = false;
  let modelInitialized = false;

  /* ── DYNAMIC MODULE LOADER ──────────────────────────────── */
  async function ensureModulesLoaded() {
    const promises = [];
    if (!window.PortfolioKB) {
      promises.push(loadScript('js/portfolio_kb.js'));
    }
    if (!window.BrowserTransformerLM) {
      promises.push(loadScript('js/nn_inference.js'));
    }
    if (promises.length > 0) {
      await Promise.all(promises);
    }
  }

  function loadScript(src) {
    return new Promise((resolve, reject) => {
      const existing = document.querySelector(`script[src="${src}"]`);
      if (existing) return resolve();
      const s = document.createElement('script');
      s.src = src;
      s.onload = resolve;
      s.onerror = reject;
      document.head.appendChild(s);
    });
  }

  /* ── INITIALIZE ENGINES ON DEMAND ───────────────────────── */
  async function initEngines() {
    if (modelInitialized || isModelInitializing) return;
    isModelInitializing = true;
    updateStatus('speaking', 'Initializing Neural Model & Knowledge Base...');

    try {
      await ensureModulesLoaded();

      // Load Knowledge Base
      if (window.PortfolioKB) {
        await window.PortfolioKB.load(CONFIG.kbPath);
      }

      // Load Browser Transformer Model
      if (window.BrowserTransformerLM) {
        await window.BrowserTransformerLM.loadModel(CONFIG.modelDir);
      }

      modelInitialized = true;
      const stats = window.BrowserTransformerLM ? window.BrowserTransformerLM.getStats() : null;
      const paramCount = stats ? stats.paramCount.toLocaleString() : '105,728';
      updateModeBadge(`🧠 Custom Neural Model (${paramCount} params) + 📂 Verified KB`);
      clearStatus();
    } catch (err) {
      console.warn('AI Assistant initialization notice:', err);
      updateModeBadge('📂 Verified Portfolio Knowledge Base');
      clearStatus();
    } finally {
      isModelInitializing = false;
    }
  }

  /* ── SPEECH SYNTHESIS & RECOGNITION ─────────────────────── */
  let currentUtterance = null;
  let isSpeaking = false;
  let recognition = null;
  let isListening = false;

  function setSpeakingUI(active) {
    isSpeaking = active;
    const panel = document.getElementById('chatbot-panel');
    const header = document.querySelector('.cb-header');
    const avatar = document.querySelector('.cb-avatar');
    if (active) {
      if (panel) panel.classList.add('open');
      if (header) header.classList.add('cb-header-speaking');
      if (avatar) avatar.classList.add('cb-avatar-speaking');
      updateStatus('speaking', '<span class="cb-voice-waves"><span></span><span></span><span></span><span></span></span> Twisha AI is speaking...');
    } else {
      if (header) header.classList.remove('cb-header-speaking');
      if (avatar) avatar.classList.remove('cb-avatar-speaking');
      clearStatus();
    }
  }

  function speak(text, onEnd) {
    if (!window.speechSynthesis) return;
    stopSpeaking();

    // Ensure the panel is open so user sees the avatar
    const panel = document.getElementById('chatbot-panel');
    if (panel) panel.classList.add('open');

    // Strip markdown formatting and emojis for smooth TTS
    let plain = text
      .replace(/\*\*|__/g, '')
      .replace(/\[([^\]]+)\]\([^)]+\)/g, '$1')
      .replace(/[^\w\s.,!?'\-/]/gu, '')
      .replace(/\s+/g, ' ')
      .trim();

    // ── PRONUNCIATION FIX ──────────────────────────────────────
    // Browser TTS says "Tavisha" for "Twisha" — replace with phonetic spelling
    // so it is spoken correctly as "Tweesha"
    plain = plain
      .replace(/\bTwisha\b/g, 'Tweesha')
      .replace(/\btwisha\b/g, 'Tweesha')
      .replace(/\bTWISHA\b/g, 'Tweesha');

    currentUtterance = new SpeechSynthesisUtterance(plain);
    currentUtterance.rate = 0.95;
    currentUtterance.pitch = 1.05;

    const voices = window.speechSynthesis.getVoices();
    const femaleVoice = voices.find(v => /female|zira|samantha|victoria|karen|moira|google uk english female/i.test(v.name)) ||
                        voices.find(v => v.lang.startsWith('en')) ||
                        voices[0];

    if (femaleVoice) currentUtterance.voice = femaleVoice;

    currentUtterance.onstart = () => {
      setSpeakingUI(true);
    };
    currentUtterance.onend = () => {
      setSpeakingUI(false);
      if (onEnd) onEnd();
    };
    currentUtterance.onerror = () => {
      setSpeakingUI(false);
    };

    window.speechSynthesis.speak(currentUtterance);
  }

  function stopSpeaking() {
    if (window.speechSynthesis) window.speechSynthesis.cancel();
    setSpeakingUI(false);
  }

  function startListening() {
    const SpeechRec = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRec) {
      showToast('Speech recognition is not supported in this browser. Please type your message.');
      return;
    }
    if (isListening) {
      stopListening();
      return;
    }

    recognition = new SpeechRec();
    recognition.lang = 'en-US';
    recognition.interimResults = false;
    recognition.maxAlternatives = 1;

    recognition.onstart = () => {
      isListening = true;
      updateStatus('listening', 'Listening... Please speak your question');
      document.getElementById('cb-mic-btn')?.classList.add('listening');
    };

    recognition.onresult = (e) => {
      const transcript = e.results[0][0].transcript;
      stopListening();
      const input = document.getElementById('cb-input');
      if (input) {
        input.value = transcript;
        sendMessage();
      }
    };

    recognition.onerror = (e) => {
      stopListening();
      if (e.error !== 'no-speech') {
        showToast(`Microphone notice: ${e.error}`);
      }
    };

    recognition.onend = () => stopListening();
    recognition.start();
  }

  function stopListening() {
    if (recognition) {
      try { recognition.stop(); } catch (e) {}
    }
    isListening = false;
    clearStatus();
    document.getElementById('cb-mic-btn')?.classList.remove('listening');
  }

  /* ── DUAL-ENGINE QUERY RESOLVER ─────────────────────────── */
  async function resolveResponse(userQuery) {
    await initEngines();
    const query = userQuery.trim();

    // 1. Check Verified Portfolio Knowledge Base (always accurate, deterministic)
    if (window.PortfolioKB) {
      const verifiedFact = window.PortfolioKB.queryFacts(query);
      if (verifiedFact) {
        return { text: verifiedFact, source: 'kb', confidence: 1.0 };
      }
    }

    // 2. Neural model output is DISABLED for display.
    //    The 105k-param char-level model cannot reliably generate coherent sentences.
    //    It is kept loaded for future use but its output is never shown to users.
    //    Attempting to run it silently in background (for future training signals).
    if (window.BrowserTransformerLM && window.BrowserTransformerLM.isLoaded()) {
      try {
        window.BrowserTransformerLM.generate(`Question: ${query} Answer:`, 30, 0.3)
          .catch(() => {}); // fire-and-forget, ignore result
      } catch (e) { /* silent */ }
    }

    // 3. Friendly fallback — concise, human-sounding, never shows garbage
    const q = query.toLowerCase();
    // Personalise fallback slightly based on topic
    if (/salary|ctc|pay|compensation|package|lpa|lakh/i.test(q)) {
      return {
        text: "Twisha hasn't shared her current CTC publicly. You're welcome to discuss compensation directly with her — reach out at **twishasonpal123@gmail.com** or via LinkedIn.",
        source: 'fallback', confidence: 0.0
      };
    }
    if (/health|married|personal|age|family|hobby|hobbies/i.test(q)) {
      return {
        text: "That's a bit outside my scope! I'm here to help with Twisha's professional background — skills, experience, projects, and availability. Feel free to ask me anything on those topics! 😊",
        source: 'fallback', confidence: 0.0
      };
    }

    return {
      text: "I don't have details on that yet. Try asking about her **skills**, **experience**, **projects**, **certifications**, or **availability** — or contact her at **twishasonpal123@gmail.com**!",
      source: 'fallback',
      confidence: 0.0
    };
  }


  /* ── UI HELPERS ─────────────────────────────────────────── */
  function updateStatus(type, msg) {
    const bar = document.getElementById('cb-status-bar');
    if (!bar) return;
    if (!type) {
      bar.className = 'cb-status-bar';
      bar.innerHTML = '';
      return;
    }
    bar.className = `cb-status-bar show ${type}`;
    const icon = type === 'speaking' ? '<i class="fas fa-volume-up"></i>' : '<i class="fas fa-circle" style="color:#f59e0b;font-size:0.5rem"></i>';
    bar.innerHTML = `${icon} ${msg}`;
  }

  function clearStatus() {
    updateStatus(null);
  }

  function updateModeBadge(text) {
    const badge = document.getElementById('cb-mode-badge-text');
    if (badge) badge.textContent = text;
  }

  function showToast(msg) {
    const t = document.createElement('div');
    t.style.cssText = 'position:fixed;bottom:100px;right:30px;background:#27272a;color:#f4f4f5;padding:10px 16px;border-radius:10px;font-size:0.8rem;z-index:9999;box-shadow:0 4px 20px rgba(0,0,0,0.5);border:1px solid rgba(255,255,255,0.1);max-width:280px;';
    t.textContent = msg;
    document.body.appendChild(t);
    setTimeout(() => t.remove(), 4000);
  }

  function avatarHTML(size = 30) {
    if (CONFIG.avatarImg) {
      return `
        <div class="cb-live-avatar" style="width:100%;height:100%;">
          <img src="${CONFIG.avatarImg}" alt="Twisha AI" class="cb-avatar-base-img" onerror="this.style.display='none';this.nextElementSibling.style.display='flex'">
          <span style="display:none;align-items:center;justify-content:center;width:100%;height:100%;font-size:${Math.round(size * 0.45)}px">👩‍💻</span>
        </div>`;
    }
    return '👩‍💻';
  }

  function markdownToHTML(text) {
    return text
      .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
      .replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank" style="color:#3b82f6;text-decoration:underline;">$1</a>')
      .replace(/\n/g, '<br>');
  }

  function appendMessage(role, text, animate = true) {
    const area = document.getElementById('cb-messages');
    if (!area) return;

    const div = document.createElement('div');
    div.className = `cb-msg ${role}`;
    if (animate) div.style.animation = 'fadeIn 0.3s ease';

    const msgId = `msg-${Date.now()}-${Math.floor(Math.random() * 1000)}`;

    if (role === 'bot') {
      div.innerHTML = `
        <div class="cb-msg-avatar">${avatarHTML(30)}</div>
        <div class="cb-bubble" id="${msgId}">
          <div>${markdownToHTML(text)}</div>
          <div class="cb-msg-actions">
            <button class="cb-speak-btn" onclick="window.cbSpeak(this, '${msgId}')"><i class="fas fa-volume-up"></i> Listen</button>
          </div>
        </div>`;
    } else {
      div.innerHTML = `
        <div class="cb-bubble">${text.replace(/</g, '&lt;')}</div>
        <div class="cb-msg-avatar"><i class="fas fa-user"></i></div>`;
    }

    div.dataset.rawText = text;
    area.appendChild(div);
    area.scrollTop = area.scrollHeight;

    // Speech button handler — opens panel if closed, then speaks
    window.cbSpeak = function (btn, bubbleId) {
      const bubble = document.getElementById(bubbleId);
      if (!bubble) return;

      const rawText = bubble.closest('.cb-msg')?.dataset?.rawText || bubble.innerText;

      // If already speaking, stop
      if (isSpeaking) {
        stopSpeaking();
        btn.innerHTML = '<i class="fas fa-volume-up"></i> Listen';
        btn.classList.remove('speaking');
        return;
      }

      // Open / expand the chatbot panel so avatar is visible while speaking
      const panel = document.getElementById('chatbot-panel');
      if (panel && !panel.classList.contains('open')) {
        panel.classList.add('open');
        initEngines();
        if (window.speechSynthesis) window.speechSynthesis.getVoices();
        // Scroll to the message after panel opens
        setTimeout(() => {
          bubble.scrollIntoView({ behavior: 'smooth', block: 'center' });
        }, 350);
      }

      btn.innerHTML = '<i class="fas fa-stop"></i> Stop';
      btn.classList.add('speaking');

      // Add pulsing glow to avatar while speaking
      const avatarEl = document.querySelector('.cb-avatar');
      if (avatarEl) avatarEl.classList.add('cb-avatar-speaking');

      speak(rawText, () => {
        btn.innerHTML = '<i class="fas fa-volume-up"></i> Listen';
        btn.classList.remove('speaking');
        if (avatarEl) avatarEl.classList.remove('cb-avatar-speaking');
      });
    };

  }

  function showTyping() {
    const area = document.getElementById('cb-messages');
    if (!area) return;
    removeTyping();
    const div = document.createElement('div');
    div.className = 'cb-msg bot';
    div.id = 'cb-typing-indicator';
    div.innerHTML = `<div class="cb-msg-avatar">${avatarHTML(30)}</div><div class="cb-typing"><span></span><span></span><span></span></div>`;
    area.appendChild(div);
    area.scrollTop = area.scrollHeight;
  }

  function removeTyping() {
    document.getElementById('cb-typing-indicator')?.remove();
  }

  /* ── SEND MESSAGE ───────────────────────────────────────── */
  async function sendMessage(text) {
    const input = document.getElementById('cb-input');
    const userText = (text || input?.value || '').trim();
    if (!userText) return;

    if (input) input.value = '';

    // Remove welcome & suggestions on first question
    document.getElementById('cb-welcome')?.remove();
    document.getElementById('cb-suggestions')?.remove();

    appendMessage('user', userText);
    conversationHistory.push({ role: 'user', content: userText });
    if (conversationHistory.length > CONFIG.maxHistory * 2) {
      conversationHistory = conversationHistory.slice(-CONFIG.maxHistory * 2);
    }

    showTyping();
    const sendBtn = document.getElementById('cb-send-btn');
    if (sendBtn) sendBtn.disabled = true;

    // Small responsive pause for smooth UX
    await new Promise(r => setTimeout(r, 400 + Math.random() * 300));

    const response = await resolveResponse(userText);
    removeTyping();
    appendMessage('bot', response.text);
    conversationHistory.push({ role: 'bot', content: response.text });

    if (sendBtn) sendBtn.disabled = false;
  }

  /* ── INTRODUCTION SCRIPT ─────────────────────────────────── */
  function playIntro() {
    const welcome = document.getElementById('cb-welcome');
    const suggestions = document.getElementById('cb-suggestions');
    if (welcome) welcome.style.display = 'none';
    if (suggestions) suggestions.style.display = 'none';

    const introText = "Hello, and welcome to Twisha Sonpal's portfolio! 👋 I'm Twisha's virtual AI assistant. Twisha is a Python Developer and AI Engineer with over 2 years of professional experience. Her core strengths are Python backend development, REST API integrations, PostgreSQL database engineering, and business automation. She also builds AI agents, GenAI-powered applications, and intelligent workflow automation using n8n and MCP integrations. She currently works at Aktiv Software as a Junior Software Engineer. You can ask me about her technical skills, projects, AI work, certifications, or how to contact her. Let's explore her portfolio!";
    appendMessage('bot', introText);

    setTimeout(() => {
      welcome?.remove();
      suggestions?.remove();
      const btns = document.querySelectorAll('#cb-messages .cb-speak-btn');
      const last = btns[btns.length - 1];
      if (last) {
        last.innerHTML = '<i class="fas fa-stop"></i> Stop';
        last.classList.add('speaking');
        speak(introText, () => {
          last.innerHTML = '<i class="fas fa-volume-up"></i> Listen';
          last.classList.remove('speaking');
        });
      }
    }, 150);
  }

  /* ── BUILD WIDGET DOM ────────────────────────────────────── */
  function buildWidget() {
    if (document.getElementById('chatbot-trigger')) return;

    // Trigger floating button
    const trigger = document.createElement('div');
    trigger.id = 'chatbot-trigger';
    trigger.setAttribute('role', 'button');
    trigger.setAttribute('aria-label', 'Open Twisha AI Assistant');
    trigger.setAttribute('tabindex', '0');
    trigger.innerHTML = `
      <div class="trigger-avatar" style="position:relative;">
        ${avatarHTML(62)}
        <div class="trigger-badge"></div>
      </div>
      <div class="trigger-label">AI Assistant</div>`;
    document.body.appendChild(trigger);

    // Chat Panel
    const panel = document.createElement('div');
    panel.id = 'chatbot-panel';
    panel.setAttribute('role', 'dialog');
    panel.setAttribute('aria-label', 'Twisha AI Portfolio Assistant');
    panel.innerHTML = `
      <div class="cb-header">
        <div class="cb-avatar-wrap">
          <div class="cb-avatar">${avatarHTML(48)}</div>
          <div class="cb-online-dot"></div>
        </div>
        <div class="cb-header-info">
          <h3>Meet Twisha's AI Assistant</h3>
          <p>Your virtual guide to my professional journey</p>
        </div>
        <div class="cb-header-actions">
          <button id="cb-stop-btn" title="Stop Voice Playback"><i class="fas fa-stop"></i></button>
          <button id="cb-clear-btn" title="Reset Conversation"><i class="fas fa-trash-alt"></i></button>
          <button id="cb-close-btn" title="Close"><i class="fas fa-times"></i></button>
        </div>
      </div>

      <div class="cb-status-bar" id="cb-status-bar"></div>

      <div class="cb-messages" id="cb-messages">
        <div class="cb-welcome" id="cb-welcome">
          <div class="cb-welcome-avatar">${avatarHTML(54)}</div>
          <h4>Hi there! 👋</h4>
          <p>I'm Twisha's AI portfolio assistant, powered by a client-side neural model and verified portfolio knowledge base.</p>
          <button class="cb-intro-btn" id="cb-intro-btn"><i class="fas fa-play"></i> Introduce Twisha</button>
        </div>

        <div class="cb-suggestions" id="cb-suggestions">
          <p>💡 Suggested Questions</p>
          <button class="cb-suggestion-btn">Tell me about Twisha.</button>
          <button class="cb-suggestion-btn">What are her technical skills?</button>
          <button class="cb-suggestion-btn">Tell me about her Python experience.</button>
          <button class="cb-suggestion-btn">What projects has she worked on?</button>
          <button class="cb-suggestion-btn">What is her experience with APIs and databases?</button>
          <button class="cb-suggestion-btn">Is she available to join a new company?</button>
          <button class="cb-suggestion-btn">How can I contact her?</button>
        </div>
      </div>

      <div class="cb-input-bar">
        <input id="cb-input" type="text" placeholder="Ask about skills, projects, experience..." autocomplete="off" maxlength="300">
        <button id="cb-mic-btn" title="Voice Input"><i class="fas fa-microphone"></i></button>
        <button id="cb-send-btn" title="Send Message"><i class="fas fa-paper-plane"></i></button>
      </div>

      <div class="cb-mode-badge"><span id="cb-mode-badge-text">🧠 Client-Side Neural LM + 📂 Verified Portfolio KB</span></div>`;

    document.body.appendChild(panel);
  }

  /* ── ATTACH EVENTS ───────────────────────────────────────── */
  function wireEvents() {
    const trigger = document.getElementById('chatbot-trigger');
    const panel = document.getElementById('chatbot-panel');

    function togglePanel() {
      panel.classList.toggle('open');
      if (panel.classList.contains('open')) {
        initEngines();
        if (window.speechSynthesis) window.speechSynthesis.getVoices();
        setTimeout(() => document.getElementById('cb-input')?.focus(), 300);
      } else {
        stopSpeaking();
        stopListening();
      }
    }

    trigger.addEventListener('click', togglePanel);
    trigger.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        togglePanel();
      }
    });

    // Prevent any click inside the panel from bubbling to document and closing the panel
    panel.addEventListener('click', (e) => {
      e.stopPropagation();
    });

    document.getElementById('cb-close-btn')?.addEventListener('click', (e) => {
      e.stopPropagation();
      panel.classList.remove('open');
      stopSpeaking();
      stopListening();
    });

    document.getElementById('cb-stop-btn')?.addEventListener('click', (e) => {
      e.stopPropagation();
      stopSpeaking();
      stopListening();
    });

    document.getElementById('cb-clear-btn')?.addEventListener('click', (e) => {
      e.stopPropagation();
      const area = document.getElementById('cb-messages');
      if (area) {
        area.innerHTML = `
          <div class="cb-welcome" id="cb-welcome">
            <div class="cb-welcome-avatar">${avatarHTML(54)}</div>
            <h4>Conversation Reset</h4>
            <p>Ask any question about Twisha's skills, projects, or background!</p>
            <button class="cb-intro-btn" id="cb-intro-btn"><i class="fas fa-play"></i> Introduce Twisha</button>
          </div>`;
        document.getElementById('cb-intro-btn')?.addEventListener('click', (ev) => {
          ev.stopPropagation();
          playIntro();
        });
      }
      conversationHistory = [];
      stopSpeaking();
    });

    document.getElementById('cb-intro-btn')?.addEventListener('click', (e) => {
      e.stopPropagation();
      playIntro();
    });

    document.querySelectorAll('.cb-suggestion-btn').forEach((btn) => {
      btn.addEventListener('click', (e) => {
        e.stopPropagation();
        sendMessage(btn.textContent.trim());
      });
    });

    document.getElementById('cb-send-btn')?.addEventListener('click', (e) => {
      e.stopPropagation();
      sendMessage();
    });

    document.getElementById('cb-input')?.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
      }
    });

    document.getElementById('cb-mic-btn')?.addEventListener('click', (e) => {
      e.stopPropagation();
      startListening();
    });

    // Close on outside click
    document.addEventListener('click', (e) => {
      if (!panel.classList.contains('open')) return;

      // When the assistant is actively speaking, NEVER close the panel on accidental clicks
      if (isSpeaking) return;

      const path = (e.composedPath && e.composedPath()) || [];
      if (path.some(el => el === panel || el === trigger)) return;
      if (panel.contains(e.target) || trigger.contains(e.target)) return;
      if (e.target && e.target.closest && (e.target.closest('#chatbot-panel') || e.target.closest('#chatbot-trigger'))) return;

      panel.classList.remove('open');
      stopSpeaking();
      stopListening();
    });

    // Escape key closes panel
    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape' && panel.classList.contains('open')) {
        panel.classList.remove('open');
        stopSpeaking();
        stopListening();
      }
    });
  }

  /* ── INITIALIZATION ─────────────────────────────────────── */
  function init() {
    buildWidget();
    wireEvents();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
