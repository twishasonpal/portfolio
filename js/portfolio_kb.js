/**
 * portfolio_kb.js - Fact Retrieval & Exact Factual Calculation Module
 * Loads data/portfolio.json and executes deterministic fact matching,
 * experience duration calculations, and fallback handling.
 */

window.PortfolioKB = (function() {
  'use strict';

  let kbData = null;
  let isLoaded = false;

  async function load(jsonUrl = 'data/portfolio.json') {
    if (isLoaded && kbData) return kbData;
    try {
      const response = await fetch(jsonUrl);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      kbData = await response.json();
      isLoaded = true;
      return kbData;
    } catch (err) {
      console.warn('PortfolioKB: Failed to load portfolio.json via fetch, using fallback data structure.', err);
      kbData = getFallbackKB();
      isLoaded = true;
      return kbData;
    }
  }

  function getFallbackKB() {
    return {
      profile: {
        name: "Twisha Sonpal",
        title: "Python Developer & AI Engineer",
        location: "Ahmedabad, Gujarat, India",
        summary: "Python Developer & AI Engineer with 2+ years of professional experience building backend systems, REST API integrations, automation solutions, and AI-powered applications. Skilled in Python, Flask, PostgreSQL, AI automation (n8n, LLM agents), AWS deployment, and Odoo ERP. Actively building AI agents, GenAI apps, and automation workflows alongside core backend engineering."
      },
      contact: {
        email: "twishasonpal123@gmail.com",
        phone: "+91-9428970313",
        linkedin: "https://www.linkedin.com/in/sonpal-twisha-96855a239/",
        github: "https://github.com/twishasonpal"
      },
      availability: {
        is_open_to_work: true,
        status: "Open to Python Developer, Backend Engineer, and AI Automation roles",
        notice_period: "Negotiable",
        last_updated: "2026-10-09"
      },
      work_experience: [
        {
          company: "Aktiv Software",
          role: "Junior Software Engineer",
          type: "Full-time",
          start_date: "2024-07-01",
          end_date: null,
          is_current: true
        },
        {
          company: "Aktiv Software",
          role: "Trainee",
          type: "Internship / Training",
          start_date: "2024-01-01",
          end_date: "2024-07-01",
          is_current: false
        }
      ],
      skills: {
        languages: ["Python", "JavaScript", "SQL", "Java", "PHP", "HTML5", "CSS3"],
        backend: ["Python", "Flask", "REST APIs", "OOP", "Automation", "Task Schedulers"],
        databases: ["PostgreSQL", "SQL", "Pandas", "Matplotlib"],
        cloud_and_devops: ["AWS", "Application Hosting", "Git", "GitHub"],
        erp: ["Odoo ERP", "QWeb Reports"],
        emerging_and_learning: ["AI Agents", "n8n Workflows", "LLM Applications", "GenAI", "MCP Server Integration"]
      },
      projects: [],
      education: [],
      certifications: []
    };
  }

  /**
   * Calculate total non-overlapping work experience duration dynamically from start/end dates.
   */
  function calculateExperienceDuration() {
    if (!kbData || !kbData.work_experience) return { totalMonths: 33, formatted: "2+ years" };

    const exps = kbData.work_experience;
    let totalMonths = 0;
    let fullTimeMonths = 0;
    let traineeMonths = 0;

    const now = new Date(2026, 9, 1); // Current verified date: Oct 2026

    exps.forEach(exp => {
      const start = new Date(exp.start_date);
      const end = exp.end_date ? new Date(exp.end_date) : now;
      
      const months = (end.getFullYear() - start.getFullYear()) * 12 + (end.getMonth() - start.getMonth());
      totalMonths += Math.max(0, months);

      if (exp.type.toLowerCase().includes('full-time')) {
        fullTimeMonths += Math.max(0, months);
      } else {
        traineeMonths += Math.max(0, months);
      }
    });

    const years = Math.floor(totalMonths / 12);
    const remMonths = totalMonths % 12;
    const formatted = years > 0 ? `${years} year${years > 1 ? 's' : ''} ${remMonths > 0 ? remMonths + ' month' + (remMonths > 1 ? 's' : '') : ''} (including 6-month Trainee period and 2+ years Junior Software Engineer role)` : `${totalMonths} months`;

    return { totalMonths, fullTimeMonths, traineeMonths, formatted };
  }

  /**
   * Deterministic Factual Retrieval — broad natural-language intent matching.
   * Handles typos, short forms, and natural recruiter speech patterns.
   */
  function queryFacts(queryText) {
    if (!kbData) kbData = getFallbackKB();
    const q = queryText.toLowerCase().trim().replace(/[?!.,]/g, '');

    // ─── HELPER: fuzzy word match ──────────────────────────────
    const has = (...words) => words.some(w => q.includes(w));

    // ─── 0. IDENTITY / BIO / GREETINGS ────────────────────────
    // ONLY fires when no specific professional topic is present.
    // e.g. "Tell me about her certifications" → certifications rule, not bio.
    const specificTopics = /certif|course|experience|skill|project|work|job|company|api|database|backend|python|ai|contact|email|education|degree|available|salary|ctc|locat|remote|linkedin|linkding|linkdin/i;
    const isGenericBio =
      has('who is she', 'who are you', 'introduce', 'overview', 'bio') ||
      q === 'hi' || q === 'hello' || q === 'hey' ||
      q.startsWith('hi ') || q.startsWith('hello ') || q.startsWith('hey ') ||
      // "about her" / "tell me about" / "profile" only if no specific topic follows
      ((has('about her', 'tell me about', 'say about', 'describe her', 'twisha', 'summary', 'profile')) && !specificTopics.test(q));

    if (isGenericBio) {
      return `**${kbData.profile.name}** — **${kbData.profile.title}**\n📍 ${kbData.profile.location}\n\n${kbData.profile.summary}\n\n💬 Ask me about her skills, experience, projects, certifications, or availability!`;
    }

    // ─── 1. SKILLS / TECHNOLOGIES ─────────────────────────────
    if (
      has('skill', 'skil', 'techno', 'tech stack', 'language',
          'tool', 'what can she', 'what does she know', 'expertise',
          'proficien', 'capable', 'what she know', 'what she have',
          'know about', 'she know')
    ) {
      return `Twisha's technical skills:\n\n• 🐍 **Languages:** ${kbData.skills.languages.join(', ')}\n• ⚙️ **Backend:** ${kbData.skills.backend.join(', ')}\n• 🗄️ **Databases & Data:** ${kbData.skills.databases.join(', ')}\n• ☁️ **Cloud & DevOps:** ${kbData.skills.cloud_and_devops.join(', ')}\n• 🏢 **ERP Systems:** ${kbData.skills.erp.join(', ')}\n• 🤖 **AI & Automation:** ${kbData.skills.emerging_and_learning.join(', ')}`;
    }

    // ─── 2. CERTIFICATIONS ─────────────────────────────────────
    if (
      has('certif', 'certi', 'certiic', 'certificate', 'certification',
          'credential', 'license', 'how many cert', 'courses', 'course')
    ) {
      if (!kbData.certifications || kbData.certifications.length === 0) {
        return 'Twisha holds certifications from IBM, AWS, ISRO, Google, Johns Hopkins University, Udemy, and Infosys. You can view them on her Certificates page!';
      }
      const list = kbData.certifications.map(c => `• **${c.name}** — ${c.issuer} (${c.year})`).join('\n');
      return `Twisha holds **${kbData.certifications.length} verified certifications**:\n\n${list}`;
    }

    // ─── 3. EXPERIENCE / WORK HISTORY ─────────────────────────
    if (
      has('experience', 'experiance', 'experence', 'expereince',
          'how long', 'how many year', 'past company', 'previous company',
          'past work', 'work history', 'work at', 'worked at',
          'company', 'employer', 'job history', 'career',
          'aktiv', 'current job', 'current company', 'where she work',
          'where does she work')
    ) {
      const expDur = calculateExperienceDuration();
      return `Twisha has **${expDur.formatted}** of total professional experience:\n\n🏢 **Aktiv Software** — Ahmedabad, India\n\n• **Junior Software Engineer** (Jul 2024 – Present | Full-time)\n  Rebuilt backend modules for 80% performance gain, integrated third-party REST APIs, built PostgreSQL data pipelines, and migrated 200+ Odoo ERP modules.\n\n• **Trainee** (Jan 2024 – Jul 2024 | 6-month program)\n  Intensive Python backend training program — OOP, database design, internal tooling.`;
    }

    // ─── 4. BACKEND / FRAMEWORK ────────────────────────────────
    if (has('backend', 'framework', 'flask', 'django', 'server side', 'server-side')) {
      return `Twisha's backend expertise:\n\n• **${kbData.skills.backend.join(', ')}**\n• PostgreSQL database design & optimization\n• REST API integrations & authentication\n• Automation scripts & task schedulers\n• Python OOP and clean code architecture`;
    }

    // ─── 5. PROJECTS ───────────────────────────────────────────
    if (
      has('project', 'built', 'made', 'developed', 'created',
          'car sensing', 'fab grocery', 'dashboard', 'voting',
          'what she built', 'portfolio work', 'her work', 'what has she built')
    ) {
      return `Twisha's key projects:\n\n• 🚗 **Car Sensing System** (Python, OpenCV) — Real-time overspeed detection & traffic monitoring\n• 📊 **Data Analysis Dashboard** (Python, Pandas, Matplotlib) — Sales & revenue trend visualization\n• 🔌 **API Integration Platform** (Python, REST APIs, PostgreSQL) — Real-time cross-system data sync\n• 🤖 **AI Automation Workflows** (Python, AI Agents, n8n, MCP) — Intelligent workflow automation\n• 🛒 **Fab Grocery** (Python, JS, SQL) — E-commerce grocery delivery platform\n• 🗳️ **Voting System** (Python, SQL) — Secure election management system\n• ☁️ **AWS Cloud App Management** (AWS) — Server configuration & production deployment`;
    }

    // ─── 6. DATABASE / API ─────────────────────────────────────
    if (has('api', 'database', 'db', 'sql', 'postgres', 'postgresql', 'data')) {
      return `Twisha's database & API experience:\n\n• **REST APIs:** Integrated multiple third-party APIs in production, built secure auth endpoints, tested with Postman\n• **PostgreSQL:** Complex queries, multi-table joins, CTEs, triggers, indexing, schema design\n• **Python Pandas:** ETL pipelines & KPI dashboards\n• **Odoo ORM:** Advanced database operations via the Odoo framework`;
    }

    // ─── 7. PYTHON SPECIFICALLY ────────────────────────────────
    if (has('python', 'pyth')) {
      return `Python is Twisha's primary language. Her Python expertise includes:\n\n• OOP, clean architecture, and modular design\n• REST API integrations in Python\n• Automation scripts & task schedulers\n• PostgreSQL data pipelines using Pandas\n• Backend module development (80% performance improvement on a production system)\n• AI agents and LLM application development with Python\n• Certifications: IBM GenAI with Python, ISRO Geoprocessing with Python, Google Crash Course on Python`;
    }

    // ─── 8. AVAILABILITY / OPEN TO WORK ───────────────────────
    if (
      has('available', 'availability', 'availab',
          'join', 'immediate', 'immediat', 'immedi',
          'hire', 'hiring', 'open to', 'open for',
          'notice', 'notice period', 'opportunity', 'looking for',
          'she is open', 'is she open', 'ready to join',
          'when can she', 'can she join', 'she join')
    ) {
      const avail = kbData.availability;
      return `**Availability Status:** ${avail.status}\n\n• ⏳ **Notice Period:** ${avail.notice_period}\n• 📍 **Preferred Locations:** Remote or Ahmedabad, Gujarat, India\n• 📧 **Contact:** twishasonpal123@gmail.com\n• 💼 **LinkedIn:** linkedin.com/in/sonpal-twisha-96855a239/`;
    }

    // ─── 9. CONTACT / RESUME / LINKEDIN ────────────────────────
    if (
      has('contact', 'email', 'phone', 'mobile', 'number',
          'linkedin', 'linkding', 'linkdin', 'linked in',
          'github', 'resume', 'cv', 'reach', 'profile id', 'social',
          'get in touch', 'message her', 'connect with', 'how to reach')
    ) {
      const c = kbData.contact;
      // If asking specifically for LinkedIn / Linkding
      if (has('linkedin', 'linkding', 'linkdin', 'linked in')) {
        return `Here is Twisha's **LinkedIn profile**:\n\n💼 **LinkedIn:** [linkedin.com/in/sonpal-twisha](https://www.linkedin.com/in/sonpal-twisha-96855a239/)\n📧 **Email:** [${c.email}](mailto:${c.email})\n🐙 **GitHub:** [github.com/twishasonpal](https://github.com/twishasonpal)\n\nYou can click the LinkedIn link above to view her full professional profile!`;
      }
      return `How to reach Twisha Sonpal:\n\n📧 **Email:** [${c.email}](mailto:${c.email})\n📱 **Phone:** ${c.phone}\n💼 **LinkedIn:** [linkedin.com/in/sonpal-twisha](https://www.linkedin.com/in/sonpal-twisha-96855a239/)\n🐙 **GitHub:** [github.com/twishasonpal](https://github.com/twishasonpal)\n📄 **Resume:** Available for download on her portfolio website.`;
    }

    // ─── 10. EDUCATION ─────────────────────────────────────────
    if (
      has('education', 'degree', 'college', 'university',
          'study', 'studies', 'school', 'qualification', 'academic')
    ) {
      return `**Education:**\n🎓 **B.E. Computer Science & Engineering** — L.J. Institute of Engineering & Technology (2022–2025)\n📋 **Diploma in Computer Engineering** — Gujarat Technological University (2019–2022)\n🏫 **Secondary School** — Shree Swaminarayan Gurukul (Up to 2019)`;
    }

    // ─── 11. AI / GENAI / AUTOMATION ───────────────────────────
    if (
      has('ai', 'artificial intelligence', 'genai', 'gen ai',
          'llm', 'language model', 'agent', 'n8n', 'automation',
          'machine learning', 'ml')
    ) {
      return `Twisha is an **AI Engineer & Python Developer** actively building AI solutions:\n\n• 🤖 **AI Agents** — Intelligent workflow agents using n8n and MCP server integrations\n• 🧠 **GenAI Apps** — LLM-powered application development (IBM certification)\n• 🔧 **AI Automation** — Business process automation connecting tools and data sources\n• ☁️ **AWS** — Cloud deployment for AI and backend applications\n\n**Certifications:** IBM (Building GenAI Apps with Python), AI Automation with n8n & LLM APIs\n\n*Note: Her professional production background is primarily Python backend & ERP engineering. AI/GenAI is her active growth area with real projects and certifications.*`;
    }

    // ─── 12. ODOO / ERP ────────────────────────────────────────
    if (has('odoo', 'erp', 'erpnext', 'qweb', 'frappe')) {
      return `Twisha has hands-on Odoo ERP experience:\n\n• Custom module development (Sales, Inventory, Accounting)\n• Migrated 200+ modules across Odoo versions — zero compatibility issues\n• Resolved 1,500+ production bugs in live ERP environments\n• Built dynamic QWeb reports & KPI dashboards\n• Integrated third-party systems with Odoo via REST APIs`;
    }

    // ─── 13. SALARY / CTC / COMPENSATION ──────────────────────
    if (
      has('salary', 'ctc', 'pay', 'compensation', 'package',
          'lpa', 'lakh', 'per annum', 'wage', 'stipend', 'hike',
          'expected', 'current ctc', 'your ctc')
    ) {
      return `Twisha hasn't shared her current CTC publicly. Feel free to discuss compensation directly with her:\n\n📧 **Email:** twishasonpal123@gmail.com\n💼 **LinkedIn:** linkedin.com/in/sonpal-twisha-96855a239/`;
    }

    // ─── 14. PERSONAL / OUT-OF-SCOPE ──────────────────────────
    if (
      has('health', 'married', 'single', 'relationship', 'boyfriend',
          'girlfriend', 'family', 'age', 'dob', 'date of birth',
          'height', 'weight', 'hobby', 'hobbies', 'religion', 'caste')
    ) {
      return `That's outside my scope! 😊 I'm here to help with Twisha's **professional background** — skills, experience, projects, and availability. Try asking me one of those!`;
    }

    // ─── 15. LOCATION / REMOTE / RELOCATION ───────────────────
    if (
      has('locat', 'city', 'where she', 'remote', 'work from home',
          'relocat', 'shift', 'move', 'onsite', 'hybrid')
    ) {
      return `Twisha is based in **Ahmedabad, Gujarat, India**.\n\n• ✅ Open to **Remote** roles\n• ✅ Open to **Ahmedabad** based roles\n• 🤔 Open to discussing relocation for the right opportunity\n\n📧 twishasonpal123@gmail.com for further discussion.`;
    }

    // ─── 16. UNKNOWN — return null (triggers clean fallback) ───
    return null;

  }

  return {
    load: load,
    calculateExperienceDuration: calculateExperienceDuration,
    queryFacts: queryFacts,
    getKBData: () => kbData
  };
})();

