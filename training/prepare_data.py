import os
import json
import pypdf

def main():
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)

    # 1. Extract and Save Resume Text
    reader = pypdf.PdfReader('Twisha_Sonpal_Resume_PythonDeveloper.pdf')
    pages_text = []
    for idx, page in enumerate(reader.pages):
        text = page.extract_text()
        pages_text.append(f"=== PAGE {idx + 1} ===\n" + text)
    full_resume_text = "\n\n".join(pages_text)

    with open('data/raw/resume_text.txt', 'w', encoding='utf-8') as f:
        f.write(full_resume_text)

    # 2. Extract and Save Portfolio Text
    with open('data/portfolio.json', 'r', encoding='utf-8') as f:
        p_data = json.load(f)

    p_lines = []
    p_lines.append("=== TWISHA SONPAL - PORTFOLIO FACTS ===")
    p_lines.append("")
    prof = p_data['profile']
    p_lines.append(f"NAME: {prof['name']}")
    p_lines.append(f"TITLE: {prof['title']}")
    p_lines.append(f"LOCATION: {prof['location']}")
    p_lines.append(f"SUMMARY: {prof['summary']}")
    p_lines.append("")

    c = p_data['contact']
    p_lines.append("CONTACT INFORMATION:")
    p_lines.append(f"- Email: {c['email']}")
    p_lines.append(f"- Phone: {c['phone']}")
    p_lines.append(f"- LinkedIn: {c['linkedin']}")
    p_lines.append(f"- GitHub: {c['github']}")
    p_lines.append(f"- Portfolio URL: {c['portfolio_url']}")
    p_lines.append("")

    avail = p_data['availability']
    p_lines.append("JOB AVAILABILITY:")
    p_lines.append(f"- Open to work: {avail['is_open_to_work']}")
    p_lines.append(f"- Status: {avail['status']}")
    p_lines.append(f"- Notice Period: {avail['notice_period']}")
    p_lines.append(f"- Last Updated: {avail['last_updated']}")
    p_lines.append("")

    p_lines.append("WORK EXPERIENCE:")
    for exp in p_data['work_experience']:
        p_lines.append(f"- Role: {exp['role']} at {exp['company']} ({exp['location']})")
        p_lines.append(f"  Type: {exp['type']} | Period: {exp['start_date']} to {exp['end_date']}")
        p_lines.append(f"  Summary: {exp['summary']}")
        p_lines.append("  Key Highlights:")
        for h in exp['highlights']:
            p_lines.append(f"    * {h}")
    p_lines.append("")

    p_lines.append("TECHNICAL SKILLS:")
    for cat, s_list in p_data['skills'].items():
        cat_name = cat.replace('_', ' ').title()
        skills_str = ", ".join(s_list)
        p_lines.append(f"- {cat_name}: {skills_str}")
    p_lines.append("")

    p_lines.append("KEY PROJECTS:")
    for proj in p_data['projects']:
        stack_str = ", ".join(proj['tech_stack'])
        p_lines.append(f"- Project: {proj['name']}")
        p_lines.append(f"  Category: {proj['category']} | Tech Stack: {stack_str}")
        p_lines.append(f"  Description: {proj['description']}")
    p_lines.append("")

    p_lines.append("EDUCATION:")
    for edu in p_data['education']:
        p_lines.append(f"- {edu['degree']} from {edu['institution']} ({edu['period']})")
    p_lines.append("")

    p_lines.append("VERIFIED CERTIFICATIONS:")
    for cert in p_data['certifications']:
        date_info = cert.get('date', cert.get('year'))
        p_lines.append(f"- {cert['name']} (Issuer: {cert['issuer']}, Year/Date: {date_info})")

    full_portfolio_text = "\n".join(p_lines)
    with open('data/raw/portfolio_text.txt', 'w', encoding='utf-8') as f:
        f.write(full_portfolio_text)

    # 3. Create cleaned combined corpus in data/processed/portfolio_corpus.txt
    corpus_blocks = [
        "### PROFESSIONAL SUMMARY",
        f"{prof['name']} is a {prof['title']} based in {prof['location']}.",
        prof['summary'],
        "",
        "### CONTACT & LINKS",
        f"Email: {c['email']}",
        f"Phone: {c['phone']}",
        f"LinkedIn: {c['linkedin']}",
        f"GitHub: {c['github']}",
        f"Portfolio Website: {c['portfolio_url']}",
        "",
        "### CURRENT AVAILABILITY",
        f"Status: {avail['status']}",
        f"Notice Period: {avail['notice_period']}",
        f"Open to Work: {'Yes' if avail['is_open_to_work'] else 'No'}",
        f"Information verified as of: {avail['last_updated']}",
        "",
        "### PROFESSIONAL EXPERIENCE",
        "Twisha Sonpal has over 2 years of professional software engineering and backend development experience at Aktiv Software in Ahmedabad, India.",
        "Role 1: Junior Software Engineer (July 2024 to Present)",
        "Key accomplishments:",
        "- Rebuilt and optimized a backend module in Python, achieving an 80% performance improvement through code refactoring and query optimization.",
        "- Integrated third-party REST APIs for real-time bidirectional data synchronization between external client systems and backend services.",
        "- Created Python automation scripts and scheduled background task jobs that automated repetitive data workflows.",
        "- Designed complex PostgreSQL database queries, joins, triggers, and indexing for optimal retrieval speeds.",
        "- Built data analytics pipelines using Python Pandas and Matplotlib to monitor KPIs and revenue metrics.",
        "- Successfully migrated 200+ Odoo ERP modules across versions and resolved 1500+ production bugs.",
        "Role 2: Trainee / Intern (January 2024 to July 2024)",
        "- Completed a rigorous 6-month intensive backend engineering program covering Python OOP, database design, and internal tool maintenance.",
        "",
        "### TECHNICAL SKILLS",
        "Programming Languages: Python, JavaScript, SQL, Java, PHP, HTML5, CSS3.",
        "Backend & Architecture: Python, Flask, REST APIs, OOP, Automation, Task Schedulers.",
        "Databases & Data: PostgreSQL, SQL, Pandas, Matplotlib, PGAdmin, Jupyter Notebook.",
        "Cloud & DevOps: AWS, Application Hosting, Domain & DNS Configuration, Server Management, Git, GitHub.",
        "ERP Systems: Odoo ERP, QWeb Reports, Business Process Consulting.",
        "AI & Automation: AI Agents, n8n Workflows, LLM Applications, GenAI, MCP Server Integration.",
        "",
        "### NOTABLE PROJECTS",
        "1. AI Automation & Intelligent Agent Solutions: Built autonomous workflows using AI agents and MCP server integrations connecting tools and data sources.",
        "2. API Integration & Data Synchronization Platform: Integrated third-party REST APIs with PostgreSQL for real-time cross-system data sync.",
        "3. Cloud Deployment & Application Management: Hosted and maintained applications on AWS infrastructure with DNS configuration and server monitoring.",
        "4. Web Development & AWS Deployment: Developed fullstack web applications and deployed them to AWS.",
        "5. End-to-End Odoo ERP Implementation: Deployed complete Odoo ERP setups configuring Sales, Purchase, Inventory, and Accounting modules.",
        "6. Car Sensing System: Python and OpenCV computer vision application detecting vehicle overspeeding and monitoring traffic flow.",
        "7. Fab Grocery: E-commerce grocery web platform with Python, JavaScript, and SQL.",
        "8. Data Analysis Dashboard: Sales demographics and revenue trend visualization using Python, Pandas, and Matplotlib.",
        "9. Voting System: Python and SQL based election management system.",
        "",
        "### EDUCATION",
        "1. Bachelor of Engineering (B.E.) in Computer Science & Engineering: L.J. Institute of Engineering & Technology (2022 - 2025).",
        "2. Diploma in Computer Engineering: Gujarat Technological University (2019 - 2022).",
        "3. Secondary School Certificate (SSC): Shree Swaminarayan Gurukul (Up to 2019).",
        "",
        "### CERTIFICATIONS",
        "- AWS Cloud Practitioner Essentials (AWS, July 2026)",
        "- AI Automation: Build LLM Apps & AI Agents with n8n & APIs (Udemy, 2024)",
        "- Building Generative AI-Powered Applications with Python (IBM, 2024)",
        "- Geoprocessing using Python (ISRO, 2023)",
        "- Fullstack Development (Johns Hopkins University, 2022)",
        "- Python Programming Comprehensive Bootcamp (Udemy, 2021)",
        "- Crash Course on Python (Google, 2021)",
        "- Basics of Python (Infosys, 2021)"
    ]

    corpus_text = "\n".join(corpus_blocks)
    with open('data/processed/portfolio_corpus.txt', 'w', encoding='utf-8') as f:
        f.write(corpus_text)

    print(f"Extraction complete!")
    print(f"- data/raw/resume_text.txt: {len(full_resume_text)} characters")
    print(f"- data/raw/portfolio_text.txt: {len(full_portfolio_text)} characters")
    print(f"- data/processed/portfolio_corpus.txt: {len(corpus_text)} characters")

if __name__ == '__main__':
    main()
