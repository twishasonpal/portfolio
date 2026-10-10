import json

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:
    torch = None
    class Dataset:
        pass

def load_corpus_from_json(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    sentences = []
    
    # Profile & Intro
    prof = data['profile']
    sentences.append(f"Twisha Sonpal is a {prof['title']} based in {prof['location']}.")
    sentences.append(f"Summary: {prof['summary']}")

    # Contact
    contact = data['contact']
    sentences.append(f"Contact Twisha at email {contact['email']} or phone {contact['phone']}.")
    sentences.append(f"LinkedIn: {contact['linkedin']}")
    sentences.append(f"GitHub: {contact['github']}")

    # Availability
    avail = data['availability']
    sentences.append(f"Availability status: {avail['status']}. Notice period: {avail['notice_period']}.")

    # Work experience
    for exp in data['work_experience']:
        sentences.append(f"At {exp['company']} ({exp['location']}), Twisha worked as {exp['role']} ({exp['type']}) starting from {exp['start_date']}.")
        sentences.append(f"Role summary: {exp['summary']}")
        for h in exp['highlights']:
            sentences.append(f"Work achievement: {h}")

    # Skills
    for category, skill_list in data['skills'].items():
        sentences.append(f"Twisha's {category.replace('_', ' ')} skills include: {', '.join(skill_list)}.")

    # Projects
    for proj in data['projects']:
        sentences.append(f"Project {proj['name']} ({proj['category']}): {proj['description']} Built with {', '.join(proj['tech_stack'])}.")

    # Education
    for edu in data['education']:
        sentences.append(f"Education: {edu['degree']} at {edu['institution']} ({edu['period']}).")

    # Certifications
    for cert in data['certifications']:
        sentences.append(f"Certification: {cert['name']} issued by {cert['issuer']} ({cert['year']}).")

    # Common Recruiter QA Pairs
    qa_pairs = [
        ("Who is Twisha Sonpal?", "Twisha Sonpal is a Python Developer and Backend Engineer from Ahmedabad, Gujarat, India with 2+ years experience."),
        ("What programming languages does she know?", "Twisha knows Python, JavaScript, SQL, Java, PHP, HTML5, and CSS3."),
        ("What backend technologies has she used?", "She uses Python, Flask, Django, REST APIs, PostgreSQL, OOP, and Task Schedulers."),
        ("How much professional experience does she have?", "Twisha has over 2 years of experience starting from Jan 2024 as Trainee and Junior Software Engineer at Aktiv Software."),
        ("What projects has she built?", "She built Fab Grocery, Car Sensing System, Data Analysis Dashboard, Voting System, QR Code Scanner, and AI Automation Workflows."),
        ("What did she do at Aktiv Software?", "She rebuilt backend modules for 80% performance gains, integrated REST APIs, built PostgreSQL pipelines, and migrated 200+ Odoo modules."),
        ("Is she available to join immediately?", "Twisha is open to select Python Developer and Backend Engineering roles. Her notice period is negotiable."),
        ("Can I view her resume?", "Yes, you can download her resume directly from the portfolio website or contact her at twishasonpal123@gmail.com."),
        ("Which projects demonstrate her Python skills?", "Car Sensing System, Data Analysis Dashboard, Voting System, and API Integration Platform all demonstrate her Python expertise."),
        ("What certifications does she hold?", "AWS Cloud Practitioner Essentials, IBM GenAI Apps with Python, ISRO Geoprocessing with Python, and n8n AI Automation.")
    ]

    for q, a in qa_pairs:
        sentences.append(f"Question: {q} Answer: {a}")

    return sentences

class PortfolioDataset(Dataset):
    def __init__(self, texts, tokenizer, seq_len=64):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.samples = []

        for text in texts:
            encoded = tokenizer.encode(text, add_special_tokens=True)
            # Create overlapping slices if sequence is longer than seq_len
            for i in range(0, max(1, len(encoded) - 1), seq_len // 2):
                chunk = encoded[i:i + seq_len + 1]
                if len(chunk) < seq_len + 1:
                    # Pad
                    chunk = chunk + [tokenizer.pad_id] * (seq_len + 1 - len(chunk))
                x = torch.tensor(chunk[:-1], dtype=torch.long)
                y = torch.tensor(chunk[1:], dtype=torch.long)
                self.samples.append((x, y))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
