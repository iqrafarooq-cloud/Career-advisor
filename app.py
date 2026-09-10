import os
import json
import re
from typing import Dict, List, Tuple

import gradio as gr
import streamlit as st
from dotenv import load_dotenv
from groq import Groq

# ---------------------------------------------------------
# CareerPath AI
# SDG 4: Quality Education
# SDG 8: Decent Work & Economic Growth
# SDG 10: Reduced Inequalities
#
# Streamlit = final deployed UI
# Gradio = reusable AI chatbot/demo layer
# Groq = generative AI engine
# ---------------------------------------------------------

load_dotenv()

st.set_page_config(
    page_title="CareerPath AI",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------
# Configuration
# -----------------------------

MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"")

CAREERS = [
    {
        "name": "Data Analyst",
        "description": "Turns data into insights that support business decisions.",
        "skills": ["Analytical Thinking", "Problem Solving", "Excel", "SQL", "Statistics", "Data Visualization"],
        "interests": ["Technology", "Business", "Finance", "Science"],
        "work_styles": ["Working with data", "Solving problems", "Working with technology"],
        "education": ["Computer Science", "Statistics", "Mathematics", "Business", "Economics"],
        "path": ["Excel & Data Fundamentals", "SQL", "Statistics", "Power BI/Tableau", "Python & Pandas", "Portfolio + Interview Preparation"],
    },
    {
        "name": "Software QA Engineer",
        "description": "Tests software products, finds defects, and helps teams deliver reliable software.",
        "skills": ["Problem Solving", "Communication", "Logical Thinking", "Testing", "Technology", "Attention to Detail"],
        "interests": ["Technology", "Business"],
        "work_styles": ["Solving problems", "Working with technology", "Working with people"],
        "education": ["Computer Science", "Software Engineering", "Information Technology", "Business"],
        "path": ["Testing Fundamentals", "Web & API Testing", "SQL", "Automation", "CI/CD + Git", "Portfolio + Interview Preparation"],
    },
    {
        "name": "Software Engineer",
        "description": "Designs, builds, tests, and maintains software applications.",
        "skills": ["Problem Solving", "Logical Thinking", "Programming", "Technology", "Mathematics"],
        "interests": ["Technology", "Science"],
        "work_styles": ["Working with technology", "Solving problems", "Creating things"],
        "education": ["Computer Science", "Software Engineering", "Information Technology"],
        "path": ["Programming Fundamentals", "Data Structures", "Git & GitHub", "Web/API Development", "Databases", "Projects + Interview Preparation"],
    },
    {
        "name": "Business Analyst",
        "description": "Connects business needs with technology and process improvements.",
        "skills": ["Communication", "Problem Solving", "Research", "Critical Thinking", "Data Analysis", "Leadership"],
        "interests": ["Business", "Technology", "Finance"],
        "work_styles": ["Working with people", "Solving problems", "Working with data"],
        "education": ["Business", "Computer Science", "Economics", "Information Technology"],
        "path": ["Business Analysis Fundamentals", "Requirements Engineering", "SQL & Data Basics", "Process Modeling", "Stakeholder Management", "Case Studies + Interview Preparation"],
    },
    {
        "name": "UX/UI Designer",
        "description": "Designs useful, accessible, and engaging digital experiences.",
        "skills": ["Creativity", "Communication", "Research", "Problem Solving", "Design"],
        "interests": ["Design", "Technology", "Marketing"],
        "work_styles": ["Creating things", "Working with people", "Solving problems"],
        "education": ["Design", "Computer Science", "Fine Arts", "Marketing"],
        "path": ["Design Principles", "Figma", "User Research", "Wireframing", "Prototyping", "Portfolio + Case Studies"],
    },
    {
        "name": "Cybersecurity Analyst",
        "description": "Protects systems, networks, and information from security threats.",
        "skills": ["Problem Solving", "Logical Thinking", "Research", "Technology", "Attention to Detail"],
        "interests": ["Technology", "Science"],
        "work_styles": ["Working with technology", "Solving problems", "Working with data"],
        "education": ["Cybersecurity", "Computer Science", "Information Technology"],
        "path": ["Networking Basics", "Linux", "Security Fundamentals", "SIEM & Monitoring", "Threat Detection", "Security Projects + Certifications"],
    },
    {
        "name": "Product Manager",
        "description": "Guides product strategy, prioritization, customer value, and delivery.",
        "skills": ["Communication", "Leadership", "Research", "Problem Solving", "Critical Thinking", "Data Analysis"],
        "interests": ["Business", "Technology", "Marketing"],
        "work_styles": ["Working with people", "Solving problems", "Creating things"],
        "education": ["Business", "Computer Science", "Marketing", "Economics"],
        "path": ["Product Fundamentals", "User Research", "Product Analytics", "Roadmapping", "Stakeholder Management", "Product Case Studies"],
    },
    {
        "name": "Digital Marketer",
        "description": "Uses digital channels and analytics to attract and engage audiences.",
        "skills": ["Communication", "Creativity", "Research", "Data Analysis", "Writing"],
        "interests": ["Marketing", "Business", "Design"],
        "work_styles": ["Creating things", "Working with people", "Working with data"],
        "education": ["Marketing", "Business", "Communications", "Design"],
        "path": ["Marketing Fundamentals", "Content Strategy", "SEO", "Social Media", "Analytics", "Campaign Portfolio"],
    },
    {
        "name": "Financial Analyst",
        "description": "Analyzes financial information to support investment and business decisions.",
        "skills": ["Analytical Thinking", "Mathematics", "Excel", "Problem Solving", "Data Analysis"],
        "interests": ["Finance", "Business"],
        "work_styles": ["Working with data", "Solving problems"],
        "education": ["Finance", "Accounting", "Economics", "Business", "Mathematics"],
        "path": ["Accounting Basics", "Advanced Excel", "Financial Modeling", "Statistics", "Valuation", "Financial Analysis Portfolio"],
    },
]

SKILLS = sorted({
    skill
    for career in CAREERS
    for skill in career["skills"]
} | {
    "Python", "SQL", "Excel", "Communication", "Leadership", "Writing",
    "Research", "Critical Thinking", "Creativity", "Problem Solving",
    "Programming", "Technology", "Mathematics", "Data Analysis",
    "Attention to Detail", "Logical Thinking", "Testing", "Design"
})

INTERESTS = [
    "Technology", "Healthcare", "Business", "Design", "Finance",
    "Education", "Science", "Marketing"
]

WORK_STYLES = [
    "Working with people",
    "Working with technology",
    "Working with data",
    "Creating things",
    "Solving problems",
]

EDUCATION_OPTIONS = [
    "High School",
    "Diploma",
    "Bachelor's",
    "Master's",
    "PhD",
    "Other",
]

GOAL_OPTIONS = [
    "Get my first job",
    "Change my career",
    "Improve my current career",
    "Become a specialist",
    "Explore career options",
    "Start a business",
]

# -----------------------------
# Groq
# -----------------------------

@st.cache_resource
def get_groq_client():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return None
    return Groq(api_key=api_key)


def call_groq(system_prompt: str, user_prompt: str, temperature: float = 0.3) -> str:
    client = get_groq_client()
    if client is None:
        return (
            "⚠️ GROQ_API_KEY is not configured. Add it to your local `.env` "
            "or Streamlit Secrets before using the AI features."
        )

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=1200,
        )
        return response.choices[0].message.content.strip()
    except Exception as exc:
        return f"⚠️ AI service error: {exc}"


# -----------------------------
# Career matching engine
# -----------------------------

def normalize(values: List[str]) -> set:
    return {v.strip().lower() for v in (values or []) if v and v.strip()}


def education_match(user_education: str, career: Dict) -> float:
    if not user_education:
        return 0.5
    career_education = normalize(career["education"])
    if user_education.lower() in career_education:
        return 1.0
    # Most careers are accessible from related education.
    return 0.65


def score_career(
    education: str,
    skills: List[str],
    interests: List[str],
    work_styles: List[str],
    goal: str,
) -> Tuple[float, Dict[str, float]]:
    user_skills = normalize(skills)
    user_interests = normalize(interests)
    user_work = normalize(work_styles)

    results = []

    for career in CAREERS:
        career_skills = normalize(career["skills"])
        career_interests = normalize(career["interests"])
        career_work = normalize(career["work_styles"])

        skill_score = len(user_skills & career_skills) / max(1, len(career_skills))
        interest_score = len(user_interests & career_interests) / max(1, len(career_interests))
        work_score = len(user_work & career_work) / max(1, len(career_work))
        edu_score = education_match(education, career)

        # Goal is deliberately a smaller factor because it should not dominate
        # the user's actual interests and skills.
        goal_score = 0.75 if goal else 0.5

        overall = (
            skill_score * 0.35
            + interest_score * 0.25
            + work_score * 0.20
            + edu_score * 0.10
            + goal_score * 0.10
        )

        results.append({
            "career": career,
            "score": round(overall * 100),
            "skill_score": round(skill_score * 100),
            "interest_score": round(interest_score * 100),
            "work_score": round(work_score * 100),
            "education_score": round(edu_score * 100),
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[0]["score"], {
        "top_score": results[0]["score"]
    } if results else {}, results


def build_profile_text(
    education: str,
    skills: List[str],
    interests: List[str],
    work_styles: List[str],
    goal: str,
    experience: str,
) -> str:
    return f"""
Education: {education}
Skills: {", ".join(skills) if skills else "Not provided"}
Interests: {", ".join(interests) if interests else "Not provided"}
Preferred work style: {", ".join(work_styles) if work_styles else "Not provided"}
Career goal: {goal or "Not provided"}
Experience: {experience or "Not provided"}
""".strip()


# -----------------------------
# AI prompts
# -----------------------------

CAREER_SYSTEM = """
You are CareerPath AI, a career-focused AI advisor.

Your ONLY domain is:
- career exploration
- career planning
- education pathways
- employability
- skills development
- professional development
- job preparation
- interview preparation
- CV/resume guidance
- career transitions
- learning roadmaps
- workplace/professional skills

You MUST refuse unrelated requests politely.

If the user asks something outside the domain, say:
"I'm CareerPath AI, a career-focused assistant. I can help with
career planning, education, skills, job preparation, and professional growth."

Never claim that a career recommendation is a guaranteed outcome.
Use language such as "strong match", "potential fit", or "worth exploring".
Do not make a final life decision for the user.

Do not invent specific salaries, employment statistics, job openings,
certification requirements, or current market facts unless they are
explicitly provided in the supplied context.

Be concise, practical, supportive, and actionable.
"""


def ai_career_explanation(profile: str, top_results: List[Dict]) -> str:
    context = "\n".join(
        f"- {r['career']['name']}: {r['score']}% match — {r['career']['description']}"
        for r in top_results
    )

    prompt = f"""
User profile:
{profile}

Top matching career paths:
{context}

Explain the top 3 career matches.

For each:
1. Why it fits the user's profile.
2. What the user already appears to have.
3. What they should develop next.
4. One realistic first step.

Finish with a short note that the recommendations are guidance,
not a final decision.
"""
    return call_groq(CAREER_SYSTEM, prompt)


def ai_skill_gap(career_name: str, skills: List[str]) -> str:
    career = next((c for c in CAREERS if c["name"] == career_name), None)
    if not career:
        return "Career not found."

    user = normalize(skills)
    required = normalize(career["skills"])
    strengths = sorted(user & required)
    gaps = sorted(required - user)

    prompt = f"""
Target career: {career_name}
Required skills from the CareerPath knowledge base:
{", ".join(career["skills"])}

User skills:
{", ".join(skills) if skills else "None provided"}

Known strengths:
{", ".join(strengths) if strengths else "No direct matches yet"}

Potential skill gaps:
{", ".join(gaps) if gaps else "No major gaps identified"}

Create a practical skill-gap analysis.
Separate "Already Strong", "Develop Next", and "Suggested Practice".
Do not invent unrelated requirements.
"""
    return call_groq(CAREER_SYSTEM, prompt)


def ai_roadmap(career_name: str, profile: str) -> str:
    career = next((c for c in CAREERS if c["name"] == career_name), None)
    if not career:
        return "Career not found."

    prompt = f"""
Create a practical 6-month roadmap for this user.

Profile:
{profile}

Target career:
{career_name}

Knowledge-base career path:
{" -> ".join(career["path"])}

Required skills:
{", ".join(career["skills"])}

Use this structure:
Month 1:
Month 2:
Month 3:
Month 4:
Month 5:
Month 6:

For every month include:
- learning focus
- practical activity/project
- a measurable outcome

End with "First step this week".
Keep the roadmap realistic for a beginner/intermediate learner.
"""
    return call_groq(CAREER_SYSTEM, prompt)


# -----------------------------
# Chatbot guardrail
# -----------------------------

CAREER_KEYWORDS = {
    "career", "job", "jobs", "work", "profession", "employment",
    "skill", "skills", "resume", "cv", "interview", "internship",
    "internships", "degree", "education", "course", "courses",
    "certification", "certifications", "roadmap", "learn", "learning",
    "promotion", "workplace", "role", "roles", "occupation",
    "career change", "career path", "career advice", "salary negotiation",
    "portfolio", "linkedin", "freelance", "freelancing", "mentor"
}


def is_likely_career_question(message: str) -> bool:
    text = message.lower().strip()

    if not text:
        return False

    # Strong domain phrases.
    if any(phrase in text for phrase in [
        "what career", "which career", "career path",
        "career change", "how to become", "how can i become",
        "what should i learn", "job interview", "my resume",
        "my cv", "skill gap", "professional development"
    ]):
        return True

    tokens = set(re.findall(r"[a-zA-Z]+", text))
    return bool(tokens & CAREER_KEYWORDS)


def chatbot_response(message: str, history=None, profile_context: str = "") -> str:
    if not is_likely_career_question(message):
        return (
            "I'm **CareerPath AI**, a career-focused assistant. "
            "I can help with career planning, education, skills, "
            "job preparation, interviews, CV/resume guidance, "
            "career transitions, and professional growth."
        )

    prompt = f"""
User profile context:
{profile_context or "No profile has been completed yet."}

User's career-related question:
{message}

Answer the question directly.
Personalize the answer when profile context is available.
If the question asks for a current fact such as live vacancies,
current salary statistics, or current market rankings and no
verified data is provided, clearly say that the app does not have
live verified data instead of inventing numbers.
"""
    return call_groq(CAREER_SYSTEM, prompt, temperature=0.4)


# Gradio-compatible wrapper.
# This allows the same chatbot logic to be used in a Gradio demo
# without duplicating the AI logic used by Streamlit.
def gradio_chat(message, history):
    return chatbot_response(message, history)


# -----------------------------
# UI
# -----------------------------

st.markdown(
    """
    <style>
    .main-title {
        font-size: 3rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        font-size: 1.1rem;
        opacity: 0.8;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        padding: 1rem;
        border-radius: 12px;
        border: 1px solid rgba(128,128,128,0.25);
        text-align: center;
    }
    .career-card {
        padding: 1rem;
        border-radius: 14px;
        border: 1px solid rgba(128,128,128,0.25);
        margin-bottom: 0.7rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="main-title">🎯 CareerPath AI</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Your AI-powered career advisor — discover paths, identify skill gaps, and build your roadmap.</div>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("🧭 Navigation")
    page = st.radio(
        "Go to",
        [
            "🏠 Home",
            "📝 Career Assessment",
            "🎯 Career Matches",
            "🧩 Skill Gap",
            "🗺️ Career Roadmap",
            "🤖 AI Career Mentor",
            "🌍 SDG Impact",
        ],
    )

    st.divider()
    api_status = "Connected" if get_groq_client() else "Not configured"
    st.caption(f"Groq AI: **{api_status}**")
    st.caption(f"Model: `{MODEL}`")

# Session defaults.
for key, default in {
    "profile": {},
    "matches": [],
    "selected_career": None,
    "explanation": "",
    "skill_gap": "",
    "roadmap": "",
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# -----------------------------
# HOME
# -----------------------------

if page == "🏠 Home":
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 🎓 SDG 4")
        st.write("Accessible career and lifelong-learning guidance.")

    with col2:
        st.markdown("### 💼 SDG 8")
        st.write("Supports employability, skill development, and career planning.")

    with col3:
        st.markdown("### 🌍 SDG 10")
        st.write("Makes basic career guidance more accessible.")

    st.divider()

    st.subheader("How CareerPath AI works")

    steps = [
        ("1", "Complete your profile", "Tell us about your education, interests, skills, and goals."),
        ("2", "Get career matches", "Our matching engine compares your profile with structured career profiles."),
        ("3", "Understand your gaps", "See which skills you already have and which ones to develop."),
        ("4", "Build your roadmap", "Generative AI creates a practical learning and career plan."),
        ("5", "Ask your AI mentor", "Get career-focused guidance while unrelated questions are blocked."),
    ]

    for number, title, description in steps:
        st.markdown(f"### {number}. {title}")
        st.write(description)

    st.info(
        "Start with **Career Assessment**. CareerPath AI provides guidance and options; "
        "it does not make a final career decision for you."
    )


# -----------------------------
# ASSESSMENT
# -----------------------------

elif page == "📝 Career Assessment":
    st.header("📝 Career Assessment")
    st.write("Answer a few questions so the system can build your initial career profile.")

    education = st.selectbox("Education level", [""] + EDUCATION_OPTIONS)

    skills = st.multiselect(
        "Which skills do you currently have?",
        SKILLS,
        help="Choose the skills you genuinely feel comfortable using.",
    )

    interests = st.multiselect(
        "What areas interest you?",
        INTERESTS,
    )

    work_styles = st.multiselect(
        "How do you prefer to work?",
        WORK_STYLES,
    )

    goal = st.selectbox(
        "What is your main career goal?",
        [""] + GOAL_OPTIONS,
    )

    experience = st.text_area(
        "Briefly describe your experience",
        placeholder="Example: I have 1 year of customer support experience and enjoy solving problems.",
    )

    if st.button("🚀 Analyze My Career Profile", type="primary", use_container_width=True):
        if not any([education, skills, interests, work_styles, goal, experience]):
            st.warning("Please provide at least some information before running the assessment.")
        else:
            st.session_state.profile = {
                "education": education,
                "skills": skills,
                "interests": interests,
                "work_styles": work_styles,
                "goal": goal,
                "experience": experience,
            }

            _, _, results = score_career(
                education,
                skills,
                interests,
                work_styles,
                goal,
            )
            st.session_state.matches = results
            st.session_state.selected_career = results[0]["career"]["name"] if results else None
            st.session_state.explanation = ""

            st.success("Your career profile has been created. Open **Career Matches** to see your recommendations.")


# -----------------------------
# MATCHES
# -----------------------------

elif page == "🎯 Career Matches":
    st.header("🎯 Your Career Matches")

    if not st.session_state.profile or not st.session_state.matches:
        st.warning("Complete the Career Assessment first.")
    else:
        profile = st.session_state.profile
        profile_text = build_profile_text(
            profile["education"],
            profile["skills"],
            profile["interests"],
            profile["work_styles"],
            profile["goal"],
            profile["experience"],
        )

        st.subheader("Your profile")
        st.code(profile_text)

        st.subheader("Top career paths")

        top_results = st.session_state.matches[:5]

        for index, result in enumerate(top_results, start=1):
            career = result["career"]
            st.markdown(
                f"""
                <div class="career-card">
                    <h3>{index}. {career['name']} — {result['score']}% match</h3>
                    <p>{career['description']}</p>
                    <b>Key skills:</b> {", ".join(career['skills'][:6])}
                </div>
                """,
                unsafe_allow_html=True,
            )

        if st.button("✨ Explain My Top Career Matches", type="primary"):
            with st.spinner("CareerPath AI is analyzing your profile..."):
                st.session_state.explanation = ai_career_explanation(
                    profile_text,
                    top_results,
                )

        if st.session_state.explanation:
            st.subheader("🤖 AI Career Analysis")
            st.markdown(st.session_state.explanation)

        selected = st.selectbox(
            "Choose a career to explore",
            [r["career"]["name"] for r in top_results],
            index=0,
        )
        st.session_state.selected_career = selected


# -----------------------------
# SKILL GAP
# -----------------------------

elif page == "🧩 Skill Gap":
    st.header("🧩 Skill Gap Analysis")

    if not st.session_state.profile:
        st.warning("Complete the Career Assessment first.")
    else:
        profile = st.session_state.profile
        options = [r["career"]["name"] for r in st.session_state.matches[:5]]
        if not options:
            st.warning("Run your Career Assessment first.")
        else:
            selected = st.selectbox("Target career", options)
            st.session_state.selected_career = selected

            career = next(c for c in CAREERS if c["name"] == selected)
            user_skills = set(profile["skills"])
            required = set(career["skills"])

            strong = sorted(user_skills & required)
            gaps = sorted(required - user_skills)

            c1, c2 = st.columns(2)
            with c1:
                st.metric("Matched skills", len(strong))
            with c2:
                st.metric("Potential gaps", len(gaps))

            st.markdown("### ✅ Already aligned")
            if strong:
                st.write(", ".join(strong))
            else:
                st.write("No direct matches identified yet.")

            st.markdown("### 📚 Develop next")
            if gaps:
                st.write(", ".join(gaps))
            else:
                st.write("No major gaps identified from the current career profile.")

            if st.button("🤖 Generate AI Skill-Gap Plan", type="primary"):
                with st.spinner("Generating your personalized skill plan..."):
                    st.session_state.skill_gap = ai_skill_gap(selected, profile["skills"])

            if st.session_state.skill_gap:
                st.divider()
                st.markdown(st.session_state.skill_gap)


# -----------------------------
# ROADMAP
# -----------------------------

elif page == "🗺️ Career Roadmap":
    st.header("🗺️ Your Career Roadmap")

    if not st.session_state.profile:
        st.warning("Complete the Career Assessment first.")
    else:
        options = [r["career"]["name"] for r in st.session_state.matches[:5]]
        if not options:
            st.warning("Run your Career Assessment first.")
        else:
            selected = st.selectbox("Choose your target career", options)
            st.session_state.selected_career = selected

            profile = st.session_state.profile
            profile_text = build_profile_text(
                profile["education"],
                profile["skills"],
                profile["interests"],
                profile["work_styles"],
                profile["goal"],
                profile["experience"],
            )

            if st.button("🚀 Generate 6-Month Roadmap", type="primary"):
                with st.spinner("Building your personalized roadmap..."):
                    st.session_state.roadmap = ai_roadmap(selected, profile_text)

            if st.session_state.roadmap:
                st.markdown(st.session_state.roadmap)
            else:
                st.info("Click the button to generate your roadmap.")


# -----------------------------
# CHATBOT
# -----------------------------

elif page == "🤖 AI Career Mentor":
    st.header("🤖 AI Career Mentor")
    st.write(
        "Ask career-related questions about skills, learning, job preparation, "
        "career transitions, interviews, CVs, or professional development."
    )

    if not get_groq_client():
        st.warning("Configure GROQ_API_KEY to enable the AI mentor.")

    profile_context = ""
    if st.session_state.profile:
        p = st.session_state.profile
        profile_context = build_profile_text(
            p["education"],
            p["skills"],
            p["interests"],
            p["work_styles"],
            p["goal"],
            p["experience"],
        )

    # The chatbot logic is implemented as a normal Python function so it can
    # be reused by both Streamlit and Gradio.
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []

    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_message = st.chat_input("Ask a career-related question...")

    if user_message:
        st.session_state.chat_messages.append(
            {"role": "user", "content": user_message}
        )

        with st.chat_message("user"):
            st.markdown(user_message)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = chatbot_response(
                    user_message,
                    st.session_state.chat_messages,
                    profile_context,
                )
                st.markdown(answer)

        st.session_state.chat_messages.append(
            {"role": "assistant", "content": answer}
        )

    with st.expander("🔎 Gradio integration"):
        st.write(
            "The chatbot function is Gradio-compatible through `gradio_chat()`. "
            "The final deployed experience uses Streamlit as the main UI, while "
            "the same chatbot backend can also be exposed as a Gradio demo."
        )


# -----------------------------
# SDG IMPACT
# -----------------------------

elif page == "🌍 SDG Impact":
    st.header("🌍 Social Impact & SDG Alignment")

    st.subheader("SDG 4 — Quality Education")
    st.write(
        "CareerPath AI makes career exploration and learning-path guidance more accessible "
        "by providing personalized educational and skill-development recommendations."
    )

    st.subheader("SDG 8 — Decent Work & Economic Growth")
    st.write(
        "The platform helps users understand employable skills, explore career options, "
        "prepare for jobs, and create actionable development roadmaps."
    )

    st.subheader("SDG 10 — Reduced Inequalities")
    st.write(
        "The product is designed to lower the access barrier to basic career guidance, "
        "especially for people who may not have access to a professional career counselor."
    )

    st.divider()

    st.subheader("What makes the AI responsible?")
    st.markdown(
        """
        - **Career-only guardrail:** unrelated questions are politely refused.
        - **Explainable matching:** recommendations are based on visible profile factors.
        - **No guaranteed outcomes:** the AI presents options rather than making life decisions.
        - **Knowledge grounding:** career paths are based on a structured career dataset.
        - **No hardcoded API key:** secrets belong in environment variables or Streamlit Secrets.
        """
    )

    st.success(
        "Hackathon message: CareerPath AI doesn't tell people what they must become — "
        "it helps them understand what they could become and how to get there."
    )
