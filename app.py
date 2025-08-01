import streamlit as st
from typing import Optional, TypedDict
import os
from groq import Groq
from langgraph.graph import StateGraph, END

class ProspectMessageState(TypedDict):
    prospect_name: Optional[str]
    designation: Optional[str]
    company: Optional[str]
    industry: Optional[str]
    prospect_background: str
    final_message: Optional[str]

GROQ_API_KEY = st.secrets["GROQ_API_KEY"]    

client = Groq(api_key=GROQ_API_KEY)


def groq_llm(prompt: str, model: str = "llama3-8b-8192", temperature: float = 0.3) -> str:
    """Generate text using Groq API"""
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    return response.choices[0].message.content.strip()

def summarizer(text: str) -> str:
    """Summarize long backgrounds into key points"""
    if not text or not isinstance(text, str):
        return "No content to summarize."

    truncated_text = text[:4000]
    prompt = f"""
Create 3 concise bullet points from this background text. Focus on key professional highlights and achievements:

{truncated_text}

Bullet points:
-"""
    try:
        return groq_llm(prompt).strip()
    except Exception as e:
        print(f"Summarization error: {e}")
        return "Background summary unavailable"



def summarize_backgrounds(state: ProspectMessageState) -> ProspectMessageState:
    """Node to summarize prospect and user backgrounds"""
    return {
        **state,
        "prospect_background": summarizer(state["prospect_background"]),
        # "my_background": summarizer(state["my_background"])
    }

import re

def extract_name_from_background(background: str) -> str:
    if not background:
        return "there"
    # Take first two capitalized words as name
    match = re.findall(r'\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)?', background)
    if match:
        return match[0]
    return "there"
def generate_message(state: ProspectMessageState) -> ProspectMessageState:
    """Node to generate LinkedIn message with event context"""
    extracted_name = extract_name_from_background(state['prospect_background'])
    prospect_first_name = extracted_name.split()[0] if extracted_name != "Unknown Prospect" else "there"
    my_name="Joseph"
    prompt = f"""
IMPORTANT: Output ONLY the message itself. 
Do NOT include explanations, labels, or introductions.

Write a LinkedIn connection note (MAX 3 LINES, under 300 characters).  
Tone: professional, conversational, concise.

MANDATORY RULES:
1. Start with: "Hi {{first_name}},"
2. Begin by acknowledging the prospect’s role, leadership, or focus using context from: {state['prospect_background']}
   - Natural openers: "Noticed your...", "Saw your...", "I noticed your...", "Your leadership in..."
3. After referencing the prospect, weave in MY BACKGROUND **using the action-driven pattern**:
   - Executive Role: "I’m focused on automating workflows across enterprise ops at Speridian, where I lead digital transformation initiatives."
   - Academic: "I’m working on workflow automation strategies that align tech with business growth."
   - Hybrid: " I’m focused on automating digital workflows to drive consistency and scale at Speridian."
   - Evangelist: "I’m working on automation strategies that modernize enterprise operations."
4. Pivot into my current focus areas: automation, lending workflows, digital banking, or transformation strategies.
5. Follow the cadence:  
   - Prospect reference → My background (action-driven) → Connection invite.
6. End with a short, professional invitation to connect. Examples:
   - "Thought it’d be great to connect."
   - "Would love to connect."
   - "It would be great to connect."
   - "Would be glad to connect."
7. Sign off with: "Best, {my_name}" (must include my_name).
8. ABSOLUTELY FORBIDDEN words: exploring, interested, learning, no easy feat, impressive, noteworthy, remarkable, fascinating, admiring, inspiring, no small feat, no easy task, stood out.

EXAMPLE STRUCTURE (do not alter format):
Hi Michael,
Noticed your journey scaling consumer and indirect lending at Gateway and TTCU. I’m focused on automating workflows across lending ops to boost speed and consistency. Thought it’d be great to connect.
Best, Joseph

Hi Jed,
Saw your long-standing leadership across mortgage, consumer, and warehouse lending at TBK. I’m working on automating lending ops to drive faster execution with better control—thought it’d be great to connect.
Best, Joseph

Hi James,
Saw your post on growing referrals and review-driven business—clearly a well-oiled engine. I’m focused on automating lending workflows to boost execution and control behind the scenes. Thought it’d be great to connect.
Best, Joseph

Hi Jim,
Saw your work leading product at Mortgage Cadence—especially on MCP Essentials and platform flexibility. I’m focused on automating lending workflows too. Thought it’d be great to connect.
Best, Joseph

Hi Ken,
Saw the momentum you’re building in reverse at US Mortgage—trusted leadership, national reach, and a sharp ops model. I’m working on automating lending workflows to support that kind of scale. Thought it’d be great to connect.
Best, Joseph

Hi George,
Your leadership at the tech–finance intersection at Antares really stood out—especially your focus on automation and scalable platforms. I’m working on loan ops automation that aligns well with that modernization vision. Thought it’d be great to connect.
Best, Joseph

Hi Corey,
Noticed your push to build highly engaged, adaptable lending teams at Sunrise Banks. I’m focused on workflow automation that helps support scale without losing that collaborative edge. Thought it’d be good to connect.
Best, Joseph

Hi Travis,
I’ve been looking into how IT leaders drive cost optimization without sacrificing agility—your approach at Lower overlaps a lot. I work on lending automation to support similar efficiency. Thought it’d be good to connect.
Best, Joseph

Hi Josh,
Noticed your focus on making lending processes more efficient and borrower-friendly at Lower. I work on automating workflows to support that level of delivery. Would be glad to connect.
Best, Joseph

Hi Diane,
Your extensive work in digital transformation and system conversions at ConnectOne Bank resonates with my focus on automating lending workflows to enhance efficiency and control. Would love to connect.
Best, Joseph

Hi Ed,
Your experience in real estate and mortgage banking, especially with strategic sales and business development, aligns with my work on automating workflows to boost operational efficiency. Would love to connect.
Best, Joseph

Hi Matthew,
Your leadership in mortgage automation and fintech innovation, especially with Big POS and the ""Farm to Table"" approach, is impressive. I’m focused on streamlining lending workflows for efficiency and control—thought it’d be great to connect.
Best, Joseph

Hi Nick,
I noticed your role in leading lending and transactions at Gordon Brothers—particularly your work with transformative financial solutions. I’ve been following trends in capital strategies for distressed corporates and would love to connect.
Best, Joseph

Hi Ken,
I see your focus on mortgage lending and community building through events at United Bank Mortgage. I’ve been exploring how process automation can improve efficiency in lending workflows. It would be great to connect.
Best, Joseph

Hi David,
I noticed your leadership in mortgage lending at Firstmark Credit Union—driving member-focused solutions with consistency. I focus on automating workflows to support that kind of execution. Would be great to connect.
Best, Joseph

Hi Jason,
Your background in retail lending and credit risk management at General Electric Credit Union stood out—leading teams and managing underwriting processes. I work on automating lending workflows to help teams scale efficiently. Would be good to connect.
Best, Joseph

Now create for:
Prospect: {state['prospect_name']} ({state['designation']} at {state['company']})
Industry: {state['industry']}
Key Highlight: {state['prospect_background']}
My Background (optional if relevant): PhD in IT | Director, Digital Transformation at Speridian | AI & Digital Strategy | Healthcare Informatics | Social Computing Scientist

Message (STRICTLY 2–3 LINES, under 300 characters):
Hi {{first_name}},"""

    
    

    try:
        response = groq_llm(prompt, temperature=0.7)
        # Clean response
        message = response.strip()
        unwanted_starts = [
            "Here is a LinkedIn connection message",
            "Here’s a LinkedIn message",
            "LinkedIn connection message:",
            "Message:",
            "Output:"
        ]
        for phrase in unwanted_starts:
            if message.lower().startswith(phrase.lower()):
                message = message.split("\n", 1)[-1].strip()

        # Ensure connection note is present
        # connection_phrases = ["look forward", "would be great", "hope to connect", "love to connect", "looking forward"]
        # if not any(phrase in message.lower() for phrase in connection_phrases):
        #     # Add connection note if missing
        #     message += "\nI'll be there too & looking forward to catching up with you at the event."
            
        # if message.count(f"Best, {my_name}") > 1:
        #     parts = message.split(f"Best, {my_name}")
        #     message = parts[0].strip() + f"\n\nBest, {my_name}"
        # if "Glad to be connected" not in message:
        #     lines = message.split("\n")
        #     if len(lines) > 1:
        #         lines.insert(1, "Glad to be connected.")
        #     else:
        #         message = f"{lines[0]}\nGlad to be connected."
        #     message = "\n".join(lines)

# Strip forbidden flattery words if the model sneaks them in
        for word in ["exploring", "interested", "learning", "impressive", "noteworthy", "remarkable", 
             "fascinating", "admiring", "inspiring", "no small feat", "no easy task", "stood out"]:
            message = re.sub(rf"\b{word}\b", "", message, flags=re.IGNORECASE).strip()


        return {**state, "final_message": message}
    except Exception as e:
        print(f"Message generation failed: {e}")
        return {**state, "final_message": "Failed to generate message"}
# =====================
# Build LangGraph Workflow
# =====================
workflow = StateGraph(ProspectMessageState)
workflow.add_node("summarize_backgrounds", summarize_backgrounds)
workflow.add_node("generate_message", generate_message)
workflow.set_entry_point("summarize_backgrounds")
workflow.add_edge("summarize_backgrounds", "generate_message")
workflow.add_edge("generate_message", END)
graph1 = workflow.compile()

# =====================
# Streamlit UI
# =====================
st.set_page_config(page_title="LinkedIn Message Generator", layout="centered")
st.title(" First Level Msgs for Speridian")

with st.form("prospect_form"):
    prospect_name = st.text_input("Prospect Name", "")
    designation =""
    company = ""
    industry =""
    prospect_background = st.text_area("Prospect Background", "Prospect professional background goes here...")

    submitted = st.form_submit_button("Generate Message")

if submitted:
    with st.spinner("Generating message..."):
        initial_state: ProspectMessageState = {
            "prospect_name": prospect_name,
            "designation": designation,
            "company": company,
            "industry": industry,
            "prospect_background": prospect_background,

        }
        result = graph1.invoke(initial_state)

    st.success(" Message Generated!")
    st.text_area("Final LinkedIn Message", result["final_message"], height=200, key="final_msg")

    # Copy button using JavaScript
    copy_code = f"""
    <script>
    function copyToClipboard() {{
        var text = `{result['final_message']}`;
        navigator.clipboard.writeText(text).then(() => {{
            alert("Message copied to clipboard!");
        }});
    }}
    </script>
    <button onclick="copyToClipboard()"> Copy Message</button>
    """

    st.components.v1.html(copy_code, height=50)
