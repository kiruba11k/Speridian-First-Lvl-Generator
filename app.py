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

Write a LinkedIn connection note (MAX 3 LINES, MAX 250 characters).  
Tone: professional, conversational, concise.

MANDATORY RULES:
1. Start with: "Hi {{first_name}},"
2. Use a dynamic opener (randomly vary among these):
   - "Noticed your..."
   - "Saw your..."
   - "I noticed your..."
   - "Your leadership in..."
   - "Came across your work on..."
   - "I’ve been following your..."
   - "It’s clear from your role in..."
   - "I find it interesting..."
   - "Came across your profile..."
   - "Impressed by your work in..."
   - "Admire the growth you’ve driven in..."
   - "Your recent contributions to..."
   - "Noticed your leadership in..."
   - "Your profile reflects deep expertise in..."
   - "The results you’ve achieved at..."
   - "Caught your recent work on..."
   - "Your experience in..."
   - "Saw your role at..."
   - "Your achievements in..."
   - "Your insights into..."
   - "Noticed your career..."
   - "Your profile reflects excellence in..."
   - "Saw the innovative strategies you’ve implemented at..."
   - "Impressed by your journey from..."
   - "Your track record in..."
   - "Your contributions to..."
   - "Saw your recognition in..."
   - "Your problem-solving approach in..."
   - "Noticed your focus on..."
   - "Your role in transforming..."
   - "Admire the strategic impact you’ve made..."
   - "Your recent post on..."
   - "Saw how you’ve scaled..."
3. Keep the prospect reference to Only **ONE key highlight** from: {state['prospect_background']} in short phrase.  
4. After referencing the prospect, weave in MY BACKGROUND using the action-driven style (randomly choose ONE):

Executive Role:
- I focus on automating retail workflows to reduce risk and improve turnaround.
- I lead automation programs that streamline operations and boost speed-to-market.
- I drive retail process efficiency to minimize delays and compliance risks.
- I specialize in operational automation to lower costs and increase reliability.
- I enable seamless retail workflows that support rapid scaling.
- I deliver automation frameworks that strengthen consistency across branches.
- I transform manual-heavy processes into agile, tech-driven workflows.
- I integrate retail tech to cut inefficiencies and improve delivery timelines.
- I guide automation initiatives to optimize turnaround and customer satisfaction.
- I align workflow automation with growth and operational excellence goals.

Academic Role:
- I’m working on workflow automation strategies that align tech with business growth.
- I research how automation can bridge operational gaps in retail and banking.
- I explore process engineering methods that improve turnaround performance.
- I study tech adoption strategies to enhance retail scalability.
- I develop models for risk reduction through workflow automation.
- I focus on academic research linking automation to transformation success.
- I investigate automation’s role in boosting operational consistency.
- I analyze digital workflow integration for sustainable business scaling.
- I evaluate automation-driven process improvements in the retail sector.
- I create frameworks combining tech innovation with business strategy.

Hybrid Role:
- I’m focused on automating digital workflows to drive consistency and scaling.
- I combine strategy and execution to deliver high-impact automation programs.
- I connect tech innovation with operational scaling in retail workflows.
- I integrate automation into core business processes to increase agility.
- I bridge business needs and tech capabilities for smooth workflow adoption.
- I design automation rollouts that fit both operational and growth needs.
- I merge leadership and tech skills to optimize turnaround processes.
- I balance execution speed with automation quality in retail operations.
- I deliver measurable gains by blending digital tools with process redesign.
- I craft automation strategies informed by both business and tech priorities.

Evangelist Role:
- I’m working on automation strategies that modernize enterprise operations.
- I advocate for automation as a driver of retail transformation.
- I share insights on scaling automation for competitive advantage.
- I champion digital workflow adoption to improve consistency.
- I inspire teams to embrace automation for operational excellence.
- I promote automation solutions that align with growth strategies.
- I spotlight best practices in transforming workflows through tech.
- I encourage industry peers to accelerate automation journeys.
- I communicate the business value of modernizing processes with automation.
- I drive conversations on automation as a catalyst for enterprise scaling.

5. Pivot into my current focus areas: automation, lending workflows, digital banking, or transformation strategies.
6. Follow this exact cadence:
   Prospect reference → My background (action-driven) → Connection invite.
7. End with a short, professional invitation to connect. Examples:
   - "Thought it’d be great to connect."
   - "Would love to connect."
   - "It would be great to connect."
   - "Would be glad to connect."
8. Sign off with: "Best, {my_name}" (must include my_name).
9. ABSOLUTELY FORBIDDEN words: exploring, interested, learning, no easy feat, impressive, noteworthy, remarkable, fascinating, admiring, inspiring, no small feat, no easy task, stood out.

EXAMPLE STRUCTURE (do not alter format):

Hi Dror,
Your focus on speeding up execution while embedding sustainable change across digital, ops, and CX caught my interest. I’m connecting with peers exploring similar shifts—would be great to stay in touch.
Best,
Joseph

Hi Sharon,
I find it interesting how mortgage lending leaders like you bring structure to such a dynamic space. Always good to connect with those shaping smart, steady growth. Let’s connect and trade notes.
Best,
Joseph

Hi Natalie, 
Working across digital and business tech must give you a unique view of how banks grow with tech. I often think about what it takes to move fast without breaking what works. As fintech professionals, I'd be delighted to connect with you.
Best,
Joseph

Hi Ronald,
Your steady hand in steering remittance payment ops—especially during crises—really stands out. I’ve always believed operational resilience in payments is both a challenge and an opportunity. Would be keen to connect.
Best, 
Joseph

Hi Lisa,
Your role leading retail ops at Simmons—after years across strategy, region ops, and project delivery—gives strong visibility into branch execution. I focus on automating retail workflows to reduce risk and improve turnaround. Would be glad to connect.
Best, 
Joseph

Hi Robert, 
Saw the 21 Locks project you supported — such a solid example of community-driven lending. I’m exploring how automation is reshaping commercial lending ops. Thought it’d be great to connect and share notes.
Best,
Joseph

Hi Maria, 
Caught your panel at COCC Amplify on multi-branding—sounds like an energizing space to be in. I’m exploring how banks are rethinking core innovation and ops with automation. Would be great to connect.
Best,
Joseph

Hi David, 
Saw your work at MissionFed and the Top Tech Awards. I’ve been exploring how credit unions are driving IT-led transformation in ops and member experience. Thought it’d be great to connect.
Best,
Joseph

Hi Vince, 
I’ve been exploring how credit unions are modernizing lending workflows across real estate, consumer, and indirect. Your role at MissionFed makes this even more relevant. Thought it’d be great to connect.
Best,
Joseph

Hi Ben, 
I’ve been exploring how mortgage lenders are simplifying workflows with automation while keeping the borrower experience personal. Your lending background at River Bank & Trust makes this especially relevant. Thought we could connect.
Best,
Joseph"

Hi Jana, 
I’ve been diving into how banks are applying AI and automation to rewire core systems and drive real CX value. Your work at CrossFirst sits right at that intersection. Thought it’d be great to connect.
Best,
Joseph"

Hi Patty, 
I’ve been exploring how credit unions are applying automation across IT and ops to strengthen resilience and streamline service. Your experience leading tech at CU SoCal makes this especially relevant. Thought it’d be great to connect.
Best,
Joseph

Hi Jay, 
Saw your session with Greenlight and your work at Bank of Ann Arbor. I’ve been digging into how community banks are using digital channels and fintech to drive CX. Thought it’d be great to connect and trade perspectives.
Best,
Joseph

Hi Mark, 
Your work with the Technology Industry Group at Bank of Ann Arbor is interesting. I’m exploring how banks supporting tech-driven businesses are modernizing lending and ops with automation. Would be great to connect.
Best,
Joseph"

Hi Duane, 
Noticed your work at First Northern Bank driving tech and innovation across ops and CX. I’m exploring how regional banks are using automation to align IT with business strategy. Thought it’d be great to connect.
Best,
Joseph

Hi Harold, 
I’ve been exploring how banks like First Northern are modernizing commercial lending to drive efficiency and better CX. Given your role and community leadership in Sacramento, thought it’d be great to connect.
Best,
Joseph

Hi Felecia, 
Your work shaping compliance programs across the mortgage space really stood out. I’ve been exploring how tech and automation are influencing compliance and ops in lending. Thought it’d be great to connect and share insights.
Best,
Joseph

Hi Dustin, 
Congrats on Archway’s momentum—your work at WaFd and Pike Street Labs is a great example of what’s possible when tech meets banking. I’ve been diving into how automation’s reshaping CX and ops. Would be great to connect.
Best,
Joseph

Hi John,
I came across your work in credit leadership at MidFirst. I’m always interested in how leaders like you balance credit soundness and growth in evolving markets. Would be great to connect and exchange ideas.
Best, 
Joseph

Now create for:
Prospect: {state['prospect_name']} ({state['designation']} at {state['company']})
Industry: {state['industry']}
Key Highlight: {state['prospect_background']}
My Background (optional if relevant): PhD in IT | Director, Digital Transformation at Speridian | AI & Digital Strategy | Healthcare Informatics | Social Computing Scientist

Message (STRICTLY 2–3 LINES, MAX 250 characters):
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

        # Ensure connection note is present# - DO NOT chain multiple achievements or roles.

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
