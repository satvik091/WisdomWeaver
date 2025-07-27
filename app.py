import streamlit as st
import os
import random
import pandas as pd
from datetime import datetime
from pydantic import BaseModel, Field
from typing import Optional
from langchain.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import Runnable
from PIL import Image
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- Configuration & Setup ---
# Securely loading API key (recommended)
# IMPORTANT: For production, remove the hardcoded key and rely *only* on .env or Streamlit secrets.
# You should ideally set this in your .env file or Streamlit secrets.
os.environ["GOOGLE_API_KEY"] = "AIzaSyA83KimfLvcMNWd7P8PEP7yzNC9V6ZnUDM" # Consider removing this line for better security

google_api_key = os.environ.get("GOOGLE_API_KEY") # Use .get() for safer access

if google_api_key is None:
    st.error("Error: GOOGLE_API_KEY environment variable not set. Please check your .env file or Streamlit secrets.")
    st.stop() # Stop the app if the key is missing

st.set_page_config(page_title="🕉️ Gita Wisdom", layout="wide")

# Constants
GITA_CSV_PATH = "bhagavad_gita_verses.csv"
IMAGE_PATH = "WhatsApp Image 2024-11-18 at 11.40.34_076eab8e.jpg"

# --- Data Loading ---
def load_verse_data(path):
    """Loads Bhagavad Gita verse data from a CSV file."""
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        st.error(f"Error: CSV file not found at {path}. Please ensure the 'try' folder and 'bhagavad_gita_verses.csv' exist.")
        st.stop()
        
    verses_db = {}
    for _, row in df.iterrows():
        chapter_num_str = str(row['chapter_number']).replace('Chapter ', '')
        
        try:
            chapter_key = f"chapter_{int(chapter_num_str)}"
        except ValueError:
            st.warning(f"Skipping row due to invalid chapter number format: {row['chapter_number']}")
            continue
        
        full_verse_ref = str(row['chapter_verse'])
        
        if chapter_key not in verses_db:
            verses_db[chapter_key] = {
                "title": f"Chapter {chapter_num_str}: {row['chapter_title']}",
                "summary": "",
                "verses": []
            }
        
        verses_db[chapter_key]["verses"].append({
            "chapter_number": chapter_num_str,
            "chapter_title": row.get("chapter_title", "N/N"),
            "chapter_verse": full_verse_ref,
            "translation": row.get("translation", "N/A"),
            "sanskrit": row.get("sanskrit_verse", "N/A")
        })
    return verses_db

# --- LLM Integration ---
class GitaVerseResponse(BaseModel):
    verse_reference: str = Field(description="Chapter and verse, e.g., 'Chapter 2, Verse 47'")
    sanskrit: str = Field(description="Original Sanskrit verse")
    translation: str = Field(description="English translation of the verse")
    explanation: str = Field(description="Explanation and context")
    application: str = Field(description="Practical application")

parser = PydanticOutputParser(pydantic_object=GitaVerseResponse)

prompt = PromptTemplate.from_template(
    """
You are a wise assistant trained in the teachings of the Bhagavad Gita.
Your goal is to provide profound and practical insights from the Bhagavad Gita.

Based on the user's question, identify the most relevant Bhagavad Gita verse (if applicable) and explain its wisdom.
Ensure the response is structured precisely according to the format instructions.
Make sure your response does not contain any stray characters, especially 'g' or 'y' at the beginning or end.

User's Question: {input}

{format_instructions}
""",
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.5, google_api_key=google_api_key)
chain: Runnable = prompt | llm | parser

# --- Session State Initialization ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "messages" not in st.session_state:
    st.session_state.messages = []

if "bot" not in st.session_state:
    st.session_state.bot = type("Bot", (), {})()
    st.session_state.bot.verses_db = load_verse_data(GITA_CSV_PATH)
    st.session_state.bot.themes = {
        "Detachment": "freedom from desire",
        "Karma Yoga": "selfless action",
        "Bhakti": "devotion",
        "Jnana": "knowledge",
        "Dharma": "duty",
        "Self-Realization": "understanding one's true nature",
        "Yoga": "union of body, mind, and spirit"
    }

if "favorite_verses" not in st.session_state:
    st.session_state.favorite_verses = []

if 'user_input_value' not in st.session_state:
    st.session_state.user_input_value = ""

# --- Helper Functions ---
def sort_key_for_verse(verse_data_item):
    """Helper function to sort verses by chapter and verse number."""
    verse_str = verse_data_item['chapter_verse']
    parts = verse_str.split('.')
    chapter_part = int(parts[0])
    
    verse_num_str = parts[1] if len(parts) > 1 else '0'
    if '–' in verse_num_str:
        try:
            main_verse_num = int(verse_num_str.split('–')[0].strip())
        except ValueError:
            main_verse_num = 0
    elif '-' in verse_num_str:
         try:
            main_verse_num = int(verse_num_str.split('-')[0].strip())
         except ValueError:
            main_verse_num = 0
    else:
        try:
            main_verse_num = int(verse_num_str.strip())
        except ValueError:
            main_verse_num = 0
    return (chapter_part, main_verse_num)

def handle_quick_actions(action_type):
    """Generates a question based on selected quick action."""
    if not hasattr(st.session_state.bot, 'verses_db') or not st.session_state.bot.verses_db:
        st.error("Verse data not loaded, cannot perform quick actions.")
        return ""

    if action_type == "random_verse":
        chapters = list(st.session_state.bot.verses_db.keys())
        if not chapters:
            return "No chapters found to pick a random verse from."
        
        chapter_key = random.choice(chapters)
        verses_in_chapter = st.session_state.bot.verses_db[chapter_key]["verses"]
        if not verses_in_chapter:
            return f"No verses found in chapter {chapter_key.split('_')[1]}."
        
        random_verse_data = random.choice(verses_in_chapter)
        return (f"Please share the wisdom from Chapter {random_verse_data['chapter_number']}, "
                f"Verse {random_verse_data['chapter_verse']} and its practical application.")

    elif action_type == "daily_reflection":
        today = datetime.now().strftime("%A")
        mood = st.session_state.get("current_mood", "seeking wisdom")
        theme = st.session_state.get("selected_theme", "any relevant theme")

        return (f"What guidance does the Bhagavad Gita offer for {today}, "
                f"considering a mood of '{mood}' and focusing on the theme of '{theme}'? "
                f"Please provide a verse for daily reflection and contemplation, along with its practical application.")
    return ""

def clear_chat_history():
    """Clears all chat history and the input text field."""
    st.session_state.messages = []
    st.session_state.chat_history = []
    st.session_state.user_input_value = "" # Ensures text input field is cleared
    st.toast("Chat history cleared!")

def clean_llm_output(text: str) -> str:
    """Removes common leading/trailing stray characters like 'g', 'y' and excessive whitespace."""
    # First, trim leading/trailing whitespace including newlines
    cleaned_text = text.strip()

    # Define characters to potentially remove from the very beginning or end
    stray_chars = ['g', 'y', 'G', 'Y', '`', '*', '_'] # Add any other common stray chars you observe

    # Remove stray characters from the beginning
    while cleaned_text and cleaned_text[0] in stray_chars:
        cleaned_text = cleaned_text[1:].strip()

    # Remove stray characters from the end
    while cleaned_text and cleaned_text[-1] in stray_chars:
        cleaned_text = cleaned_text[:-1].strip()
        
    return cleaned_text

# --- UI Renderers ---
def render_enhanced_sidebar():
    """Renders the interactive sidebar for Browse chapters and viewing journey details."""
    with st.sidebar:
        st.title("📖 Browse Sacred Texts")
        
        if not hasattr(st.session_state.bot, 'verses_db') or not st.session_state.bot.verses_db:
            st.info("Loading verse data...")
            st.session_state.bot.verses_db = load_verse_data(GITA_CSV_PATH)
            if not st.session_state.bot.verses_db:
                st.error("Failed to load Bhagavad Gita verses. Please check the CSV file and path.")
                return

        chapters = list(st.session_state.bot.verses_db.keys())
        chapters.sort(key=lambda x: int(x.split('_')[1]))

        selected_chapter_key = st.selectbox(
            "Select Chapter",
            chapters,
            format_func=lambda x: f"Ch. {x.split('_')[1]}: {st.session_state.bot.verses_db[x]['title']}"
        )

        if selected_chapter_key:
            chapter_data = st.session_state.bot.verses_db[selected_chapter_key]
            st.markdown(f"### {chapter_data['title']}")
            if chapter_data.get('summary'):
                st.markdown(f"*{chapter_data.get('summary')}*")
            
            sorted_verses = sorted(chapter_data['verses'], key=sort_key_for_verse)

            st.info(f"📊 {len(sorted_verses)} verses in this chapter")

            st.markdown("#### All Verses:")
            for verse_data in sorted_verses:
                with st.expander(f"Verse {verse_data['chapter_verse']}"):
                    st.markdown(f"**Chapter:** {verse_data['chapter_number']}")
                    st.markdown(f"**Verse:** {verse_data['chapter_verse']}")
                    if verse_data.get('sanskrit') and verse_data['sanskrit'] != "N/A":
                        st.markdown(f"**Sanskrit:** *{verse_data['sanskrit']}*")
                    st.markdown(f"**Translation:** {verse_data['translation']}")
                    
                    if st.button(f"Ask about {verse_data['chapter_verse']}", key=f"ask_verse_sidebar_{verse_data['chapter_verse']}"):
                        st.session_state.user_input_value = (
                            f"Please explain Chapter {verse_data['chapter_number']}, Verse {verse_data['chapter_verse']} "
                            f"and its practical application in modern life."
                        )
                        st.rerun()

        st.markdown("---")
        st.title("💭 Your Spiritual Journey")
        user_qs = [m["content"] for m in st.session_state.messages if m["role"] == "user"]
        if user_qs:
            for i, q in enumerate(user_qs[-5:], 1):
                with st.expander(f"Question {len(user_qs) - len(user_qs[-5:]) + i}"):
                    st.markdown(f"*{q}*")
        else:
            st.info("No questions asked yet.")

        st.markdown("---")
        st.title("⭐ Favorite Verses")
        if st.session_state.favorite_verses:
            for fav in st.session_state.favorite_verses:
                st.markdown(f"• {fav}")
        else:
            st.info("No favorites saved yet. (Feature coming soon!)")

def render_additional_options():
    """Renders personalization options and quick action buttons."""
    st.markdown("### 🎯 Personalize Your Spiritual Journey")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.selectbox("🎭 Current Mood", ["Seeking Wisdom", "Feeling Confused", "Need Motivation", "Seeking Peace",
            "Facing Challenges", "Grateful", "Contemplative"], key="current_mood")

    with col2:
        st.selectbox("📚 Focus Theme", list(st.session_state.bot.themes.keys()), key="selected_theme")

    with col3:
        st.selectbox("🌐 Response Style", ["Detailed", "Concise", "Contemplative", "Practical"], key="response_style")

    st.markdown("### ⚡ Quick Actions")
    action_col1, action_col2, _, _ = st.columns(4)

    with action_col1:
        if st.button("🎲 Random Verse"):
            st.session_state.user_input_value = handle_quick_actions("random_verse")
            st.rerun()
    with action_col2:
        if st.button("💭 Daily Reflection"):
            st.session_state.user_input_value = handle_quick_actions("daily_reflection")
            st.rerun()

# --- MAIN APP LAYOUT AND LOGIC ---

# Apply custom CSS for a rounder input field and buttons
st.markdown(
    """
    <style>
    /* Targeting the input element within the text input widget */
    div.stTextInput > label > div > div > input {
        border-radius: 20px !important;
        padding: 10px 15px !important;
        border: 1px solid #d3d3d3;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    /* To round the button as well */
    div.stButton > button {
        border-radius: 20px !important;
        padding: 10px 20px !important;
    }
    /* CSS for centering the image */
    .centered-image {
        display: flex;
        justify-content: center;
        width: 100%;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Image above title, centered
col_left, col_center, col_right = st.columns([1,2,1]) 

with col_center:
    try:
        st.image(IMAGE_PATH, width=200) # Adjust width as needed
    except FileNotFoundError:
        st.warning(f"Image not found at {IMAGE_PATH}. Please ensure the file exists.")

st.title("🕉️ Bhagavad Gita Wisdom Weaver")

# Render sidebar
render_enhanced_sidebar()

# Static Personalization Options at the Top of Main Content
render_additional_options()

# Chat History Display Area Placeholder
chat_history_placeholder = st.empty()

# Input and Button Section
col1, col2 = st.columns([8, 1]) 

with col1:
    user_query = st.text_input(
        "🙏 Ask about the Gita:",
        value=st.session_state.user_input_value, # This links the input to session_state
        key="main_input",
        placeholder="Type your question here...",
    )

with col2:
    st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True) 
    submit_button = st.button("Ask")

# Submission Logic
if submit_button and user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    st.session_state.chat_history.append(("You", user_query))

    with st.spinner("Meditating on the verses..."):
        try:
            mood = st.session_state.get("current_mood", "neutral")
            theme = st.session_state.get("selected_theme", "general wisdom")
            response_style = st.session_state.get("response_style", "detailed")

            full_prompt_input = (
                f"User's mood: {mood}. Focus theme: {theme}. Response style: {response_style}.\n\n"
                f"Question: {user_query}"
            )

            result: GitaVerseResponse = chain.invoke({"input": full_prompt_input})
            
            # Apply cleaning to each field before forming the final response text
            cleaned_verse_reference = clean_llm_output(result.verse_reference)
            cleaned_sanskrit = clean_llm_output(result.sanskrit)
            cleaned_translation = clean_llm_output(result.translation)
            cleaned_explanation = clean_llm_output(result.explanation)
            cleaned_application = clean_llm_output(result.application)

            response_text = f"""
📖 **{cleaned_verse_reference}**

🕉️ **Sanskrit:** {cleaned_sanskrit}

🌐 **Translation:** {cleaned_translation}

🧠 **Explanation:** {cleaned_explanation}

💡 **Application:** {cleaned_application}
""".strip()
            
            # One final pass over the whole response text just in case
            response_text = clean_llm_output(response_text)

            st.session_state.messages.append({"role": "assistant", "content": response_text})
            st.session_state.chat_history.append(("GitaBot", response_text))

        except Exception as e:
            st.error(f"Error invoking the wisdom: {e}. Please try again or rephrase your question.")
            st.error(f"Detailed error: {e}")

    # This line ensures the input field is cleared AFTER a query is processed.
    st.session_state.user_input_value = ""
    st.rerun() # Forces a re-render to reflect the cleared input field

# Display Chat History
with chat_history_placeholder.container():
    st.markdown("---") # Separator before the chat history
    
    # Reset History Button (aligned left, just above chat history text)
    if st.button("🔄 Reset History", on_click=clear_chat_history, key="reset_history_button"):
        pass # The on_click handler will trigger clear_chat_history() and a rerun

    st.markdown("## 🗣️ Chat History")
    for i, (sender, msg) in enumerate(reversed(st.session_state.chat_history)):
        with st.chat_message(name=sender):
            st.markdown(msg, unsafe_allow_html=True)
            if sender == "GitaBot":
                if msg not in st.session_state.favorite_verses:
                    if st.button("⭐ Save to Favorites", key=f"fav_btn_{i}"):
                        st.session_state.favorite_verses.append(msg)
                        st.toast("Verse added to favorites!")
                        st.rerun()

# --- ABOUT THIS APPLICATION (at the very bottom) ---
st.markdown("---") # Optional separator
st.markdown(
    """
    <div style="text-align: center; margin-top: 100px; font-size: big; color: grey;">
    💫 About This Application<br>
    This application uses Google's Gemini AI to provide insights from the Bhagavad Gita. The wisdom shared here is meant for reflection and guidance. For deeper spiritual understanding, please consult with qualified spiritual teachers and study the original texts.<br>
    Built with ❤️ for spiritual seekers everywhere
    </div>
    """,
    unsafe_allow_html=True
)