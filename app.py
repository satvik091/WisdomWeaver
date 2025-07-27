import requests
import pandas as pd
import streamlit as st
import os
import google.generativeai as genai
from typing import Dict, List
import json
import asyncio
from PIL import Image
import time
from datetime import datetime
import re
from dotenv import load_dotenv

load_dotenv()

# Constants
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "YOUR_API_KEY_HERE")
GITA_CSV_PATH = "bhagavad_gita_verses.csv"
IMAGE_PATH = "WhatsApp Image 2024-11-18 at 11.40.34_076eab8e.jpg"

def initialize_session_state():
    """Initialize Streamlit session state variables with better defaults."""
    default_states = {
        'messages': [],
        'bot': None,
        'selected_theme': 'Life Guidance',
        'question_history': [],
        'favorite_verses': [],
        'current_mood': 'Seeking Wisdom',
        'language_preference': 'English'
    }
    
    for key, default_value in default_states.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

    # Initialize bot if not already done
    if st.session_state.bot is None:
        if not GEMINI_API_KEY:
            st.error("Please set the GEMINI_API_KEY in your configuration.")
            st.stop()
        st.session_state.bot = GitaGeminiBot(GEMINI_API_KEY)

class GitaGeminiBot:
    def __init__(self, api_key: str):
        """Initialize the Gita bot with Gemini API and enhanced features."""
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash')
        self.verses_db = self.load_gita_database()
        self.themes = {
            'Life Guidance': 'guidance for life decisions and personal growth',
            'Dharma & Ethics': 'understanding of duty, righteousness, and moral conduct',
            'Spiritual Growth': 'spiritual development and self-realization',
            'Relationships': 'wisdom for interpersonal relationships and social harmony',
            'Work & Career': 'guidance for professional life and service',
            'Inner Peace': 'achieving mental tranquility and emotional balance',
            'Devotion & Love': 'understanding devotion, love, and surrender'
        }

    @st.cache_data
    def load_gita_database(_self) -> Dict:
        """Load the Bhagavad Gita dataset with caching for better performance."""
        try:
            verses_df = pd.read_csv(GITA_CSV_PATH)
        except FileNotFoundError:
            st.error(f"Gita database file '{GITA_CSV_PATH}' not found. Please ensure the file is in the correct location.")
            st.stop()
        except Exception as e:
            st.error(f"Error loading Gita database: {str(e)}")
            st.stop()

        verses_db = {}
        for _, row in verses_df.iterrows():
            chapter = f"chapter_{row['chapter_number']}"
            if chapter not in verses_db:
                verses_db[chapter] = {
                    "title": row['chapter_title'],
                    "verses": {},
                    "summary": _self._get_chapter_summary(row['chapter_number'])
                }
            verse_num = str(row['chapter_verse'])
            verses_db[chapter]["verses"][verse_num] = {
                "translation": row['translation']
            }
        return verses_db

    def _get_chapter_summary(self, chapter_num: int) -> str:
        """Get a brief summary for each chapter."""
        summaries = {
            1: "Arjuna's moral dilemma and the beginning of Krishna's counsel",
            2: "The fundamental teachings on the soul, duty, and the path of knowledge",
            3: "The path of selfless action and karma yoga",
            4: "Divine knowledge, incarnation, and the evolution of dharma",
            5: "The harmony between action and renunciation",
            6: "The practice of meditation and self-control",
            7: "Knowledge of the Absolute and devotion to the Divine",
            8: "The imperishable Brahman and the path at the time of death",
            9: "Royal knowledge and the most confidential wisdom",
            10: "Divine manifestations and infinite glories",
            11: "The cosmic vision of the universal form",
            12: "The path of devotion and love",
            13: "The field of activity and the knower of the field",
            14: "The three modes of material nature",
            15: "The supreme person and the cosmic tree",
            16: "Divine and demonic natures in human beings",
            17: "The three divisions of faith and their characteristics",
            18: "The perfection of renunciation and complete surrender"
        }
        return summaries.get(chapter_num, "Eternal wisdom and guidance")

    def format_response(self, raw_text: str) -> Dict:
        """Enhanced response formatting with better error handling."""
        try:
            # Try JSON parsing first
            if raw_text.strip().startswith('{') and raw_text.strip().endswith('}'):
                try:
                    return json.loads(raw_text)
                except json.JSONDecodeError:
                    pass

            # Enhanced text parsing
            response = {
                "verse_reference": "",
                "sanskrit": "",
                "translation": "",
                "explanation": "",
                "application": "",
                "keywords": []
            }

            lines = [line.strip() for line in raw_text.split('\n') if line.strip()]
            current_section = None
            
            for line in lines:
                line_lower = line.lower()
                
                # Better pattern matching
                if re.search(r'chapter\s+\d+.*verse\s+\d+', line_lower):
                    response["verse_reference"] = line
                elif line_lower.startswith(('sanskrit:', 'verse:')):
                    response["sanskrit"] = re.sub(r'^(sanskrit:|verse:)\s*', '', line, flags=re.IGNORECASE)
                elif line_lower.startswith('translation:'):
                    response["translation"] = re.sub(r'^translation:\s*', '', line, flags=re.IGNORECASE)
                elif line_lower.startswith(('explanation:', 'meaning:')):
                    current_section = "explanation"
                    response["explanation"] = re.sub(r'^(explanation:|meaning:)\s*', '', line, flags=re.IGNORECASE)
                elif line_lower.startswith(('application:', 'practical:')):
                    current_section = "application"
                    response["application"] = re.sub(r'^(application:|practical:)\s*', '', line, flags=re.IGNORECASE)
                elif current_section and line:
                    response[current_section] += " " + line

            # Extract keywords for better searchability
            text_content = f"{response['translation']} {response['explanation']} {response['application']}"
            response["keywords"] = self._extract_keywords(text_content)

            return response

        except Exception as e:
            st.error(f"Error formatting response: {str(e)}")
            return {
                "verse_reference": "Error in parsing",
                "sanskrit": "",
                "translation": raw_text[:500] + "..." if len(raw_text) > 500 else raw_text,
                "explanation": "Please try rephrasing your question.",
                "application": "",
                "keywords": []
            }

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract relevant keywords from the response text."""
        common_gita_keywords = [
            'dharma', 'karma', 'moksha', 'yoga', 'devotion', 'meditation', 'duty', 
            'righteousness', 'soul', 'divine', 'surrender', 'detachment', 'wisdom',
            'knowledge', 'action', 'service', 'love', 'peace', 'truth'
        ]
        
        text_lower = text.lower()
        found_keywords = [keyword for keyword in common_gita_keywords if keyword in text_lower]
        return found_keywords[:5]

    async def get_response(self, question: str, theme: str = None, mood: str = None) -> Dict:
        """Enhanced response generation with theme and mood context."""
        try:
            # Build context-aware prompt
            theme_context = ""
            if theme and theme in self.themes:
                theme_context = f"Focus on {self.themes[theme]}. "
            
            mood_context = ""
            if mood:
                mood_context = f"The user is currently {mood.lower()}. "

            prompt = f"""
            {theme_context}{mood_context}Based on the Bhagavad Gita's teachings, provide guidance for this question:
            {question}

            Please format your response exactly like this:
            Chapter X, Verse Y
            Sanskrit: [Sanskrit verse if available]
            Translation: [Clear English translation]
            Explanation: [Detailed explanation of the verse's meaning and context]
            Application: [Practical guidance for applying this wisdom in modern life]

            Make the response comprehensive but accessible to modern readers.
            """

            # Add retry logic for API calls
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = self.model.generate_content(prompt)
                    if response.text:
                        break
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise e
                    time.sleep(1)  # Brief pause before retry

            if not response.text:
                raise ValueError("Empty response received from the model")

            formatted_response = self.format_response(response.text)
            
            # Add metadata
            formatted_response["timestamp"] = datetime.now().isoformat()
            formatted_response["theme"] = theme
            formatted_response["mood"] = mood
            
            return formatted_response

        except Exception as e:
            st.error(f"Error getting response: {str(e)}")
            return {
                "verse_reference": "Service Temporarily Unavailable",
                "sanskrit": "",
                "translation": "We're experiencing technical difficulties. Please try again in a moment.",
                "explanation": "The wisdom of the Gita teaches us patience in times of difficulty.",
                "application": "Take this moment to practice patience and try your question again.",
                "keywords": ["patience", "perseverance"],
                "timestamp": datetime.now().isoformat(),
                "theme": theme,
                "mood": mood
            }

def render_additional_options():
    """Render additional options below the image."""
    st.markdown("### 🎯 Personalize Your Spiritual Journey")
    
    # Create columns for better layout
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.selectbox(
            "🎭 Current Mood",
            ["Seeking Wisdom", "Feeling Confused", "Need Motivation", "Seeking Peace", 
             "Facing Challenges", "Grateful", "Contemplative"],
            key="current_mood",
            help="Your current state of mind helps tailor the guidance"
        )
    
    with col2:
        st.selectbox(
            "📚 Focus Theme",
            list(st.session_state.bot.themes.keys()),
            key="selected_theme",
            help="Choose the area where you seek guidance"
        )
    
    with col3:
        st.selectbox(
            "🌐 Response Style",
            ["Detailed", "Concise", "Contemplative", "Practical"],
            key="response_style",
            help="How would you like the wisdom to be presented?"
        )

    # Quick action buttons
    st.markdown("### ⚡ Quick Actions")
    
    action_col1, action_col2, action_col3, action_col4 = st.columns(4)
    
    with action_col1:
        if st.button("🎲 Random Verse", help="Get a random verse for inspiration"):
            return "random_verse"
    
    with action_col2:
        if st.button("💭 Daily Reflection", help="Get guidance for daily contemplation"):
            return "daily_reflection"
    
    return None

def handle_quick_actions(action_type):
    """Handle quick action button clicks."""
    if action_type == "random_verse":
        # Get random verse
        import random
        chapters = list(st.session_state.bot.verses_db.keys())
        random_chapter = random.choice(chapters)
        verses = list(st.session_state.bot.verses_db[random_chapter]["verses"].keys())
        random_verse = random.choice(verses)
        
        chapter_num = random_chapter.split('_')[1]
        question = f"Please share the wisdom from Chapter {chapter_num}, Verse {random_verse} and its practical application."
        return question
    
    elif action_type == "daily_reflection":
        today = datetime.now().strftime("%A")
        question = f"What guidance does the Bhagavad Gita offer for {today}? Please provide a verse for daily reflection and contemplation."
        return question
    
    elif action_type == "verse_search":
        st.session_state.show_search = True
        return None
    
    elif action_type == "chapter_summary":
        st.session_state.show_chapter_summary = True
        return None
    
    return None

def render_enhanced_sidebar():
    """Enhanced sidebar with better organization."""
    st.sidebar.title("📖 Browse Sacred Texts")
    
    # Chapter browser with enhanced info
    chapters = list(st.session_state.bot.verses_db.keys())
    selected_chapter = st.sidebar.selectbox(
        "Select Chapter",
        chapters,
        format_func=lambda x: f"Ch. {x.split('_')[1]}: {st.session_state.bot.verses_db[x]['title']}"
    )

    if selected_chapter:
        chapter_data = st.session_state.bot.verses_db[selected_chapter]
        st.sidebar.markdown(f"### {chapter_data['title']}")
        st.sidebar.markdown(f"{chapter_data.get('summary', '')}")
        
        # Show verse count
        verse_count = len(chapter_data['verses'])
        st.sidebar.info(f"📊 {verse_count} verses in this chapter")
        
        # Expandable verses
        verses = chapter_data['verses']
        for verse_num, verse_data in list(verses.items())[:5]:
            with st.sidebar.expander(f"Verse {verse_num}"):
                st.markdown(verse_data['translation'][:150] + "..." if len(verse_data['translation']) > 150 else verse_data['translation'])
        
        if len(verses) > 5:
            st.sidebar.info(f"+ {len(verses) - 5} more verses in this chapter")

    # Enhanced question history
    st.sidebar.markdown("---")
    st.sidebar.title("💭 Your Spiritual Journey")
    
    user_questions = [msg["content"] for msg in st.session_state.messages if msg["role"] == "user"]
    if user_questions:
        st.sidebar.markdown(f"*Questions Asked:* {len(user_questions)}")
        
        # Show recent questions with timestamps
        for i, q in enumerate(user_questions[-5:], 1):
            with st.sidebar.expander(f"Question {len(user_questions) - 5 + i}"):
                st.markdown(f"{q}")
    else:
        st.sidebar.info("🌱 Begin your journey by asking a question")

    # Favorites section (placeholder for future enhancement)
    st.sidebar.markdown("---")
    st.sidebar.title("⭐ Favorite Verses")
    if st.session_state.favorite_verses:
        for fav in st.session_state.favorite_verses:
            st.sidebar.markdown(f"• {fav}")
    else:
        st.sidebar.info("No favorites saved yet")

def get_chat_history_as_text():
    history = st.session_state.get('messages', [])
    text = ""
    for message in history:
        if message["role"] == "user":
            text += f"User: {message['content']}\n\n"
        elif message["role"] == "assistant":
            # Format the assistant's detailed response
            formatted_response = ""
            if message.get("verse_reference"):
                formatted_response += f"📖 {message['verse_reference']}\n"
            if message.get('sanskrit'):
                formatted_response += f"Sanskrit: {message['sanskrit']}\n"
            if message.get('translation'):
                formatted_response += f"Translation: {message['translation']}\n"
            if message.get('explanation'):
                formatted_response += f"Understanding: {message['explanation']}\n"
            if message.get('application'):
                formatted_response += f"Modern Application: {message['application']}\n"
            if message.get('keywords'):
                formatted_response += f"Key Concepts: {' • '.join(message['keywords'])}\n"
            text += f"WisdomWeaver:\n{formatted_response}\n"
    
    return text

def main():
    """Enhanced main Streamlit application."""
    st.set_page_config(
        page_title="Bhagavad Gita Wisdom Weaver",
        page_icon="🕉",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Load and display image
    if os.path.exists(IMAGE_PATH):
        try:
            image = Image.open(IMAGE_PATH)
            max_width = 800
            aspect_ratio = image.height / image.width
            resized_image = image.resize((max_width, int(max_width * aspect_ratio)))
            st.image(resized_image, use_container_width=True, caption="Bhagavad Gita - Eternal Wisdom")
        except Exception as e:
            st.error(f"Error loading image: {str(e)}")
    else:
        st.warning("Image file not found. Please ensure the image is in the correct location.")

    initialize_session_state()

    # Render additional options below image
    quick_action = render_additional_options()
    
    # Handle quick actions
    if quick_action:
        auto_question = handle_quick_actions(quick_action)
        if auto_question:
            st.session_state.messages.append({"role": "user", "content": auto_question})
            with st.spinner("Contemplating your question..."):
                response = asyncio.run(st.session_state.bot.get_response(
                    auto_question, 
                    st.session_state.selected_theme,
                    st.session_state.current_mood
                ))
                st.session_state.messages.append({
                    "role": "assistant",
                    **response
                })
                st.rerun()

    # Main content area
    col1, col2 = st.columns([2, 1])

    with col1:
        st.title("🕉 Bhagavad Gita Wisdom")
        st.markdown("""
        Ask questions about life, dharma, and spirituality to receive guidance from the timeless wisdom of the Bhagavad Gita.
        Personalize your experience using the options above.
        """)

        # This section below is where the conflicts were. I have merged your code with a change from 'main'.
        # Your download button code is kept and is now below the main chat messages.
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                if message["role"] == "user":
                    st.markdown(message["content"])
                else:
                    # Enhanced assistant message display
                    if message.get("verse_reference"):
                        st.markdown(f"📖 {message['verse_reference']}")
                    
                    if message.get('sanskrit'):
                        st.markdown(f"Sanskrit: {message['sanskrit']}")
                    
                    if message.get('translation'):
                        st.markdown(f"*Translation:* {message['translation']}")
                    
                    if message.get('explanation'):
                        st.markdown("### 🧠 Understanding")
                        st.markdown(message["explanation"])
                    
                    if message.get('application'):
                        st.markdown("### 🌟 Modern Application")
                        st.markdown(message["application"])
                    
                    # Show keywords if available
                    if message.get('keywords'):
                        st.markdown("*Key Concepts:* " + " • ".join([f"{kw}" for kw in message['keywords']]))

        # Add the download button
        chat_text_to_download = get_chat_history_as_text()
        if chat_text_to_download:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filename = f"WisdomWeaver_Chat_{timestamp}.txt"
            
            st.download_button(
                label="Download Chat History",
                data=chat_text_to_download,
                file_name=filename,
                mime="text/plain",
                key="download_chat_button"
            )

        # Enhanced chat input
        if question := st.chat_input("Ask your question here..."):
            st.session_state.messages.append({"role": "user", "content": question})

            with st.spinner("🧘 Contemplating your question..."):
                response = asyncio.run(st.session_state.bot.get_response(
                    question,
                    st.session_state.selected_theme,
                    st.session_state.current_mood
                ))
                st.session_state.messages.append({
                    "role": "assistant",
                    **response
                })
                st.rerun()

    with col2:
        render_enhanced_sidebar()

    # Enhanced footer
    st.markdown("---")
    st.markdown(
        """
        💫 *About This Application*
        
        This application uses Google's Gemini AI to provide insights from the Bhagavad Gita. 
        The wisdom shared here is meant for reflection and guidance. For deeper spiritual 
        understanding, please consult with qualified spiritual teachers and study the 
        original texts.
        
        Built with ❤ for spiritual seekers everywhere
        """
    )

if __name__ == "__main__":
    main()