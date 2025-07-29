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
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
from collections import Counter, deque

# Import the advanced emotion detector
from emotion_advanced import AdvancedEmotionDetector

load_dotenv()

# Constants
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "YOUR_API_KEY")
GITA_CSV_PATH = "bhagavad_gita_verses.csv"
IMAGE_PATH = "Public/Images/WhatsApp Image 2024-11-18 at 11.40.34_076eab8e.jpg"

def initialize_session_state():
    """Initialize Streamlit session state variables with better defaults."""
    default_states = {
        'messages': [],
        'bot': None,
        'selected_theme': 'Life Guidance',
        'question_history': [],
        'favorite_verses': [],
        'current_mood': 'Seeking Wisdom',
        'emotional_state': 'Neutral',
        'language_preference': 'English',
        'webcam_enabled': False,
        'emotion_detector': None,
        'emotion_log': deque(maxlen=300),
        'last_detected_emotion': None
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

    # Initialize emotion detector once
    if st.session_state.emotion_detector is None:
        st.session_state.emotion_detector = AdvancedEmotionDetector()


def dominant_emotion(window_sec: int = 5) -> str:
    """
    Return the emotion that occurred most often in the last <window_sec> seconds
    on the webcam feed.  Falls back to the sidebar selection when nothing found.
    """
    ctx = st.session_state.get("webrtc_ctx")
    if ctx and ctx.state.playing and ctx.video_processor:
        cutoff = time.time() - window_sec
        recent = [e for ts, e in ctx.video_processor.emotion_history if ts >= cutoff]
        if recent:
            dom = Counter(recent).most_common(1)[0][0]
            # Don't modify session state - just return the detected emotion
            return dom
    return st.session_state.emotional_state


class EmotionTransformer(VideoTransformerBase):
    """WebRTC video transformer for emotion detection."""
    
    def __init__(self):
        # One detector per peer‑connection
        self.detector = AdvancedEmotionDetector()
        # Ring‑buffer of (timestamp, emotion) tuples – 5 s at 30 fps ≈ 150
        self.emotion_history: deque = deque(maxlen=150)
    
    def recv(self, frame):
        """Process each frame for emotion detection."""
        try:
            # Convert frame to numpy array
            img = frame.to_ndarray(format="bgr24")
            
            # Flip frame horizontally for mirror effect
            img = cv2.flip(img, 1)
            
            # Only proceed if detector is available
            if self.detector is not None:
                # Detect faces
                faces = self.detector.detect_faces_optimized(img)
                
                if len(faces) > 0:
                    # Process only the best face
                    x, y, w, h = faces[0]
                    padding = 30
                    y1 = max(0, y - padding)
                    y2 = min(img.shape[0], y + h + padding)
                    x1 = max(0, x - padding)
                    x2 = min(img.shape[1], x + w + padding)
                    face_roi = img[y1:y2, x1:x2]
                    
                    if face_roi.size > 0:
                        self.detector.update_emotion_async(face_roi)
                        # Save latest emotion for the GUI thread
                        if hasattr(self.detector, "current_emotion") and self.detector.current_emotion:
                            self.emotion_history.append(
                                (time.time(), self.detector.current_emotion)
                            )
                    
                    # Draw results on frame
                    self.detector.draw_advanced_results(img, faces)
            
            # Try different VideoFrame import approaches
            try:
                from streamlit_webrtc.models import VideoFrame
                return VideoFrame.from_ndarray(img, format="bgr24")
            except ImportError:
                try:
                    import av
                    # Create VideoFrame using av library
                    av_frame = av.VideoFrame.from_ndarray(img, format="bgr24")
                    return av_frame
                except ImportError:
                    # Fallback: return the original frame
                    return frame
            
        except Exception as e:
            print(f"Error in emotion detection: {e}")
            # Return original frame on error
            return frame

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
        return found_keywords[:5]  # Return top 5 relevant keywords

    async def get_response(self, question: str, theme: str = None, mood: str = None, emotional_state: str = None) -> Dict:
        """Enhanced response generation with theme, mood, and emotional state context."""
        try:
            # Build context-aware prompt
            theme_context = ""
            if theme and theme in self.themes:
                theme_context = f"Focus on {self.themes[theme]}. "
            
            mood_context = ""
            if mood:
                mood_context = f"The user is currently {mood.lower()}. "

            emotional_context = ""
            if emotional_state:
                emotional_context = f"The user's emotional state is {emotional_state.lower()}. Please provide guidance that acknowledges and addresses this emotional state. "

            prompt = f"""
            {theme_context}{mood_context}{emotional_context}Based on the Bhagavad Gita's teachings, provide guidance for this question:
            {question}

            Please format your response exactly like this:
            Chapter X, Verse Y
            Sanskrit: [Sanskrit verse if available]
            Translation: [Clear English translation]
            Explanation: [Detailed explanation of the verse's meaning and context, considering the user's emotional state]
            Application: [Practical guidance for applying this wisdom in modern life, tailored to the user's current emotional state]

            Make the response comprehensive but accessible to modern readers, with special attention to providing comfort and guidance appropriate for someone who is {emotional_state.lower() if emotional_state else 'seeking wisdom'}.
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
            formatted_response["emotional_state"] = emotional_state
            
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
                "mood": mood,
                "emotional_state": emotional_state
            }

def render_additional_options():
    """Render additional options below the image, including webcam with emotion detection."""
    
    # --- NEW: keep UI in-sync with webcam ---
    if st.session_state.get("webcam_enabled"):
        detected = dominant_emotion()
        if detected and detected != st.session_state.get("last_detected_emotion"):
            st.session_state.emotional_state = detected
            st.session_state.last_detected_emotion = detected
    # ----------------------------------------
    
    st.markdown("### 🎯 Personalize Your Spiritual Journey")
    
    # Create columns for better layout - now 4 columns instead of 3
    col1, col2, col3, col4 = st.columns(4)
    
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
    
    with col4:
        st.selectbox(
            "💭 Emotional State",
            ["Neutral", "Happy", "Sad", "Angry", "Fear", "Surprise", "Disgust"],
            key="emotional_state",
            help="Your current emotional state for personalized guidance"
        )

    # Webcam Section
    st.markdown("### 📹 Spiritual Presence & Emotion Detection")
    webcam_col1, webcam_col2 = st.columns([1, 3])
    with webcam_col1:
        webcam_enabled = st.checkbox(
            "Enable Webcam with Emotion Detection",
            key="webcam_enabled",
            help="Enable webcam and emotion detection for mindful presence"
        )

    if webcam_enabled:
        # WebRTC Configuration for better connectivity
        rtc_configuration = RTCConfiguration({
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        })
        
        st.info("🎥 Webcam with emotion detection is now active. You can continue chatting while the camera runs!")
        
        # Create a smaller container for the webcam feed
        webcam_container = st.container()
        with webcam_container:
            # Use columns to control the width - making it smaller
            cam_col1, cam_col2, cam_col3 = st.columns([1, 2, 1])
            with cam_col2:
                # Start WebRTC streamer with emotion detection and higher resolution
                ctx = webrtc_streamer(
                    key="gita_webcam",
                    video_transformer_factory=EmotionTransformer,
                    rtc_configuration=rtc_configuration,
                    media_stream_constraints={
                        "video": {
                            "width": {"ideal": 1280, "min": 640, "max": 1920},
                            "height": {"ideal": 720, "min": 480, "max": 1080},
                            "frameRate": {"ideal": 30, "min": 15, "max": 60}
                        }, 
                        "audio": False
                    },
                    async_processing=True
                )
                # Expose ctx so the main thread can read the emotion history
                if ctx:
                    st.session_state["webrtc_ctx"] = ctx
    
    # Quick action buttons
    st.markdown("### ⚡ Quick Actions")
    
    action_col1, action_col2, action_col3, action_col4 = st.columns(4)
    
    with action_col1:
        if st.button("🎲 Random Verse", help="Get a random verse for inspiration"):
            return "random_verse"
    
    with action_col2:
        if st.button("💭 Daily Reflection", help="Get guidance for daily contemplation"):
            return "daily_reflection"
    
    with action_col3:
        if st.button("🔍 Verse Search", help="Search for specific verses"):
            return "verse_search"
    
    with action_col4:
        if st.button("📖 Chapter Summary", help="Get a summary of any chapter"):
            return "chapter_summary"

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
    """Enhanced sidebar with better organization - showing ALL verses."""
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
        st.sidebar.markdown(f"*{chapter_data.get('summary', '')}*")
        
        # Show verse count
        verse_count = len(chapter_data['verses'])
        st.sidebar.info(f"📊 {verse_count} verses in this chapter")
        
        # Show ALL verses instead of just top 5
        verses = chapter_data['verses']
        st.sidebar.markdown("#### All Verses:")
        
        # Create a scrollable container for all verses
        for verse_num, verse_data in verses.items():
            with st.sidebar.expander(f"Verse {verse_num}"):
                # Show full translation for shorter verses, truncate longer ones
                translation = verse_data['translation']
                if len(translation) > 200:
                    st.markdown(translation[:200] + "...")
                else:
                    st.markdown(translation)
                
                # Add a button to use this verse for questioning
                if st.button(f"Ask about this verse", key=f"ask_verse_{selected_chapter}_{verse_num}"):
                    chapter_num = selected_chapter.split('_')[1]
                    question = f"Please explain Chapter {chapter_num}, Verse {verse_num} and its practical application in modern life."
                    st.session_state.auto_question = question

    # Enhanced question history
    st.sidebar.markdown("---")
    st.sidebar.title("💭 Your Spiritual Journey")
    
    user_questions = [msg["content"] for msg in st.session_state.messages if msg["role"] == "user"]
    if user_questions:
        st.sidebar.markdown(f"**Questions Asked:** {len(user_questions)}")
        
        # Show recent questions with timestamps
        for i, q in enumerate(user_questions[-5:], 1):  # Show last 5 questions
            with st.sidebar.expander(f"Question {len(user_questions) - 5 + i}"):
                st.markdown(f"*{q}*")
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

def create_downloadable_content(chat_history: List[Dict]) -> str:
    """Formats the chat history into a readable string for download."""
    content = f"--- Wisdom Weaver Chat History - {datetime.now().strftime('%Y-%m-%d %H:%M')} ---\n\n"
    for message in chat_history:
        role = message["role"]
        
        if role == "user":
            content += f"User: {message['content']}\n\n"
        else:
            # Handle the structured AI response
            content += f"Wisdom Weaver: "
            if message.get("verse_reference"):
                content += f"📖 {message['verse_reference']}\n"
            if message.get("sanskrit"):
                content += f"Sanskrit: {message['sanskrit']}\n"
            if message.get("translation"):
                content += f"Translation: {message['translation']}\n"
            if message.get("explanation"):
                content += f"Explanation: {message['explanation']}\n"
            if message.get("application"):
                content += f"Modern Application: {message['application']}\n"
            content += "\n"
            
    content += "--- End of Chat History ---"
    return content

def main():
    """Enhanced main Streamlit application."""
    st.set_page_config(
        page_title="Bhagavad Gita Wisdom Weaver",
        page_icon="🕉️",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Initialize session state first, before any other operations
    initialize_session_state()

    # Load and display image with reduced width
    if os.path.exists(IMAGE_PATH):
        try:
            image = Image.open(IMAGE_PATH)
            # Increased max_width from 500 to 850 for larger image
            max_width = 1800
            aspect_ratio = image.height / image.width
            resized_image = image.resize((max_width, int(max_width * aspect_ratio)))
            
            # Center the image by using columns with better proportions
            col_img1, col_img2, col_img3 = st.columns([2, 1, 2])
            with col_img2:
                st.image(resized_image, use_container_width=True, caption="Bhagavad Gita - Eternal Wisdom")
        except Exception as e:
            st.error(f"Error loading image: {str(e)}")
    else:
        st.warning("Image file not found. Please ensure the image is in the correct location.")

    # Check for auto question from sidebar verse buttons
    if hasattr(st.session_state, 'auto_question'):
        st.session_state.messages.append({"role": "user", "content": st.session_state.auto_question})
        with st.spinner("Contemplating your question..."):
            response = asyncio.run(st.session_state.bot.get_response(
                st.session_state.auto_question, 
                st.session_state.selected_theme,
                st.session_state.current_mood,
                dominant_emotion()
            ))
            st.session_state.messages.append({
                "role": "assistant",
                **response
            })
            del st.session_state.auto_question  # Clear the auto question
            st.rerun()

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
                    st.session_state.current_mood,
                    dominant_emotion()
                ))
                st.session_state.messages.append({
                    "role": "assistant",
                    **response
                })
                st.rerun()

    # Main content area - adjusted column widths: wider sidebar, narrower main content
    col1, col2 = st.columns([3, 2])

    with col1:
        if st.button("🔄 Reset Chat", help="Clear all chat history and start fresh"):
            for key in ['messages', 'question_history']:
                if key in st.session_state:
                    st.session_state[key] = []
            st.rerun()

        st.title("🕉️ Bhagavad Gita Wisdom")

        st.markdown("""
        Ask questions about life, dharma, and spirituality to receive guidance from the timeless wisdom of the Bhagavad Gita.
        *Personalize your experience using the options above.*
        """)

        # Enhanced message display
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                if message["role"] == "user":
                    st.markdown(message["content"])
                else:
                    # Enhanced assistant message display
                    if message.get("verse_reference"):
                        st.markdown(f"**📖 {message['verse_reference']}**")
                    
                    if message.get('sanskrit'):
                        st.markdown(f"*Sanskrit:* {message['sanskrit']}")
                    
                    if message.get('translation'):
                        st.markdown(f"**Translation:** {message['translation']}")
                    
                    if message.get('explanation'):
                        st.markdown("### 🧠 Understanding")
                        st.markdown(message["explanation"])
                    
                    if message.get('application'):
                        st.markdown("### 🌟 Modern Application")
                        st.markdown(message["application"])
                    
                    # Show keywords if available
                    if message.get('keywords'):
                        st.markdown("**Key Concepts:** " + " • ".join([f"`{kw}`" for kw in message['keywords']]))
                    
                    # Show context values that were passed to LLM
                    context_parts = []
                    if message.get('theme'):
                        context_parts.append(f"🎯 {message['theme']}")
                    if message.get('mood'):
                        context_parts.append(f"🎭 {message['mood']}")
                    if message.get('emotional_state'):
                        context_parts.append(f"💭 {message['emotional_state']}")
                    
                    if context_parts:
                        st.markdown("**Response Context:** " + " • ".join(context_parts))

        # Add the download button after the chat messages
        if st.session_state.messages:
            chat_content = create_downloadable_content(st.session_state.messages)
            st.download_button(
                label="📥 Download Chat History",
                data=chat_content,
                file_name=f"WisdomWeaver_Chat_History_{datetime.now().strftime('%Y-%m-%d')}.txt",
                mime="text/plain",
                help="Download your entire chat conversation as a text file"
            )

        # Enhanced chat input
        if "submission_in_progress" not in st.session_state:
            st.session_state.submission_in_progress = False

        if not st.session_state.submission_in_progress:
            question = st.chat_input("Ask your question here...")
            if question:
                st.session_state.messages.append({"role": "user", "content": question})
                st.session_state.submission_in_progress = True
                st.rerun()
        else:
            with st.spinner("🧘 Contemplating your question..."):
                last_user_msg = st.session_state.messages[-1]["content"]
                response = asyncio.run(st.session_state.bot.get_response(
                    last_user_msg,
                    st.session_state.selected_theme,
                    st.session_state.current_mood,
                    dominant_emotion()
                ))
                st.session_state.messages.append({
                    "role": "assistant",
                    **response
                })
                st.session_state.submission_in_progress = False
                st.rerun()

        with col2:
         render_enhanced_sidebar()

    # --- About Us Section ---
    st.markdown("---")
    with st.expander("💫 About Wisdom Weaver", expanded=True):
        st.markdown("""
## About Wisdom Weaver

**Wisdom Weaver** is a thoughtful AI-driven spiritual guide rooted in the timeless wisdom of the *Bhagavad Gita*. Created for modern seekers navigating life’s complexities, this platform offers personalized guidance, daily reflection, and the ability to connect with the deeper meaning behind ancient teachings.

### 🌱 Our Vision
To bridge ancient spiritual insight with today’s challenges—offering clarity, strength, and inner peace through meaningful interaction.

### 🔍 What We Offer
- **AI-Powered Insights:** Harnessing Google’s Gemini AI to interpret Gita verses in ways that resonate with your current state of mind.
- **Verse Exploration:** Access verses across all 18 chapters with translations, transliterations, and simplified meaning.
- **Theme-Based Guidance:** Whether it’s anxiety, purpose, relationships, or grief—we help you reflect and grow.
- **Interactive Tools:** Save favorite verses, revisit reflections, or receive a random verse tailored to your need.
- **Community-Centric Design:** Built by people who believe spirituality is a journey best shared.

### 🌟 Why Bhagavad Gita?
In every era, humanity has faced the same questions: Who am I? What is my purpose? Why do I suffer? The Gita doesn’t provide fixed answers—it offers a path. A mirror. A gentle but firm invitation to understand the self and act with awareness.

### 🧭 Meet the Team
- **Satvik gupta & Contributors:** Students of life, seekers of clarity—dedicated to merging tradition with technology.
- **Spiritual Mentors & Advisors:** Guiding the app’s soul to ensure authenticity and reverence.



*Wisdom Weaver is more than an app. It’s a living dialogue between past and present—a companion for every soul who believes that wisdom is not something we learn, but something we remember.*
""")


if __name__ == "__main__":
    main()
