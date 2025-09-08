import streamlit as st

st.set_page_config(
    page_title="Wisdom Weaver",
    layout="wide",
    initial_sidebar_state="expanded"
)
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
import plotly.graph_objects as go
import plotly.express as px

# Import the advanced emotion detector
from emotion_advanced import AdvancedEmotionDetector

load_dotenv()

# Constants
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "YOUR_API_KEY")
GITA_CSV_PATH = "bhagavad_gita_verses.csv"
IMAGE_PATH = "Public/Images/WhatsApp Image 2024-11-18 at 11.40.34_076eab8e.jpg"


def create_confidence_visualization(confidence_data: Dict[str, float], title: str = "Model Confidence") -> go.Figure:
    """
    Create a confidence visualization using Plotly.
    
    Args:
        confidence_data: Dictionary with emotion names as keys and confidence scores as values
        title: Title for the chart
    
    Returns:
        Plotly figure object
    """
    if not confidence_data:
        return None
    
    # Sort emotions by confidence score
    sorted_emotions = sorted(confidence_data.items(), key=lambda x: x[1], reverse=True)
    emotions, scores = zip(*sorted_emotions)
    
    # Create color mapping based on confidence levels
    colors = []
    for score in scores:
        if score >= 80:
            colors.append('#2E8B57')  # Sea Green - High confidence
        elif score >= 60:
            colors.append('#FFD700')  # Gold - Medium confidence
        elif score >= 40:
            colors.append('#FF8C00')  # Dark Orange - Low confidence
        else:
            colors.append('#DC143C')  # Crimson - Very low confidence
    
    # Create horizontal bar chart
    fig = go.Figure(data=[
        go.Bar(
            x=scores,
            y=emotions,
            orientation='h',
            marker_color=colors,
            text=[f'{score:.1f}%' for score in scores],
            textposition='auto',
            hovertemplate='<b>%{y}</b><br>Confidence: %{x:.1f}%<extra></extra>'
        )
    ])
    
    # Update layout for better appearance
    fig.update_layout(
        title={
            'text': title,
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 16, 'color': '#333333'}
        },
        xaxis_title="Confidence Score (%)",
        yaxis_title="Emotions",
        xaxis=dict(range=[0, 100]),
        height=300,
        margin=dict(l=20, r=20, t=40, b=20),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        showlegend=False
    )
    
    return fig


def create_confidence_interval_display(confidence_data: Dict[str, float], primary_emotion: str) -> str:
    """
    Create a text-based confidence interval display.
    
    Args:
        confidence_data: Dictionary with emotion names and confidence scores
        primary_emotion: The primary detected emotion
    
    Returns:
        Formatted string showing confidence interval
    """
    if not confidence_data or primary_emotion not in confidence_data:
        return "Confidence: Not available"
    
    primary_confidence = confidence_data[primary_emotion]
    
    # Calculate confidence interval based on the score
    if primary_confidence >= 90:
        interval = f"95% CI: {primary_confidence-5:.1f}%–{min(100, primary_confidence+5):.1f}%"
        confidence_level = "Very High"
    elif primary_confidence >= 75:
        interval = f"90% CI: {primary_confidence-8:.1f}%–{min(100, primary_confidence+8):.1f}%"
        confidence_level = "High"
    elif primary_confidence >= 60:
        interval = f"85% CI: {primary_confidence-10:.1f}%–{min(100, primary_confidence+10):.1f}%"
        confidence_level = "Medium"
    elif primary_confidence >= 40:
        interval = f"80% CI: {primary_confidence-12:.1f}%–{min(100, primary_confidence+12):.1f}%"
        confidence_level = "Low"
    else:
        interval = f"75% CI: {primary_confidence-15:.1f}%–{min(100, primary_confidence+15):.1f}%"
        confidence_level = "Very Low"
    
    return f"**{primary_emotion.title()}** ({primary_confidence:.1f}%) - {confidence_level} Confidence\n{interval}"


def render_confidence_dashboard(emotion_detector):
    """
    Render a comprehensive confidence dashboard for emotion detection.
    
    Args:
        emotion_detector: The AdvancedEmotionDetector instance
    """
    if not emotion_detector:
        return
    
    # Get current emotion and confidence data
    current_emotion, confidence_data = emotion_detector.get_current_emotion()
    
    if not confidence_data:
        st.info("🔍 Waiting for emotion detection data...")
        return
    
    st.markdown("### 📊 Emotion Detection Confidence Dashboard")
    
    # Create two columns for the dashboard
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Display confidence visualization
        if confidence_data:
            fig = create_confidence_visualization(confidence_data, "Emotion Confidence Scores")
            if fig:
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    
    with col2:
        # Display confidence interval and summary
        if current_emotion and confidence_data:
            confidence_text = create_confidence_interval_display(confidence_data, current_emotion)
            st.markdown(confidence_text)
            
            # Add confidence level indicator
            primary_confidence = confidence_data.get(current_emotion, 0)
            if primary_confidence >= 80:
                st.success("✅ High Confidence Detection")
            elif primary_confidence >= 60:
                st.warning("⚠️ Medium Confidence Detection")
            else:
                st.error("❌ Low Confidence Detection")
            
            # Show top 3 emotions
            sorted_emotions = sorted(confidence_data.items(), key=lambda x: x[1], reverse=True)[:3]
            st.markdown("**Top 3 Emotions:**")
            for i, (emotion, score) in enumerate(sorted_emotions, 1):
                st.markdown(f"{i}. {emotion.title()}: {score:.1f}%")
    
    # Add helpful tooltip
    with st.expander("ℹ️ Understanding Confidence Scores"):
        st.markdown("""
        **What do these confidence scores mean?**
        
        - **90-100%**: Very High Confidence - The model is very certain about this emotion
        - **75-89%**: High Confidence - The model is confident but there's some uncertainty
        - **60-74%**: Medium Confidence - Moderate certainty, consider the context
        - **40-59%**: Low Confidence - Significant uncertainty, results should be interpreted carefully
        - **Below 40%**: Very Low Confidence - High uncertainty, consider manual assessment
        
        **Tips for better accuracy:**
        - Ensure good lighting conditions
        - Face the camera directly
        - Maintain a neutral expression initially
        - Avoid rapid movements
        """)


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
    on the webcam feed. Falls back to the sidebar selection when nothing found.
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

    def _calculate_response_confidence(self, response: Dict, question: str, theme: str = None, mood: str = None, emotional_state: str = None) -> float:
        """
        Calculate confidence score for the AI response based on various factors.
        
        Args:
            response: The formatted response dictionary
            question: The user's question
            theme: Selected theme
            mood: User's mood
            emotional_state: Detected emotional state
            
        Returns:
            Confidence score between 0 and 100
        """
        confidence_score = 0.0
        
        # Base confidence from response completeness (40% weight)
        completeness_score = 0.0
        required_fields = ['verse_reference', 'translation', 'explanation', 'application']
        present_fields = sum(1 for field in required_fields if response.get(field) and len(str(response[field]).strip()) > 10)
        completeness_score = (present_fields / len(required_fields)) * 40
        
        # Content quality score (30% weight)
        quality_score = 0.0
        if response.get('translation'):
            quality_score += 10
        if response.get('explanation') and len(response['explanation']) > 50:
            quality_score += 10
        if response.get('application') and len(response['application']) > 50:
            quality_score += 10
        
        # Context relevance score (20% weight)
        relevance_score = 0.0
        if theme and theme in self.themes:
            relevance_score += 5
        if mood:
            relevance_score += 5
        if emotional_state:
            relevance_score += 5
        if response.get('keywords'):
            relevance_score += 5
        
        # Response length and detail score (10% weight)
        detail_score = 0.0
        total_length = sum(len(str(response.get(field, ''))) for field in ['translation', 'explanation', 'application'])
        if total_length > 200:
            detail_score = 10
        elif total_length > 100:
            detail_score = 7
        elif total_length > 50:
            detail_score = 5
        
        confidence_score = completeness_score + quality_score + relevance_score + detail_score
        
        # Ensure score is between 0 and 100
        return min(100.0, max(0.0, confidence_score))

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
            
            # Calculate confidence score based on response quality and completeness
            confidence_score = self._calculate_response_confidence(formatted_response, question, theme, mood, emotional_state)
            formatted_response["confidence_score"] = confidence_score
            
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
    
    # Create columns for better layout
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
        
        # Create a container for the webcam feed and confidence dashboard
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
        
        # Add confidence dashboard below the webcam
        if st.session_state.emotion_detector:
            render_confidence_dashboard(st.session_state.emotion_detector)
    
    # Quick action buttons - All 5 on one line, centered
    st.markdown("### ⚡ Quick Actions")
    
    # Adjust column widths to center and evenly space 5 buttons with text + emoji
    # The middle columns are now slightly larger to accommodate text, and side columns balance.
    # The sum of these ratios is 11.
    col_empty_left_qa, col_qa1, col_qa2, col_qa3, col_qa4, col_qa5, col_empty_right_qa = st.columns([0.75, 2, 2, 2, 2, 2, 0.75]) 


    with col_qa1:
        if st.button("🎲 Random Verse", help="Get a random verse for inspiration"):
            return "random_verse"
    
    with col_qa2:
        if st.button("💭 Daily Reflection", help="Get guidance for daily contemplation"):
            return "daily_reflection"
    
    with col_qa3:
        if st.button("🔍 Verse Search", help="Search for specific verses"):
            return "verse_search"
    
    with col_qa4:
        if st.button("📖 Chapter Summary", help="Get a summary of any chapter"):
            return "chapter_summary"


    with col_qa5:
        if st.button("⭐ Favorites Marked", help="View your saved favorite verses"):
            return "view_favorites"
            
    # Reset Chat button - on its own centered line directly below Quick Actions
    # Use columns to center a single button effectively.
    col_empty_left_reset, col_reset, col_empty_right_reset = st.columns([3, 2, 3]) 
    with col_reset:
        if st.button("🔄 Reset Chat", help="Clear all chat history and start fresh"):
            return "reset_chat"


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
    
    elif action_type == "view_favorites":  # Handle the new action type
        st.info("Displaying your favorite verses...")
        return None 
    
    elif action_type == "reset_chat":  # Handle the reset_chat action type
        for key in ['messages', 'question_history']:
            if key in st.session_state:
                st.session_state[key] = []
        st.experimental_rerun()  # Rerun to clear chat
        return None  # No question to generate after reset
    
    return None


# 📚 Blog List Section
st.header("📚 Blog List")

blogs = []

if not blogs:
    st.info("No blogs available yet.")
else:
    for blog in blogs:
        st.subheader(blog['title'])
        st.write(blog['content'])


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

    # Enhanced question history with confidence indicators
    st.sidebar.markdown("---")
    st.sidebar.title("💭 Your Spiritual Journey")
    
    user_questions = [msg["content"] for msg in st.session_state.messages if msg["role"] == "user"]
    ai_responses = [msg for msg in st.session_state.messages if msg["role"] == "assistant"]
    
    if user_questions:
        st.sidebar.markdown(f"**Questions Asked:** {len(user_questions)}")
        
        # Show recent questions with confidence indicators
        for i, q in enumerate(user_questions[-5:], 1):  # Show last 5 questions
            with st.sidebar.expander(f"Question {len(user_questions) - 5 + i}"):
                st.markdown(f"*{q}*")
                
                # Show confidence for corresponding AI response if available
                if i <= len(ai_responses):
                    response = ai_responses[-(i)]
                    if response.get('confidence_score') is not None:
                        confidence_score = response['confidence_score']
                        
                        # Small confidence badge
                        if confidence_score >= 80:
                            st.sidebar.success(f"🤖 {confidence_score:.0f}%")
                        elif confidence_score >= 60:
                            st.sidebar.warning(f"🤖 {confidence_score:.0f}%")
                        else:
                            st.sidebar.error(f"🤖 {confidence_score:.0f}%")
    else:
        st.sidebar.info("🌱 Begin your journey by asking a question")
    
    # Confidence summary
    if ai_responses:
        avg_confidence = sum(msg.get('confidence_score', 0) for msg in ai_responses) / len(ai_responses)
        st.sidebar.markdown("---")
        st.sidebar.markdown("**📊 Response Quality Summary:**")
        
        if avg_confidence >= 80:
            st.sidebar.success(f"Average Confidence: {avg_confidence:.1f}%")
        elif avg_confidence >= 60:
            st.sidebar.warning(f"Average Confidence: {avg_confidence:.1f}%")
        else:
            st.sidebar.error(f"Average Confidence: {avg_confidence:.1f}%")

    # Favorites section (placeholder for future enhancement)
    st.sidebar.markdown("---")
    st.sidebar.title("⭐ Favorite Verses")
    if st.session_state.favorite_verses:
        for fav in st.session_state.favorite_verses:
            st.sidebar.markdown(f"• {fav}")
    else:
        st.sidebar.info("No favorites saved yet")




def show_preloader():
    st.markdown("""
        <div style="text-align:center; margin-top:40px;">
            <div style="font-size:60px; margin-bottom:10px;">🕉️</div>
            <div style="font-size:22px; color:#7b3f00;">Contemplating your journey...</div>
        </div>
    """, unsafe_allow_html=True)

def login_signup_page():
    st.markdown("""
        <style>
        body, .stApp, div[data-testid="stAppViewContainer"] > div:first-child {
            background-color: #e6e2d3 !important;
        }
        .login-card {
            background-color: #e6e2d3;
            border-radius: 16px;
            padding: 32px;
            margin: auto;
            max-width: 400px;
            border: 1px solid #d1cfc7;
            box-shadow: 0 2px 8px #d1cfc7;
        }
        </style>
    """, unsafe_allow_html=True)
    st.markdown("""
        <div class="login-card">
            <div style="text-align:center;">
                <div style="font-size:50px; margin-bottom:10px;">🕉️</div>
                <h2 style="color:#7b3f00;">Wisdom Weaver</h2>
                <p style="font-size:18px;"><em>Welcome! Please login or sign up to begin your spiritual journey.</em></p>
            </div>
        </div>
    """, unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["Login", "Sign Up"])
    if "users" not in st.session_state:
        st.session_state.users = {"demo": "demo123"}  # demo user

    with tab1:
        username = st.text_input("Username", key="login_user")
        password = st.text_input("Password", type="password", key="login_pass")
        if st.button("Login", key="login_btn"):
            show_preloader()
            time.sleep(1)
            if username in st.session_state.users and st.session_state.users[username] == password:
                st.session_state.logged_in = True
                st.session_state.current_user = username
                st.success("Login successful!")
                st.rerun()
            else:
                st.error("Invalid username or password.")

    with tab2:
        new_user = st.text_input("Choose a Username", key="signup_user")
        new_pass = st.text_input("Choose a Password", type="password", key="signup_pass")
        if st.button("Sign Up", key="signup_btn"):
            show_preloader()
            time.sleep(1)
            if new_user in st.session_state.users:
                st.error("Username already exists.")
            elif len(new_user) < 3 or len(new_pass) < 3:
                st.error("Username and password must be at least 3 characters.")
            else:
                st.session_state.users[new_user] = new_pass
                st.success("Sign up successful! Please login.")
                st.rerun()
# ...rest of your code remains unchanged...

def main_app():
    # Your original main() code here

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
    """Enhanced main Streamlit appli
    st.set_page_config(
        page_title="Bhagavad Gita Wisdom Weaver",
        page_icon="🕉️",
        layout="wide",  # This is key for full-width layout
        initial_sidebar_state="expanded"
    )
    st.markdown("""
        <style>
        div[data-testid="stAppViewContainer"] > div:first-child {
            background-color: #e6e2d3 !important;
        }
        section[data-testid="stSidebar"] {
            background-color: #e6e2d3 !important;
        }
        .streamlit-expanderHeader {
            background-color: #e6e2d3 !important;
        }
        .stMarkdown, .stExpanderContent {
            background-color: #e6e2d3 !important;
        }
        </style>
    """, unsafe_allow_html=True)

    # Initialize session state first, before any other operations
    initialize_session_state()

    # Load and display image with reduced width
    if os.path.exists(IMAGE_PATH):
        try:
            image = Image.open(IMAGE_PATH)
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

    quick_action = render_additional_options()
    if quick_action:

        auto_question = handle_quick_actions(quick_action)
        if auto_question:
            st.session_state.messages.append({"role": "user", "content": auto_question})
            show_preloader()
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

    col1, col2 = st.columns([2, 1])
    with col1:
        st.title("🕉️ Bhagavad Gita Wisdom")
        st.markdown("""
        Ask questions about life, dharma, and spirituality to receive guidance from the timeless wisdom of the Bhagavad Gita.
        *Personalize your experience using the options above.*
        """)
        for message in st.session_state.messages:

        # Handle reset action directly here to avoid re-generating a question
        if quick_action == "reset_chat":
            handle_quick_actions("reset_chat")  # This will clear messages and rerun
        else:
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
                    st.experimental_rerun()

    st.markdown("<h1 style='text-align: center; white-space: nowrap;'>🕉️ Bhagavad Gita Wisdom</h1>", unsafe_allow_html=True)

    st.markdown("""
        <p style='text-align: center; font-size: 1.1em; line-height: 1.6; padding-left: 5%; padding-right: 5%;'>
        Ask questions about life, dharma, and spirituality to receive guidance from the timeless wisdom of the Bhagavad Gita. Personalize your experience using the options above.
        </p>
        """, unsafe_allow_html=True)

    left_empty_for_center, main_chat_col, sidebar_col = st.columns([1, 3, 1]) 

    with main_chat_col:
        if "submission_in_progress" not in st.session_state:
            st.session_state.submission_in_progress = False

        if not st.session_state.submission_in_progress:
            question = st.chat_input("Ask your question here...")
            if question:
                st.session_state.messages.append({"role": "user", "content": question})
                st.session_state.submission_in_progress = True
                st.experimental_rerun()
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
                st.experimental_rerun()

        # Enhanced message display (now below the input)
        for i, message in enumerate(st.session_state.messages):

            with st.chat_message(message["role"]):
                if message["role"] == "user":
                    st.markdown(message["content"])
                else:
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
                    if message.get('keywords'):
                        st.markdown("**Key Concepts:** " + " • ".join([f"`{kw}`" for kw in message['keywords']]))

        if question := st.chat_input("Ask your question here..."):
            st.session_state.messages.append({"role": "user", "content": question})
            show_preloader()
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
                    
                    # Display confidence score for AI responses
                    if message.get('confidence_score') is not None:
                        confidence_score = message['confidence_score']
                        
                        # Create confidence indicator
                        col1, col2, col3 = st.columns([1, 3, 1])
                        with col2:
                            st.markdown("**🤖 AI Response Confidence:**")
                            
                            # Confidence bar
                            confidence_color = "green" if confidence_score >= 80 else "orange" if confidence_score >= 60 else "red"
                            st.progress(confidence_score / 100, text=f"{confidence_score:.1f}%")
                            
                            # Confidence level indicator
                            if confidence_score >= 80:
                                st.success(f"✅ High Confidence ({confidence_score:.1f}%)")
                            elif confidence_score >= 60:
                                st.warning(f"⚠️ Medium Confidence ({confidence_score:.1f}%)")
                            else:
                                st.error(f"❌ Low Confidence ({confidence_score:.1f}%)")
                            
                            # Confidence explanation tooltip
                            with st.expander("ℹ️ What does this confidence score mean?"):
                                st.markdown(f"""
                                **Response Quality Assessment:**
                                
                                This confidence score ({confidence_score:.1f}%) indicates how well the AI was able to:
                                - **Provide complete information** (verse reference, translation, explanation, application)
                                - **Match your context** (theme: {message.get('theme', 'N/A')}, mood: {message.get('mood', 'N/A')})
                                - **Address your emotional state** ({message.get('emotional_state', 'N/A')})
                                - **Offer detailed and relevant guidance**
                                
                                **Confidence Levels:**
                                - **80-100%**: Excellent response quality with comprehensive guidance
                                - **60-79%**: Good response with room for improvement
                                - **Below 60%**: Basic response, consider rephrasing your question
                                """)

                    # NEW: Add Favorite button for assistant responses
                    unique_key = f"favorite_btn_{i}_{message.get('verse_reference', '').replace(' ', '_')}"
                    if st.button("⭐ Add to Favorites", key=unique_key):
                        verse_info = ""
                        if message.get("verse_reference"):
                            verse_info += message["verse_reference"]
                        if message.get("translation"):
                            if verse_info:
                                verse_info += " - "
                            verse_info += message["translation"]
                        
                        if verse_info and verse_info not in st.session_state.favorite_verses:
                            st.session_state.favorite_verses.append(verse_info)
                            st.success("Verse added to favorites!")
                        elif verse_info:
                            st.warning("This verse is already in your favorites!")

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

    with sidebar_col:
        render_enhanced_sidebar()


    st.markdown("---")
    with st.expander("💫 About Wisdom Weaver", expanded=True):
        st.markdown("""
        <div style="background-color:#e6e2d3; border-radius:15px; padding:24px; border:1px solid #d1cfc7;">
        <h2 style="text-align:center;">🕉️ <span style="color:#7b3f00;">Wisdom Weaver</span> 🕉️</h2>
        <p style="text-align:center; font-size:18px;">
        <em>“Let the light of ancient wisdom guide your modern journey.”</em>
        </p>
        <hr>
        ### 🌱 Our Mission
        <span style="font-size:16px;">
        To bridge the timeless teachings of the <strong>Bhagavad Gita</strong> with the challenges of today, nurturing clarity, strength, and inner peace for every seeker.
        </span>
        ### ✨ Features
        - 🧘 <strong>Spiritual Guidance:</strong> Personalized answers powered by Google's Gemini AI.
        - 📖 <strong>Verse Explorer:</strong> Browse, search, and reflect on verses from all 18 chapters.
        - 🎯 <strong>Quick Actions:</strong> Random verse, daily reflection, and more.
        - ⭐ <strong>Favorites:</strong> Save and revisit your most inspiring verses.
        - 🫶 <strong>Community:</strong> Connect, share, and grow with fellow seekers.
        ### 📚 Why the Bhagavad Gita?
        <span style="font-size:16px;">
        The Gita is a universal scripture, a dialogue of the soul, offering wisdom for self-discovery, resilience, and harmony. Its teachings transcend boundaries, inviting all to walk the path of awareness.
        </span>
        ### 👥 Meet the Team
        - <strong>Satvik & Contributors:</strong> Spiritual technologists and lifelong learners.
        - <strong>Advisors:</strong> Gita scholars and meditation mentors.
        ### 🤝 Connect & Community
        - 📧 Email: <a href="mailto:support@wisdomweaver.app">support@wisdomweaver.app</a>
        - 📸 Instagram: <a href="https://instagram.com/wisdomweaver.ai">@wisdomweaver.ai</a>
        - 💬 Discord: <a href="https://discord.gg/yourcommunity">Join our Community</a>
        - 💡 Feedback: We welcome your ideas and stories!
        <hr>
        <p style="text-align:center; font-size:18px;">
        <em>“You are not alone on this journey. May the wisdom of the Gita illuminate your path.”</em><br>
        <span style="font-size:22px;">🙏</span>
        </p>
        </div>
        """, unsafe_allow_html=True)

def main():
    if not st.session_state.get("logged_in", False):
        login_signup_page()
    else:
        main_app()

if __name__ == "__main__":
    main()


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