import requests
import pandas as pd
import streamlit as st
import os
import google.generativeai as genai
from typing import Dict
import json
import asyncio
from PIL import Image
import nest_asyncio
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")



def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    if 'bot' not in st.session_state:
        if not api_key:
            st.error("Please set the GOOGLE_API_KEY in your .env file.")
            st.stop()
        st.session_state.bot = GitaGeminiBot(api_key)

class GitaGeminiBot:
    def __init__(self, api_key: str):
        """Initialize the Gita bot with Gemini API."""
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash')
        self.verses_db = self.load_gita_database()

    def load_gita_database(self) -> Dict:
        """Load the Bhagavad Gita dataset."""
        path = "bhagavad_gita_verses.csv"
        verses_df = pd.read_csv(path)

        verses_db = {}
        for _, row in verses_df.iterrows():
            chapter = f"chapter_{row['chapter_number']}"
            if chapter not in verses_db:
                verses_db[chapter] = {
                    "title": row['chapter_title'],
                    "verses": {}
                }
            verse_num = str(row['chapter_verse'])
            verses_db[chapter]["verses"][verse_num] = {
                "translation": row['translation']
            }
        return verses_db

    def format_response(self, raw_text: str) -> Dict:
        """Format the response into a structured dictionary."""
        try:
            try:
                return json.loads(raw_text)
            except json.JSONDecodeError:
                pass

            if '{' in raw_text and '}' in raw_text:
                json_start = raw_text.find('{')
                json_end = raw_text.rfind('}') + 1
                json_str = raw_text[json_start:json_end]
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    pass

            lines = raw_text.split('\n')
            response = {
                "verse_reference": "",
                "sanskrit": "",
                "translation": "",
                "explanation": "",
                "application": ""
            }

            current_section = None
            for line in lines:
                line = line.strip()
                if not line:
                    continue

                if "Chapter" in line and "Verse" in line:
                    response["verse_reference"] = line
                elif line.startswith("Sanskrit:"):
                    response["sanskrit"] = line.replace("Sanskrit:", "").strip()
                elif line.startswith("Translation:"):
                    response["translation"] = line.replace("Translation:", "").strip()
                elif line.startswith("Explanation:"):
                    current_section = "explanation"
                    response["explanation"] = line.replace("Explanation:", "").strip()
                elif line.startswith("Application:"):
                    current_section = "application"
                    response["application"] = line.replace("Application:", "").strip()
                elif current_section:
                    response[current_section] += " " + line

            return response

        except Exception as e:
            print(f"Error formatting response: {str(e)}")
            return {
                "verse_reference": "Unable to parse verse",
                "sanskrit": "",
                "translation": raw_text,
                "explanation": "",
                "application": ""
            }

    async def get_response(self, question: str) -> Dict:
        """Generate response using Gemini API."""
        try:
            prompt = f"""
            Based on the Bhagavad Gita's teachings, provide guidance for this question:
            {question}

            Format your response like this:
            Chapter X, Verse Y
            Sanskrit: [Sanskrit verse]
            Translation: [English translation]
            Explanation: [Clear explanation of the verse's meaning]
            Application: [How to apply this teaching in modern life]

            Keep the format strict and consistent.
            """

            response = await self.model.generate_content_async(prompt)

            if not response.text:
                raise ValueError("Empty response received from the model")

            return self.format_response(response.text)

        except Exception as e:
            print(f"Error getting response: {str(e)}")
            return {
                "verse_reference": "Error",
                "sanskrit": "",
                "translation": f"An error occurred: {str(e)}",
                "explanation": "Please try again.",
                "application": ""
            }

async def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Bhagavad Gita Wisdom Weaver",
        page_icon="🕉️",
        layout="wide"
    )

    image_path = "WhatsApp Image 2024-11-18 at 11.40.34_076eab8e.jpg"
    if os.path.exists(image_path):
        image = Image.open(image_path)
        max_width = 800
        aspect_ratio = image.height / image.width
        resized_image = image.resize((max_width, int(max_width * aspect_ratio)))
        st.image(resized_image, use_container_width=True, caption="Bhagavad Gita - Eternal Wisdom")
    else:
        st.error("Image file not found. Please upload the image.")

    initialize_session_state()

    col1, col2 = st.columns([2, 1])

    with col1:
        st.title("🕉️ Bhagavad Gita Wisdom")
        st.markdown("""Ask questions about life, dharma, and spirituality to receive guidance from the timeless wisdom of the Bhagavad Gita.""")

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                if message["role"] == "user":
                    st.markdown(message["content"])
                else:
                    if message["verse_reference"]:
                        st.markdown(f"**{message['verse_reference']}**")
                    if message.get('sanskrit'):
                        st.markdown(f"*{message['sanskrit']}*")
                    if message.get('translation'):
                        st.markdown(message["translation"])
                    if message.get('explanation'):
                        st.markdown("### Understanding")
                        st.markdown(message["explanation"])
                    if message.get('application'):
                        st.markdown("### Modern Application")
                        st.markdown(message["application"])

        if question := st.chat_input("Ask your question here..."):
            st.session_state.messages.append({"role": "user", "content": question})

            with st.spinner("Contemplating your question..."):
                response = await st.session_state.bot.get_response(question)
                st.session_state.messages.append({
                    "role": "assistant",
                    **response
                })
                st.rerun()

    with col2:
        st.sidebar.title("Browse Chapters")
        chapters = list(st.session_state.bot.verses_db.keys())
        selected_chapter = st.sidebar.selectbox(
            "Select Chapter",
            chapters,
            format_func=lambda x: f"{x}: {st.session_state.bot.verses_db[x]['title']}"
        )

        if selected_chapter:
            st.sidebar.markdown(f"### {st.session_state.bot.verses_db[selected_chapter]['title']}")
            verses = st.session_state.bot.verses_db[selected_chapter]['verses']
            for verse_num, verse_data in verses.items():
                with st.sidebar.expander(f"Verse {verse_num}"):
                    st.markdown(verse_data['translation'])

        # --- New Section for User Question History ---
        st.sidebar.markdown("---")
        st.sidebar.title("Your Questions History")
        user_questions = [msg["content"] for msg in st.session_state.messages if msg["role"] == "user"]
        if user_questions:
            for i, q in enumerate(user_questions):
                st.sidebar.markdown(f"**{i+1}.** {q}")
        else:
            st.sidebar.info("No questions asked yet in this session.")
            
        # --- End of New Section ---


    st.markdown("---")
    st.markdown(
        "💫 This application uses the Gemini API to provide insights from the Bhagavad Gita. "
        "For deeper understanding, please consult with qualified spiritual teachers."
    )

if __name__ == "__main__":
    nest_asyncio.apply()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(main())