# 🕉️ Bhagavad Gita Wisdom Weaver

A real-time, AI-powered chatbot that provides **mental health support and spiritual guidance** using teachings from the **Bhagavad Gita**. Ask life questions and receive structured answers powered by **Google Gemini API**, displayed in a clean and friendly **Streamlit interface**.

---
## ❓ Why Use WisdomWeaver?

In today’s fast-paced world, we often face stress, confusion, and emotional challenges. **WisdomWeaver** bridges ancient spiritual wisdom with modern AI to help you:

- 🧘‍♀️ Reflect deeply on life problems with timeless Gita teachings.
- 💡 Get practical and philosophical advice tailored to your questions.
- 🌿 Improve mental well-being with spiritually grounded responses.
- 🔄 Understand the Gita verse-by-verse with contextual insights.

Whether you're spiritually inclined, curious about the Gita, or just looking for calm guidance — this tool is made for **you**.

---

## 📽️ Demo :
https://bkins-wisdomweaver.streamlit.app/
 <img width="1024" height="1536" alt="image" src="https://github.com/user-attachments/assets/9ab2ad92-9c96-4230-9aec-b8ad881253c0" />

> 🙏 Ask any question like: *"Mujhe anxiety ho rahi hai, kya karun?"*  
> 📜 Get a reply from the Gita like:  
> **Chapter 2, Verse 47**  
> *Karmanye vadhikaraste ma phaleshu kadachana...*  
> _"Do your duty without attachment to outcomes."_  
> 💡 With Explanation + Real-life Application!

---

## 🧠 Features

- 🧘‍♂️ **Ask Anything**: Get spiritual & practical guidance based on Bhagavad Gita.
- 🔍 **Chapter/Verse Browser**: View any shloka translation chapter-wise.
- 🧾 **Structured Response**: AI responds with:
  - Chapter & Verse
  - Sanskrit Shloka
  - Translation
  - Explanation
  - Modern Life Application
- 💬 **Chat History**: See your past questions in the sidebar.
- 🌐 **Streamlit UI**: Responsive, clean, and user-friendly.
- ⚡ **Powered by Gemini AI**: Uses Google’s Gemini 2.0 Flash model.

---

## 🛠️ Tech Stack

| Feature       | Tech Used           |
|---------------|---------------------|
| UI/Frontend   | Streamlit           |
| AI Backend    | Google Gemini API   |
| Language      | Python              |
| Data Handling | Pandas              |
| Image         | PIL (Pillow)        |
| Async Support | asyncio             |
| Data Source   | Bhagavad Gita CSV   |

---

## ⚙️ Setup Instructions

### 📦 Prerequisites

- Python 3.9 or higher
- pip (Python package installer)
- Google Gemini API Key ([Get one here](https://aistudio.google.com/app/apikey))

## 🔑 Generating Your Google Gemini API Key

To use the Google Gemini API, follow these steps to generate your API key:

1. Go to the [Google AI Studio](https://makersuite.google.com/app) website.
2. Sign in with your Google account.
3. Click on **"Create API Key in new project"** or select an existing project to generate a new key.
4. Copy the generated API key.  
   📌 **Note:** You’ll need this key for authentication in the next step.

### 🚀 Installation

1. **Clone the repository**
```bash
git clone https://github.com/satvik091/WisdomWeaver.git
cd WisdomWeaver
```

2. **Create a virtual environment (recommended)**
```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

3. **Install required Python packages**
```bash
pip install -r requirements.txt
```

## 🔑 API Key Configuration

To securely use your Google Gemini API key in the **WisdomWeaver** project:

### 1. Create a `.env` file  
In the root directory of your project (where `main.py` and `requirements.txt` are located), create a new file named `.env`.

### 2. Add your API key to `.env`  
Open the `.env` file and add the following line (replace `your_api_key_here` with the actual key you generated earlier):
-change .env.example to .env
```env
GOOGLE_API_KEY=your_api_key_here

### 🔔 Important Notes

- 🔒 **Never share your API key publicly.**
- ✅ **Make sure your `.env` file is excluded from version control** (e.g., Git).
- 📁 **The `.gitignore` file should already contain an entry for `.env`.** Double-check if you're unsure.
```
---

### ▶️ Run the Application

1. **Make sure your virtual environment is activated**
```bash
# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

2. **Run the Streamlit app**
```bash
streamlit run app.py
```
### 🌐 Open in Browser

Once the app starts, **WisdomWeaver** will automatically open in your default web browser at:

[http://localhost:8501](http://localhost:8501)

If it doesn’t open automatically, simply copy and paste the URL into your browser.


### 🔧 Troubleshooting

**Issue: Module not found errors**
- Make sure your virtual environment is activated
- Run `pip install -r requirements.txt` again

**Issue: API key not working**
- Verify your API key in the `.env` file
- Make sure the `.env` file is in the root directory
- Check that your Google AI API key is valid

**Issue: Streamlit not starting**
- Make sure you're in the correct directory
- Try running `streamlit --version` to verify installation

---
## 📂 Folder Structure

```plaintext
gita-gemini-bot/
├── main.py                  # Streamlit app file
├── bhagavad_gita_verses.csv # Bhagavad Gita verse data
├── requirements.txt         # Python dependencies
├── README.md                # You're here!
├── .env.example             # Sample environment config
└── .streamlit/              # Streamlit config folder
```

---
## 💻 Sample Question

**Q:** *Zindagi ka purpose kya hai?*

**Output:**

- 📖 **Chapter 3, Verse 30**
- 🕉️ *Mayi sarvani karmani sannyasyadhyatmacetasa...*

**Translation:**  
*Dedicate all actions to me with full awareness of the Self.*

**Explanation:**  
Lord Krishna advises detachment and devotion in duty.

**Application:**  
Focus on sincere efforts, not selfish rewards.

---
## 🤝 Contributing

We welcome contributions as part of **GirlScript Summer of Code 2025 (GSSoC'25)** and beyond!

### 📌 Steps to Contribute

1. **Fork** this repo 🍴  
2. **Create a branch**  
  ```bash
   git checkout -b feat/amazing-feature
  ```
3. **Make your changes** ✨
4. **Commit your changes**
  ```bash
  git commit -m 'Add: Amazing Feature'
  ```
5. **Push to your branch**
  ```bash
  git push origin feat/amazing-feature
  ```
6. **Open a Pull Request and link the related issue**
  ```bash
  Closes #6
  ```

---
## 🌸 GirlScript Summer of Code 2025

This project is proudly part of **GSSoC '25**!  
Thanks to the amazing open-source community, contributors, and mentors for your valuable support.

---
## 📄 License

This project is licensed under the **MIT License**.  
See the [LICENSE](LICENSE) file for full details.

---

## 🙏 Acknowledgements

- 📜 **Bhagavad Gita** – Eternal source of wisdom  
- 🧠 **Google Gemini API** – AI backend for responses  
- 🌐 **Streamlit Team** – For the interactive app framework  
- 👥 **GSSoC 2025 Community** – For mentorship and collaboration  

---

## 📬 Contact

Have ideas, feedback, or just want to say hi?

- 🛠️ Open an issue in the repository  
- 📧 Contact our mentor:

**Mentor**: Harmanpreet  
**GitHub**: [Harman-2](https://github.com/Harman-2)

---

Thank you for visiting! 🙏


