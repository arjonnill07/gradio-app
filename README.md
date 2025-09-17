# Prohori (প্রহরী)

A powerful web application for scraping, analyzing, and visualizing data from Google News and YouTube. This tool provides deep insights into media trends, public discourse, and content performance with a specialized focus on Bengali-language analytics.

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/Arjon07CSE/Prohori)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🚀 Live Demo

Experience the application live without any installation. The project is proudly hosted on Hugging Face Spaces.

**[Click here to launch the Social Perception Analyzer](https://huggingface.co/spaces/Arjon07CSE/Prohori)**

---

## 🌟 Overview

The Social Perception Analyzer is an all-in-one analytics dashboard designed for researchers, analysts, and strategists. It automates the process of gathering and interpreting data from two of the world's largest information sources: Google News and YouTube. By providing interactive visualizations and data export capabilities, it empowers users to make data-driven decisions, track public narratives, and understand audience engagement with unparalleled clarity.

The application is production-ready, featuring robust error handling, production-grade logging, and a fully refactored, stable codebase.

---

## ✨ Key Features

The application is divided into two powerful, interconnected modules:

### 📰 Google News Scraper & Analytics
-   **Targeted Scraping**: Execute searches using specific keywords, date ranges, and target media domains.
-   **Advanced Filtering**: Refine search results with secondary keywords to isolate the most relevant articles.
-   **Dynamic Dashboard**:
    -   **KPI Metrics**: At-a-glance view of total articles, unique media sources, and date ranges.
    -   **News Volume Timeline**: Track publication frequency over time to identify trends and spikes.
    -   **Top Media Sources**: Instantly see which outlets are dominating the conversation.
    -   **Bengali Word Cloud**: Visualize the most prominent terms in headlines for a rapid thematic overview.
-   **Data Export**: Download all scraped news data to a `.csv` file for further offline analysis.

### 📺 YouTube Topic Analysis
-   **Comprehensive Video Search**: Find relevant YouTube videos based on your query and filter by publication date.
-   **In-Depth Statistics**: Gather detailed statistics for dozens of top videos, including view, like, and comment counts.
-   **Audience Insight Engine**: Scrape comments from top-performing videos to analyze audience sentiment and discussion points.
-   **Rich Visualization Suite**:
    -   **Performance KPIs**: Track total videos found, cumulative views, and comments scraped.
    -   **Channel Dominance Analysis**: Identify top channels by both video volume and total view count.
    -   **Content Performance Quadrant**: A powerful scatter plot that maps videos by views vs. engagement to reveal high-impact content.
    -   **Bengali Comment Word Cloud**: Generate a word cloud from comments to understand the voice of the audience.
-   **Data Export**: Download video statistics and comment data in separate, organized `.csv` files.

---

## 🖼️ Application Screenshots

*(Here, you can insert screenshots of your application to give users a visual preview. Replace the placeholder links with links to your actual images.)*

| News Analytics Dashboard | YouTube Content Quadrant |
| :----------------------: | :----------------------: |
| <img width="1862" height="812" alt="image" src="https://github.com/user-attachments/assets/75183ff2-bf41-4abe-8e5a-8f55a53ddc8b" />
 | <img width="1876" height="905" alt="image" src="https://github.com/user-attachments/assets/91087e81-1968-402e-8e47-2dd15460b86f" />
 |

| Top Media Sources Plot | Bengali Word Cloud |
| :--------------------: | :--------------------: |
| <img width="2400" height="1800" alt="image" src="https://github.com/user-attachments/assets/535881fc-e557-4b6e-b137-261877a809bb" />
 | <img width="4500" height="2400" alt="image" src="https://github.com/user-attachments/assets/5329d2e9-f419-4fb7-a2f9-6dde796a5db9" />
 |

---

## 🛠️ Technical Stack

This project is built with a modern, robust stack of technologies:

-   **Backend:** Python
-   **Web UI Framework:** Gradio
-   **Data Manipulation:** Pandas, NumPy
-   **Web Scraping:** GoogleNews, Google API Python Client
-   **Data Visualization:** Matplotlib, Seaborn, WordCloud
-   **Date Handling:** python-dateutil

---

## ⚙️ Installation & Local Setup

Follow these steps to run the application on your local machine.

### Prerequisites
-   Python 3.8 or higher
-   A Google account and a **YouTube Data API v3 key**.

### 1. Clone the Repository
```bash
git clone https://github.com/arjonnill07/gradio-app.git
cd gradio-app
2. Create and Activate a Virtual Environment
This keeps your project dependencies isolated.
code
Bash
# For macOS/Linux
python3 -m venv venv
source venv/bin/activate

# For Windows
python -m venv venv
venv\Scripts\activate
3. Install Dependencies
The project includes a requirements.txt file to install all necessary libraries.
code
Bash
pip install -r requirements.txt```

### 4. Configure Environment Variables
The YouTube analysis module requires an API key. It's best practice to set this as an environment variable rather than hardcoding it.

-   **Get your API Key**: Follow the instructions from the [Google Cloud Console](https://console.cloud.google.com/) to enable the "YouTube Data API v3" and create an API key.
-   **Set the Environment Variable**:
    ```bash
    # For macOS/Linux
    export YOUTUBE_API_KEY="YOUR_API_KEY"

    # For Windows
    set YOUTUBE_API_KEY="YOUR_API_KEY"
    ```
    The application will automatically detect and use this key.

### 5. Download the Bengali Font
For the word clouds and plots to render Bengali text correctly, you need the `NotoSansBengali-Regular.ttf` font.
-   Download it from [Google Fonts](https://fonts.google.com/specimen/Noto+Sans+Bengali).
-   Place the `.ttf` file in the root directory of the project.

### 6. Run the Application
```bash
python app.py
Open your web browser and navigate to the local URL provided in the terminal (usually http://127.0.0.1:7860).
---
##🤝 Contributing
Contributions are welcome and highly appreciated! If you have an idea for a new feature, find a bug, or want to improve the documentation, please feel free to:
Open an issue to discuss your ideas.
Submit a pull request with your changes.


