# ==============================================================================
# SOCIAL PERCEPTION ANALYZER - FINAL COMPLETE APPLICATION
# Version: 4.1 (Fully Refactored, Production-Ready)
# ==============================================================================
# --- IMPORTS ---
import re
from GoogleNews import GoogleNews
from requests.exceptions import HTTPError
import pandas as pd
import logging
import time
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
import gradio as gr
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties, fontManager
import seaborn as sns
from wordcloud import WordCloud
import dateparser
import numpy as np
import os

# ==============================================================================
# SETUP PRODUCTION-GRADE LOGGING & CONFIGURATION
# ==============================================================================
log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
log_handler = RotatingFileHandler('app.log', maxBytes=5*1024*1024, backupCount=2)
log_handler.setFormatter(log_formatter)
logger = logging.getLogger()
logger.setLevel(logging.INFO)
if not logger.handlers:
    logger.addHandler(log_handler)
logger.info("Application starting up.")

# --- APPLICATION CONFIGURATION ---
APP_TITLE = "Social Perception Analyzer"
APP_TAGLINE = "Prepared for the Policymakers of Bangladesh Nationalist Party (BNP)"
APP_FOOTER = "Developed by CDSR"

# --- FONT CONFIGURATION ---
FONT_PATH = 'NotoSansBengali-Regular.ttf'
BANGLA_FONT = None

def setup_bangla_font():
    """Properly set up Bengali font for all visualizations"""
    global BANGLA_FONT
    # Strictly enforce NotoSansBengali-Regular.ttf for all Bengali text
    if os.path.exists(FONT_PATH):
        try:
            fontManager.addfont(FONT_PATH)
            BANGLA_FONT = FontProperties(fname=FONT_PATH)
            plt.rcParams['font.family'] = BANGLA_FONT.get_name()
            plt.rcParams['axes.unicode_minus'] = False
            logger.info(f"Successfully loaded '{FONT_PATH}' for Bengali text.")
            return True
        except Exception as e:
            logger.error(f"Error loading Bengali font: {e}")
            return False
    else:
        logger.error(f"Font file {FONT_PATH} not found. Bengali text will not render correctly.")
        BANGLA_FONT = None
        plt.rcParams['font.family'] = 'sans-serif'
        return False

# Initialize font system
font_loaded = setup_bangla_font()

# ==============================================================================
# CORE HELPER FUNCTIONS
# ==============================================================================
def clean_bengali_text(text):
    """Remove non-Bengali characters except spaces and underscores (for joined phrases)"""
    cleaned = re.sub(r'[^\u0980-\u09FF_\s]', '', str(text))
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned

# Comprehensive stopword list for Bengali text analysis
BANGLA_STOP_WORDS = [
    'অতএব', 'অথচ', 'অথবা', 'অনুযায়ী', 'অনেক', 'অনেকে', 'অনেকেই', 'অন্তত', 'অন্য', 'অবধি', 'অবশ্য',
    'অভিপ্রায়', 'একে', 'একই', 'একেবারে', 'একটি', 'একবার', 'এখন', 'এখনও', 'এখানে', 'এখানেই', 'এটি',
    'এতটাই', 'এতদূর', 'এতটুকু', 'এক', 'এবং', 'এবার', 'এমন', 'এমনভাবে', 'এর', 'এরা', 'এঁরা', 'এঁদের',
    'এই', 'এইভাবে', 'ও', 'ওঁরা', 'ওঁর', 'ওঁদের', 'ওকে', 'ওখানে', 'ওদের', 'ওর', 'কাছ', 'কাছে', 'কাজ',
    'কারণ', 'কিছু', 'কিছুই', 'কিন্তু', 'কিভাবে', 'কেন', 'কোন', 'কোনও', 'কোনো', 'ক্ষেত্রে', 'খুব',
    'গুলি', 'গিয়ে', 'চায়', 'ছাড়া', 'জন্য', 'জানা', 'ঠিক', 'তিনি', 'তিন', 'তিনিও', 'তাকে', 'তাঁকে',
    'তার', 'তাঁর', 'তারা', 'তাঁরা', 'তাদের', 'তাঁদের', 'তাহলে', 'থাকলেও', 'থেকে', 'মধ্যেই', 'মধ্যে',
    'দ্বারা', 'নয়', 'না', 'নিজের', 'নিজে', 'নিয়ে', 'পারেন', 'পারা', 'পারে', 'পরে', 'পর্যন্ত', 'পুনরায়',
    'ফলে', 'বজায়', 'বা', 'বাদে', 'বার', 'বিশেষ', 'বিভিন্ন', 'ব্যবহার', 'ব্যাপারে', 'ভাবে', 'ভাবেই', 'মাধ্যমে',
    'মতো', 'মতোই', 'যখন', 'যদি', 'যদিও', 'যা', 'যাকে', 'যাওয়া', 'যায়', 'যে', 'যেখানে', 'যেতে', 'যেমন',
    'যেহেতু', 'রহিছে', 'শিক্ষা', 'শুধু', 'সঙ্গে', 'সব', 'সমস্ত', 'সম্প্রতি', 'সহ', 'সাধারণ', 'সামনে', 'হতে',
    'হতেই', 'হবে', 'হয়', 'হয়তো', 'হয়', 'হচ্ছে', 'হত', 'হলে', 'হলেও', 'হয়নি', 'হাজার', 'হোওয়া', 'আরও', 'আমরা',
    'আমার', 'আমি', 'আর', 'আগে', 'আগেই', 'আছে', 'আজ', 'তাকে', 'তাতে', 'তাদের', 'তাহার', 'তাহাতে', 'তাহারই',
    'তথা', 'তথাপি', 'সে', 'সেই', 'সেখান', 'সেখানে', 'থেকে', 'নাকি', 'নাগাদ', 'দু', 'দুটি', 'সুতরাং',
    'সম্পর্কে', 'সঙ্গেও', 'সর্বাধিক', 'সর্বদা', 'সহ', 'হৈতে', 'হইবে', 'হইয়া', 'হৈল', 'জানিয়েছেন', 'প্রতিবেদক'
]

COMBINED_STOPWORDS = set(BANGLA_STOP_WORDS)

PHRASES_TO_JOIN = {
    "তারেক রহমান": "তারেক_রহমান",
    "খালেদা জিয়া": "খালেদা_জিয়া",
    "বিএনপি জিন্দাবাদ": "বিএনপি_জিন্দাবাদ"
}

def get_dynamic_time_agg(start_date, end_date):
    """Determine appropriate time aggregation level based on date range"""
    if not isinstance(start_date, pd.Timestamp) or not isinstance(end_date, pd.Timestamp):
        return 'D', 'Daily'  # Graceful fallback
    
    delta = end_date - start_date
    if delta.days <= 2: 
        return 'H', 'Hourly'
    if delta.days <= 90: 
        return 'D', 'Daily'
    if delta.days <= 730: 
        return 'W', 'Weekly'
    return 'M', 'Monthly'

def kpi_badge_html(value, label, threshold_high=None, threshold_low=None):
    """
    Returns HTML for a color-coded KPI badge.
    Green for high, red for low, yellow for medium.
    """
    try:
        # Handle comma-separated numbers
        if isinstance(value, str) and ',' in value:
            val = float(value.replace(',', ''))
        else:
            val = float(value)
    except (TypeError, ValueError, AttributeError):
        val = value
    
    color = '#e0e0e0'  # default
    if threshold_high is not None and isinstance(val, (int, float)) and val >= threshold_high:
        color = '#4caf50'  # green
    elif threshold_low is not None and isinstance(val, (int, float)) and val <= threshold_low:
        color = '#f44336'  # red
    elif threshold_high is not None and threshold_low is not None and isinstance(val, (int, float)):
        color = '#ffeb3b'  # yellow
    
    # Format value with commas for large numbers
    if isinstance(value, (int, float)):
        formatted_value = f"{value:,.0f}"
    else:
        formatted_value = str(value)
    
    return f"<div style='display:inline-block;padding:8px 16px;border-radius:8px;background:{color};color:#222;font-weight:bold;margin:2px;'>{label}: {formatted_value}</div>"

def set_plot_style():
    """Configure consistent matplotlib style for all visualizations"""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['figure.figsize'] = (10, 6)
    # Always use NotoSansBengali-Regular.ttf for Bengali text
    if BANGLA_FONT and BANGLA_FONT.get_name():
        plt.rcParams['font.family'] = BANGLA_FONT.get_name()
    else:
        plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['axes.unicode_minus'] = False  # Fix for minus sign rendering

def cleanup_figures(*figures):
    """Properly close matplotlib figures to prevent memory leaks"""
    for fig in figures:
        if fig is not None:
            try:
                plt.close(fig)
            except:
                pass

# ==============================================================================
# NEWS SCRAPER BACKEND
# ==============================================================================
def run_news_scraper_pipeline(search_keywords, sites, start_date_str, end_date_str, interval, max_pages, filter_keys, progress=gr.Progress()):
    """Full implementation of the news scraper with robust error handling."""
    # Input validation and sanitization
    search_keywords = str(search_keywords).strip() if search_keywords else ""
    sites = str(sites).strip() if sites else ""
    start_date_str = str(start_date_str).strip() if start_date_str else ""
    end_date_str = str(end_date_str).strip() if end_date_str else ""
    filter_keys = str(filter_keys).strip() if filter_keys else ""
    
    if not all([search_keywords, start_date_str, end_date_str]):
        raise gr.Error("Search Keywords, Start Date, and End Date are required.")
    
    start_dt = dateparser.parse(start_date_str)
    end_dt = dateparser.parse(end_date_str)
    
    if not all([start_dt, end_dt]):
        raise gr.Error("Invalid date format. Please use a recognizable format like YYYY-MM-DD or '2 weeks ago'.")
    
    # Ensure start date is before end date
    if start_dt > end_dt:
        start_dt, end_dt = end_dt, start_dt
        gr.Warning("Start date was after end date. Dates have been swapped.")
    
    all_articles, current_dt = [], start_dt
    total_intervals = (end_dt - start_dt).days // interval + 1
    
    while current_dt <= end_dt:
        try:
            interval_end_dt = min(current_dt + pd.Timedelta(days=interval - 1), end_dt)
            start_str, end_str = current_dt.strftime('%Y-%m-%d'), interval_end_dt.strftime('%Y-%m-%d')
            
            progress((current_dt - start_dt).days / (end_dt - start_dt).days, 
                    desc=f"Fetching news from {start_str} to {end_str}")
            
            site_query = f"({' OR '.join(['site:' + s.strip() for s in sites.split(',') if s.strip()])})" if sites else ""
            final_query = f'"{search_keywords}" {site_query} after:{start_str} before:{end_str}'
            
            googlenews = GoogleNews(lang='bn', region='BD', period='1d')
            googlenews.search(final_query)
            
            for page in range(1, max_pages + 1):
                try:
                    results = googlenews.results()
                    if not results: 
                        break
                    all_articles.extend(results)
                    
                    if page < max_pages:
                        googlenews.getpage(page + 1)
                        time.sleep(0.3)  # Reduced sleep for performance
                except HTTPError as e:
                    if e.response.status_code == 429:
                        wait_time = 3  # Reduced wait for optimization
                        gr.Warning(f"Rate limited by Google News. Pausing for {wait_time} seconds.")
                        time.sleep(wait_time)
                    else:
                        logger.error(f"HTTP Error fetching news: {e}")
                        break
                except Exception as e:
                    logger.error(f"An error occurred fetching news: {e}")
                    break
            
            current_dt += pd.Timedelta(days=interval)
        except Exception as e:
            logger.error(f"Error in news scraping loop: {e}")
            break
    
    if not all_articles: 
        return pd.DataFrame(), pd.DataFrame()
    
    # Create DataFrame and clean data
    df = pd.DataFrame(all_articles).drop_duplicates(subset=['link'])
    
    # Parse dates safely
    df['published_date'] = df['date'].apply(lambda x: dateparser.parse(x, languages=['bn']) if pd.notna(x) else None)
    
    # Drop rows with missing critical data
    df = df.dropna(subset=['published_date', 'title'])
    
    # Apply advanced filtering if filter keywords are provided
    if filter_keys and filter_keys.strip():
        def match_complex_query(text, query):
            """Advanced query parser supporting AND, OR, NOT logic"""
            if not text or not query:
                return False
            
            text = str(text).lower()
            query = query.lower()
            
            # Simple tokenization that preserves phrases in quotes
            tokens = re.findall(r'"[^"]+"|\S+', query)
            
            # Build a regex pattern from the tokens
            patterns = []
            for token in tokens:
                if token == 'and':
                    continue  # We'll handle this with the final pattern
                elif token == 'or':
                    patterns.append('|')
                elif token == 'not':
                    patterns.append('(?=^(?!.*')
                else:
                    # Clean token and convert to regex pattern
                    clean_token = token.strip('"')
                    if clean_token.startswith('"') and clean_token.endswith('"'):
                        clean_token = clean_token[1:-1]
                    patterns.append(re.escape(clean_token))
            
            # Join patterns and handle negation
            final_pattern = ''.join(patterns)
            if '(?=' in final_pattern:
                final_pattern += '))'
            
            try:
                return bool(re.search(final_pattern, text))
            except:
                # Fallback to simple substring match if regex fails
                return any(token in text for token in tokens if token not in ['and', 'or', 'not'])
        
        # Apply filtering to title and description
        mask = df.apply(lambda row: match_complex_query(
            str(row['title']) + ' ' + str(row.get('desc', '')), 
            filter_keys
        ), axis=1)
        
        df = df[mask]
    
    # Return both full dataset and filtered display dataset
    return df, df[['published_date', 'title', 'media', 'desc', 'link']].sort_values(by='published_date', ascending=False)

# ==============================================================================
# YOUTUBE ANALYZER BACKEND
# ==============================================================================
def run_youtube_analysis_pipeline(api_key, query, max_videos_for_stats, num_videos_for_comments, max_comments_per_video, published_after, progress=gr.Progress()):
    """Complete YouTube analysis pipeline with robust error handling."""
    # Use integrated API key for seamless experience
    api_key = os.getenv("YOUTUBE_API_KEY", "AIzaSyB_f3uROqZfwBWsc_sDEV63WmUHBgvGGqw")
    
    if not query: 
        raise gr.Error("Search Keywords are required.")
    
    try:
        from googleapiclient.discovery import build
        from googleapiclient.errors import HttpError
        youtube = build('youtube', 'v3', developerKey=api_key)
    except ImportError:
        logger.error("Required YouTube API libraries not installed")
        raise gr.Error("YouTube analysis requires additional libraries. Please install google-api-python-client.")
    except HttpError as e:
        raise gr.Error(f"Failed to initialize YouTube service. Check API Key. Error: {e}")
    except Exception as e:
        raise gr.Error(f"An unexpected error occurred during API initialization: {e}")
    
    progress(0.1, desc="Performing broad scan for videos...")
    all_video_ids, next_page_token, total_results_estimate = [], None, 0
    PAGES_TO_FETCH = min(15, (max_videos_for_stats // 50) + 1)
    
    search_params = {
        'q': query, 
        'part': 'id', 
        'maxResults': 50, 
        'type': 'video', 
        'order': 'relevance'
    }
    
    if published_after:
        parsed_date = dateparser.parse(published_after)
        if parsed_date:
            search_params['publishedAfter'] = parsed_date.replace(tzinfo=timezone.utc).isoformat()
        else:
            gr.Warning(f"Could not parse date: '{published_after}'. Ignoring filter.")
    
    for page in range(PAGES_TO_FETCH):
        try:
            if next_page_token: 
                search_params['pageToken'] = next_page_token
            
            response = youtube.search().list(**search_params).execute()
            
            if page == 0:
                total_results_estimate = response.get('pageInfo', {}).get('totalResults', 0)
            
            # Extract valid video IDs
            valid_ids = []
            for item in response.get('items', []):
                if 'id' in item and 'videoId' in item['id']:
                    valid_ids.append(item['id']['videoId'])
            
            all_video_ids.extend(valid_ids)
            
            next_page_token = response.get('nextPageToken')
            progress(0.1 + (0.3 * (page / PAGES_TO_FETCH)), 
                    desc=f"Broad scan: Found {len(all_video_ids)} videos...")
            
            if not next_page_token: 
                break
        except HttpError as e:
            if "quotaExceeded" in str(e):
                raise gr.Error("CRITICAL: YouTube API daily quota exceeded. Try again tomorrow.")
            logger.error(f"HTTP error during video search: {e}")
            break
        except Exception as e:
            logger.error(f"Unexpected error during YouTube search: {e}")
            break
    
    if not all_video_ids:
        return pd.DataFrame(), pd.DataFrame(), ""
    
    # Fetch video details in batches
    progress(0.4, desc=f"Fetching details for {len(all_video_ids)} videos...")
    
    def _fetch_video_details(youtube_service, video_ids: list):
        """Fetch detailed information for a batch of video IDs"""
        all_videos_data = []
        try:
            for i in range(0, len(video_ids), 50):
                id_batch = video_ids[i:i+50]
                video_request = youtube_service.videos().list(
                    part="snippet,statistics", 
                    id=",".join(id_batch)
                )
                video_response = video_request.execute()
                
                for item in video_response.get('items', []):
                    stats = item.get('statistics', {})
                    all_videos_data.append({
                        'video_id': item['id'], 
                        'video_title': item['snippet']['title'],
                        'channel': item['snippet']['channelTitle'], 
                        'published_date': item['snippet']['publishedAt'],
                        'view_count': int(stats.get('viewCount', 0)), 
                        'like_count': int(stats.get('likeCount', 0)),
                        'comment_count': int(stats.get('commentCount', 0))
                    })
        except Exception as e:
            logger.error(f"Could not fetch video details: {e}")
        
        return all_videos_data
    
    videos_df_full_scan = pd.DataFrame(_fetch_video_details(youtube, all_video_ids))
    
    if videos_df_full_scan.empty:
        return pd.DataFrame(), pd.DataFrame(), ""
    
    # Process and clean video data
    videos_df_full_scan['published_date'] = pd.to_datetime(videos_df_full_scan['published_date'])
    
    # Calculate engagement rate safely
    videos_df_full_scan['engagement_rate'] = (
        (videos_df_full_scan['like_count'] + videos_df_full_scan['comment_count']) / 
        videos_df_full_scan['view_count'].replace(0, 1)
    ).fillna(0)
    
    videos_df_full_scan = videos_df_full_scan.sort_values(
        by='view_count', 
        ascending=False
    ).reset_index(drop=True)
    
    # Fetch comments for top videos
    videos_to_scrape_df = videos_df_full_scan.head(int(num_videos_for_comments))
    all_comments = []
    
    def _scrape_single_video_comments(youtube_service, video_id, max_comments):
        """Scrape comments for a single video with error handling"""
        comments_list = []
        try:
            request = youtube_service.commentThreads().list(
                part="snippet", 
                videoId=video_id, 
                maxResults=min(max_comments, 100),
                order='relevance', 
                textFormat="plainText"
            )
            response = request.execute()
            
            for item in response.get('items', []):
                snippet = item['snippet']['topLevelComment']['snippet']
                comments_list.append({
                    'author': snippet['authorDisplayName'], 
                    'published_date_comment': snippet['publishedAt'],
                    'comment_text': snippet['textDisplay'], 
                    'likes': snippet['likeCount'],
                    'replies': item['snippet']['totalReplyCount']
                })
        except Exception as e:
            logger.warning(f"Could not retrieve comments for video {video_id}: {e}")
        
        return comments_list
    
    for index, row in videos_to_scrape_df.iterrows():
        progress(0.7 + (0.3 * (index / len(videos_to_scrape_df))), 
                desc=f"Deep dive: Scraping comments from video {index+1}/{len(videos_to_scrape_df)}...")
        
        comments_for_video = _scrape_single_video_comments(
            youtube, 
            row['video_id'], 
            max_comments_per_video
        )
        
        if comments_for_video:
            for comment in comments_for_video:
                comment.update({
                    'video_id': row['video_id'], 
                    'video_title': row['video_title']
                })
            all_comments.extend(comments_for_video)
    
    comments_df = pd.DataFrame(all_comments)
    if not comments_df.empty:
        comments_df['published_date_comment'] = pd.to_datetime(comments_df['published_date_comment'])
    
    logger.info(f"YouTube analysis complete. Est. total videos: {total_results_estimate}. "
               f"Scanned: {len(videos_df_full_scan)}. Comments: {len(comments_df)}.")
    
    # Create summary HTML
    summary_html = f"""
    <div style='background:#f5f5f5;padding:16px;border-radius:12px;margin-bottom:12px;box-shadow:0 2px 8px #eee;'>
        <h3 style='margin:0 0 8px 0;'>YouTube Analytics Summary</h3>
        <ul style='margin:0;padding-left:18px;'>
            <li><b>Total Videos:</b> {len(videos_df_full_scan):,}</li>
            <li><b>Total Comments:</b> {len(comments_df):,}</li>
            <li><b>Total Views:</b> {videos_df_full_scan['view_count'].sum():,}</li>
        </ul>
    </div>
    """
    
    return videos_df_full_scan, comments_df, summary_html

# ==============================================================================
# ADVANCED ANALYTICS MODULE
# ==============================================================================
def generate_scraper_dashboard(df: pd.DataFrame):
    """Generate comprehensive dashboard from news scraper results."""
    if df.empty:
        # Return empty dashboard components
        return {
            "kpi_total_articles": gr.HTML(""),
            "kpi_unique_media": gr.HTML(""),
            "kpi_date_range": gr.HTML(""),
            "dashboard_timeline_plot": None,
            "dashboard_media_plot": None,
            "dashboard_wordcloud_plot": None
        }
    
    set_plot_style()
    
    # Calculate KPIs
    total_articles, unique_media = len(df), df['media'].nunique()
    start_date, end_date = pd.to_datetime(df['published_date']).min(), pd.to_datetime(df['published_date']).max()
    date_range_str = f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
    
    # Color-coded KPI badges
    kpi_total_articles_html = kpi_badge_html(
        total_articles, 'Total Articles', threshold_high=100, threshold_low=10
    )
    kpi_unique_media_html = kpi_badge_html(
        unique_media, 'Unique Media', threshold_high=10, threshold_low=2
    )
    kpi_date_range_html = kpi_badge_html(
        date_range_str, 'Date Range', threshold_high=None, threshold_low=None
    )
    
    # Time series visualization - FIXED GRADIO API USAGE
    agg_code, agg_name = get_dynamic_time_agg(start_date, end_date)
    timeline_df = df.set_index(pd.to_datetime(df['published_date'])).resample(agg_code).size().reset_index(name='count')
    timeline_df.rename(columns={'published_date': 'date'}, inplace=True)
    timeline_plot = gr.LinePlot(
        value=timeline_df,
        x='date',
        y='count',
        title=f'{agg_name} News Volume',
        tooltip=['date', 'count'],
        x_title="Date",
        y_title="Number of Articles"
    )
    
    # Media source analysis
    media_counts = df['media'].dropna().value_counts().nlargest(15).sort_values()
    fig_media = None
    if not media_counts.empty:
        fig_media, ax = plt.subplots(figsize=(8, 6))
        media_counts.plot(kind='barh', ax=ax, color='skyblue')
        ax.set_title("Top 15 Media Sources", fontproperties=BANGLA_FONT)
        ax.set_yticklabels(media_counts.index, fontproperties=BANGLA_FONT)
    ax.set_xlabel("Article Count", fontproperties=BANGLA_FONT)
    plt.tight_layout()
    
    # Word cloud generation
    fig_wc = None
    try:
        # Combine all titles and clean text
        text = " ".join(title for title in df['title'].astype(str))
        text = clean_bengali_text(text)
        
        # Join special phrases
        for phrase, joined in PHRASES_TO_JOIN.items():
            text = text.replace(phrase, joined)
        
        # Extract and filter words
        words = re.findall(r'[\u0980-\u09FF_]{2,}', text)
        words = [w for w in words if w not in COMBINED_STOPWORDS]
        words = [w for w in words if len(w) > 1]
        words = [w for w in words if not re.search(r'[a-zA-Z]', w)]
        
        # Filter by frequency
        from collections import Counter
        word_freq = Counter(words)
        min_freq = 2
        most_common = set([w for w, _ in word_freq.most_common(3)])
        filtered_words = [w for w in words if word_freq[w] >= min_freq and w not in most_common]
        wc_text = " ".join(filtered_words)
        
        # Generate word cloud
        if wc_text.strip():
            wc = WordCloud(
                font_path=FONT_PATH,
                width=1600,
                height=900,
                background_color='white',
                stopwords=COMBINED_STOPWORDS,
                collocations=False,
                colormap='plasma',
                max_words=200,
                contour_width=2,
                contour_color='steelblue',
                regexp=r"[\u0980-\u09FF_]+"
            ).generate(wc_text)
            
            fig_wc, ax = plt.subplots(figsize=(15, 8))
            ax.imshow(wc, interpolation='bilinear')
            ax.axis("off")
            ax.set_title("Bengali Headline Word Cloud", fontproperties=BANGLA_FONT, fontsize=22)
            plt.tight_layout()
    except Exception as e:
        logger.error(f"WordCloud failed: {e}")
        gr.Warning(f"WordCloud generation failed: {str(e)}")
    
    return {
        "kpi_total_articles": gr.HTML(kpi_total_articles_html),
        "kpi_unique_media": gr.HTML(kpi_unique_media_html),
        "kpi_date_range": gr.HTML(kpi_date_range_html),
        "dashboard_timeline_plot": timeline_plot,
        "dashboard_media_plot": fig_media,
        "dashboard_wordcloud_plot": fig_wc
    }

def generate_youtube_dashboard(videos_df, comments_df):
    """Generate comprehensive dashboard from YouTube analysis results."""
    # Initialize all dashboard components
    dashboard_components = {
        "kpi_yt_videos_found": gr.HTML(""),
        "kpi_yt_views_scanned": gr.HTML(""),
        "kpi_yt_comments_scraped": gr.HTML(""),
        "yt_channel_plot": None,
        "yt_wordcloud_plot": None,
        "yt_top_videos_plot": None,
        "yt_engagement_plot": None,
        "yt_time_series_plot": None
    }
    
    # Generate KPIs if data exists
    if videos_df is not None and not videos_df.empty:
        dashboard_components["kpi_yt_videos_found"] = gr.HTML(
            kpi_badge_html(len(videos_df), 'Videos Found', threshold_high=50, threshold_low=5)
        )
        dashboard_components["kpi_yt_views_scanned"] = gr.HTML(
            kpi_badge_html(videos_df['view_count'].sum(), 'Views Scanned', threshold_high=100000, threshold_low=1000)
        )
    
    if comments_df is not None and not comments_df.empty:
        dashboard_components["kpi_yt_comments_scraped"] = gr.HTML(
            kpi_badge_html(len(comments_df), 'Comments Scraped', threshold_high=100, threshold_low=10)
        )
    
    # Channel analysis
    fig_channels = None
    if videos_df is not None and not videos_df.empty and 'channel' in videos_df.columns:
        channel_counts = videos_df['channel'].value_counts().nlargest(15).sort_values()
        if not channel_counts.empty:
            fig_channels, ax = plt.subplots(figsize=(8, 6))
            channel_counts.plot(kind='barh', ax=ax, color='coral')
            ax.set_title("Top 15 Channels by Video Volume", fontproperties=BANGLA_FONT)
            ax.set_yticklabels(channel_counts.index, fontproperties=BANGLA_FONT)
            ax.set_xlabel("Video Count", fontproperties=BANGLA_FONT)
            plt.tight_layout()
    dashboard_components["yt_channel_plot"] = fig_channels
    
    # Word cloud from comments
    fig_wc = None
    if comments_df is not None and not comments_df.empty and 'comment_text' in comments_df.columns:
        try:
            text = " ".join(comment for comment in comments_df['comment_text'].astype(str))
            text = clean_bengali_text(text)
            
            # Join special phrases
            for phrase, joined in PHRASES_TO_JOIN.items():
                text = text.replace(phrase, joined)
            
            # Extract and filter words
            words = re.findall(r'[\u0980-\u09FF_]{2,}', text)
            words = [w for w in words if w not in COMBINED_STOPWORDS]
            words = [w for w in words if len(w) > 1]
            words = [w for w in words if not re.search(r'[a-zA-Z]', w)]
            
            # Filter by frequency
            from collections import Counter
            word_freq = Counter(words)
            min_freq = 2
            most_common = set([w for w, _ in word_freq.most_common(3)])
            filtered_words = [w for w in words if word_freq[w] >= min_freq and w not in most_common]
            wc_text = " ".join(filtered_words)
            
            # Generate word cloud
            if wc_text.strip():
                wc = WordCloud(
                    font_path=FONT_PATH,
                    width=1600,
                    height=900,
                    background_color='white',
                    stopwords=COMBINED_STOPWORDS,
                    collocations=False,
                    colormap='plasma',
                    max_words=250,
                    contour_width=2,
                    contour_color='darkorange',
                    regexp=r"[\u0980-\u09FF_]+"
                ).generate(wc_text)
                
                fig_wc, ax = plt.subplots(figsize=(15, 8))
                ax.imshow(wc, interpolation='bilinear')
                ax.axis("off")
                ax.set_title("Bengali Word Cloud from YouTube Comments", fontproperties=BANGLA_FONT, fontsize=22)
                plt.tight_layout()
        except Exception as e:
            logger.error(f"YouTube WordCloud failed: {e}")
    dashboard_components["yt_wordcloud_plot"] = fig_wc
    
    # Top commented videos
    fig_top_videos = None
    if comments_df is not None and not comments_df.empty and 'video_title' in comments_df.columns:
        top_videos = comments_df['video_title'].value_counts().nlargest(10)
        if not top_videos.empty:
            fig_top_videos, ax = plt.subplots(figsize=(10, 6))
            top_videos.plot(kind='barh', ax=ax, color='dodgerblue')
            ax.set_title("Top 10 Videos by Comment Count", fontproperties=BANGLA_FONT)
            ax.set_xlabel("Comment Count", fontproperties=BANGLA_FONT)
            ax.set_yticklabels(top_videos.index, fontproperties=BANGLA_FONT)
            plt.tight_layout()
    dashboard_components["yt_top_videos_plot"] = fig_top_videos
    
    # Engagement rate per video
    fig_engagement = None
    if videos_df is not None and not videos_df.empty and comments_df is not None and not comments_df.empty:
        if 'video_id' in videos_df.columns and 'video_id' in comments_df.columns:
            try:
                # Count comments per video
                comment_counts = comments_df['video_id'].value_counts().reset_index()
                comment_counts.columns = ['video_id', 'comment_count']
                # Ensure 'comment_count' column exists in videos_df
                merged = videos_df.merge(comment_counts, on='video_id', how='left')
                if 'comment_count' not in merged.columns:
                    merged['comment_count'] = 0
                merged['comment_count'] = merged['comment_count'].fillna(0)
                # Calculate engagement rate
                merged['engagement_rate'] = merged['comment_count'] / merged['view_count'].replace(0, 1)
                # Get top 10 videos by engagement
                top_engagement = merged.nlargest(10, 'engagement_rate')
                if not top_engagement.empty:
                    fig_engagement, ax = plt.subplots(figsize=(10, 6))
                    ax.barh(top_engagement['video_title'], top_engagement['engagement_rate'], color='mediumseagreen')
                    ax.set_title("Top 10 Videos by Engagement Rate", fontproperties=BANGLA_FONT)
                    ax.set_xlabel("Engagement Rate (Comments / Views)", fontproperties=BANGLA_FONT)
                    ax.set_yticklabels(top_engagement['video_title'], fontproperties=BANGLA_FONT)
                    plt.tight_layout()
            except Exception as e:
                logger.error(f"Engagement rate calculation failed: {e}")
    dashboard_components["yt_engagement_plot"] = fig_engagement
    
    # Comment activity over time
    fig_time_series = None
    if comments_df is not None and not comments_df.empty and 'published_date_comment' in comments_df.columns:
        try:
            comments_df['published_date_comment'] = pd.to_datetime(comments_df['published_date_comment'])
            time_series = comments_df.set_index('published_date_comment').resample('D').size().reset_index()
            time_series.columns = ['date', 'count']
            
            if not time_series.empty:
                fig_time_series = gr.LinePlot(
                    value=time_series,
                    x='date',
                    y='count',
                    title="Comment Activity Over Time",
                    tooltip=['date', 'count'],
                    x_title="Date",
                    y_title="Number of Comments"
                )
        except Exception as e:
            logger.error(f"Error in comment activity plot: {e}")
    dashboard_components["yt_time_series_plot"] = fig_time_series
    
    return dashboard_components

# ==============================================================================
# GRADIO UI DEFINITION
# ==============================================================================
with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue", secondary_hue="orange"), title=APP_TITLE) as app:
    gr.Markdown(f"# {APP_TITLE}\n*{APP_TAGLINE}*")
    
    # --- STATE MANAGEMENT ---
    scraper_results_state = gr.State()
    youtube_results_state = gr.State()
    
    with gr.Tabs():
        with gr.TabItem("1. News Scraper", id=0):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Search Criteria")
                    search_keywords_textbox = gr.Textbox(
                        label="Search Keywords", 
                        placeholder="e.g., বিএনপি সমাবেশ", 
                        info="Keywords to search for in news articles."
                    )
                    sites_to_search_textbox = gr.Textbox(
                        label="Target Sites (Optional, comma-separated)", 
                        placeholder="e.g., prothomalo.com", 
                        info="Limit search to specific news sites."
                    )
                    start_date_textbox = gr.Textbox(
                        label="Start Date", 
                        placeholder="YYYY-MM-DD or 'last week'", 
                        info="Start date for news scraping."
                    )
                    end_date_textbox = gr.Textbox(
                        label="End Date", 
                        placeholder="YYYY-MM-DD or 'today'", 
                        info="End date for news scraping."
                    )
                    
                    gr.Markdown("### Scraping Parameters")
                    interval_days_slider = gr.Slider(
                        1, 7, 3, step=1, 
                        label="Days per Interval", 
                        info="How many days to group each scraping interval."
                    )
                    max_pages_slider = gr.Slider(
                        1, 10, 5, step=1, 
                        label="Max Pages per Interval", 
                        info="Maximum number of pages to fetch per interval."
                    )
                    filter_keywords_textbox = gr.Textbox(
                        label="Filter Keywords (comma-separated, optional)", 
                        placeholder="e.g., নির্বাচন, সরকার", 
                        info="Filter results by these keywords."
                    )
                    
                    start_scraper_button = gr.Button("Start Scraping & Analysis", variant="primary")
                    scraper_progress = gr.Progress()
                
                with gr.Column(scale=2):
                    scraper_results_df = gr.DataFrame(
                        label="Filtered Results", 
                        interactive=True
                    )
                    scraper_download_file = gr.File(
                        label="Download Filtered Results CSV"
                    )
        
        with gr.TabItem("2. News Analytics", id=1):
            gr.Markdown("### News Analytics Dashboard")
            
            with gr.Group():
                news_summary_card = gr.HTML(
                    "<div style='background:#f5f5f5;padding:16px;border-radius:12px;margin-bottom:12px;box-shadow:0 2px 8px #eee;'>"
                    "<h3 style='margin:0 0 8px 0;'>Key Findings</h3>"
                    "<ul style='margin:0;padding-left:18px;'>"
                    "<li><b>Total Articles:</b> <span id='news_total_articles'></span></li>"
                    "<li><b>Unique Media:</b> <span id='news_unique_media'></span></li>"
                    "<li><b>Date Range:</b> <span id='news_date_range'></span></li>"
                    "</ul></div>"
                )
                
                kpi_total_articles = gr.HTML()
                kpi_unique_media = gr.HTML()
                kpi_date_range = gr.HTML()
                
                with gr.Row():
                    with gr.Column():
                        dashboard_timeline_plot = gr.LinePlot(
                            label="News Volume Timeline"
                        )
                    with gr.Column():
                        dashboard_media_plot = gr.Plot(
                            label="Top Media Sources by Article Count"
                        )
                
                dashboard_wordcloud_plot = gr.Plot(
                    label="Headline Word Cloud"
                )
        
        with gr.TabItem("3. YouTube Topic Analysis", id=2):
            gr.Markdown("## YouTube Topic Analysis")
            
            with gr.Row():
                with gr.Column(scale=1):
                    yt_search_keywords = gr.Textbox(
                        label="YouTube Search Keywords", 
                        placeholder="e.g., BNP Rally", 
                        info="Keywords to search for in YouTube videos."
                    )
                    yt_max_videos_slider = gr.Slider(
                        10, 100, 30, step=5, 
                        label="Max Videos for Stats", 
                        info="Maximum number of videos to scan for statistics."
                    )
                    yt_num_videos_comments_slider = gr.Slider(
                        1, 20, 5, step=1, 
                        label="Videos for Comments", 
                        info="Number of top videos to scrape comments from."
                    )
                    yt_max_comments_slider = gr.Slider(
                        10, 200, 50, step=10, 
                        label="Max Comments per Video", 
                        info="Maximum number of comments to fetch per video."
                    )
                    yt_published_after = gr.Textbox(
                        label="Published After (Optional)", 
                        placeholder="YYYY-MM-DD", 
                        info="Only include videos published after this date."
                    )
                    
                    start_youtube_analysis_button = gr.Button(
                        "Start YouTube Analysis", 
                        variant="primary"
                    )
                    yt_progress = gr.Progress()
                
                with gr.Column(scale=2):
                    yt_results_df = gr.DataFrame(
                        label="YouTube Video Results", 
                        interactive=True
                    )
                    yt_comments_df = gr.DataFrame(
                        label="YouTube Comments Results", 
                        interactive=True
                    )
                    
                    yt_dashboard_html = gr.HTML()
                    
                    with gr.Group():
                        kpi_yt_videos_found = gr.HTML()
                        kpi_yt_views_scanned = gr.HTML()
                        kpi_yt_comments_scraped = gr.HTML()
                        
                        with gr.Row():
                            with gr.Column():
                                yt_channel_plot = gr.Plot(
                                    label="Top Channels by Video Volume"
                                )
                            with gr.Column():
                                yt_time_series_plot = gr.LinePlot(
                                    label="Comment Activity Over Time"
                                )
                        
                        with gr.Row():
                            with gr.Column():
                                yt_top_videos_plot = gr.Plot(
                                    label="Top Videos by Comment Count"
                                )
                            with gr.Column():
                                yt_engagement_plot = gr.Plot(
                                    label="Top Videos by Engagement Rate"
                                )
                        
                        yt_wordcloud_plot = gr.Plot(
                            label="Bengali Word Cloud from Comments"
                        )

    # --- EVENT HANDLERS ---
    def scraper_button_handler(search_keywords, sites, start_date, end_date, interval, max_pages, filter_keys):
        """Handle news scraper button click event."""
        try:
            df, filtered_df = run_news_scraper_pipeline(
                search_keywords, sites, start_date, end_date, 
                interval, max_pages, filter_keys
            )
            
            # Update the state with the full results
            scraper_results_state = df
            
            # Generate dashboard visualizations
            dashboard = generate_scraper_dashboard(df)
            
            # Prepare download file
            if not df.empty:
                csv_path = "news_results.csv"
                df.to_csv(csv_path, index=False)
                scraper_download_file = gr.File(value=csv_path, visible=True)
            else:
                scraper_download_file = gr.File(visible=False)
            
            return (
                filtered_df,
                scraper_download_file,
                dashboard["kpi_total_articles"],
                dashboard["kpi_unique_media"],
                dashboard["kpi_date_range"],
                dashboard["dashboard_timeline_plot"],
                dashboard["dashboard_media_plot"],
                dashboard["dashboard_wordcloud_plot"]
            )
        except Exception as e:
            logger.error(f"Error in scraper button handler: {str(e)}")
            gr.Error(f"An error occurred during scraping: {str(e)}")
            # Return empty values to reset the UI
            return (
                pd.DataFrame(), 
                gr.File(visible=False),
                gr.HTML(""), gr.HTML(""), gr.HTML(""),
                None, None, None
            )
    
    start_scraper_button.click(
        fn=scraper_button_handler,
        inputs=[
            search_keywords_textbox, 
            sites_to_search_textbox, 
            start_date_textbox, 
            end_date_textbox, 
            interval_days_slider, 
            max_pages_slider, 
            filter_keywords_textbox
        ],
        outputs=[
            scraper_results_df,
            scraper_download_file,
            kpi_total_articles,
            kpi_unique_media,
            kpi_date_range,
            dashboard_timeline_plot,
            dashboard_media_plot,
            dashboard_wordcloud_plot
        ]
    )
    
    def youtube_button_handler(keywords, max_videos, num_comments_videos, max_comments, published_after):
        """Handle YouTube analysis button click event."""
        try:
            videos_df, comments_df, summary_html = run_youtube_analysis_pipeline(
                api_key=None,
                query=keywords,
                max_videos_for_stats=max_videos,
                num_videos_for_comments=num_comments_videos,
                max_comments_per_video=max_comments,
                published_after=published_after
            )
            
            # Update the state with the results
            youtube_results_state = (videos_df, comments_df)
            
            # Generate dashboard visualizations
            dashboard = generate_youtube_dashboard(videos_df, comments_df)
            
            return (
                videos_df,
                comments_df,
                summary_html,
                dashboard["kpi_yt_videos_found"],
                dashboard["kpi_yt_views_scanned"],
                dashboard["kpi_yt_comments_scraped"],
                dashboard["yt_channel_plot"],
                dashboard["yt_time_series_plot"],
                dashboard["yt_top_videos_plot"],
                dashboard["yt_engagement_plot"],
                dashboard["yt_wordcloud_plot"]
            )
        except Exception as e:
            logger.error(f"Error in YouTube button handler: {str(e)}")
            gr.Error(f"An error occurred during YouTube analysis: {str(e)}")
            # Return empty values to reset the UI
            return (
                pd.DataFrame(), 
                pd.DataFrame(),
                gr.HTML(""),
                gr.HTML(""), gr.HTML(""), gr.HTML(""),
                None, None, None, None, None
            )
    
    start_youtube_analysis_button.click(
        fn=youtube_button_handler,
        inputs=[
            yt_search_keywords, 
            yt_max_videos_slider, 
            yt_num_videos_comments_slider, 
            yt_max_comments_slider, 
            yt_published_after
        ],
        outputs=[
            yt_results_df,
            yt_comments_df,
            yt_dashboard_html,
            kpi_yt_videos_found,
            kpi_yt_views_scanned,
            kpi_yt_comments_scraped,
            yt_channel_plot,
            yt_time_series_plot,
            yt_top_videos_plot,
            yt_engagement_plot,
            yt_wordcloud_plot
        ]
    )

# ==============================================================================
# LAUNCH THE APP
# ==============================================================================
if __name__ == "__main__":
    app.launch( debug=True,share=True)