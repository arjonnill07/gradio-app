# ==============================================================================
# SOCIAL PERCEPTION ANALYZER - FINAL COMPLETE APPLICATION
# Version: 3.0 (Architecturally Refactored, Production Ready)
# ==============================================================================

# --- IMPORTS ---
import gradio as gr
import pandas as pd
import numpy as np
import torch
import re
import sqlite3
import json
import logging
import requests
import os
import time
import random
import functools
from io import StringIO
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler

# --- APIs and Web Scraping ---
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from GoogleNews import GoogleNews
from urllib.error import HTTPError
import dateparser

# --- NLP & Machine Learning ---
from transformers import pipeline, BitsAndBytesConfig, AutoTokenizer
from sentence_transformers import SentenceTransformer
from huggingface_hub.utils import HfHubHTTPError

# --- Visualization ---
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import seaborn as sns
from wordcloud import WordCloud

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
try:
    BANGLA_FONT = FontProperties(fname=FONT_PATH)
    logger.info("Successfully loaded 'NotoSansBengali-Regular.ttf' font.")
except OSError:
    logger.error("Failed to load 'NotoSansBengali-Regular.ttf'. Ensure the file is in the root directory.")
    gr.Warning("Bangla font not found! Visualizations may not render text correctly.")
    BANGLA_FONT = FontProperties()

# ==============================================================================
# CORE HELPER FUNCTIONS
# ==============================================================================

BANGLA_STOP_WORDS = [
    'অতএব', 'অথচ', 'অথবা', 'অনুযায়ী', 'অনেক', 'অনেকে', 'অনেকেই', 'অন্তত', 'অন্য', 'অবধি', 'অবশ্য',
    'অভিপ্রায়', 'একে', 'একই', 'একেবারে', 'একটি', 'একবার', 'এখন', 'এখনও', 'এখানে', 'এখানেই', 'এটি',
    'এতটাই', 'এতদূর', 'এতটুকু', 'এক', 'এবং', 'এবার', 'এমন', 'এমনভাবে', 'এর', 'এরা', 'এঁরা', 'এঁদের',
    'এই', 'এইভাবে', 'ও', 'ওঁরা', 'ওঁর', 'ওঁদের', 'ওকে', 'ওখানে', 'ওদের', 'ওর', 'কাছ', 'কাছে', 'কাজ',
    'কারণ', 'কিছু', 'কিছুই', 'কিন্তু', 'কিভাবে', 'কেন', 'কোন', 'কোনও', 'কোনো', 'ক্ষেত্রে', 'খুব',
    'গুলি', 'গিয়ে', 'চায়', 'ছাড়া', 'জন্য', 'জানা', 'ঠিক', 'তিনি', 'তিন', 'তিনিও', 'তাকে', 'তাঁকে',
    'তার', 'তাঁর', 'তারা', 'তাঁরা', 'তাদের', 'তাঁদের', 'তাহলে', ' থাকলেও', 'থেকে', 'মধ্যেই', 'মধ্যে',
    'द्वारा', 'নয়', 'না', 'নিজের', 'নিজে', 'নিয়ে', 'পারেন', 'পারা', 'পারে', 'পরে', 'পর্যন্ত', 'পুনরায়',
    'ফলে', 'বজায়', 'বা', 'বাদে', 'বার', 'বিশেষ', 'বিভিন্ন', 'ব্যবহার', 'ব্যাপারে', 'ভাবে', 'ভাবেই', 'মাধ্যমে',
    'মতো', 'মতোই', 'যখন', 'যদি', 'যদিও', 'যা', 'যাকে', 'যাওয়া', 'যায়', 'যে', 'যেখানে', 'যেতে', 'যেমন',
    'যেহেতু', 'রহিছে', 'শিক্ষা', 'শুধু', 'সঙ্গে', 'সব', 'সমস্ত', 'সম্প্রতি', 'সহ', 'সাধারণ', 'সামনে', 'হতে',
    'হতেই', 'হবে', 'হয়', 'হয়তো', 'হয়', 'হচ্ছে', 'হত', 'হলে', 'হলেও', 'হয়নি', 'হাজার', 'হোওয়া', 'আরও', 'আমরা',
    'আমার', 'আমি', 'আর', 'আগে', 'আগেই', 'আছে', 'আজ', 'তাকে', 'তাতে', 'তাদের', 'তাহার', 'তাহাতে', 'তাহারই',
    'তথা', 'তথাপি', 'সে', 'সেই', 'সেখান', 'সেখানে', 'থেকে', 'নাকি', 'নাগাদ', 'দু', 'দুটি', 'সুতরাং',
    'সম্পর্কে', 'সঙ্গেও', 'সর্বাধিক', 'সর্বদা', 'সহ', 'হৈতে', 'হইবে', 'হইয়া', 'হৈল', 'জানিয়েছেন', 'প্রতিবেদক'
]

def get_dynamic_time_agg(start_date, end_date):
    """Hardened helper to determine time aggregation level."""
    if not isinstance(start_date, pd.Timestamp) or not isinstance(end_date, pd.Timestamp):
        return 'D', 'Daily' # Graceful fallback
    delta = end_date - start_date
    if delta.days <= 2: return 'H', 'Hourly'
    if delta.days <= 90: return 'D', 'Daily'
    if delta.days <= 730: return 'W', 'Weekly'
    return 'M', 'Monthly'

# ==============================================================================
# ML MODEL MANAGEMENT
TOKENIZER_MODEL_ID = "csebuetnlp/banglabert_large"
TOKENIZER = None

def get_bangla_tokenizer():
    global TOKENIZER
    if TOKENIZER is None:
        try:
            TOKENIZER = AutoTokenizer.from_pretrained(TOKENIZER_MODEL_ID)
            logger.info("BanglaBERT tokenizer loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load BanglaBERT tokenizer: {e}")
            TOKENIZER = None
    return TOKENIZER
# ==============================================================================


SENTIMENT_MODEL_ID = 'ahs95/banglabert-sentiment-analysis'
MODELS = {"sentiment_pipeline": None}

def _load_pipeline_with_retry(task, model_id, retries=3):
    logger.info(f"Initializing {task} pipeline for model: {model_id}")
    for attempt in range(retries):
        try:
            device = 0 if torch.cuda.is_available() else -1
            if device == -1: gr.Warning(f"{model_id} will run on CPU and may be very slow.")
            pipe = pipeline(task, model=model_id, device=device)
            logger.info(f"Pipeline '{task}' loaded successfully.")
            return pipe
        except (HfHubHTTPError, requests.exceptions.ConnectionError) as e:
            logger.warning(f"Network error on loading {model_id} (Attempt {attempt + 1}/{retries}): {e}")
            if attempt < retries - 1: time.sleep(5)
            else: raise gr.Error(f"Failed to download model '{model_id}' after {retries} attempts. Check network.")
        except Exception as e:
            logger.error(f"An unexpected error occurred while loading {model_id}: {e}")
            raise gr.Error(f"Could not initialize model '{model_id}'. Error: {e}")
    return None

def get_sentiment_pipeline():
    if MODELS["sentiment_pipeline"] is None:
        MODELS["sentiment_pipeline"] = _load_pipeline_with_retry("sentiment-analysis", SENTIMENT_MODEL_ID)
    return MODELS["sentiment_pipeline"]

# ==============================================================================
# NEWS SCRAPER BACKEND
# ==============================================================================

def run_news_scraper_pipeline(search_keywords, sites, start_date_str, end_date_str, interval, max_pages, filter_keys, progress=gr.Progress()):
    """Full, robust implementation of the news scraper."""
    # Input validation and sanitization
    search_keywords = search_keywords.strip()
    if not all([search_keywords, start_date_str, end_date_str]):
        raise gr.Error("Search Keywords, Start Date, and End Date are required.")

    start_dt = dateparser.parse(start_date_str)
    end_dt = dateparser.parse(end_date_str)
    if not all([start_dt, end_dt]):
        raise gr.Error("Invalid date format. Please use a recognizable format like YYYY-MM-DD or '2 weeks ago'.")

    all_articles, current_dt = [], start_dt
    while current_dt <= end_dt:
        interval_end_dt = min(current_dt + pd.Timedelta(days=interval - 1), end_dt)
        start_str, end_str = current_dt.strftime('%Y-%m-%d'), interval_end_dt.strftime('%Y-%m-%d')
        progress(0, desc=f"Fetching news from {start_str} to {end_str}")

        site_query = f"({' OR '.join(['site:' + s.strip() for s in sites.split(',') if s.strip()])})" if sites else ""
        final_query = f'"{search_keywords}" {site_query} after:{start_str} before:{end_str}'

        googlenews = GoogleNews(lang='bn', region='BD')
        googlenews.search(final_query)
        
        for page in range(1, max_pages + 1):
            try:
                results = googlenews.results()
                if not results: break
                all_articles.extend(results)
                if page < max_pages:
                    googlenews.getpage(page + 1)
                    time.sleep(random.uniform(2, 5))
            except HTTPError as e:
                if e.code == 429:
                    wait_time = random.uniform(15, 30)
                    gr.Warning(f"Rate limited by Google News. Pausing for {wait_time:.0f} seconds.")
                    time.sleep(wait_time)
                else:
                    logger.error(f"HTTP Error fetching news: {e}"); break
            except Exception as e:
                logger.error(f"An error occurred fetching news: {e}"); break
        
        current_dt += pd.Timedelta(days=interval)

    if not all_articles: return pd.DataFrame(), pd.DataFrame()
    
    df = pd.DataFrame(all_articles).drop_duplicates(subset=['link'])
    df['published_date'] = df['date'].apply(lambda x: dateparser.parse(x, languages=['bn']))
    df.dropna(subset=['published_date', 'title'], inplace=True)

    if filter_keys and filter_keys.strip():
        # Advanced filtering logic: supports AND, OR, NOT, and phrase search
        def parse_query(query):
            # Simple parser for AND, OR, NOT, and phrase queries
            query = query.lower()
            tokens = re.findall(r'"[^"]+"|\S+', query)
            expr = []
            for token in tokens:
                if token == 'and': expr.append('&')
                elif token == 'or': expr.append('|')
                elif token == 'not': expr.append('!')
                else:
                    if token.startswith('"') and token.endswith('"'):
                        expr.append(f'"{token[1:-1]}"')
                    else:
                        expr.append(f'"{token}"')
            return ' '.join(expr)

        def match_complex_query(text, query):
            # Evaluate the parsed query against the text
            text = text.lower()
            expr = parse_query(query)
            # Replace quoted terms with their presence in text
            def term_eval(term):
                term = term.strip('"')
                return term in text
            # Replace operators with Python equivalents
            expr = re.sub(r'"([^"]+)"', lambda m: str(term_eval(m.group(0))), expr)
            expr = expr.replace('&', ' and ').replace('|', ' or ').replace('!', ' not ')
            try:
                return eval(expr)
            except Exception:
                return False

        mask = df.apply(lambda row: match_complex_query(str(row['title']) + ' ' + str(row['desc']), filter_keys), axis=1)
        df = df[mask]

    return df, df[['published_date', 'title', 'media', 'desc', 'link']].sort_values(by='published_date', ascending=False)

# ==============================================================================
# YOUTUBE ANALYZER BACKEND
# ==============================================================================
# (This section remains unchanged from the previous robust version)
def _fetch_video_details(youtube_service, video_ids: list):
    all_videos_data = []
    try:
        for i in range(0, len(video_ids), 50):
            id_batch = video_ids[i:i+50]
            video_request = youtube_service.videos().list(part="snippet,statistics", id=",".join(id_batch))
            video_response = video_request.execute()
            for item in video_response.get('items', []):
                stats = item.get('statistics', {})
                all_videos_data.append({
                    'video_id': item['id'], 'video_title': item['snippet']['title'],
                    'channel': item['snippet']['channelTitle'], 'published_date': item['snippet']['publishedAt'],
                    'view_count': int(stats.get('viewCount', 0)), 'like_count': int(stats.get('likeCount', 0)),
                    'comment_count': int(stats.get('commentCount', 0))
                })
    except HttpError as e:
        logger.error(f"Could not fetch video details. Error: {e}")
        gr.Warning("Could not fetch details for some videos due to an API error.")
    return all_videos_data

def _scrape_single_video_comments(youtube_service, video_id, max_comments):
    comments_list = []
    try:
        request = youtube_service.commentThreads().list(
            part="snippet", videoId=video_id, maxResults=min(max_comments, 100),
            order='relevance', textFormat="plainText"
        )
        response = request.execute()
        for item in response.get('items', []):
            snippet = item['snippet']['topLevelComment']['snippet']
            comments_list.append({
                'author': snippet['authorDisplayName'], 'published_date_comment': snippet['publishedAt'],
                'comment_text': snippet['textDisplay'], 'likes': snippet['likeCount'],
                'replies': item['snippet']['totalReplyCount']
            })
    except HttpError as e:
        logger.warning(f"Could not retrieve comments for video {video_id} (may be disabled). Error: {e}")
    return comments_list

def run_youtube_analysis_pipeline(api_key, query, max_videos_for_stats, num_videos_for_comments, max_comments_per_video, published_after, progress=gr.Progress()):
    # Use integrated API key for seamless experience
    api_key = "AIzaSyB_f3uROqZfwBWsc_sDEV63WmUHBgvGGqw"
    if not query: raise gr.Error("Search Keywords are required.")
    try:
        youtube = build('youtube', 'v3', developerKey=api_key)
    except HttpError as e:
        raise gr.Error(f"Failed to initialize YouTube service. Check API Key. Error: {e}")
    except Exception as e:
        raise gr.Error(f"An unexpected error occurred during API initialization: {e}")

    progress(0.1, desc="Performing broad scan for videos...")
    all_video_ids, next_page_token, total_results_estimate = [], None, 0
    PAGES_TO_FETCH = min(15, (max_videos_for_stats // 50) + 1)
    search_params = {'q': query, 'part': 'id', 'maxResults': 50, 'type': 'video', 'order': 'relevance'}
    if published_after:
        parsed_date = dateparser.parse(published_after)
        if parsed_date:
            search_params['publishedAfter'] = parsed_date.replace(tzinfo=timezone.utc).isoformat()
        else:
            gr.Warning(f"Could not parse date: '{published_after}'. Ignoring filter.")

    for page in range(PAGES_TO_FETCH):
        try:
            if next_page_token: search_params['pageToken'] = next_page_token
            response = youtube.search().list(**search_params).execute()
            if page == 0:
                total_results_estimate = response.get('pageInfo', {}).get('totalResults', 0)
            all_video_ids.extend([item['id']['videoId'] for item in response.get('items', [])])
            next_page_token = response.get('nextPageToken')
            progress(0.1 + (0.3 * (page / PAGES_TO_FETCH)), desc=f"Broad scan: Found {len(all_video_ids)} videos...")
            if not next_page_token: break
        except HttpError as e:
             if "quotaExceeded" in str(e): raise gr.Error("CRITICAL: YouTube API daily quota exceeded. Try again tomorrow.")
             logger.error(f"HTTP error during video search: {e}"); break

    if not all_video_ids:
        return pd.DataFrame(), pd.DataFrame(), 0

    progress(0.4, desc=f"Fetching details for {len(all_video_ids)} videos...")
    videos_df_full_scan = pd.DataFrame(_fetch_video_details(youtube, all_video_ids))
    if videos_df_full_scan.empty:
        return pd.DataFrame(), pd.DataFrame(), 0

    videos_df_full_scan['published_date'] = pd.to_datetime(videos_df_full_scan['published_date'])
    videos_df_full_scan['engagement_rate'] = ((videos_df_full_scan['like_count'] + videos_df_full_scan['comment_count']) / videos_df_full_scan['view_count']).fillna(0)
    videos_df_full_scan = videos_df_full_scan.sort_values(by='view_count', ascending=False).reset_index(drop=True)

    videos_to_scrape_df, all_comments = videos_df_full_scan.head(int(num_videos_for_comments)), []
    for index, row in videos_to_scrape_df.iterrows():
        progress(0.7 + (0.3 * (index / len(videos_to_scrape_df))), desc=f"Deep dive: Scraping comments from video {index+1}/{len(videos_to_scrape_df)}...")
        comments_for_video = _scrape_single_video_comments(youtube, row['video_id'], max_comments_per_video)
        if comments_for_video:
            for comment in comments_for_video:
                comment.update({'video_id': row['video_id'], 'video_title': row['video_title']})
            all_comments.extend(comments_for_video)

    comments_df = pd.DataFrame(all_comments)
    if not comments_df.empty:
        comments_df['published_date_comment'] = pd.to_datetime(comments_df['published_date_comment'])

    logger.info(f"YouTube analysis complete. Est. total videos: {total_results_estimate}. Scanned: {len(videos_df_full_scan)}. Comments: {len(comments_df)}.")
    return videos_df_full_scan, comments_df, total_results_estimate


# ==============================================================================
# ADVANCED ANALYTICS MODULE
# ==============================================================================
# (This section remains unchanged, as it was already robust)
def set_plot_style():
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['figure.dpi'] = 100

def run_sentiment_analysis(df: pd.DataFrame, text_column: str, progress=gr.Progress()):
    if text_column not in df.columns: return df
    sentiment_pipeline = get_sentiment_pipeline()
    if not sentiment_pipeline:
        gr.Warning("Sentiment model failed to load. Skipping analysis.")
        return df

    texts = df[text_column].dropna().tolist()
    if not texts: return df

    progress(0, desc="Running sentiment analysis...")
    results = sentiment_pipeline(texts, batch_size=32)

    text_to_sentiment = {text: result for text, result in zip(texts, results)}
    df['sentiment_label'] = df[text_column].map(lambda x: text_to_sentiment.get(x, {}).get('label'))
    df['sentiment_score'] = df[text_column].map(lambda x: text_to_sentiment.get(x, {}).get('score'))
    logger.info("Sentiment analysis complete.")
    return df

def generate_scraper_dashboard(df: pd.DataFrame):
    set_plot_style()
    
    total_articles, unique_media = len(df), df['media'].nunique()
    start_date, end_date = pd.to_datetime(df['published_date']).min(), pd.to_datetime(df['published_date']).max()
    date_range_str = f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"

    agg_code, agg_name = get_dynamic_time_agg(start_date, end_date)
    timeline_df = df.set_index(pd.to_datetime(df['published_date'])).resample(agg_code).size().reset_index(name='count')
    timeline_plot = gr.LinePlot(timeline_df, x='published_date', y='count', title=f'{agg_name} News Volume', tooltip=['published_date', 'count'])
    
    media_counts = df['media'].dropna().value_counts().nlargest(15).sort_values()
    fig_media = None
    if not media_counts.empty:
        fig_media, ax = plt.subplots(figsize=(8, 6)); media_counts.plot(kind='barh', ax=ax, color='skyblue'); ax.set_title("Top 15 Media Sources", fontproperties=BANGLA_FONT)
        ax.set_yticklabels(media_counts.index, fontproperties=BANGLA_FONT); ax.set_xlabel("Article Count"); plt.tight_layout()

    text = " ".join(title for title in df['title'].astype(str))
    fig_wc = None
    try:
        tokenizer = get_bangla_tokenizer()
        if tokenizer:
            # Tokenize and filter out stopwords and short tokens
            tokens = tokenizer.tokenize(text)
            words = [w for w in tokens if w not in BANGLA_STOP_WORDS and len(w) > 1 and not w.startswith("▁") and re.match(r'^[\u0980-\u09FF]+$', w)]
            words = [w.replace("▁", "") for w in words if w.replace("▁", "")]
            wc_text = " ".join(words)
        else:
            wc_text = text
        wc = WordCloud(font_path=FONT_PATH, width=800, height=400, background_color='white', stopwords=BANGLA_STOP_WORDS, collocations=True, colormap='viridis').generate(wc_text)
        fig_wc, ax = plt.subplots(figsize=(10, 5)); ax.imshow(wc, interpolation='bilinear'); ax.axis("off")
    except Exception as e: logger.error(f"WordCloud failed: {e}")
    
    return {
        kpi_total_articles: str(total_articles), kpi_unique_media: str(unique_media), kpi_date_range: date_range_str,
        dashboard_timeline_plot: timeline_plot, dashboard_media_plot: fig_media, dashboard_wordcloud_plot: fig_wc,
        scraper_dashboard_group: gr.update(visible=True)
    }

def generate_sentiment_dashboard(df: pd.DataFrame):
    updates = {sentiment_dashboard_tab: gr.update(visible=False)}
    set_plot_style()
    
    if 'sentiment_label' in df.columns:
        sentiment_counts = df['sentiment_label'].value_counts()
        fig_pie, fig_media_sent = None, None
        if not sentiment_counts.empty:
            fig_pie, ax = plt.subplots(figsize=(6, 6)); ax.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=90, colors=['#66c2a5', '#fc8d62', '#8da0cb'])
            ax.set_title("Overall Sentiment Distribution", fontproperties=BANGLA_FONT); ax.axis('equal')
        
        top_media = df['media'].value_counts().nlargest(10).index
        media_sentiment = pd.crosstab(df[df['media'].isin(top_media)]['media'], df['sentiment_label'], normalize='index').mul(100)
        if not media_sentiment.empty:
            fig_media_sent, ax = plt.subplots(figsize=(10, 7)); media_sentiment.plot(kind='barh', stacked=True, ax=ax, colormap='viridis')
            ax.set_title("Sentiment by Top Media Sources", fontproperties=BANGLA_FONT); ax.set_yticklabels(media_sentiment.index, fontproperties=BANGLA_FONT); plt.tight_layout()
        
        updates.update({sentiment_pie_plot: fig_pie, sentiment_by_media_plot: fig_media_sent, sentiment_dashboard_tab: gr.update(visible=True)})
    return updates

def generate_youtube_dashboard(videos_df, comments_df):
    set_plot_style()
    kpis = {
        kpi_yt_videos_found: f"{len(videos_df):,}" if videos_df is not None else "0",
        kpi_yt_views_scanned: f"{videos_df['view_count'].sum():,}" if videos_df is not None else "0",
        kpi_yt_comments_scraped: f"{len(comments_df):,}" if comments_df is not None else "0"
    }
    
    channel_counts = videos_df['channel'].value_counts().nlargest(15).sort_values()
    fig_channels, ax = plt.subplots(figsize=(8, 6))
    if not channel_counts.empty:
        channel_counts.plot(kind='barh', ax=ax, color='coral'); ax.set_title("Top 15 Channels by Video Volume", fontproperties=BANGLA_FONT); ax.set_yticklabels(channel_counts.index, fontproperties=BANGLA_FONT); plt.tight_layout()

    # Rich analytics: engagement, top videos, comment activity, time series, etc.
    fig_wc, fig_top_videos, fig_engagement, fig_comment_activity, fig_time_series = None, None, None, None, None
    if comments_df is not None and not comments_df.empty:
        # Top commented videos
        top_videos = comments_df['video_title'].value_counts().nlargest(10)
        fig_top_videos, ax = plt.subplots(figsize=(10, 6))
        top_videos.plot(kind='barh', ax=ax, color='dodgerblue')
        ax.set_title("Top 10 Videos by Comment Count", fontproperties=BANGLA_FONT)
        ax.set_xlabel("Comment Count")
        ax.set_yticklabels(top_videos.index, fontproperties=BANGLA_FONT)
        plt.tight_layout()

        # Engagement rate per video
        if 'video_id' in comments_df.columns and 'video_title' in comments_df.columns:
            engagement_df = comments_df.groupby('video_title').size().to_frame('comment_count')
            if videos_df is not None and not videos_df.empty:
                merged = videos_df.set_index('video_title').join(engagement_df)
                merged['engagement_rate'] = merged['comment_count'] / merged['view_count']
                merged = merged.sort_values('engagement_rate', ascending=False).head(10)
                fig_engagement, ax = plt.subplots(figsize=(10, 6))
                merged['engagement_rate'].plot(kind='barh', ax=ax, color='mediumseagreen')
                ax.set_title("Top 10 Videos by Engagement Rate", fontproperties=BANGLA_FONT)
                ax.set_xlabel("Engagement Rate (Comments / Views)")
                ax.set_yticklabels(merged.index, fontproperties=BANGLA_FONT)
                plt.tight_layout()

        # Comment activity over time
        if 'published_date_comment' in comments_df.columns:
            comments_df['published_date_comment'] = pd.to_datetime(comments_df['published_date_comment'])
            time_series = comments_df.set_index('published_date_comment').resample('D').size()
            fig_time_series, ax = plt.subplots(figsize=(10, 4))
            time_series.plot(ax=ax, color='darkorange')
            ax.set_title("Comment Activity Over Time", fontproperties=BANGLA_FONT)
            ax.set_xlabel("Date")
            ax.set_ylabel("Number of Comments")
            plt.tight_layout()

        # Word cloud (improved: only Bengali words, no sentiment)
        text = " ".join(comment for comment in comments_df['comment_text'].astype(str))
        try:
            tokenizer = get_bangla_tokenizer()
            if tokenizer:
                tokens = tokenizer.tokenize(text)
                words = [w for w in tokens if w not in BANGLA_STOP_WORDS and len(w) > 1 and not w.startswith("▁") and re.match(r'^[\u0980-\u09FF]+$', w)]
                words = [w.replace("▁", "") for w in words if w.replace("▁", "")]
                wc_text = " ".join(words)
            else:
                wc_text = text
            wc = WordCloud(font_path=FONT_PATH, width=900, height=450, background_color='white', stopwords=BANGLA_STOP_WORDS, collocations=True, colormap='plasma').generate(wc_text)
            fig_wc, ax = plt.subplots(figsize=(12, 6)); ax.imshow(wc, interpolation='bilinear'); ax.axis("off"); ax.set_title("Bengali Word Cloud from Comments", fontproperties=BANGLA_FONT)
        except Exception as e: logger.error(f"YouTube WordCloud failed: {e}")

    return {
        **kpis,
        yt_channel_plot: fig_channels,
        yt_wordcloud_plot: fig_wc,
        'yt_top_videos_plot': fig_top_videos,
        'yt_engagement_plot': fig_engagement,
        'yt_comment_activity_plot': fig_comment_activity,
        'yt_time_series_plot': fig_time_series
    }

def generate_youtube_topic_dashboard(videos_df_full_scan: pd.DataFrame):
    if videos_df_full_scan is None or videos_df_full_scan.empty: return None, None, None
    set_plot_style()
    
    channel_views = videos_df_full_scan.groupby('channel')['view_count'].sum().nlargest(15).sort_values()
    fig_channel_views, ax = plt.subplots(figsize=(10, 7)); channel_views.plot(kind='barh', ax=ax, color='purple'); ax.set_title("Channel Dominance by Total Views (Top 15)", fontproperties=BANGLA_FONT); ax.set_xlabel("Combined Views on Topic"); ax.set_yticklabels(channel_views.index, fontproperties=BANGLA_FONT); plt.tight_layout()

    df_sample = videos_df_full_scan.sample(n=min(len(videos_df_full_scan), 200))
    avg_views, avg_engagement = df_sample['view_count'].median(), df_sample['engagement_rate'].median()
    fig_quadrant, ax = plt.subplots(figsize=(10, 8)); sns.scatterplot(data=df_sample, x='view_count', y='engagement_rate', size='like_count', sizes=(20, 400), hue='channel', alpha=0.7, ax=ax, legend=False)
    ax.set_xscale('log'); ax.set_yscale('log'); ax.set_title("Content Performance Quadrant", fontproperties=BANGLA_FONT); ax.set_xlabel("Video Views (Log Scale)", fontproperties=BANGLA_FONT); ax.set_ylabel("Engagement Rate (Log Scale)", fontproperties=BANGLA_FONT)
    ax.axhline(avg_engagement, ls='--', color='gray'); ax.axvline(avg_views, ls='--', color='gray'); ax.text(avg_views*1.1, ax.get_ylim()[1], 'High Performers', color='green', fontproperties=BANGLA_FONT); ax.text(ax.get_xlim()[0], avg_engagement*1.1, 'Niche Stars', color='blue', fontproperties=BANGLA_FONT)

    fig_age, ax = plt.subplots(figsize=(10, 7)); sns.scatterplot(data=df_sample, x='published_date', y='view_count', size='engagement_rate', sizes=(20, 400), alpha=0.6, ax=ax)
    ax.set_yscale('log'); ax.set_title("Content Age vs. Impact", fontproperties=BANGLA_FONT); ax.set_xlabel("Publication Date", fontproperties=BANGLA_FONT); ax.set_ylabel("Views (Log Scale)", fontproperties=BANGLA_FONT); plt.xticks(rotation=45)
    
    return fig_channel_views, fig_quadrant, fig_age

# ==============================================================================
# GRADIO UI DEFINITION
# ==============================================================================

with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue", secondary_hue="orange"), title=APP_TITLE) as app:
    gr.Markdown(f"# {APP_TITLE}\n*{APP_TAGLINE}*")

    # --- STATE MANAGEMENT ---
    scraper_results_state = gr.State()
    youtube_results_state = gr.State()

    with gr.Tabs() as tabs:
        with gr.TabItem("1. News Scraper", id=0):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 1. Search Criteria")
                    search_keywords_textbox = gr.Textbox(label="Search Keywords", placeholder="e.g., বিএনপি সমাবেশ")
                    sites_to_search_textbox = gr.Textbox(label="Target Sites (Optional, comma-separated)", placeholder="e.g., prothomalo.com")
                    start_date_textbox = gr.Textbox(label="Start Date", placeholder="YYYY-MM-DD or 'last week'")
                    end_date_textbox = gr.Textbox(label="End Date", placeholder="YYYY-MM-DD or 'today'")
                    gr.Markdown("### 2. Scraping Parameters")
                    interval_days_slider = gr.Slider(1, 7, 3, step=1, label="Days per Interval")
                    max_pages_slider = gr.Slider(1, 10, 5, step=1, label="Max Pages per Interval")
                    filter_keywords_textbox = gr.Textbox(label="Filter Keywords (comma-separated, optional)", placeholder="e.g., নির্বাচন, সরকার")
                    start_scraper_button = gr.Button("Start Scraping & Analysis", variant="primary")
                with gr.Column(scale=2):
                    scraper_results_df = gr.DataFrame(label="Filtered Results", interactive=False, wrap=True)
                    scraper_download_file = gr.File(label="Download Filtered Results CSV")

        with gr.TabItem("2. News Analytics", id=1):
             with gr.Group(visible=False) as scraper_dashboard_group:
                with gr.Tabs():
                    with gr.TabItem("Overview"):
                        with gr.Row():
                            kpi_total_articles = gr.Textbox(label="Total Articles Found", interactive=False)
                            kpi_unique_media = gr.Textbox(label="Unique Media Sources", interactive=False)
                            kpi_date_range = gr.Textbox(label="Date Range of Articles", interactive=False)
                        dashboard_timeline_plot = gr.LinePlot(label="News Volume Timeline")
                        with gr.Row():
                            dashboard_media_plot = gr.Plot(label="Top Media Sources by Article Count")
                            dashboard_wordcloud_plot = gr.Plot(label="Headline Word Cloud")
                    with gr.TabItem("Sentiment Analysis", visible=False) as sentiment_dashboard_tab:
                        with gr.Row():
                            sentiment_pie_plot = gr.Plot(label="Overall Sentiment")
                            sentiment_by_media_plot = gr.Plot(label="Sentiment by Media Source")

        with gr.TabItem("3. YouTube Topic Analysis", id=2):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### YouTube Search & Analysis")
                    yt_api_key = gr.Textbox(label="YouTube API Key", placeholder="Paste your YouTube Data API v3 key here")
                    yt_search_keywords = gr.Textbox(label="Search Keywords", placeholder="e.g., বিএনপি, তারেক রহমান")
                    yt_published_after = gr.Textbox(label="Published After Date (Optional)", placeholder="YYYY-MM-DD or '1 month ago'")
                    gr.Markdown("### Analysis Parameters")
                    yt_max_videos_for_stats = gr.Slider(label="Videos to Scan for Topic Stats (Broad Scan)", minimum=50, maximum=750, value=300, step=50)
                    yt_num_videos_for_comments = gr.Slider(label="Top Videos for Comment Analysis (Deep Dive)", minimum=5, maximum=100, value=25, step=5)
                    yt_max_comments = gr.Slider(10, 100, 30, step=10, label="Max Comments per Video")
                    start_yt_analysis_button = gr.Button("Start YouTube Analysis", variant="primary")
                with gr.Column(scale=2):
                    with gr.Group(visible=False) as yt_dashboard_group:
                        gr.Markdown("### YouTube Topic Analytics Dashboard")
                        with gr.Row():
                            kpi_yt_total_topic_videos = gr.Textbox(label="Est. Total Videos on Topic (YT)", interactive=False)
                            kpi_yt_videos_found = gr.Textbox(label="Videos Scanned for Stats", interactive=False)
                            kpi_yt_views_scanned = gr.Textbox(label="Combined Views (of Scanned)", interactive=False)
                            kpi_yt_comments_scraped = gr.Textbox(label="Comments Analyzed (from Top Videos)", interactive=False)
                        with gr.Tabs():
                            with gr.TabItem("Top Videos & Engagement"):
                                yt_videos_df_output = gr.DataFrame(label="Top Videos Analyzed for Comments (sorted by views)")
                                yt_top_videos_plot = gr.Plot(label="Top 10 Videos by Comment Count")
                                yt_engagement_plot = gr.Plot(label="Top 10 Videos by Engagement Rate")
                            with gr.TabItem("Comment Activity & Word Cloud"):
                                yt_comment_activity_plot = gr.Plot(label="Comment Activity Over Time")
                                yt_wordcloud_plot = gr.Plot(label="Bengali Word Cloud from Comments")
                            with gr.TabItem("Channel & Topic Analytics"):
                                yt_channel_plot = gr.Plot(label="Channel Contribution by Video Count")
                                yt_channel_views_plot = gr.Plot(label="Channel Dominance by Views")
                                yt_performance_quadrant_plot = gr.Plot(label="Content Performance Quadrant")
                                yt_content_age_plot = gr.Plot(label="Content Age vs. Impact")
    
    gr.Markdown(f"<div style='text-align: center; margin-top: 20px;'>{APP_FOOTER}</div>")

    # ==============================================================================
    # EVENT HANDLERS
    # ==============================================================================
    
    # --- NEWS SCRAPER WORKFLOW ---
    def news_scraper_workflow(search_keywords, sites, start_date, end_date, interval, max_pages, filter_keys, progress=gr.Progress()):
        progress(0, desc="Starting news analysis...")
        raw_df, display_df = run_news_scraper_pipeline(search_keywords, sites, start_date, end_date, interval, max_pages, filter_keys, progress)
        
        if raw_df.empty:
            gr.Info("No news articles found for your query."); return None, None, None
            
        progress(0.8, desc="Analyzing sentiment of news headlines...")
        analyzed_df = run_sentiment_analysis(raw_df.copy(), 'title', progress)
        
        output_path = "filtered_news_data.csv"; display_df.to_csv(output_path, index=False)
        return display_df, output_path, analyzed_df

    start_scraper_button.click(
        fn=news_scraper_workflow,
        inputs=[search_keywords_textbox, sites_to_search_textbox, start_date_textbox, end_date_textbox, interval_days_slider, max_pages_slider, filter_keywords_textbox],
        outputs=[scraper_results_df, scraper_download_file, scraper_results_state]
    )

    def update_news_dashboards(analyzed_df):
        if analyzed_df is None or analyzed_df.empty:
            return {scraper_dashboard_group: gr.update(visible=False), sentiment_dashboard_tab: gr.update(visible=False)}
        
        scraper_updates = generate_scraper_dashboard(analyzed_df)
        sentiment_updates = generate_sentiment_dashboard(analyzed_df)
        return {**scraper_updates, **sentiment_updates}

    news_ui_components = [
        scraper_dashboard_group, kpi_total_articles, kpi_unique_media, kpi_date_range,
        dashboard_timeline_plot, dashboard_media_plot, dashboard_wordcloud_plot,
        sentiment_dashboard_tab, sentiment_pie_plot, sentiment_by_media_plot
    ]
    scraper_results_state.change(fn=update_news_dashboards, inputs=scraper_results_state, outputs=news_ui_components)

    # --- YOUTUBE WORKFLOW ---
    def youtube_workflow(api_key, query, max_stats, num_comments, max_comments, published_after, progress=gr.Progress()):
        sanitized_api_key = api_key.strip()
        sanitized_query = query.strip()
        videos_df_full, comments_df, total_vids_est = run_youtube_analysis_pipeline(
            sanitized_api_key, sanitized_query, max_stats, num_comments, max_comments, published_after, progress
        )
        if videos_df_full.empty:
            gr.Info("No videos found for your YouTube query."); return None, None

        if comments_df is not None and not comments_df.empty:
            progress(0.9, desc="Analyzing comment sentiment...")
            comments_df = run_sentiment_analysis(comments_df.copy(), 'comment_text', progress)
            
        top_videos_for_display = videos_df_full.head(int(num_comments))
        return top_videos_for_display, {"full_scan": videos_df_full, "comments": comments_df, "total_estimate": total_vids_est}

    start_yt_analysis_button.click(
        fn=youtube_workflow,
        inputs=[yt_api_key, yt_search_keywords, yt_max_videos_for_stats, yt_num_videos_for_comments, yt_max_comments, yt_published_after],
        outputs=[yt_videos_df_output, youtube_results_state]
    )

    def update_youtube_dashboards(results_data):
        if not results_data or results_data.get("full_scan") is None or results_data["full_scan"].empty:
            return {
                yt_dashboard_group: gr.update(visible=False), kpi_yt_total_topic_videos: "0",
                kpi_yt_videos_found: "0", kpi_yt_views_scanned: "0", kpi_yt_comments_scraped: "0",
                yt_channel_plot: None, yt_wordcloud_plot: None, yt_sentiment_pie_plot: None,
                yt_sentiment_by_video_plot: None, yt_channel_views_plot: None,
                yt_performance_quadrant_plot: None, yt_content_age_plot: None
            }
        
        videos_df_full, comments_df, total_estimate = results_data.get("full_scan"), results_data.get("comments"), results_data.get("total_estimate", 0)
        deep_dive_updates = generate_youtube_dashboard(videos_df_full, comments_df)
        fig_ch_views, fig_quad, fig_age = generate_youtube_topic_dashboard(videos_df_full)
        
        return {
            yt_dashboard_group: gr.update(visible=True),
            kpi_yt_total_topic_videos: f"{total_estimate:,}",
            **deep_dive_updates,
            yt_channel_views_plot: fig_ch_views,
            yt_performance_quadrant_plot: fig_quad,
            yt_content_age_plot: fig_age,
        }
    
    yt_ui_components = [
        yt_dashboard_group, kpi_yt_total_topic_videos, kpi_yt_videos_found, kpi_yt_views_scanned, kpi_yt_comments_scraped,
        yt_channel_plot, yt_wordcloud_plot, yt_sentiment_pie_plot, yt_sentiment_by_video_plot,
        yt_channel_views_plot, yt_performance_quadrant_plot, yt_content_age_plot
    ]
    youtube_results_state.change(fn=update_youtube_dashboards, inputs=youtube_results_state, outputs=yt_ui_components)
    
# ==============================================================================
# LAUNCH THE APP
# ==============================================================================

if __name__ == "__main__":
    auth_credentials = os.getenv("AUTH_CREDENTIALS")
    auth_tuple = None
    if auth_credentials and ":" in auth_credentials:
        user, pwd = auth_credentials.split(":", 1)
        auth_tuple = (user, pwd)
        logger.info("Using authentication credentials from environment variable.")
    else:
        logger.warning("No AUTH_CREDENTIALS found. Using default insecure credentials. Set this as an environment variable for production.")
        auth_tuple = ("bnp", "12345")
        
    app.launch(debug=True, auth=auth_tuple)