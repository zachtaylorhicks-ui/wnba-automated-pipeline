
# src/data_prep.py
import pandas as pd
import numpy as np
import time
import logging
import os
import requests
from bs4 import BeautifulSoup
import re
import unicodedata
import json
from typing import Optional

# This will be installed from requirements.txt in the real environment
try:
    from curl_cffi import requests as curl_requests
except ImportError:
    # Fallback for environments where it might not be pre-installed
    import requests as curl_requests

# --- Configuration & Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S', force=True)

# --- Scraper Classes & Functions ---
class WNBA_Production_Scraper:
    BASE_URL = "https://stats.wnba.com/stats/playergamelogs"
    HEADERS = {'Accept': 'application/json, text/plain, */*', 'Origin': 'https://www.wnba.com', 'Referer': 'https://www.wnba.com/', 'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'}
    def __init__(self, timeout: int = 45, max_retries: int = 3, backoff_factor: int = 5):
        self.session = curl_requests.Session(impersonate="chrome120", headers=self.HEADERS)
        self.timeout, self.max_retries, self.backoff_factor = timeout, max_retries, backoff_factor
    def _fetch_season_data(self, year: int, season_type: str) -> Optional[pd.DataFrame]:
        params = {'LeagueID': '10', 'Season': str(year), 'SeasonType': season_type, 'MeasureType': 'Base', 'PerMode': 'PerGame'}
        for attempt in range(self.max_retries):
            try:
                response = self.session.get(self.BASE_URL, params=params, timeout=self.timeout)
                response.raise_for_status()
                data = response.json()
                if not data.get('resultSets') or not data['resultSets'][0].get('rowSet'): return None
                return pd.DataFrame(data['resultSets'][0]['rowSet'], columns=data['resultSets'][0]['headers'])
            except Exception as e:
                logging.error(f"Attempt {attempt + 1}/{self.max_retries} for {year} {season_type} failed: {e}")
                if attempt < self.max_retries - 1: time.sleep(self.backoff_factor * (attempt + 1))
        return None

def get_current_season_data(current_year: int) -> Optional[pd.DataFrame]:
    logging.info(f"--- LIVE REFRESH: Scraping data for {current_year}... ---")
    scraper = WNBA_Production_Scraper()
    season_types = ['Regular Season', 'Playoffs']
    refreshed_data = []
    for st in season_types:
        df = scraper._fetch_season_data(year=current_year, season_type=st)
        if df is not None and not df.empty:
            df['YEAR'], df['SEASON_TYPE'] = current_year, st
            refreshed_data.append(df)
            time.sleep(2)
    return pd.concat(refreshed_data, ignore_index=True) if refreshed_data else None

# --- Feature Engineering Functions ---
def calculate_advanced_stats(df):
    logging.info("Calculating advanced team-level stats...")
    team_game_stats = df.groupby(['gameId', 'playerteamName']).agg(
        FGA=('fieldGoalsAttempted', 'sum'), FTA=('freeThrowsAttempted', 'sum'),
        OREB=('reboundsOffensive', 'sum'), TOV=('turnovers', 'sum')
    ).reset_index()
    team_game_stats['Poss'] = 0.96 * (team_game_stats['FGA'] - team_game_stats['OREB'] + team_game_stats['TOV'] + (0.44 * team_game_stats['FTA']))
    team_points = df.groupby(['gameId', 'playerteamName'])['points'].sum().reset_index().rename(columns={'points': 'team_points'})
    team_game_stats = pd.merge(team_game_stats, team_points, on=['gameId', 'playerteamName'])
    team_game_stats['OffRtg'] = (team_game_stats['team_points'] / team_game_stats['Poss']) * 100
    game_dates = df[['gameId', 'gameDate']].drop_duplicates()
    team_game_stats = pd.merge(team_game_stats, game_dates, on='gameId')
    team_game_stats.sort_values(by=['playerteamName', 'gameDate'], inplace=True)
    for stat in ['Poss', 'OffRtg']:
        team_game_stats[f'roll_{stat}'] = team_game_stats.groupby('playerteamName')[stat].transform(lambda x: x.rolling(window=10, min_periods=1).mean().shift(1))
    team_game_stats.fillna(0, inplace=True)
    df = pd.merge(df, team_game_stats[['gameId', 'playerteamName', 'roll_Poss', 'roll_OffRtg']], on=['gameId', 'playerteamName'], how='left')
    opp_stats = df[['gameId', 'roll_Poss', 'roll_OffRtg', 'opponentteamName']].rename(columns={'roll_Poss': 'opp_roll_Poss', 'roll_OffRtg': 'opp_roll_OffRtg', 'opponentteamName': 'playerteamName'}).drop_duplicates()
    df = pd.merge(df, opp_stats, on=['gameId', 'playerteamName'], how='left')
    df.fillna(0, inplace=True)
    logging.info("Advanced stats calculation complete.")
    return df

def engineer_player_features(df):
    logging.info("Engineering player-level features...")
    df.sort_values(by=['personId', 'gameDate'], inplace=True)
    stats_to_average = ['points', 'assists', 'reboundsTotal', 'steals', 'blocks', 'turnovers', 'fieldGoalsAttempted', 'threePointersAttempted']
    windows = [3, 5, 10]
    for col in stats_to_average:
        for w in windows:
            df[f"{col}_roll_{w}"] = df.groupby('personId')[col].transform(lambda x: x.rolling(window=w, min_periods=1).mean().shift(1))
    df['days_rest'] = df.groupby('personId')['gameDate'].diff().dt.days
    df[['days_rest'] + [f"{s}_roll_{w}" for s in stats_to_average for w in windows]] = df[['days_rest'] + [f"{s}_roll_{w}" for s in stats_to_average for w in windows]].fillna(0)
    df.loc[df['days_rest'] > 14, 'days_rest'] = 14
    return df

# --- Wiki Bio Scraping & Enrichment ---
def run_wiki_enrichment(base_data_path, cache_path):
    # This function is adapted from your Jostler script
    WNBA_ALL_PLAYERS_URL = "https://en.wikipedia.org/wiki/List_of_Women%27s_National_Basketball_Association_players"
    PLAYER_ID_NAME_MAP, NORMALIZED_NAME_TO_ID_MAP, NORMALIZED_CANONICAL_NAMES = {}, {}, []

    def normalize_name(name):
        if not isinstance(name, str): return ""
        name = "".join(c for c in unicodedata.normalize('NFKD', name.lower()) if not unicodedata.combining(c))
        return name if name.strip() else ""

    def build_player_universe(hist_df):
        nonlocal PLAYER_ID_NAME_MAP, NORMALIZED_NAME_TO_ID_MAP, NORMALIZED_CANONICAL_NAMES
        hist_players = hist_df[['personId', 'fullName']].dropna().drop_duplicates(subset='personId')
        hist_players['personId'] = pd.to_numeric(hist_players['personId'], errors='coerce').dropna().astype(int)
        for _, row in hist_players.iterrows():
            pid, name = int(row['personId']), row['fullName']
            PLAYER_ID_NAME_MAP[pid] = name
            norm_name = normalize_name(str(name))
            if norm_name: NORMALIZED_NAME_TO_ID_MAP[norm_name] = pid
        NORMALIZED_CANONICAL_NAMES = list(NORMALIZED_NAME_TO_ID_MAP.keys())

    if os.path.exists(cache_path):
        logging.info(f"Loading player profiles from cache: {cache_path}")
        with open(cache_path, 'r') as f:
            return json.load(f)

    # If cache doesn't exist, build it
    logging.info("Wiki cache not found. Scraping Wikipedia for Player Info...")
    hist_df = pd.read_csv(base_data_path, usecols=['personId', 'fullName'], low_memory=False)
    build_player_universe(hist_df)
    
    # FuzzyWuzzy needs to be installed for this part
    try:
        from fuzzywuzzy import process, fuzz
    except ImportError:
        logging.error("FuzzyWuzzy not found. Cannot perform wiki scraping. Please install it.")
        return {}

    all_player_profiles = {}
    response = requests.get(WNBA_ALL_PLAYERS_URL, headers={'User-Agent': 'Mozilla/5.0'})
    soup = BeautifulSoup(response.content, 'html.parser'); content_div = soup.find('div', class_='mw-parser-output')
    for link_tag in content_div.find_all('a', href=re.compile(r'^/wiki/')):
        player_name = link_tag.get_text(strip=True)
        # We need a way to get player ID from name. We will build a temporary map.
        match, score = process.extractOne(normalize_name(player_name), NORMALIZED_CANONICAL_NAMES, scorer=fuzz.token_set_ratio)
        pid = NORMALIZED_NAME_TO_ID_MAP.get(match) if match and score >= 95 else None

        if pid and pid not in all_player_profiles:
             all_player_profiles[pid] = {'personId': pid, 'playerName': PLAYER_ID_NAME_MAP.get(pid, player_name), 'wikiUrl': f"https://en.wikipedia.org{link_tag['href']}"}
    
    logging.info(f"Found {len(all_player_profiles)} potential player profiles. Now enriching with details...")
    for pid, profile in all_player_profiles.items():
        if 'wikiUrl' not in profile or not profile['wikiUrl']: continue
        try:
            time.sleep(0.01) # Be respectful
            response = requests.get(profile['wikiUrl'], headers={'User-Agent': 'Mozilla/5.0'}); response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser'); infobox = soup.find('table', class_='infobox')
            if not infobox: continue
            def get_info(label_regex):
                header = infobox.find('th', string=re.compile(label_regex, re.I))
                return header.find_next_sibling('td').get_text(strip=True, separator=' ') if header and header.find_next_sibling('td') else 'N/A'
            profile.update({'height': get_info(r'Listed height'), 'born': get_info(r'Born'), 'draftInfo': get_info(r'WNBA draft')})
        except Exception: pass
    
    with open(cache_path, 'w') as f:
        json.dump(all_player_profiles, f, indent=2)
    logging.info(f"Wiki cache created and saved to {cache_path}")
    return all_player_profiles

def main():
    base_file = 'data/wnba_all_player_boxscores_1997-2025.csv'
    wiki_cache_file = 'data/wnba_wiki_cache.json'
    output_path = 'data/wnba_model_ready_data_with_features.csv'

    # Run the wiki scraper first to create the cache if it doesn't exist
    run_wiki_enrichment(base_file, wiki_cache_file)

    df_raw = pd.read_csv(base_file, low_memory=False)
    rename_map = {'PLAYER_ID':'personId','GAME_ID':'gameId','GAME_DATE':'gameDate','PLAYER_NAME':'fullName','TEAM_NAME':'playerteamName','WL':'win','MIN':'numMinutes','PTS':'points','AST':'assists','BLK':'blocks','STL':'steals','FGA':'fieldGoalsAttempted','FGM':'fieldGoalsMade','FG_PCT':'fieldGoalsPercentage','FG3A':'threePointersAttempted','FG3M':'threePointersMade','FG3_PCT':'threePointersPercentage','FTA':'freeThrowsAttempted','FTM':'freeThrowsMade','FT_PCT':'freeThrowsPercentage','DREB':'reboundsDefensive','OREB':'reboundsOffensive','REB':'reboundsTotal','PF':'foulsPersonal','TOV':'turnovers','PLUS_MINUS':'plusMinusPoints','SEASON_TYPE':'gameType', 'TEAM_ABBREVIATION': 'teamAbbreviation'}
    df = df_raw.rename(columns=rename_map)
    df['gameDate'] = pd.to_datetime(df['gameDate'])
    df['MATCHUP'] = df['MATCHUP'].astype(str)
    df['home'] = df['MATCHUP'].apply(lambda x: 1 if 'vs.' in x else 0)
    temp_teams = df['MATCHUP'].str.replace('vs. ', '@').str.split(' @ ', expand=True)
    df['opponentteamName'] = np.where(df['teamAbbreviation'] == temp_teams[0], temp_teams[1], temp_teams[0])
    df['personId'] = pd.to_numeric(df['personId'], errors='coerce')
    df.dropna(subset=['personId'], inplace=True)

    df = engineer_player_features(df)
    df = calculate_advanced_stats(df)

    percentage_cols = ['fieldGoalsPercentage', 'threePointersPercentage', 'freeThrowsPercentage']
    df[percentage_cols] = df[percentage_cols].fillna(0.0)
    df.dropna(subset=['numMinutes', 'points'], inplace=True)
    df.sort_values(by='gameDate', ascending=False, inplace=True)
    
    df.to_csv(output_path, index=False)
    logging.info(f"Success! Feature-engineered data saved to '{output_path}'")
    
if __name__ == '__main__':
    main()

