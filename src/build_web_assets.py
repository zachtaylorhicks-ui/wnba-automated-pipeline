
# src/build_web_assets.py

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
import logging
import warnings
import shutil
import gzip

try:
    from fuzzywuzzy import process, fuzz
except ImportError:
    logging.error("FuzzyWuzzy not installed. Some name matching may fail.")

# --- Configuration ---
DATA_DIR = 'data'
DIST_DIR = 'dist'
HISTORICAL_DATA_FILE = f'{DATA_DIR}/wnba_all_player_boxscores_1997-2025.csv'
RAW_MODEL_DATA_FILE = f'{DATA_DIR}/wnba_model_ready_data.csv'
WIKI_CACHE_FILE = f'{DATA_DIR}/wnba_wiki_cache.json'
PREDICTIONS_INPUT_FILE = f'{DATA_DIR}/final_blended_predictions.pkl'
PROJECTION_YEAR = 2025
ROOKIE_MINUTES_PLACEHOLDER = 18.0

# --- Fantasy Stat Configuration ---
FANTASY_STATS_COUNTING = ['PTS', 'REB', 'AST', 'STL', 'BLK', '3PM', 'TOV']
FANTASY_STATS_PERCENTAGE = ['FG_impact', 'FT_impact']
ALL_FANTASY_STATS = FANTASY_STATS_COUNTING + FANTASY_STATS_PERCENTAGE
HISTORICAL_STAT_MAP = {'points':'PTS','reboundsTotal':'REB','assists':'AST','steals':'STL','blocks':'BLK','threePointersMade':'3PM','turnovers':'TOV','fieldGoalsMade':'FGM','fieldGoalsAttempted':'FGA','freeThrowsMade':'FTM','freeThrowsAttempted':'FTA','personId':'personId','fullName':'playerName','playerteamName':'team','numMinutes':'MIN'}

# --- Team Name Mapping ---
WNBA_TEAM_NAME_MAP = {"Atlanta Dream":"ATL","Chicago Sky":"CHI","Connecticut Sun":"CON","Dallas Wings":"DAL","Indiana Fever":"IND","Las Vegas Aces":"LVA","Los Angeles Sparks":"LAS","Minnesota Lynx":"MIN","New York Liberty":"NYL","Phoenix Mercury":"PHO","Seattle Storm":"SEA","Washington Mystics":"WAS","Golden State Valkyries":"GSV"}
REVERSE_TEAM_MAP = {v:k for k,v in WNBA_TEAM_NAME_MAP.items()}

# --- Setup & Utility Functions ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)-8s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S', force=True)
warnings.simplefilter(action='ignore', category=FutureWarning)

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj) if np.isfinite(obj) else None
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, np.bool_): return bool(obj)
        if pd.isna(obj): return None
        return super(NumpyEncoder, self).default(obj)

def setup_environment():
    if os.path.exists(DIST_DIR): shutil.rmtree(DIST_DIR)
    os.makedirs(os.path.join(DIST_DIR, DATA_DIR, 'player_history'), exist_ok=True)
    logging.info(f"Output directory '{DIST_DIR}' cleaned and ready.")

def get_team_abbr(team_name):
    if not isinstance(team_name, str): return 'FA'
    return WNBA_TEAM_NAME_MAP.get(team_name, 'FA')

def calculate_z_scores(df_per_game):
    if df_per_game.empty: return df_per_game
    df = df_per_game.copy(); qualified_players = df[df['MIN'] > 5];
    if qualified_players.empty: qualified_players = df
    league_avg_fga=qualified_players['FGA'].mean(); league_avg_fta=qualified_players['FTA'].mean()
    league_avg_fg_pct=qualified_players['FGM'].sum()/qualified_players['FGA'].sum() if qualified_players['FGA'].sum()>0 else 0
    league_avg_ft_pct=qualified_players['FTM'].sum()/qualified_players['FTA'].sum() if qualified_players['FTA'].sum()>0 else 0
    for stat in FANTASY_STATS_COUNTING:
        mean, std = qualified_players[stat].mean(), qualified_players[stat].std(); multiplier = -1 if stat == 'TOV' else 1
        df[f'z_{stat}'] = ((df[stat] - mean) / (std if std > 1e-6 else 1)) * multiplier
    df['FG_pct']=df['FGM']/df['FGA'].replace(0,np.nan); df['FT_pct']=df['FTM']/df['FTA'].replace(0,np.nan)
    fg_impact=(df['FG_pct']-league_avg_fg_pct)*(df['FGA']/(league_avg_fga if league_avg_fga>0 else 1))
    ft_impact=(df['FT_pct']-league_avg_ft_pct)*(df['FTA']/(league_avg_fta if league_avg_fta>0 else 1))
    df['FG_impact']=fg_impact; df['FT_impact']=ft_impact
    df['z_FG_impact']=(fg_impact-fg_impact.mean())/(fg_impact.std() if fg_impact.std()>1e-6 else 1)
    df['z_FT_impact']=(ft_impact-ft_impact.mean())/(ft_impact.std() if ft_impact.std()>1e-6 else 1)
    df['custom_z_score']=df[[f'z_{stat}' for stat in ALL_FANTASY_STATS]].sum(axis=1)
    return df

def process_historical_data(df_hist_raw, master_player_df):
    logging.info("Processing historical data for season-long rankings...")
    df = df_hist_raw.rename(columns=HISTORICAL_STAT_MAP); df['gameDate']=pd.to_datetime(df['gameDate']); df['YEAR']=df['gameDate'].dt.year; df['personId']=pd.to_numeric(df['personId'],errors='coerce').dropna().astype(int)
    stat_cols = ['PTS','REB','AST','STL','BLK','3PM','TOV','FGM','FGA','FTM','FTA','MIN']
    all_season_data = {}
    for year in sorted(df['YEAR'].unique(), reverse=True):
        season_df = df[df['YEAR']==year]; player_gp = season_df.groupby('personId')['gameId'].nunique().reset_index(name='GP')
        if player_gp.empty: continue
        df_total_stats = season_df.groupby('personId')[stat_cols].sum().reset_index(); df_total = pd.merge(player_gp, df_total_stats, on='personId')
        df_total = pd.merge(master_player_df[['personId','playerName','position']], df_total, on='personId', how='right')
        last_team = season_df.sort_values('gameDate').groupby('personId')['team'].last().reset_index(); df_total = pd.merge(df_total, last_team, on='personId', how='left'); df_total['team'] = df_total['team'].apply(get_team_abbr)
        df_per_game = df_total.copy()
        for col in stat_cols: df_per_game[col] = df_per_game[col] / df_per_game['GP'].replace(0,1)
        all_season_data[year] = {'per_game': df_per_game, 'total': df_total}
    return all_season_data

def generate_daily_games_and_grades(all_projections_df, df_hist_raw, historical_minutes_map, master_player_df):
    logging.info("Generating multi-model daily games and grades...")
    df_pred = all_projections_df.copy(); df_pred['game_date'] = pd.to_datetime(df_pred['game_date']).dt.strftime('%Y-%m-%d')
    df_actual = df_hist_raw.copy().rename(columns={'points':'PTS','reboundsTotal':'REB','assists':'AST'}); df_actual['gameDate'] = pd.to_datetime(df_actual['gameDate']).dt.strftime('%Y-%m-%d'); df_actual['personId'] = pd.to_numeric(df_actual['personId'], errors='coerce').dropna().astype(int)
    df_pred['home_team_abbr'] = df_pred['home_team'].apply(get_team_abbr); df_pred['away_team_abbr'] = df_pred['away_team'].apply(get_team_abbr)
    df_actual['team_abbr'] = df_actual['playerteamName'].apply(get_team_abbr); df_actual['opp_abbr'] = df_actual['opponentteamName'].apply(get_team_abbr)
    df_actual['home_team_abbr'] = np.where(df_actual['home'] == 1, df_actual['team_abbr'], df_actual['opp_abbr'])
    df_actual['away_team_abbr'] = np.where(df_actual['home'] == 0, df_actual['team_abbr'], df_actual['opp_abbr'])
    df_pred.dropna(subset=['home_team_abbr', 'away_team_abbr'], inplace=True)
    df_pred['game_key'] = df_pred.apply(lambda r: f"{r['game_date']}_{'_'.join(sorted([r.home_team_abbr, r.away_team_abbr]))}", axis=1)
    df_actual.dropna(subset=['home_team_abbr', 'away_team_abbr'], inplace=True)
    df_actual['game_key'] = df_actual.apply(lambda r: f"{r['gameDate']}_{'_'.join(sorted([r.home_team_abbr, r.away_team_abbr]))}", axis=1)
    daily_games, historical_grades = {}, []
    all_game_keys = sorted(pd.concat([df_pred['game_key'], df_actual['game_key']]).dropna().unique())
    master_player_df_indexed = master_player_df.set_index('personId')
    for game_key in tqdm(all_game_keys, desc="Processing Daily Games"):
        game_all_models_df = df_pred[df_pred['game_key'] == game_key]; game_actual_df = df_actual[df_actual['game_key'] == game_key]
        if game_all_models_df.empty and game_actual_df.empty: continue
        date = game_key.split('_')[0]
        home_abbr, away_abbr = (game_all_models_df.iloc[0]['home_team_abbr'], game_all_models_df.iloc[0]['away_team_abbr']) if not game_all_models_df.empty else (game_actual_df.iloc[0]['home_team_abbr'], game_actual_df.iloc[0]['away_team_abbr'])
        all_model_projections = {}
        if not game_all_models_df.empty:
            for model_name, game_pred_df in game_all_models_df.groupby('model_source'):
                def format_team_projs(team_abbr_in):
                    team_df = game_pred_df[game_pred_df.teamName.apply(get_team_abbr) == team_abbr_in]
                    players = []
                    for _, p in team_df.iterrows():
                        try: player_name = master_player_df_indexed.loc[p['personId']]['playerName']
                        except (KeyError, AttributeError): player_name = p['playerName']
                        players.append({'personId':p['personId'],'Player_Name':player_name,'Predicted_Minutes':historical_minutes_map.get(p['personId'],ROOKIE_MINUTES_PLACEHOLDER),'points':p.get('PTS',0),'reb':p.get('REB',0),'ast':p.get('AST',0)})
                    return {'teamName':REVERSE_TEAM_MAP.get(team_abbr_in,team_abbr_in),'totalPoints':team_df.get('PTS', 0).sum(),'players':sorted(players,key=lambda x:x['Predicted_Minutes'],reverse=True)}
                all_model_projections[model_name] = [format_team_projs(home_abbr), format_team_projs(away_abbr)]
        grade_obj = {"isGraded": False}
        if not game_actual_df.empty:
            actual_scores = game_actual_df.groupby('team_abbr')['PTS'].sum().to_dict(); player_actuals = {int(p['personId']):{s:p[s] for s in ['PTS','REB','AST'] if s in p and pd.notna(p[s])} for _,p in game_actual_df.iterrows()}
            model_grades = {}
            for model_name, proj in all_model_projections.items():
                if not proj: continue
                pred_home, pred_away = proj[0]['totalPoints'], proj[1]['totalPoints']
                proj_players_df = pd.DataFrame([p for t in proj for p in t['players']]); actual_players_df = game_actual_df[['personId', 'PTS', 'REB', 'AST']]; merged_mae = pd.merge(proj_players_df, actual_players_df, on='personId')
                model_grades[model_name] = {"correctWinner":bool((pred_home > pred_away) == (actual_scores.get(home_abbr,0) > actual_scores.get(away_abbr,0))),"scoreCloseness":abs((pred_home + pred_away) - sum(actual_scores.values())),"PTS":(merged_mae['points'] - merged_mae['PTS']).abs().mean() if not merged_mae.empty else None,"REB":(merged_mae['reb'] - merged_mae['REB']).abs().mean() if not merged_mae.empty else None,"AST":(merged_mae['ast'] - merged_mae['AST']).abs().mean() if not merged_mae.empty else None}
            grade_obj = {"isGraded":True,"playerActuals":player_actuals,"model_grades":model_grades,"gameSummary":{"actual":{k:v for k,v in actual_scores.items() if k in [home_abbr, away_abbr]}}}
            historical_grades.append({"date":date,"model_grades":model_grades})
        if all_model_projections: daily_games.setdefault(date, []).append({'projections': all_model_projections, 'grade': grade_obj})
    return daily_games, historical_grades

def create_player_performance_dataframe(hist_df_raw, master_player_df):
    df = hist_df_raw.rename(columns={'points':'PTS', 'reboundsTotal':'REB', 'assists':'AST', 'steals':'STL', 'blocks':'BLK', 'threePointersMade':'3PM', 'turnovers':'TOV', 'numMinutes':'MIN'}); df['personId'] = pd.to_numeric(df['personId'], errors='coerce').dropna().astype(int)
    df = df.merge(master_player_df[['personId', 'birthDate_iso', 'draftCategory', 'team']], on='personId', how='left'); df['date_dt'] = pd.to_datetime(df['gameDate']); df['birthDate_iso_dt'] = pd.to_datetime(df['birthDate_iso'], errors='coerce')
    df['age'] = (df['date_dt'] - df['birthDate_iso_dt']).dt.days / 365.25; df.sort_values(by=['personId', 'date_dt'], inplace=True); df['game_number'] = df.groupby('personId').cumcount() + 1; df['date'] = df['date_dt'].dt.strftime('%Y-%m-%d'); return df.drop(columns=['date_dt', 'birthDate_iso_dt'])

def main():
    setup_environment()
    try:
        hist_df_raw = pd.read_csv(HISTORICAL_DATA_FILE, low_memory=False); raw_model_df = pd.read_csv(RAW_MODEL_DATA_FILE, low_memory=False)
        with open(WIKI_CACHE_FILE, 'r') as f: enriched_profiles = {int(k): v for k, v in json.load(f).items()}
        all_projections_df = pd.read_pickle(PREDICTIONS_INPUT_FILE)
    except FileNotFoundError as e: logging.error(f"FATAL: Missing input file: {e}. Run prior scripts."); return

    master_player_df = pd.DataFrame.from_dict(enriched_profiles, orient='index').reset_index(drop=True); master_player_df['personId'] = master_player_df['personId'].astype(int); master_player_df['birthDate_iso'] = pd.to_datetime(master_player_df.get('born', pd.Series()).str.extract(r'\((\d{4}-\d{2}-\d{2})\)')[0], errors='coerce')

    manifest = {}; historical_seasons = process_historical_data(raw_model_df, master_player_df)
    for year, data in historical_seasons.items():
        per_game_z = calculate_z_scores(data['per_game']); total_df_with_z = pd.merge(data['total'], per_game_z[['personId'] + [f'z_{stat}' for stat in ALL_FANTASY_STATS] + ['custom_z_score']], on='personId', how='left')
        per_game_key=f'actuals_{year}_full_per_game'; total_key=f'actuals_{year}_full_total'
        with open(os.path.join(DIST_DIR,DATA_DIR,f'{per_game_key}.json'),'w') as f: json.dump(per_game_z.to_dict('records'),f,cls=NumpyEncoder)
        with open(os.path.join(DIST_DIR,DATA_DIR,f'{total_key}.json'),'w') as f: json.dump(total_df_with_z.to_dict('records'),f,cls=NumpyEncoder)
        manifest[per_game_key]={'label':f'{year} Actuals (Per Game)','split':'actuals'}; manifest[total_key]={'label':f'{year} Actuals (Total)','split':'actuals'}

    model_blend_map = {"Ensemble": "amalgamated_pred", "Smart Blend": "smart_weighted_pred", "Lowest MAE": "lowest_mae_pred", "Base Transformer": "base_preds:best_transformer", "Bestest Transformer": "base_preds:bestest_transformer"}
    stat_cols_agg = ['PTS', 'REB', 'AST', 'STL', 'BLK', '3PM', 'TOV', 'FGM', 'FGA', 'FTM', 'FTA']
    rolling_mins = raw_model_df.sort_values('gameDate').groupby('personId')['numMinutes'].rolling(5, 1).mean().reset_index(); historical_minutes_map = rolling_mins.loc[rolling_mins.groupby('personId')['level_1'].idxmax()].set_index('personId')['numMinutes'].to_dict()
    player_meta_cols = ['personId', 'playerName', 'position', 'team']
    
    for model_name, col_info in model_blend_map.items():
        proj_records = []
        for _, row in all_projections_df.iterrows():
            if ':' in col_info: base, specific = col_info.split(':'); pred_dict = row.get(base, {}).get(specific, {})
            else: pred_dict = row.get(col_info, {})
            if pred_dict and isinstance(pred_dict, dict): record = {'personId': row['player_id'], 'game_date': row['game_date']}; record.update({stat.upper():val for stat,val in pred_dict.items() if stat.upper() in stat_cols_agg}); proj_records.append(record)
        if not proj_records: continue
        proj_df = pd.DataFrame(proj_records); proj_df = proj_df[proj_df.game_date.dt.year == PROJECTION_YEAR]
        future_totals = proj_df.groupby('personId')[stat_cols_agg].sum().reset_index(); gp_proj = proj_df.groupby('personId')['game_date'].nunique().reset_index(name='GP_proj'); future_totals=pd.merge(future_totals,gp_proj,on='personId',how='left'); future_totals=pd.merge(future_totals,master_player_df[player_meta_cols],on='personId',how='left')
        future_totals['team'] = future_totals['personId'].map(master_player_df.set_index('personId')['team']); future_totals['team'].fillna('FA', inplace=True)
        hybrid_total = future_totals.copy(); hybrid_total['GP']=hybrid_total.get('GP_proj',0); hybrid_total['MIN']=hybrid_total['GP']*hybrid_total['personId'].map(historical_minutes_map).fillna(ROOKIE_MINUTES_PLACEHOLDER); hybrid_per_game=hybrid_total.copy()
        for col in stat_cols_agg+['MIN']: hybrid_per_game[col] = hybrid_per_game[col] / hybrid_per_game['GP'].replace(0,1)
        per_game_z = calculate_z_scores(hybrid_per_game); total_df_with_z = pd.merge(hybrid_total, per_game_z[['personId'] + [f'z_{stat}' for stat in ALL_FANTASY_STATS] + ['custom_z_score']], on='personId', how='left')
        model_id = model_name.replace(" ","_").replace(".",""); per_game_key=f'projections_{PROJECTION_YEAR}_{model_id}_hybrid_per_game'; total_key=f'projections_{PROJECTION_YEAR}_{model_id}_hybrid_total'
        manifest[per_game_key]={'label':f'{PROJECTION_YEAR} Projections ({model_name}) (Per Game)','split':'projections'}; manifest[total_key]={'label':f'{PROJECTION_YEAR} Projections ({model_name}) (Total)','split':'projections'}
        with open(os.path.join(DIST_DIR,DATA_DIR,f'{per_game_key}.json'),'w') as f: json.dump(per_game_z.to_dict('records'),f,cls=NumpyEncoder)
        with open(os.path.join(DIST_DIR,DATA_DIR,f'{total_key}.json'),'w') as f: json.dump(total_df_with_z.to_dict('records'),f,cls=NumpyEncoder)
    
    # Restructure all_projections_df for daily games
    daily_proj_records = []
    for model_name, col_info in model_blend_map.items():
        for _, row in all_projections_df.iterrows():
            if ':' in col_info: base, specific = col_info.split(':'); pred_dict = row.get(base, {}).get(specific, {})
            else: pred_dict = row.get(col_info, {})
            if pred_dict and isinstance(pred_dict, dict):
                record = {'personId':row['player_id'], 'playerName':row['player_name'], 'teamName':row['teamName'], 'game_date':row['game_date'], 'home_team':row['home_team'], 'away_team':row['away_team'], 'model_source': model_name}
                record.update({stat.upper():val for stat,val in pred_dict.items() if stat.upper() in stat_cols_agg}); daily_proj_records.append(record)
    daily_projs_df = pd.DataFrame(daily_proj_records)
    daily_games_data, historical_grades_data = generate_daily_games_and_grades(daily_projs_df, hist_df_raw, historical_minutes_map, master_player_df)

    player_perf_hist_df = create_player_performance_dataframe(raw_model_df, master_player_df)
    last_game_numbers = player_perf_hist_df.groupby('personId')['game_number'].max().to_dict()
    all_projections_df.sort_values(by=['player_id','game_date'],inplace=True); all_projections_df['proj_game_order']=all_projections_df.groupby('player_id').cumcount(); all_projections_df['game_number']=all_projections_df.apply(lambda r: last_game_numbers.get(r['player_id'],0)+1+r['proj_game_order'],axis=1)
    
    player_history_dir = os.path.join(DIST_DIR, DATA_DIR, 'player_history')
    for pid, profile in enriched_profiles.items():
        player_data_for_file = {'performanceHistory': sorted(player_perf_hist_df[player_perf_hist_df['personId']==pid].to_dict('records'), key=lambda x:x['date'])} if pid in player_perf_hist_df['personId'].values else {}
        # Add future projections for ALL models
        future_projs = []
        for model_name, col_info in model_blend_map.items():
            player_model_projs = all_projections_df[all_projections_df['player_id']==pid]
            for _, row in player_model_projs.iterrows():
                if ':' in col_info: base, specific = col_info.split(':'); pred_dict = row.get(base, {}).get(specific, {})
                else: pred_dict = row.get(col_info, {})
                if pred_dict and isinstance(pred_dict, dict):
                    proj = {'game_date':row['game_date'].strftime('%Y-%m-%d'), 'game_number':row['game_number'], 'model_source':model_name}
                    proj.update({stat.upper():val for stat,val in pred_dict.items() if stat.upper() in ['PTS','REB','AST','STL','BLK','3PM']}); future_projs.append(proj)
        if future_projs: player_data_for_file['futureProjections'] = sorted(future_projs, key=lambda x:x['game_date'])
        if player_data_for_file:
            with open(os.path.join(player_history_dir, f"{pid}.json"), 'w') as f: json.dump(player_data_for_file, f, cls=NumpyEncoder)

    logging.info("Pre-aggregating and compressing career data...")
    career_data_payload={}; draft_categories=['All']+[cat for cat in player_perf_hist_df['draftCategory'].unique() if pd.notna(cat)]; minutes_filters=['0','15_career','15_game']
    for draft_cat in draft_categories:
        for min_filter in minutes_filters:
            filtered_df = player_perf_hist_df.copy();
            if draft_cat!='All': filtered_df=filtered_df[filtered_df['draftCategory']==draft_cat]
            if min_filter=='15_career': career_mpg=filtered_df.groupby('personId')['MIN'].mean(); qualified_players=career_mpg[career_mpg>15].index; filtered_df=filtered_df[filtered_df['personId'].isin(qualified_players)]
            elif min_filter=='15_game': filtered_df=filtered_df[filtered_df['MIN']>=15]
            stats_payload={}; stats_to_process=['PTS','REB','AST','STL','BLK','3PM']
            for stat in stats_to_process:
                all_player_lines_age, all_player_lines_games = [], []; stat_filtered_df=filtered_df.dropna(subset=[stat,'age','game_number','team'])
                for _, player_df in stat_filtered_df.groupby('personId'):
                    player_df_sorted_age=player_df.sort_values('age'); player_df_sorted_games=player_df.sort_values('game_number'); player_team=player_df_sorted_age['team'].iloc[0] if not player_df_sorted_age.empty else None
                    all_player_lines_age.extend(player_df_sorted_age[['age',stat]].rename(columns={'age':'x',stat:'y'}).assign(team=player_team).to_dict('records')); all_player_lines_games.extend(player_df_sorted_games[['game_number',stat]].rename(columns={'game_number':'x',stat:'y'}).assign(team=player_team).to_dict('records'))
                    all_player_lines_age.append(None); all_player_lines_games.append(None)
                stats_payload[stat]={'age':all_player_lines_age,'game_number':all_player_lines_games}
            career_data_payload[f"{draft_cat}_{min_filter}"]=stats_payload
    career_data_path=os.path.join(DIST_DIR,DATA_DIR,'career_data.json.gz')
    with gzip.GzipFile(career_data_path,'wb') as fout: fout.write(json.dumps(career_data_payload,cls=NumpyEncoder).encode('utf-8'))

    master_json = {"lastUpdated": datetime.now().isoformat(), "seasonLongDataManifest": manifest, "playerProfiles": enriched_profiles, "dailyGamesByDate": daily_games_data, "historicalGrades": historical_grades_data}
    with open(os.path.join(DIST_DIR, 'predictions.json'), 'w') as f: json.dump(master_json, f, cls=NumpyEncoder)
    logging.info("--- Website Asset Generation Complete ---")

if __name__ == '__main__':
    main()
