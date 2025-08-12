
# src/predict.py

# --- Imports & Setup ---
import pandas as pd
import numpy as np
import os
import logging
import gc
import sys
import re
import time
import json
import warnings
import joblib
from datetime import datetime, date
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

# ML Imports
import torch
import torch.nn as nn
import lightgbm as lgb
from fuzzywuzzy import process
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Local Module Imports
from train import (
    WNBATransformer, WNBAGNN, BlueprintWNBAModel,
    STATS_COLS_15, STATS_COLS_19, BASE_FEATURE_COLS, ADVANCED_FEATURE_COLS,
    WNBAGraphDataset, WNBADataset, prepare_transformer_tensors, custom_collate_transformer
)

try:
    from torch_geometric.loader import DataLoader as GraphDataLoader
    from torch_geometric.nn import to_hetero
    from torch.utils.data import DataLoader as BasicDataLoader
except ImportError:
    logging.error("PyTorch Geometric not found. GNN predictions will fail.")

# --- Global Configuration & Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)-8s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S', force=True)
warnings.simplefilter(action='ignore', category=FutureWarning)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RANDOM_SEED = 42

# --- Constants & Paths ---
BASE_PATH = "./"
DATA_PATH = os.path.join(BASE_PATH, 'data')
MODEL_DATA_FEATURES_PATH = os.path.join(DATA_PATH, 'wnba_model_ready_data_with_features.csv')
WIKI_CACHE_PATH = os.path.join(DATA_PATH, 'wnba_wiki_cache.json')
BLUEPRINT_MODEL_PATH = os.path.join(BASE_PATH, 'best_pro_plus_blueprint_model_v4.9.pth')
BLUEPRINT_PLAYER_MAP_PATH = os.path.join(BASE_PATH, 'pro_plus_blueprint_player_map.pkl')
BLUEPRINT_SCALERS_PATH = os.path.join(BASE_PATH, 'pro_plus_blueprint_scalers.pkl')

WNBA_SCHEDULE_URL = "https://en.wikipedia.org/wiki/2025_WNBA_season"
WNBA_ROSTER_URL = "https://en.wikipedia.org/wiki/List_of_current_WNBA_team_rosters"
INJURY_URL = "https://www.covers.com/sport/basketball/wnba/injuries"
WNBA_TEAM_NAME_MAP = {"Atlanta Dream": "Atlanta Dream", "Chicago Sky": "Chicago Sky", "Connecticut Sun": "Connecticut Sun", "Dallas Wings": "Dallas Wings", "Golden State Valkyries": "Golden State Valkyries", "Indiana Fever": "Indiana Fever", "Las Vegas Aces": "Las Vegas Aces", "Los Angeles Sparks": "Los Angeles Sparks", "Minnesota Lynx": "Minnesota Lynx", "New York Liberty": "New York Liberty", "Phoenix Mercury": "Phoenix Mercury", "Seattle Storm": "Seattle Storm", "Washington Mystics": "Washington Mystics", "Atlanta":"Atlanta Dream", "Chicago":"Chicago Sky", "Connecticut":"Connecticut Sun", "Dallas":"Dallas Wings", "Golden State":"Golden State Valkyries", "Indiana":"Indiana Fever", "Las Vegas":"Las Vegas Aces", "Los Angeles":"Los Angeles Sparks", "Minnesota":"Minnesota Lynx", "New York":"New York Liberty", "Phoenix":"Phoenix Mercury", "Seattle":"Seattle Storm", "Washington":"Washington Mystics"}
COMPARISON_STATS = [s for s in STATS_COLS_19 if s not in ['plusMinusPoints', 'fieldGoalsPercentage', 'threePointersPercentage', 'freeThrowsPercentage']]

# --- Live Data Scraping Functions ---
def scrape_wnba_rosters()->pd.DataFrame:
    player_data = [];logging.info("Scraping Rosters...")
    try:r=requests.get(WNBA_ROSTER_URL,headers={'User-Agent':'Mozilla/5.0'});r.raise_for_status()
    except Exception as e:logging.error(f"Roster scrape failed: {e}");return pd.DataFrame()
    soup=BeautifulSoup(r.content,'lxml')
    for container in soup.find_all('table',class_='toccolours'):
        team_name_tag=container.find('th').find('a') if container.find('th') else None; canonical_name=WNBA_TEAM_NAME_MAP.get(team_name_tag.get_text(strip=True)) if team_name_tag else None
        if canonical_name and(player_table:=container.find('table',class_='sortable')):
            for row in player_table.find_all('tr')[1:]:
                if len(cells:=row.find_all('td'))>=9:player_data.append({'team_name':canonical_name,'player_name':cells[3].get_text(strip=True)})
    logging.info(f"Successfully scraped {len(player_data)} roster spots.");return pd.DataFrame(player_data)

def scrape_wnba_schedule()->pd.DataFrame:
    game_data=[];logging.info("Scraping FULL season schedule for forecasting...")
    try:r=requests.get(WNBA_SCHEDULE_URL,headers={'User-Agent':'Mozilla/5.0'});r.raise_for_status()
    except Exception as e:logging.error(f"Schedule scrape failed: {e}");return pd.DataFrame()
    soup=BeautifulSoup(r.content,'lxml')
    if not(schedule_headline:=soup.find('h3',id='Schedule_2'))or not(main_container:=schedule_headline.find_next('table')):logging.error("Could not find schedule container.");return pd.DataFrame()
    for section in main_container.find_all('div',class_='mw-collapsible'):
        if not(game_table:=section.find('table',class_='wikitable')):continue
        current_date_str=None
        for row in game_table.find_all('tr')[1:]:
            cells=row.find_all('td'); offset = 0
            if len(cells) == 11: current_date_str, offset = cells[0].get_text(strip=True), 0
            elif len(cells) == 10: offset = -1
            else: continue
            if current_date_str:
                try: game_date=datetime.strptime(f"{current_date_str}, 2025","%A, %B %d, %Y").date(); away,home=WNBA_TEAM_NAME_MAP.get(cells[2+offset].get_text(strip=True)),WNBA_TEAM_NAME_MAP.get(cells[4+offset].get_text(strip=True))
                except: continue
                if away and home:game_data.append({'date':game_date.strftime('%Y-%m-%d'),'away_team':away,'home_team':home})
    logging.info(f"Successfully scraped {len(game_data)} total season games.");return pd.DataFrame(game_data).drop_duplicates()

def scrape_wnba_injuries()->pd.DataFrame:
    all_injuries=[];logging.info("Scraping Injuries...")
    try:r=requests.get(INJURY_URL,headers={'User-Agent':'Mozilla/5.0'},timeout=15);r.raise_for_status()
    except Exception as e:logging.error(f"Injury scrape failed: {e}");return pd.DataFrame()
    soup=BeautifulSoup(r.text,'html.parser')
    for container in soup.find_all('div',class_='covers-CoversSeasonInjuries-blockContainer'):
        if not(injury_table:=container.find('table',class_='covers-CoversMatchups-Table')):continue
        for row in injury_table.find('tbody').find_all('tr'):
            if'No injuries'not in row.text and len(cells:=row.find_all('td'))==4:
                if(player_link_tag:=cells[0].find('a'))and("Out"in(status_text:=re.sub(r'\s+',' ',cells[2].get_text(strip=True)))or"Out For Season"in status_text):all_injuries.append({'player_name':player_link_tag['href'].split('/')[-1].replace('-',' ').title(),'status':'Out'})
    logging.info(f"Successfully scraped {len(all_injuries)} injured players.");return pd.DataFrame(all_injuries)

# --- Prediction Helper Functions ---
def extract_birth_year(born_string):
    if not isinstance(born_string, str): return None
    match = re.search(r'\((\d{4})-\d{2}-\d{2}\)', born_string) or re.search(r'(\d{4})', born_string)
    return int(match.group(1)) if match else None

def run_evaluation_for_meta_training(model,loader,device,is_gnn=False):
    model.to(device).eval();all_preds=[]
    with torch.no_grad():
        for batch in loader:
            if is_gnn:batch_dev=batch.to(device);pred=model(batch_dev.x_dict,batch_dev.edge_index_dict)['player']
            else:p_indices,h_indicators,_,mask=[t.to(device) for t in batch];pred=model(p_indices,h_indicators,mask);pred=pred[mask]
            if pred.shape[0]>0:all_preds.append(pred.cpu().numpy())
    return np.concatenate(all_preds) if all_preds else np.array([])

def run_blueprint_predictions(schedule_df,roster_df,injury_df,name_to_id_map,official_player_names,df_features_master):
    logging.info("\n--- Running ISOLATED BLUEPRINT PIPELINE for Predictions ---")
    try:
        blueprint_player_map = joblib.load(BLUEPRINT_PLAYER_MAP_PATH)
        blueprint_scalers = joblib.load(BLUEPRINT_SCALERS_PATH)
        num_blueprint_players = len(blueprint_player_map)
        latest_player_stats=df_features_master.sort_values('gameDate').drop_duplicates('personId',keep='last')
        perf_cols_for_ewma = STATS_COLS_15 + ['usage_rate', 'ast_pct', 'Game_Score']
        BLUEPRINT_FEATURE_SETS={'player':['age','age_sq','height_cm','draft_pick','is_G','is_F','is_C'],'performance':[f'ewma_{c}'for c in perf_cols_for_ewma],'context':['home','days_rest','is_b2b','is_3in4','teammate_game_score_avg']+[f'matchup_avg_{stat}' for stat in STATS_COLS_15]}
        blueprint_feature_dims={s:len(f)for s,f in BLUEPRINT_FEATURE_SETS.items()}
        blueprint_model=BlueprintWNBAModel(num_blueprint_players,blueprint_feature_dims,64,256,0.4).to(device)
        blueprint_model.load_state_dict(torch.load(BLUEPRINT_MODEL_PATH,map_location=device));blueprint_model.eval()
        all_blueprint_outputs=[];injured_player_names={p['player_name'] for p in injury_df.to_dict('records')} if not injury_df.empty else set()
        for _,game in tqdm(schedule_df.iterrows(),total=len(schedule_df),desc="Blueprint Predictions"):
            gpd=[]
            for tn,ih in[(game['home_team'],1),(game['away_team'],0)]:
                for _,p in roster_df[roster_df['team_name']==tn].iterrows():
                    if p['player_name']in injured_player_names:continue
                    bm,s=process.extractOne(p['player_name'],official_player_names)
                    if s>85 and(pid:=name_to_id_map.get(bm))and pid in latest_player_stats['personId'].values:
                        ls=latest_player_stats[latest_player_stats['personId']==pid].iloc[0].copy(); ls['home']=ih; ls['days_rest']=3.0; ls['player_name']=bm; ls['team_name']=tn; ls['game_date']=game['date']; ls['home_team']=game['home_team']; ls['away_team']=game['away_team']
                        for col in BLUEPRINT_FEATURE_SETS['performance']+BLUEPRINT_FEATURE_SETS['context']:
                            if col not in ['home','days_rest'] and pd.isna(ls[col]): ls[col]=latest_player_stats[col].median()
                        gpd.append(ls)
            if not gpd:continue
            gdf=pd.DataFrame(gpd).reset_index(drop=True)
            with torch.no_grad():
                bi={'player_idx':torch.tensor([blueprint_player_map.get(pid,num_blueprint_players-1)for pid in gdf['personId']],dtype=torch.long).to(device)}
                for st,fs in BLUEPRINT_FEATURE_SETS.items():
                    if fs:
                        missing_cols = set(fs) - set(gdf.columns)
                        if missing_cols: raise ValueError(f"Blueprint Pipeline Error: Missing required feature columns: {missing_cols}")
                        bi[st]=torch.tensor(blueprint_scalers[st].transform(gdf[fs]),dtype=torch.float32).to(device)
                pd_preds=blueprint_model(**bi);bpdf=pd.DataFrame({s:p.squeeze().cpu().numpy()for s,p in pd_preds.items()})
            for i,pr in gdf.iterrows():
                fp={'player_id':pr['personId'],'player_name':pr['player_name'],'game_date':pr['game_date'],'home_team':pr['home_team'],'away_team':pr['away_team'],'team_name':pr['team_name']}
                fp['blueprint_pred']=bpdf.iloc[i].to_dict();all_blueprint_outputs.append(fp)
        logging.info("--- BLUEPRINT PREDICTION PIPELINE FINISHED ---");return pd.DataFrame(all_blueprint_outputs)
    except Exception as e:logging.error(f"FATAL ERROR in Blueprint pipeline: {e}",exc_info=True);return pd.DataFrame()

# --- Main Execution Block ---
def main():
    logging.info("\n" + "="*50 + "\n--- RUNNING PREDICTION & BLENDING ENGINE --- \n" + "="*50)
    
    # Load data
    df_features_master = pd.read_csv(MODEL_DATA_FEATURES_PATH, low_memory=False)
    df_features_master['gameDate'] = pd.to_datetime(df_features_master['gameDate'])
    with open(WIKI_CACHE_PATH, 'r') as f: data = json.load(f)
    player_bio_df = pd.DataFrame.from_dict(data, orient='index'); player_bio_df['personId'] = pd.to_numeric(player_bio_df['personId'], errors='coerce').dropna().astype(int)
    df_features_master = pd.merge(df_features_master, player_bio_df.drop_duplicates(subset=['personId']), on='personId', how='left')
    df_features_master['draft_pick']=df_features_master['draftInfo'].str.extract(r'P(\\d+)').astype(float); df_features_master['height_cm']=df_features_master['height'].str.extract(r'\\((\\d+)\\s*cm\\)').astype(float); df_features_master['birth_year']=df_features_master['born'].apply(extract_birth_year)
    for col in['draft_pick','height_cm','birth_year']:df_features_master[col].fillna(df_features_master[col].median(),inplace=True)
    df_features_master['season']=df_features_master['gameDate'].dt.year; df_features_master['age']=df_features_master['season']-df_features_master['birth_year']; df_features_master['age_sq'] = df_features_master['age']**2
    df_features_master['position'].fillna('N/A',inplace=True); df_features_master['is_G']=df_features_master['position'].str.contains('G').astype(int); df_features_master['is_F']=df_features_master['position'].str.contains('F').astype(int); df_features_master['is_C']=df_features_master['position'].str.contains('C').astype(int)

    player_id_to_name = df_features_master.drop_duplicates(subset='personId').set_index('personId')['fullName'].to_dict()
    name_to_id_map = {v: k for k, v in player_id_to_name.items()}
    official_player_names = list(name_to_id_map.keys())

    logging.info("--- Training Meta-Learner on Historical Data ---")
    train_df,test_df=train_test_split(df_features_master.dropna(subset=STATS_COLS_19),test_size=0.2,random_state=RANDOM_SEED)
    scalers={'stats15':StandardScaler().fit(train_df[STATS_COLS_15]),'stats19':StandardScaler().fit(train_df[STATS_COLS_19]),'base_feat':StandardScaler().fit(train_df[BASE_FEATURE_COLS]), 'adv_feat':StandardScaler().fit(train_df[ADVANCED_FEATURE_COLS])}
    player_to_idx={p:i for i,p in enumerate(df_features_master['personId'].unique())};num_players,max_players=len(player_to_idx),df_features_master.groupby('gameId').size().max()
    model_configs=[{'name':'bestest_transformer','path':'bestest_model_transformer.pth','outputs':19,'is_gnn':False,'model_class':WNBATransformer,'scaler':scalers['stats19'],'targets':STATS_COLS_19},{'name':'best_transformer','path':'best_model_transformer.pth','outputs':19,'is_gnn':False,'model_class':WNBATransformer,'scaler':scalers['stats19'],'targets':STATS_COLS_19},{'name':'bestest_base_gnn','path':'bestest_model_base_gnn.pth','outputs':15,'is_gnn':True,'features':BASE_FEATURE_COLS,'scaler':scalers['stats15'],'feature_scaler':scalers['base_feat'],'model_class':WNBAGNN,'targets':STATS_COLS_15},{'name':'best_base_gnn','path':'best_model_base_gnn.pth','outputs':19,'is_gnn':True,'features':BASE_FEATURE_COLS,'scaler':scalers['stats19'],'feature_scaler':scalers['base_feat'],'model_class':WNBAGNN,'targets':STATS_COLS_19}, {'name':'bestest_advanced_gnn','path':'bestest_model_advanced_gnn.pth','outputs':15,'is_gnn':True,'features':ADVANCED_FEATURE_COLS,'scaler':scalers['stats15'],'feature_scaler':scalers['adv_feat'],'model_class':WNBAGNN,'targets':STATS_COLS_15}, {'name':'best_advanced_gnn','path':'best_model_advanced_gnn.pth','outputs':19,'is_gnn':True,'features':ADVANCED_FEATURE_COLS,'scaler':scalers['stats19'],'feature_scaler':scalers['adv_feat'],'model_class':WNBAGNN,'targets':STATS_COLS_19}]
    sample_data = WNBAGraphDataset(f"{BASE_PATH}/data/meta_sample", test_df.head(40), player_to_idx, BASE_FEATURE_COLS, {'feature': scalers['base_feat'], 'stats': scalers['stats15']}, STATS_COLS_15)[0]
    hetero_metadata = sample_data.metadata()
    base_models={}; base_preds_for_meta={}; meta_learners={}; ground_truth_meta=test_df.reset_index(drop=True)
    for config in model_configs:
        model_path = os.path.join(BASE_PATH, config['path']); model_args={'num_players':num_players,'out_features':config['outputs']}
        if config.get('is_gnn'):model_args['feature_dim']=len(config['features'])
        model=config['model_class'](**model_args)
        if config.get('is_gnn'):model=to_hetero(model,hetero_metadata,aggr='sum')
        model.load_state_dict(torch.load(model_path,map_location=device));model.eval().to(device);base_models[config['name']]={'model':model,'config':config}

    with tqdm(total=len(base_models),desc="V10 Meta-Training Preds") as pbar:
        for name,data in base_models.items():
            config,model=data['config'],data['model'];loader_df=test_df
            if config.get('is_gnn'):loader=GraphDataLoader(WNBAGraphDataset(f"{BASE_PATH}/data/meta/{name}",loader_df,player_to_idx,config['features'],{'feature':config['feature_scaler'],'stats':config['scaler']},config['targets']),batch_size=64,shuffle=False)
            else:loader=BasicDataLoader(WNBADataset(prepare_transformer_tensors(loader_df,player_to_idx,max_players,config['targets'])),batch_size=64,collate_fn=custom_collate_transformer,shuffle=False)
            preds_scaled=run_evaluation_for_meta_training(model,loader,device,config.get('is_gnn',False));base_preds_for_meta[name]=pd.DataFrame(config['scaler'].inverse_transform(preds_scaled),columns=config['targets']);pbar.update(1)
    
    context_features_to_add = ['age', 'height_cm', 'is_G', 'is_F', 'is_C', 'days_rest', 'home']
    for stat in tqdm(COMPARISON_STATS,desc="Training Quantile Meta-Learners"):
        X_meta_list=[preds[stat].rename(name) for name,preds in base_preds_for_meta.items() if stat in preds.columns];
        if not X_meta_list: continue
        X_meta=pd.concat(X_meta_list,axis=1); X_meta[f'{stat}_preds_mean']=X_meta.mean(axis=1); X_meta[f'{stat}_preds_std']=X_meta.std(axis=1).fillna(0)
        context_df = ground_truth_meta.loc[X_meta.index][context_features_to_add].fillna(ground_truth_meta[context_features_to_add].median())
        X_meta_enhanced = pd.concat([X_meta, context_df.reset_index(drop=True)], axis=1)
        y_meta=ground_truth_meta.loc[X_meta_enhanced.index][stat]
        model=lgb.LGBMRegressor(objective='quantile',metric='quantile',alpha=0.5,n_estimators=1000,learning_rate=0.02,num_leaves=40,verbose=-1,n_jobs=-1,seed=RANDOM_SEED).fit(X_meta_enhanced,y_meta)
        meta_learners[stat] = model

    logging.info("\n--- Starting Live Prediction and Blending Phase ---")
    schedule_df, roster_df, injury_df = scrape_wnba_schedule(), scrape_wnba_rosters(), scrape_wnba_injuries()
    blueprint_predictions_df = run_blueprint_predictions(schedule_df, roster_df, injury_df, name_to_id_map, official_player_names, df_features_master)
    
    all_game_outputs = []
    today = date.today()
    future_schedule_df = schedule_df[pd.to_datetime(schedule_df['date']).dt.date >= today]
    if not future_schedule_df.empty:
        logging.info(f"--- Running V10 FORECAST on {len(future_schedule_df)} Future Games ---")
        injured_player_names = set(injury_df['player_name']) if not injury_df.empty else set();player_bio_lookup = df_features_master.drop_duplicates(subset='personId').set_index('personId');forecast_medians = train_df[ADVANCED_FEATURE_COLS].median().to_dict()
        for _, game in tqdm(future_schedule_df.iterrows(), total=len(future_schedule_df), desc="Forecasting Predictions"):
            game_players_data = []
            for team_name, is_home in [(game['home_team'], 1), (game['away_team'], 0)]:
                for _, player in roster_df[roster_df['team_name'] == team_name].iterrows():
                    best_match, score = process.extractOne(player['player_name'], official_player_names)
                    if score > 85 and (player_id := name_to_id_map.get(best_match)) and player['player_name'] not in injured_player_names:
                        player_entry = {'personId': player_id, 'home': is_home, 'player_name': best_match, 'teamName': team_name, 'days_rest': 3.0}
                        if player_id in player_bio_lookup.index:
                            for col in context_features_to_add:
                                if col in player_bio_lookup.columns: player_entry[col] = player_bio_lookup.loc[player_id, col]
                        else:
                            for col in context_features_to_add: player_entry.setdefault(col, train_df[col].median())
                        for col in ADVANCED_FEATURE_COLS: player_entry.setdefault(col, forecast_medians.get(col, 0))
                        game_players_data.append(player_entry)
            if not game_players_data: continue
            game_df = pd.DataFrame(game_players_data).reset_index(drop=True)
            local_home_idx = game_df.index[game_df['home'] == 1].tolist(); local_away_idx = game_df.index[game_df['home'] == 0].tolist()
            if not local_home_idx or not local_away_idx: continue
            base_game_preds = {}
            with torch.no_grad():
                for name,data in base_models.items():
                    model,config=data['model'],data['config'];p_indices_long=torch.tensor([player_to_idx.get(pid,num_players)for pid in game_df['personId']],dtype=torch.long)
                    if config.get('is_gnn'):
                        feature_tensor=torch.tensor(config['feature_scaler'].transform(game_df[config['features']]),dtype=torch.float32);x_dict={'player':torch.cat([p_indices_long.float().unsqueeze(1),feature_tensor],dim=1).to(device)};edge_index_dict={('player','teammate','player'):torch.tensor([[h,a] for h in local_home_idx for a in local_home_idx if h!=a] + [[h,a] for h in local_away_idx for a in local_away_idx if h!=a],dtype=torch.long).t().contiguous().to(device),('player','opponent','player'):torch.tensor([[h,a] for h in local_home_idx for a in local_away_idx] + [[a,h] for h in local_home_idx for a in local_away_idx],dtype=torch.long).t().contiguous().to(device)};preds_scaled=model(x_dict,edge_index_dict)['player']
                    else:
                        h_indicators=torch.tensor(game_df['home'].values,dtype=torch.float32).unsqueeze(1);mask=torch.ones(len(game_df),dtype=torch.bool);preds_scaled=model(p_indices_long.unsqueeze(0).to(device),h_indicators.unsqueeze(0).to(device),mask.unsqueeze(0).to(device)).squeeze(0)
                    base_game_preds[name]=pd.DataFrame(config['scaler'].inverse_transform(preds_scaled.cpu().numpy()),columns=config['targets'], index=game_df.index)
            for i, player_row in game_df.iterrows():
                amalgamated_results, base_results = {}, {};
                for model_name, preds_df in base_game_preds.items(): base_results[model_name] = preds_df.loc[i].to_dict()
                for stat, m_learner in meta_learners.items():
                    if stat in base_results.get(list(base_results.keys())[0], {}):
                        feature_vector,column_order=[],[];
                        for model_name,preds_df in base_game_preds.items():
                            if stat in preds_df.columns:feature_vector.append(preds_df.loc[i, stat]);column_order.append(model_name)
                        lgbm_input_df=pd.DataFrame([feature_vector],columns=column_order);lgbm_input_df[f'{stat}_preds_mean'] = lgbm_input_df.mean(axis=1); lgbm_input_df[f'{stat}_preds_std'] = lgbm_input_df.std(axis=1).fillna(0)
                        for col in context_features_to_add: lgbm_input_df[col] = player_row[col]
                        lgbm_input_df = lgbm_input_df[m_learner.feature_name_];amalgamated_results[stat] = m_learner.predict(lgbm_input_df)[0]
                all_game_outputs.append({"game_date":game['date'],"home_team":game['home_team'],"away_team":game['away_team'],"player_id":player_row['personId'],"player_name":player_row['player_name'],"teamName":player_row['teamName'],"base_preds":base_results,"amalgamated_pred":amalgamated_results})

    logging.info("\n--- MERGING, BLENDING & FINALIZING PREDICTIONS ---")
    outputs_df = pd.DataFrame(all_game_outputs)
    if not outputs_df.empty:
      outputs_df['game_date_str'] = pd.to_datetime(outputs_df['game_date']).dt.strftime('%Y-%m-%d')
    if not blueprint_predictions_df.empty:
        bp_df_copy = blueprint_predictions_df.copy();bp_df_copy.rename(columns={'player_id':'personId', 'game_date':'game_date_str'}, inplace=True);bp_df_copy['game_date_str'] = pd.to_datetime(bp_df_copy['game_date_str']).dt.strftime('%Y-%m-%d')
        super_amalgam_df = pd.merge(outputs_df, bp_df_copy[['personId', 'game_date_str', 'blueprint_pred']], left_on=['player_id', 'game_date_str'], right_on=['personId', 'game_date_str'], how='left')
    else:
        super_amalgam_df = outputs_df.copy();super_amalgam_df['blueprint_pred'] = [{} for _ in range(len(super_amalgam_df))]

    super_amalgam_preds = []
    for _, row in super_amalgam_df.iterrows():
        blend = {};lgbm_pred, bp_pred = row.get('amalgamated_pred', {}), row.get('blueprint_pred') if pd.notna(row.get('blueprint_pred')) else {};all_stats = set(lgbm_pred.keys()) | set(bp_pred.keys())
        for stat in all_stats:
            p1, p2 = lgbm_pred.get(stat), bp_pred.get(stat)
            if p1 is not None and p2 is not None: blend[stat] = (p1 + p2) / 2
            else: blend[stat] = p1 if p1 is not None else p2
        super_amalgam_preds.append(blend)
    super_amalgam_df['super_amalgam_pred'] = super_amalgam_preds

    best_model_map = {'points': 'blueprint_pred', 'assists': 'blueprint_pred', 'reboundsTotal': 'blueprint_pred', 'steals': 'blueprint_pred', 'blocks': 'amalgamated_pred'}
    cherry_pick_preds = []
    for _, row in super_amalgam_df.iterrows():
        final_preds = {}
        for stat, model_key in best_model_map.items():
            model_output = row.get(model_key, {}); final_preds[stat] = model_output.get(stat) if isinstance(model_output, dict) else None
        cherry_pick_preds.append(final_preds)
    super_amalgam_df['lowest_mae_pred'] = cherry_pick_preds
    
    mae_values = {'Ensemble_Blueprint': {'points': 4.76, 'assists': 1.33, 'reboundsTotal': 1.99, 'steals': 0.74, 'blocks': 0.47},'Ensemble_V10': {'points': 5.49, 'assists': 1.47, 'reboundsTotal': 2.23, 'steals': 0.75, 'blocks': 0.40}}
    stat_weights = {}
    for stat in mae_values['Ensemble_Blueprint'].keys():
        inv_mae_bp = 1/mae_values['Ensemble_Blueprint'][stat]; inv_mae_v10 = 1/mae_values['Ensemble_V10'][stat]; total_inv_mae = inv_mae_bp + inv_mae_v10
        stat_weights[stat] = {'blueprint_pred': inv_mae_bp / total_inv_mae, 'amalgamated_pred': inv_mae_v10 / total_inv_mae}
    smart_weighted_preds = []
    for _, row in super_amalgam_df.iterrows():
        final_preds = {}; bp_preds, v10_preds = row.get('blueprint_pred') if pd.notna(row.get('blueprint_pred')) else {}, row.get('amalgamated_pred', {})
        for stat, weights in stat_weights.items():
            pred_bp, pred_v10 = bp_preds.get(stat), v10_preds.get(stat)
            if pred_bp is not None and pred_v10 is not None: final_preds[stat] = (pred_bp * weights['blueprint_pred']) + (pred_v10 * weights['amalgamated_pred'])
        smart_weighted_preds.append(final_preds)
    super_amalgam_df['smart_weighted_pred'] = smart_weighted_preds

    final_df_path = os.path.join(DATA_PATH, "final_blended_predictions.pkl")
    super_amalgam_df.to_pickle(final_df_path)
    logging.info(f"Final blended predictions saved to {final_df_path}")

if __name__ == '__main__':
    # Set project dir for standalone execution in this script's context
    project_dir = os.getcwd() 
    main()
