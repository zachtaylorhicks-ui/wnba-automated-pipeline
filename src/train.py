
# src/train.py

# --- Imports & Setup ---
import pandas as pd
import numpy as np
import os
import logging
import gc
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader as BasicDataLoader
from torch.optim.lr_scheduler import OneCycleLR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# PyTorch Geometric imports
try:
    import torch_geometric
    from torch_geometric.data import HeteroData, Dataset as GraphDataset
    from torch_geometric.loader import DataLoader as GraphDataLoader
    from torch_geometric.nn import GATv2Conv, to_hetero, Linear
except ImportError:
    print("PyTorch Geometric not found, GNN models will not be trainable.")
    GATv2Conv, to_hetero, Linear, GraphDataset, GraphDataLoader, HeteroData = (None, None, None, object, object, object)

# --- Global Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)-8s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S', force=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RANDOM_SEED = 42
GAME_ID_COL, PLAYER_ID_COL, GAME_DATE_COL = 'gameId', 'personId', 'gameDate'

# --- Feature & Stat Lists ---
STATS_COLS_15 = ['points', 'assists', 'blocks', 'steals', 'fieldGoalsAttempted', 'fieldGoalsMade', 'threePointersAttempted', 'threePointersMade', 'freeThrowsAttempted', 'freeThrowsMade', 'reboundsDefensive', 'reboundsOffensive', 'reboundsTotal', 'foulsPersonal', 'turnovers']
STATS_COLS_19 = STATS_COLS_15 + ['fieldGoalsPercentage', 'threePointersPercentage', 'freeThrowsPercentage', 'plusMinusPoints']
BASE_FEATURE_COLS = ['home', 'days_rest']
try:
    df_cols = pd.read_csv('data/wnba_model_ready_data_with_features.csv', nrows=0).columns
    ADV_PLAYER_COLS = [c for c in df_cols if '_roll_' in c]
    ADV_TEAM_COLS = ['roll_Poss', 'roll_OffRtg', 'opp_roll_Poss', 'opp_roll_OffRtg']
    ADVANCED_FEATURE_COLS = BASE_FEATURE_COLS + ADV_PLAYER_COLS + ADV_TEAM_COLS
except FileNotFoundError:
    ADVANCED_FEATURE_COLS = BASE_FEATURE_COLS

# --- Model Architectures (VERBATIM from your scripts) ---
class WNBATransformer(nn.Module):
    def __init__(self, num_players, out_features, embedding_dim=63, hidden_dim=256, num_heads=4):
        super().__init__();self.player_embedding = nn.Embedding(num_players + 1, embedding_dim);d_model = embedding_dim + 1;encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=hidden_dim, dropout=0.15, batch_first=True, activation='gelu');self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=3);self.output_layer = nn.Linear(d_model, out_features)
    def forward(self, player_indices, home_indicators, mask):
        inputs = torch.cat([self.player_embedding(player_indices), home_indicators], dim=-1);return self.output_layer(self.transformer_encoder(inputs, src_key_padding_mask=~mask))

class WNBAGNN(nn.Module):
    def __init__(self, num_players, feature_dim, out_features, hidden_dim=128):
        super().__init__();self.player_embedding = nn.Embedding(num_players + 1, hidden_dim);self.feature_proj = Linear(hidden_dim + feature_dim, hidden_dim);self.conv1 = GATv2Conv(hidden_dim, hidden_dim, heads=4, dropout=0.2);self.conv2 = GATv2Conv(hidden_dim * 4, hidden_dim, heads=1, dropout=0.2);self.output_layer = Linear(hidden_dim, out_features)
    def forward(self, x_player, edge_index):
        combined = torch.cat([self.player_embedding(x_player[:, 0].long()), x_player[:, 1:]], dim=1);x = self.feature_proj(combined).relu();x = self.conv1(x, edge_index).relu();return self.output_layer(self.conv2(x, edge_index))

class BlueprintWNBAModel(nn.Module):
    def __init__(self, num_players, feature_set_dims, embedding_dim, hidden_dim, dropout):
        super().__init__();self.player_embedding = nn.Embedding(num_players, embedding_dim);self.player_stream = nn.Sequential(nn.Linear(embedding_dim + feature_set_dims['player'], hidden_dim), nn.BatchNorm1d(hidden_dim), nn.GELU(), nn.Dropout(dropout));self.performance_stream = nn.Sequential(nn.Linear(feature_set_dims['performance'], hidden_dim), nn.BatchNorm1d(hidden_dim), nn.GELU(), nn.Dropout(dropout));self.context_stream = nn.Sequential(nn.Linear(feature_set_dims['context'], hidden_dim), nn.BatchNorm1d(hidden_dim), nn.GELU(), nn.Dropout(dropout));self.fusion = nn.Sequential(nn.Linear(hidden_dim * 3, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim // 2), nn.BatchNorm1d(hidden_dim // 2), nn.GELU(), nn.Dropout(dropout));self.output_heads = nn.ModuleDict({stat: nn.Linear(hidden_dim // 2, 1) for stat in STATS_COLS_15})
    def forward(self, player_idx, player, performance, context):
        embedded_player = self.player_embedding(player_idx);player_combined = torch.cat([embedded_player, player], dim=1);player_rep = self.player_stream(player_combined);performance_rep = self.performance_stream(performance);context_rep = self.context_stream(context);fused_input = torch.cat([player_rep, performance_rep, context_rep], dim=1);fused_rep = self.fusion(fused_input);return {stat: head(fused_rep) for stat, head in self.output_heads.items()}

# --- Data Handling Classes & Functions ---
def custom_collate_transformer(batch): p,h,s,m=zip(*batch);return torch.stack(p),torch.stack(h),torch.stack(s),torch.stack(m)
class WNBADataset(BasicDataLoader):
    def __init__(self, tensors): self.tensors = tensors
    def __len__(self): return len(self.tensors)
    def __getitem__(self, idx): return self.tensors[idx]

def prepare_transformer_tensors(df, player_to_idx, max_players, target_cols):
    game_tensors = []
    for _, game_data in tqdm(df.groupby(GAME_ID_COL), desc="Pre-processing Transformer Tensors", leave=False):
        n_p=len(game_data);p_indices=torch.tensor([player_to_idx.get(pid,len(player_to_idx)) for pid in game_data[PLAYER_ID_COL]],dtype=torch.long);h_indicators=torch.tensor(game_data['home'].values,dtype=torch.float32).unsqueeze(-1);stats=torch.tensor(game_data[target_cols].values,dtype=torch.float32);pad=max_players-n_p;p_indices=torch.nn.functional.pad(p_indices,(0,pad));h_indicators=torch.nn.functional.pad(h_indicators,(0,0,0,pad));stats=torch.nn.functional.pad(stats,(0,0,0,pad));mask=torch.ones(max_players,dtype=torch.bool);mask[n_p:]=False;game_tensors.append((p_indices,h_indicators,stats,mask))
    return game_tensors

class WNBAGraphDataset(GraphDataset):
    def __init__(self, root, df, p_to_idx, f_cols, scalers, target_cols):
        self.df, self.p_to_idx, self.f_cols, self.scalers, self.target_cols = df, p_to_idx, f_cols, scalers, target_cols
        self.game_groups = df.groupby(GAME_ID_COL); self.game_ids = list(self.game_groups.groups.keys()); self.unknown_p_idx = len(p_to_idx); super().__init__(root)
    def len(self): return len(self.game_ids)
    def get(self, idx):
        g_data=self.game_groups.get_group(self.game_ids[idx]);data=HeteroData();p_indices_long=torch.tensor([self.p_to_idx.get(pid,self.unknown_p_idx) for pid in g_data[PLAYER_ID_COL]],dtype=torch.long).unsqueeze(1);f_tensor=torch.tensor(self.scalers['feature'].transform(g_data[self.f_cols]),dtype=torch.float32);data['player'].x=torch.cat([p_indices_long.float(),f_tensor],dim=1);data['player'].y=torch.tensor(self.scalers['stats'].transform(g_data[self.target_cols]),dtype=torch.float32);idx_map={gid:lid for lid,gid in enumerate(g_data.index)};h_loc,a_loc=[idx_map[i] for i in g_data.index[g_data['home']==1]],[idx_map[i] for i in g_data.index[g_data['home']==0]];t_edges,o_edges=[],[];[t_edges.extend([[team[i],team[j]],[team[j],team[i]]]) for team in [h_loc,a_loc] for i in range(len(team)) for j in range(i+1,len(team))];[o_edges.extend([[h_idx,a_idx],[a_idx,h_idx]]) for h_idx in h_loc for a_idx in a_loc];data['player','teammate','player'].edge_index=torch.tensor(t_edges,dtype=torch.long).t().contiguous();data['player','opponent','player'].edge_index=torch.tensor(o_edges,dtype=torch.long).t().contiguous();return data

class WNBAPlayerDataset(Dataset):
    def __init__(self, df, player_to_idx, feature_sets):
        self.player_indices = torch.tensor([player_to_idx[pid] for pid in df['personId']], dtype=torch.long)
        self.targets = torch.tensor(df[feature_sets['targets']].values, dtype=torch.float32)
        self.feature_tensors = { stream: torch.tensor(df[features].values, dtype=torch.float32) for stream, features in feature_sets.items() if stream != 'targets' }
    def __len__(self): return len(self.player_indices)
    def __getitem__(self, idx):
        item = {stream: tensors[idx] for stream, tensors in self.feature_tensors.items()}
        item['player_idx'] = self.player_indices[idx]
        return item, self.targets[idx]

# --- Generic Training Loop ---
def execute_training_loop(model, model_name, train_loader, epochs, lr, is_gnn, target_cols):
    model_path = f"best_model_{model_name.replace(' ', '_').lower()}.pth"
    logging.info(f"Starting training for {model_name}...")
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.HuberLoss().to(device)
    for epoch in range(epochs):
        model.train()
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - {model_name}", leave=False):
            optimizer.zero_grad()
            if is_gnn:
                batch_dev = batch.to(device)
                pred = model(batch_dev.x_dict, batch_dev.edge_index_dict)['player']
                loss = criterion(pred, batch_dev.y_dict['player'])
            else: # Transformer
                p_idx, h_ind, stats, mask = [b.to(device) for b in batch]
                pred = model(p_idx, h_ind, mask)
                loss = criterion(pred[mask], stats[mask])
            loss.backward(); optimizer.step()
    torch.save(model.state_dict(), model_path)
    logging.info(f"Finished training for {model_name} and saved to {model_path}")

# --- Blueprint Training Loop ---
def execute_blueprint_training_loop(model, train_loader, val_loader, epochs, lr, model_save_path):
    logging.info(f"Starting training for {model_save_path}...")
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    criterion = nn.HuberLoss()
    scheduler = OneCycleLR(optimizer, max_lr=lr, total_steps=len(train_loader) * epochs)
    best_val_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        for inputs, targets in pbar:
            inputs = {k: v.to(device) for k, v in inputs.items()}; targets = targets.to(device)
            predictions = model(**inputs)
            loss = sum(criterion(predictions[stat].squeeze(), targets[:, i]) for i, stat in enumerate(STATS_COLS_15))
            optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step(); scheduler.step()
            pbar.set_postfix(loss=loss.item())
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = {k: v.to(device) for k, v in inputs.items()}; targets = targets.to(device)
                predictions = model(**inputs)
                loss = sum(criterion(predictions[stat].squeeze(), targets[:, i]) for i, stat in enumerate(STATS_COLS_15))
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss; torch.save(model.state_dict(), model_save_path)
            logging.info(f"Epoch {epoch+1} | Val Loss improved to {avg_val_loss:.4f}. Model saved.")
    logging.info(f"Finished training for {model_save_path}. Best Val Loss: {best_val_loss:.4f}")

# --- Main Training Orchestration Functions ---
def train_v10_models(df, player_to_idx, num_players, max_players):
    logging.info("\n" + "="*20 + " TRAINING V10 MODELS (GNNs & TRANSFORMERS) " + "="*20)
    train_df, _ = train_test_split(df, test_size=0.2, random_state=RANDOM_SEED, stratify=df[GAME_DATE_COL].dt.year)
    scalers={'stats15':StandardScaler().fit(train_df[STATS_COLS_15]),'stats19':StandardScaler().fit(train_df[STATS_COLS_19]),'base_feat':StandardScaler().fit(train_df[BASE_FEATURE_COLS]), 'adv_feat':StandardScaler().fit(train_df[ADVANCED_FEATURE_COLS])}
    model_configs = [
        {'name': 'bestest_transformer', 'outputs': 19, 'is_gnn': False, 'model_class': WNBATransformer, 'scaler': scalers['stats19'], 'targets': STATS_COLS_19},
        {'name': 'best_transformer', 'outputs': 19, 'is_gnn': False, 'model_class': WNBATransformer, 'scaler': scalers['stats19'], 'targets': STATS_COLS_19},
        {'name': 'bestest_base_gnn', 'outputs': 15, 'is_gnn': True, 'features': BASE_FEATURE_COLS, 'scaler': scalers['stats15'], 'feature_scaler': scalers['base_feat'], 'model_class': WNBAGNN, 'targets': STATS_COLS_15},
        {'name': 'best_base_gnn', 'outputs': 19, 'is_gnn': True, 'features': BASE_FEATURE_COLS, 'scaler': scalers['stats19'], 'feature_scaler': scalers['base_feat'], 'model_class': WNBAGNN, 'targets': STATS_COLS_19},
        {'name': 'bestest_advanced_gnn', 'outputs': 15, 'is_gnn': True, 'features': ADVANCED_FEATURE_COLS, 'scaler': scalers['stats15'], 'feature_scaler': scalers['adv_feat'], 'model_class': WNBAGNN, 'targets': STATS_COLS_15},
        {'name': 'best_advanced_gnn', 'outputs': 19, 'is_gnn': True, 'features': ADVANCED_FEATURE_COLS, 'scaler': scalers['stats19'], 'feature_scaler': scalers['adv_feat'], 'model_class': WNBAGNN, 'targets': STATS_COLS_19}
    ]
    train_tensors_t_19 = prepare_transformer_tensors(train_df, player_to_idx, max_players, STATS_COLS_19)
    train_loader_t_19 = BasicDataLoader(WNBADataset(train_tensors_t_19), batch_size=32, shuffle=True, num_workers=2, pin_memory=True, collate_fn=custom_collate_transformer)
    sample_data = WNBAGraphDataset(f"{project_dir}/data/gnn_sample", train_df.head(40), player_to_idx, BASE_FEATURE_COLS, {'feature': scalers['base_feat'], 'stats': scalers['stats15']}, STATS_COLS_15)[0]
    hetero_metadata = sample_data.metadata()
    for config in model_configs:
        model_args = {'num_players': num_players, 'out_features': config['outputs']}
        if config.get('is_gnn'): model_args['feature_dim'] = len(config['features'])
        model = config['model_class'](**model_args).to(device)
        if config.get('is_gnn'): model = to_hetero(model, hetero_metadata, aggr='sum').to(device)
        if config.get('is_gnn'):
            root_dir = f"{project_dir}/data/gnn/{config['name']}"
            train_loader = GraphDataLoader(WNBAGraphDataset(root_dir, train_df, player_to_idx, config['features'], {'feature': config['feature_scaler'], 'stats': config['scaler']}, config['targets']), batch_size=32, shuffle=True)
        else: train_loader = train_loader_t_19
        execute_training_loop(model, config['name'], train_loader, epochs=25, lr=0.001, is_gnn=config.get('is_gnn'), target_cols=config['targets'])
    gc.collect()

def train_blueprint_model(df, player_to_idx, num_players):
    logging.info("\n" + "="*20 + " TRAINING BLUEPRINT MODEL " + "="*20)
    df_model = df.copy()
    # Feature Engineering specific to blueprint
    df_model['age_sq'] = df_model['age'] ** 2
    df_model['position'].fillna('N/A', inplace=True); df_model['is_G'] = df_model['position'].str.contains('G').astype(int); df_model['is_F'] = df_model['position'].str.contains('F').astype(int); df_model['is_C'] = df_model['position'].str.contains('C').astype(int)
    perf_cols_for_ewma = STATS_COLS_15 + ['usage_rate', 'ast_pct', 'Game_Score']
    FEATURE_SETS = {'player': ['age', 'age_sq', 'height_cm', 'draft_pick', 'is_G', 'is_F', 'is_C'], 'performance': [f'ewma_{col}' for col in perf_cols_for_ewma if f'ewma_{col}' in df_model.columns], 'context': ['home', 'days_rest', 'is_b2b', 'is_3in4', 'teammate_game_score_avg'] + [f'matchup_avg_{stat}' for stat in STATS_COLS_15 if f'matchup_avg_{stat}' in df_model.columns], 'targets': STATS_COLS_15}
    all_features = [item for sublist in FEATURE_SETS.values() for item in sublist if item in df_model.columns]
    df_model.dropna(subset=all_features, inplace=True)
    train_val_df, test_df = train_test_split(df_model, test_size=0.2, random_state=RANDOM_SEED)
    train_df, val_df = train_test_split(train_val_df, test_size=0.15, random_state=RANDOM_SEED)
    scalers = {}
    for stream, features in FEATURE_SETS.items():
        if stream != 'targets' and features:
            scaler = StandardScaler(); train_df.loc[:, features] = scaler.fit_transform(train_df[features]); val_df.loc[:, features] = scaler.transform(val_df[features]); scalers[stream] = scaler
    train_dataset = WNBAPlayerDataset(train_df, player_to_idx, FEATURE_SETS)
    val_dataset = WNBAPlayerDataset(val_df, player_to_idx, FEATURE_SETS)
    train_loader = BasicDataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = BasicDataLoader(val_dataset, batch_size=256)
    feature_dims = {stream: len(features) for stream, features in FEATURE_SETS.items() if stream != 'targets'}
    model = BlueprintWNBAModel(num_players, feature_dims, 64, 256, 0.4).to(device)
    execute_blueprint_training_loop(model, train_loader, val_loader, epochs=50, lr=0.005, model_save_path='best_pro_plus_blueprint_model_v4.9.pth')
    joblib.dump(scalers, 'pro_plus_blueprint_scalers.pkl')
    joblib.dump(player_to_idx, 'pro_plus_blueprint_player_map.pkl')

def main():
    logging.info("--- Starting Unified Training Pipeline ---")
    data_path = f'{project_dir}/data/wnba_model_ready_data_with_features.csv'
    if not os.path.exists(data_path):
        logging.error(f"FATAL: Main data file not found at '{data_path}'. Please run data_prep.py first.")
        return
    df = pd.read_csv(data_path, low_memory=False)
    df[GAME_DATE_COL] = pd.to_datetime(df[GAME_DATE_COL])
    player_to_idx = {p: i for i, p in enumerate(df[PLAYER_ID_COL].unique())}; num_players = len(player_to_idx); max_players = df.groupby(GAME_ID_COL).size().max()
    train_v10_models(df.copy(), player_to_idx, num_players, max_players)
    player_bio_df = pd.read_json(f'{project_dir}/data/wnba_wiki_cache.json', orient='index'); player_bio_df['personId'] = player_bio_df['personId'].astype(int)
    df_for_blueprint = pd.merge(df, player_bio_df, on='personId', how='left')
    train_blueprint_model(df_for_blueprint, player_to_idx, num_players)
    logging.info("--- ALL TRAINING PROCESSES COMPLETE ---")

if __name__ == '__main__':
    project_dir = "/content/wnba-automated-pipeline" # Set project dir for standalone execution
    main()
