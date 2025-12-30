import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, precision_recall_curve, auc, fbeta_score
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold, train_test_split
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
import time
import tqdm
import os
import argparse
import copy
import csv
from datetime import datetime
import itertools

# -------------------------
# 1. tool functions
# -------------------------
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def calculate_metrics(labels, probs):
    probs = np.nan_to_num(probs, nan=0.0)
    
    if np.sum(probs) == 0 and np.max(probs) == 0:
        return {k: 0.0 for k in ["Accuracy", "F1", "AUC_PR", "F0.5", "Precision", "Recall"]}
        
    preds = (np.array(probs) >= 0.5).astype(int)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, zero_division=0)
    precision, recall, _ = precision_recall_curve(labels, probs)
    try:
        auc_pr = auc(recall, precision)
    except:
        auc_pr = 0.0
    f05 = fbeta_score(labels, preds, beta=0.5, zero_division=0)
    return {
        "Accuracy": acc,
        "F1": f1,
        "AUC_PR": auc_pr,
        "F0.5": f05,
        "Precision": precision_score(labels, preds, zero_division=0),
        "Recall": recall_score(labels, preds, zero_division=0)
    }

def load_dataset(pair_path, code_path):
    print(f"Loading pairs from: {pair_path}")
    pairs_df = pd.read_csv(pair_path)
    print(f"Loading code from: {code_path}")
    code_df = pd.read_csv(code_path)
    code_map = {row['id']: row['code'] for _, row in code_df.iterrows()}
    
    results = []
    for _, row in pairs_df.iterrows():
        try:
            label = row['label']
            if pd.isna(label): continue
            
            if isinstance(label, str):
                label = label.strip()
                if label not in ["0", "1"]: continue
                label = int(label)
            elif isinstance(label, (float, np.floating)):
                label = int(label)
                
            if label not in [0, 1]: continue
            
            code1 = code_map.get(row['code_id_1'], "")
            code2 = code_map.get(row['code_id_2'], "")
            
            project_name = row.get('project', 'unknown_project')

            if code1 and code2:
                results.append({
                    'pair_id': row['id'],
                    'code1': code1,
                    'code2': code2,
                    'label': label,
                    'project': project_name 
                })
        except: continue
    print(f"Loaded {len(results)} valid samples.")
    return pd.DataFrame(results)

# -------------------------
# 2. save results
# -------------------------
def save_detailed_predictions(save_dir, strategy, input_type, ids, labels, probs, note=""):
    os.makedirs(save_dir, exist_ok=True)
    filename = f"pred_{strategy}_{input_type}_{note}.csv"
    path = os.path.join(save_dir, filename)
    
    preds = (np.array(probs) >= 0.5).astype(int)
    
    df = pd.DataFrame({
        'pair_id': ids,
        'true_label': labels,
        'predicted_prob': probs,
        'predicted_label': preds
    })
    df.to_csv(path, index=False)
    print(f"  [Output] Detailed predictions saved to: {path}")

def append_summary_table(save_dir, model_name, strategy, input_type, metrics, train_time, test_time, note=""):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, "summary_results.csv")
    
    file_exists = os.path.isfile(path)
    
    row_data = {
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Model": model_name,
        "Strategy": strategy,
        "Input_Type": input_type,
        "Note": note,
        "Accuracy": f"{metrics['Accuracy']:.4f}",
        "F1": f"{metrics['F1']:.4f}",
        "AUC_PR": f"{metrics['AUC_PR']:.4f}",
        "F0.5": f"{metrics['F0.5']:.4f}",
        "CL_Gap": f"{metrics.get('CL_Gap', 0):.4f}", 
        "Train_Time(s)": f"{train_time:.2f}",
        "Test_Time(s)": f"{test_time:.2f}"
    }
    
    fieldnames = list(row_data.keys())
    
    with open(path, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_data)
    print(f"  [Output] Summary appended to: {path}")

def save_trainable_only(model, path):
    state_dict = {k: v.cpu() for k, v in model.state_dict().items() if v.requires_grad}
    torch.save(state_dict, path)
    print(f"  >> Model saved to {path} (Size: {len(state_dict)} tensors)")

def load_trainable_only(model, path):
    print(f"  >> Loading model from {path}")
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint, strict=False)

# -------------------------
# 3. Dataset [Fix Applied Here]
# -------------------------
class UnifiedCodeDataset(Dataset):
    def __init__(self, data, tokenizer, test_only, mode, max_length=1024, is_train=True):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.test_only = test_only 
        self.mode = mode           
        self.is_train = is_train
        
    def __len__(self):
        return len(self.data)
    
    def _tokenize(self, text):
        return self.tokenizer(text, truncation=True, max_length=self.max_length, padding='max_length', return_tensors='pt')

    def _get_text(self, item):
        """Helper to handle concatenation logic"""
        code1 = item['code1']
        code2 = item['code2']
        if self.test_only:
            return code1
        else:
            return code1 + " " + code2

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        label = float(item['label'])
        pair_id = item.get('pair_id', f"idx_{idx}")
        
        anchor_text = self._get_text(item)

        # ------------------------------------------------
        # mode1: SFT
        # ------------------------------------------------

        if self.mode == 'sft':
            enc1 = self._tokenize(anchor_text)
            res = {
                'input_ids1': enc1['input_ids'][0],       
                'attention_mask1': enc1['attention_mask'][0],
                'label': torch.tensor([label], dtype=torch.float32),
                'pair_id': pair_id
            }
            return res

        # ------------------------------------------------
        # mode2: contrastive
        # ------------------------------------------------
        else: 
            if self.is_train:
                rand_idx = np.random.randint(0, len(self.data))
                while rand_idx == idx: rand_idx = np.random.randint(0, len(self.data))
                rand_item = self.data.iloc[rand_idx]
                
                rand_text = self._get_text(rand_item)
                
                flag = 1.0 if label == float(rand_item['label']) else 0.0
                
                enc_anchor = self._tokenize(anchor_text)
                enc_rand = self._tokenize(rand_text)
                
                return {
                    'input_ids1': enc_anchor['input_ids'][0],
                    'attention_mask1': enc_anchor['attention_mask'][0],
                    'input_ids2': enc_rand['input_ids'][0],
                    'attention_mask2': enc_rand['attention_mask'][0],
                    'flag': torch.tensor(flag, dtype=torch.float32),
                    'label': torch.tensor([label], dtype=torch.float32)
                }
            else:
                # Eval/Test (Non-SFT mode fallback)
                enc = self._tokenize(anchor_text)
                res = {
                    'input_ids1': enc['input_ids'][0],
                    'attention_mask1': enc['attention_mask'][0],
                    'label': torch.tensor([label], dtype=torch.float32),
                    'pair_id': pair_id
                }
                return res

# -------------------------
# 4. Model
# -------------------------
class UnifiedCodeModel(nn.Module):
    def __init__(self, llama_model_path, hidden_size=4096, dropout=0.3, temperature=0.07, proj_dim=256):
        super().__init__()
        print(f"Init Model from: {llama_model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(llama_model_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_use_double_quant=True, 
            bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            llama_model_path, quantization_config=bnb_config, device_map='auto', trust_remote_code=True
        )
        base_model.gradient_checkpointing_enable()
        base_model = prepare_model_for_kbit_training(base_model)
        
        # LoRA Config
        peft_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        self.llama_model = get_peft_model(base_model, peft_config)
        self.llama_model.print_trainable_parameters()
        
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.temperature = temperature
        self.proj_dim = proj_dim
        
        self._init_heads()

    def _init_heads(self):
        self.sft_classifier_single = self._build_head(self.hidden_size, self.dropout)
        
        self.projector = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size), nn.LayerNorm(self.hidden_size), nn.GELU(), nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size, self.proj_dim)
        )
        self.con_classifier = self._build_head(self.hidden_size, self.dropout)
        self.pair_projector = nn.Sequential(
            nn.Linear(self.hidden_size * 4, self.hidden_size), nn.GELU(), nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size, self.proj_dim)
        )
        self.logit_scale_pair = nn.Parameter(torch.tensor(np.log(1/0.07), dtype=torch.float32))

    def reset_heads(self):
        print("  >> Resetting trainable heads...")
        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        
        self.sft_classifier_single.apply(init_weights)
        self.projector.apply(init_weights)
        self.con_classifier.apply(init_weights)
        self.pair_projector.apply(init_weights)
        with torch.no_grad():
            self.logit_scale_pair.fill_(np.log(1/0.07))

    def _build_head(self, in_dim, dropout):
        return nn.Sequential(
            nn.Linear(in_dim, in_dim), nn.LayerNorm(in_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(in_dim, in_dim // 2), nn.LayerNorm(in_dim // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(in_dim // 2, 1)
        ).float()

    def get_code_rep(self, ids, mask):
        out = self.llama_model(input_ids=ids, attention_mask=mask, output_hidden_states=True)
        hidden = out.hidden_states[-1].to(torch.float32)
        mask_expanded = mask.unsqueeze(-1).float()
        sum_hidden = torch.sum(hidden * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        return (sum_hidden / sum_mask).float()

    def forward(self, ids1, mask1, ids2=None, mask2=None, test_only=False, strategy='sft', phase='train'):
        rep1 = self.get_code_rep(ids1, mask1)
        
        if strategy == 'sft':
            return self.sft_classifier_single(rep1)
        
        elif strategy == 'con':
            logits = self.con_classifier(rep1)
            if phase == 'train':
                rep2 = self.get_code_rep(ids2, mask2)
                z1 = F.normalize(self.projector(rep1), p=2, dim=1)
                z2 = F.normalize(self.projector(rep2), p=2, dim=1)
                return logits, z1, z2, None
            return logits
            
        elif strategy == 'full_con':
            if phase == 'train':
                rep2 = self.get_code_rep(ids2, mask2)
                z1 = F.normalize(self.projector(rep1), p=2, dim=1)
                z2 = F.normalize(self.projector(rep2), p=2, dim=1)
                return None, z1, z2, None
            else:
                return self.con_classifier(rep1)

        elif strategy == 'im_con':
            logits = self.con_classifier(rep1)
            if phase == 'train':
                rep2 = self.get_code_rep(ids2, mask2)
                z1 = F.normalize(self.projector(rep1), p=2, dim=1)
                z2 = F.normalize(self.projector(rep2), p=2, dim=1)
                
                # 特征交互
                pair_feat = torch.cat([rep1, rep2, torch.abs(rep1 - rep2), rep1 * rep2], dim=1)
                z_pair = F.normalize(self.pair_projector(pair_feat), p=2, dim=1)
                return logits, z1, z2, z_pair
            return logits
        return None

# -------------------------
# 5. Loss Calculator
# -------------------------
class LossCalculator:
    def __init__(self, model, device):
        self.model = model
        self.device = device

    def calc_con_loss(self, z1, z2, labels, flag):
        sim = F.cosine_similarity(z1, z2)
        logits = sim / self.model.temperature
        loss = F.binary_cross_entropy_with_logits(logits, flag)
        return loss

    def calc_sigmoid_loss(self, z_pair, labels):
        if z_pair is None: return torch.tensor(0.0, device=self.device)
        B = z_pair.size(0)
        z = F.normalize(z_pair, p=2, dim=1)
        scale = self.model.logit_scale_pair.exp().clamp(max=100.0)
        logits = scale * (z @ z.t())
        L = labels.view(-1, 1) == labels.view(1, -1)
        eye = torch.eye(B, device=self.device).bool()
        pos = (L & (~eye)).float()
        neg = (1.0 - pos) * (~eye).float()
        bce = F.binary_cross_entropy_with_logits(logits, pos, weight=(pos + neg), reduction='none')
        return (bce * (~eye).float()).sum() / ((~eye).float().sum() + 1e-9)

    def calc_supcon_loss(self, z, labels):
        if z is None: return torch.tensor(0.0, device=self.device)
        sim = torch.matmul(z, z.T) / self.model.temperature
        sim_max, _ = torch.max(sim, dim=1, keepdim=True)
        exp_sim = torch.exp(sim - sim_max.detach())
        mask_same = (labels.view(-1, 1) == labels.view(1, -1)).float()
        mask_eye = (~torch.eye(z.size(0), device=self.device).bool()).float()
        denom = (exp_sim * mask_eye).sum(dim=1) + 1e-9
        log_prob = sim - torch.log(denom).unsqueeze(1)
        mask_pos = mask_same * mask_eye
        loss = -(log_prob * mask_pos).sum(dim=1) / (mask_pos.sum(dim=1) + 1e-9)
        return loss.mean()

# -------------------------
# 6. epoch
# -------------------------
def run_epoch(model, loader, optimizer, criterion, strategy, test_only, device, phase='train', return_details=False):
    start_time = time.time()
    if phase == 'train': model.train()
    else: model.eval()
    
    loss_calc = LossCalculator(model, device)
    total_loss = 0
    all_preds, all_labels = [], []
    all_ids = [] 
    
    for batch in tqdm.tqdm(loader, desc=f"{phase} ({strategy})", leave=False):
        ids1 = batch['input_ids1'].to(device)
        mask1 = batch['attention_mask1'].to(device)
        
        ids2 = batch.get('input_ids2').to(device) if 'input_ids2' in batch else None
        mask2 = batch.get('attention_mask2').to(device) if 'attention_mask2' in batch else None
            
        labels = batch['label'].to(device)
        flag = batch.get('flag').to(device) if 'flag' in batch else None
        
        if 'pair_id' in batch: all_ids.extend(batch['pair_id'])

        if phase == 'train': optimizer.zero_grad()
        
        with torch.set_grad_enabled(phase == 'train'):
            out = model(ids1, mask1, ids2, mask2, test_only, strategy=strategy, phase=phase)
            
            loss = 0
            logits = None
            
            if strategy == 'sft':
                logits = out
                loss = criterion(logits.view(-1), labels.view(-1))
                
            elif strategy == 'con':
                if phase == 'train':
                    logits, z1, z2, _ = out
                    l_cls = criterion(logits.view(-1), labels.view(-1))
                    l_con = loss_calc.calc_con_loss(z1, z2, labels, flag)
                    loss = 0.2 * l_cls + 0.8 * l_con
                else:
                    logits = out
                    loss = criterion(logits.view(-1), labels.view(-1))
            
            elif strategy == 'full_con':
                if phase == 'train':
                    _, z1, z2, _ = out
                    loss = loss_calc.calc_con_loss(z1, z2, labels, flag)
                    logits = torch.zeros_like(labels) 
                else:
                    logits = out
                    loss = criterion(logits.view(-1), labels.view(-1))

            elif strategy == 'im_con':
                if phase == 'train':
                    logits, z1, z2, z_pair = out
                    l_cls = criterion(logits.view(-1), labels.view(-1))
                    l_sig = loss_calc.calc_sigmoid_loss(z_pair, labels)
                    l_sup = loss_calc.calc_supcon_loss(z_pair, labels)
                    loss = l_cls + 0.8 * l_sig + 0.5 * l_sup
                else:
                    logits = out
                    loss = criterion(logits.view(-1), labels.view(-1))
            
            if phase == 'train':
                loss.backward()
                optimizer.step()
                
        total_loss += loss.item()
        probs = torch.sigmoid(logits.view(-1)).detach().cpu().numpy()
        all_preds.extend(probs)
        all_labels.extend(labels.view(-1).cpu().numpy())
    
    elapsed_time = time.time() - start_time
    metrics = calculate_metrics(all_labels, all_preds)
    avg_loss = total_loss / len(loader)
    
    if return_details:
        return avg_loss, metrics, elapsed_time, (all_ids, all_labels, all_preds)
    
    return avg_loss, metrics, elapsed_time

# -------------------------
# 7. Grid Search & Runner
# -------------------------
def train_and_evaluate_config(model, train_df, val_df, args, params, device, criterion, global_tracker, allow_saving=True):
    curr_lr = params.get('lr', args.lr)
    curr_bs = params.get('batch_size', args.batch_size)
    curr_epochs = params.get('epochs', args.epochs)
    
    train_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(train_params, lr=curr_lr)
    
    best_local_metric = -float('inf')
    
    for ep in range(curr_epochs):
        if args.training_mode == 'cl_sft':
            if ep < (curr_epochs // 2): 
                # stage1: contrastive
                current_strategy = 'full_con'
                ds_mode = 'full_con' 
            else: 
                # stage2: SFT
                current_strategy = 'sft'
                ds_mode = 'sft'
        else:
            current_strategy = args.training_mode
            ds_mode = args.training_mode
            
        train_ds = UnifiedCodeDataset(train_df, model.tokenizer, args.test_only, ds_mode, max_length=args.max_length, is_train=True)
        train_loader = DataLoader(train_ds, batch_size=curr_bs, shuffle=True)
        
        val_mode = 'sft'
        val_ds = UnifiedCodeDataset(val_df, model.tokenizer, args.test_only, val_mode, max_length=args.max_length, is_train=False)
        val_loader = DataLoader(val_ds, batch_size=curr_bs, shuffle=False)

        _, t_metrics, _ = run_epoch(model, train_loader, optimizer, criterion, current_strategy, args.test_only, device, 'train')
        
        val_strategy = current_strategy
        _, v_metrics, _ = run_epoch(model, val_loader, optimizer, criterion, val_strategy, args.test_only, device, 'val')
        
        current_score = v_metrics.get('F1', 0.0)
            
        if current_score > best_local_metric:
            best_local_metric = current_score
            
        if allow_saving:
            if current_score > global_tracker['score']:
                print(f"  [New Global Best] Score improved: {global_tracker['score']:.4f} -> {current_score:.4f}. Saving model...")
                global_tracker['score'] = current_score
                save_trainable_only(model, global_tracker['path'])
                
    return best_local_metric

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--llama_model_path", type=str, required=True)
    parser.add_argument("--train_pair_path", type=str, required=True)
    parser.add_argument("--test_pair_path", type=str, required=True)
    parser.add_argument("--code_path", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="saved_models/")
    
    parser.add_argument("--batch_size", default=2, type=int)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--search_epochs", default=4, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--max_length", default=1024, type=int)
    parser.add_argument("--k_folds", default=5, type=int)
    
    parser.add_argument("--test_only", action='store_true', help="Single code mode")
    parser.add_argument("--training_mode", type=str, choices=['sft', 'con', 'im_con', 'full_con', 'cl_sft'], required=True)
    
    parser.add_argument("--project_split", action='store_true')
    
    args = parser.parse_args()
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)
    results_dir = os.path.join(args.save_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    
    # load data
    full_train_df = load_dataset(args.train_pair_path, args.code_path)
    test_df_raw = load_dataset(args.test_pair_path, args.code_path)
    
    model = UnifiedCodeModel(args.llama_model_path)
    model.to(device)
    
    pos_weight = torch.tensor([1.0]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    input_type = "Single" if args.test_only else "Concat"
    
    best_model_path = os.path.join(args.save_dir, f"best_{args.training_mode}_{input_type}.pth")
    global_tracker = {
        'score': -float('inf'),
        'path': best_model_path
    }
    
    param_grid = {
        'lr': [1e-4, 2e-4],
        'batch_size': [2, 4], 
    }
    
    keys, values = zip(*param_grid.items())
    param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    full_train_df = full_train_df.sample(frac=1, random_state=42).reset_index(drop=True)
    y_all = full_train_df['label'].values
    
    # ----------------------------------------------------
    # Phase 1: Grid Search
    # ----------------------------------------------------
    print(f"\n>>> Phase 1: Fast Grid Search (Epochs={args.search_epochs}, Single Split)")
    
    if args.project_split:
        if 'project' not in full_train_df.columns: raise ValueError("Project missing")
        groups = full_train_df['project'].values
        splitter = StratifiedGroupKFold(n_splits=5)
        train_idx, val_idx = next(splitter.split(full_train_df, y_all, groups=groups))
    else:
        train_idx, val_idx = train_test_split(np.arange(len(full_train_df)), test_size=0.2, stratify=y_all, random_state=42)
        
    search_train_df = full_train_df.iloc[train_idx]
    search_val_df = full_train_df.iloc[val_idx]
    
    best_search_score = -1
    best_params = None
    
    for i, params in enumerate(param_combinations):
        curr_params = params.copy()
        curr_params['epochs'] = args.search_epochs
        
        print(f"  > Checking Params: {curr_params}")
        model.reset_heads() 
        
        score = train_and_evaluate_config(
            model, search_train_df, search_val_df, args, curr_params, 
            device, criterion, global_tracker, allow_saving=False
        )
        print(f"    Result: F1 {score:.4f}")
        
        if score > best_search_score:
            best_search_score = score
            best_params = params
            
    print(f"\n>>> Grid Search Best Params: {best_params} (Search F1: {best_search_score:.4f})")

    # ----------------------------------------------------
    # Phase 2: Full CV
    # ----------------------------------------------------
    best_params['epochs'] = args.epochs
    print(f"\n>>> Phase 2: {args.k_folds}-Fold CV with Best Params (Epochs={args.epochs})")
    
    if args.project_split:
        groups = full_train_df['project'].values 
        splitter = StratifiedGroupKFold(n_splits=args.k_folds)
        split_generator = splitter.split(full_train_df, y_all, groups=groups)
    else:
        splitter = StratifiedKFold(n_splits=args.k_folds, shuffle=True, random_state=42)
        split_generator = splitter.split(np.zeros(len(y_all)), y_all)
        
    fold_f1s = []
    
    for fold, (train_idx, val_idx) in enumerate(split_generator):
        train_sub = full_train_df.iloc[train_idx]
        val_sub = full_train_df.iloc[val_idx]
        
        print(f"  Fold {fold+1}/{args.k_folds} ...", end=" ")
        model.reset_heads()
        
        f1 = train_and_evaluate_config(
            model, train_sub, val_sub, args, best_params, 
            device, criterion, global_tracker, allow_saving=True
        )
        fold_f1s.append(f1)
        print(f"Score: {f1:.4f}")
        
    avg_f1 = np.mean(fold_f1s)
    print(f"\n>>> Final CV Avg F1: {avg_f1:.4f}")
    
    # ----------------------------------------------------
    # Final Test
    # ----------------------------------------------------
    print(f">>> Loading Global Best Model from {global_tracker['path']}")
    if os.path.exists(global_tracker['path']):
        load_trainable_only(model, global_tracker['path'])

    test_ds_mode = 'sft' 
    test_strategy = 'sft' if args.training_mode == 'cl_sft' else args.training_mode
    
    test_loader = DataLoader(UnifiedCodeDataset(test_df_raw, model.tokenizer, args.test_only, test_ds_mode, max_length=args.max_length, is_train=False), batch_size=best_params['batch_size'], shuffle=False)
    
    # dummy optimizer
    dummy_opt = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-5)

    _, test_metrics, test_time, details = run_epoch(
        model, test_loader, dummy_opt, criterion, test_strategy, args.test_only, device, 'test', return_details=True
    )
    
    save_detailed_predictions(results_dir, args.training_mode, input_type, details[0], details[1], details[2], note="final_test_best")
    append_summary_table(results_dir, args.llama_model_path, args.training_mode, input_type, test_metrics, 0, test_time, note=f"Final Test Result")
    print(f"Final Test Results: Acc {test_metrics['Accuracy']:.4f} F1 {test_metrics['F1']:.4f}")

if __name__ == "__main__":
    main()