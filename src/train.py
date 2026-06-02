"""
DistilBERT + Our Proposed Method for Bengali Emotion Classification


Single experiment only:
1. Fine-tune teacher: bert-base-multilingual-cased
2. Train student: distilbert-base-multilingual-cased
3. Use Knowledge Distillation + Emotion-Aware Attention +
   Layer-wise Distillation + Dynamic Temperature Scaling +
   Contrastive Emotion Learning + Combined Multi-Loss Optimization


Dataset: UBMEC.csv with columns: text, classes
Labels: anger, disgust, fear, joy, sadness, surprise


Google Colab compatible. Upload UBMEC.csv to the Colab runtime before running.
"""


# =============================================================================
# Optional Colab package installation
# =============================================================================


import importlib.util
import subprocess
import sys




def install_if_missing(package_name, import_name=None):
    import_name = import_name or package_name
    if importlib.util.find_spec(import_name) is None:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", package_name]
        )




install_if_missing("transformers", "transformers")
install_if_missing("accelerate", "accelerate")
install_if_missing("scikit-learn", "sklearn")
install_if_missing("pandas", "pandas")
install_if_missing("matplotlib", "matplotlib")
install_if_missing("seaborn", "seaborn")
install_if_missing("tqdm", "tqdm")


# =============================================================================
# Imports
# =============================================================================


import os
import random
import warnings
from contextlib import nullcontext
from dataclasses import dataclass


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import (
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)


warnings.filterwarnings("ignore")




# =============================================================================
# Configuration
# =============================================================================


@dataclass
class Config:
    # Use "test" for a very fast Colab smoke test. Accuracy is not meaningful.
    # Use "quick" for a shorter sanity run with more data.
    # Use "live" for the full experiment reported in the paper/thesis.
    run_mode: str = "live"


    experiment_name: str = "DistilBERT + Our Proposed Method"
    dataset_path: str = "data/UBMEC.csv"


    teacher_model: str = "bert-base-multilingual-cased"
    student_model: str = "distilbert-base-multilingual-cased"


    max_length: int = 96
    batch_size: int = 8
    gradient_accumulation_steps: int = 2
    teacher_epochs: int = 3
    student_epochs: int = 8
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0


    alpha_hard: float = 0.40
    alpha_soft: float = 0.25
    alpha_layer: float = 0.15
    alpha_emotion: float = 0.12
    alpha_contrast: float = 0.08


    min_temperature: float = 1.0
    max_temperature: float = 4.0
    contrastive_temperature: float = 0.5


    test_size: float = 0.3
    random_state: int = 42
    num_workers: int = 0


    test_mode_train_samples: int = 120
    test_mode_test_samples: int = 60
    quick_mode_train_samples: int = 2500
    quick_mode_test_samples: int = 800

    teacher_save_path: str = "models/teacher_finetuned.pt"
    student_save_path: str = "models/distilbert_proposed.pt"

    results_csv_path: str = "results/distilbert_proposed_results.csv"
    confusion_matrix_path: str = "results/confusion_matrix_distilbert_proposed.png"


    def __post_init__(self):
        valid_modes = {"test", "quick", "live"}
        if self.run_mode not in valid_modes:
            raise ValueError(f"run_mode must be one of {valid_modes}")


        if self.run_mode == "test":
            self.max_length = 64
            self.batch_size = 16
            self.gradient_accumulation_steps = 1
            self.teacher_epochs = 1
            self.student_epochs = 1
        elif self.run_mode == "quick":
            self.max_length = 96
            self.batch_size = 8
            self.gradient_accumulation_steps = 2
            self.teacher_epochs = 1
            self.student_epochs = 2




config = Config()


LABELS = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]
LABEL2ID = {label: idx for idx, label in enumerate(LABELS)}
ID2LABEL = {idx: label for label, idx in LABEL2ID.items()}


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_amp = torch.cuda.is_available()




def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False




seed_everything(config.random_state)


print("=" * 80)
print(config.experiment_name)
print("=" * 80)
print(f"Run mode: {config.run_mode.upper()}")
if config.run_mode == "test":
    print(
        "Fast smoke test enabled: using tiny stratified subsets, "
        "1 teacher epoch, and 1 student epoch. "
        "This only checks whether the full code runs; accuracy is not meaningful."
    )
elif config.run_mode == "quick":
    print(
        "Quick sanity mode enabled: using larger stratified subsets, "
        "1 teacher epoch, and 2 student epochs. "
        "This is faster than live mode but still not a final-paper result."
    )
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(
        "GPU memory: "
        f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
    )
    torch.cuda.empty_cache()
print(f"Mixed precision enabled: {use_amp}")




# =============================================================================
# Dataset Loading
# =============================================================================


def load_ubmec_csv(path):
    encodings = ["utf-8", "utf-8-sig", "latin-1", "iso-8859-1"]
    last_error = None
    for encoding in encodings:
        try:
            data = pd.read_csv(path, encoding=encoding)
            print(f"Loaded {path} with encoding: {encoding}")
            return data
        except UnicodeDecodeError as exc:
            last_error = exc


    raise RuntimeError(f"Could not read {path}. Last error: {last_error}")




def stratified_subset(dataframe, label_column, max_samples, random_state):
    if max_samples is None or len(dataframe) <= max_samples:
        return dataframe.reset_index(drop=True)


    label_counts = dataframe[label_column].value_counts().sort_index()
    class_count = len(label_counts)
    per_class = max(1, max_samples // class_count)


    sampled_parts = []
    for label_value, count in label_counts.items():
        take_count = min(per_class, count)
        sampled_parts.append(
            dataframe[dataframe[label_column] == label_value].sample(
                n=take_count,
                random_state=random_state,
            )
        )


    subset = pd.concat(sampled_parts, axis=0)
    remaining = max_samples - len(subset)
    if remaining > 0:
        leftover = dataframe.drop(index=subset.index)
        if len(leftover) > 0:
            subset = pd.concat(
                [
                    subset,
                    leftover.sample(
                        n=min(remaining, len(leftover)),
                        random_state=random_state,
                    ),
                ],
                axis=0,
            )


    return subset.sample(frac=1.0, random_state=random_state).reset_index(drop=True)




df = load_ubmec_csv(config.dataset_path)
required_columns = {"text", "classes"}
missing_columns = required_columns.difference(df.columns)
if missing_columns:
    raise ValueError(
        f"UBMEC.csv must contain columns {required_columns}. "
        f"Missing: {missing_columns}. Found: {df.columns.tolist()}"
    )


df = df[["text", "classes"]].dropna().copy()
df["classes"] = df["classes"].astype(str).str.strip().str.lower()
df = df[df["classes"].isin(LABEL2ID)].copy()
df["label"] = df["classes"].map(LABEL2ID).astype(int)


if df.empty:
    raise ValueError("No valid rows found after filtering emotion labels.")


train_df, test_df = train_test_split(
    df,
    test_size=config.test_size,
    random_state=config.random_state,
    stratify=df["label"],
)
train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)


if config.run_mode == "test":
    train_df = stratified_subset(
        train_df,
        label_column="label",
        max_samples=config.test_mode_train_samples,
        random_state=config.random_state,
    )
    test_df = stratified_subset(
        test_df,
        label_column="label",
        max_samples=config.test_mode_test_samples,
        random_state=config.random_state,
    )
elif config.run_mode == "quick":
    train_df = stratified_subset(
        train_df,
        label_column="label",
        max_samples=config.quick_mode_train_samples,
        random_state=config.random_state,
    )
    test_df = stratified_subset(
        test_df,
        label_column="label",
        max_samples=config.quick_mode_test_samples,
        random_state=config.random_state,
    )


print(f"Total samples: {len(df)}")
print(f"Train samples: {len(train_df)}")
print(f"Test samples: {len(test_df)}")
print("\nClass distribution:")
print(df["classes"].value_counts().reindex(LABELS))




# =============================================================================
# Emotion-Aware Attention
# =============================================================================


EMOTION_KEYWORDS = {
    "anger": ["রাগ", "ক্রোধ", "বিরক্ত", "ক্ষোভ", "রাগান্বিত"],
    "disgust": ["ঘৃণা", "বিতৃষ্ণা", "জঘন্য", "বিরক্তিকর", "অরুচি"],
    "fear": ["ভয়", "ভয়", "আতঙ্ক", "শঙ্কা", "ভীত", "ডর"],
    "joy": ["খুশি", "আনন্দ", "হাসি", "সুখ", "উল্লাস", "ভালো"],
    "sadness": ["দুঃখ", "দুঃখিত", "কষ্ট", "বিষণ্ণ", "কান্না", "হতাশ"],
    "surprise": ["অবাক", "বিস্ময়", "বিস্ময়", "আশ্চর্য", "চমক"],
}
ALL_EMOTION_WORDS = sorted(
    {word for words in EMOTION_KEYWORDS.values() for word in words},
    key=len,
    reverse=True,
)




def build_emotion_mask(text, tokenizer, max_length):
    text_value = str(text)
    keyword_spans = []
    for keyword in ALL_EMOTION_WORDS:
        start = text_value.find(keyword)
        while start != -1:
            keyword_spans.append((start, start + len(keyword)))
            start = text_value.find(keyword, start + len(keyword))


    encoding = tokenizer(
        text_value,
        add_special_tokens=True,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_offsets_mapping=True,
    )


    offsets = encoding["offset_mapping"]
    mask = []
    for start, end in offsets:
        if start == end:
            mask.append(0)
            continue
        overlaps_emotion_span = any(
            token_start < end and start < token_end
            for token_start, token_end in keyword_spans
        )
        mask.append(1 if overlaps_emotion_span else 0)


    return mask[:max_length] + [0] * max(0, max_length - len(mask))




class EmotionAttentionLayer(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.score = nn.Linear(hidden_size, 1)
        self.projection = nn.Linear(hidden_size, hidden_size)


    def forward(self, hidden_states, attention_mask=None, emotion_mask=None):
        scores = self.score(hidden_states).squeeze(-1)


        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask == 0, -1e4)


        if emotion_mask is not None:
            scores = scores + emotion_mask.float() * 2.0


        weights = F.softmax(scores, dim=1)
        pooled = torch.bmm(weights.unsqueeze(1), hidden_states).squeeze(1)
        pooled = self.projection(pooled)
        return pooled, weights




# =============================================================================
# Layer-wise Distillation
# =============================================================================


class LayerWiseDistillationLoss(nn.Module):
    """
    DistilBERT has 6 transformer layers and mBERT has 12 transformer layers.
    Hidden states include the embedding output at index 0.


    Student layers 1..6 map to teacher layers 2,4,6,8,10,12.
    """


    def __init__(self, student_hidden_size, teacher_hidden_size, student_num_layers):
        super().__init__()
        self.projections = nn.ModuleList(
            [
                nn.Linear(student_hidden_size, teacher_hidden_size)
                for _ in range(student_num_layers)
            ]
        )


    def forward(self, student_hidden_states, teacher_hidden_states, attention_mask=None):
        student_layers = list(student_hidden_states[1:])
        teacher_layers = list(teacher_hidden_states[2::2])


        num_pairs = min(len(student_layers), len(teacher_layers), len(self.projections))
        if num_pairs == 0:
            return torch.tensor(0.0, device=student_hidden_states[0].device)


        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            denom = mask.sum(dim=1).clamp_min(1.0)
        else:
            mask = None
            denom = None


        losses = []
        for idx in range(num_pairs):
            student_state = student_layers[idx]
            teacher_state = teacher_layers[idx].detach()


            if mask is not None:
                student_pooled = (student_state * mask).sum(dim=1) / denom
                teacher_pooled = (teacher_state * mask).sum(dim=1) / denom
            else:
                student_pooled = student_state.mean(dim=1)
                teacher_pooled = teacher_state.mean(dim=1)


            student_projected = self.projections[idx](student_pooled)
            losses.append(F.mse_loss(student_projected, teacher_pooled))


        return torch.stack(losses).mean()




# =============================================================================
# Dynamic Temperature Scaling and Contrastive Emotion Learning
# =============================================================================


def calculate_dynamic_temperature(logits, min_temperature, max_temperature):
    with torch.no_grad():
        probabilities = F.softmax(logits.float(), dim=-1)
        confidence = probabilities.max(dim=-1).values
        difficulty = 1.0 - confidence
        temperature = min_temperature + (max_temperature - min_temperature) * difficulty
    return temperature.clamp(min_temperature, max_temperature)




class SupervisedContrastiveEmotionLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature


    def forward(self, embeddings, labels):
        embeddings = F.normalize(embeddings.float(), p=2, dim=1)
        labels = labels.view(-1, 1)
        batch_size = embeddings.size(0)


        similarity = torch.matmul(embeddings, embeddings.T) / self.temperature
        logits_mask = torch.ones_like(similarity) - torch.eye(
            batch_size, device=embeddings.device
        )
        positive_mask = (labels == labels.T).float() * logits_mask


        positive_count = positive_mask.sum(dim=1)
        valid_rows = positive_count > 0
        if valid_rows.sum() == 0:
            return torch.tensor(0.0, device=embeddings.device)


        similarity = similarity - similarity.max(dim=1, keepdim=True).values.detach()
        exp_logits = torch.exp(similarity) * logits_mask
        log_prob = similarity - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-8)


        mean_log_prob_pos = (positive_mask * log_prob).sum(dim=1) / positive_count.clamp_min(1.0)
        return -mean_log_prob_pos[valid_rows].mean()




# =============================================================================
# Student Model with Proposed Modules
# =============================================================================


class DistilBertProposedStudent(nn.Module):
    def __init__(self, model_name, num_labels):
        super().__init__()
        self.base = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.base.config.hidden_size
        self.num_hidden_layers = self.base.config.num_hidden_layers


        self.dropout = nn.Dropout(getattr(self.base.config, "dropout", 0.1))
        self.classifier = nn.Linear(self.hidden_size, num_labels)
        self.emotion_attention = EmotionAttentionLayer(self.hidden_size)
        self.emotion_classifier = nn.Linear(self.hidden_size, num_labels)
        self.contrastive_projection = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size, 256),
        )


    def forward(
        self,
        input_ids,
        attention_mask,
        emotion_mask=None,
        output_hidden_states=True,
    ):
        outputs = self.base(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )


        sequence_output = outputs.last_hidden_state
        cls_output = sequence_output[:, 0]


        logits = self.classifier(self.dropout(cls_output))


        emotion_pooled, emotion_weights = self.emotion_attention(
            sequence_output,
            attention_mask=attention_mask,
            emotion_mask=emotion_mask,
        )
        emotion_logits = self.emotion_classifier(self.dropout(emotion_pooled))
        embeddings = self.contrastive_projection(cls_output)


        return {
            "logits": logits,
            "emotion_logits": emotion_logits,
            "embeddings": embeddings,
            "emotion_weights": emotion_weights,
            "hidden_states": outputs.hidden_states,
        }




# =============================================================================
# Combined Multi-Loss Optimization
# =============================================================================


class ProposedDistillationLoss(nn.Module):
    def __init__(self, cfg, student_hidden_size, teacher_hidden_size, student_num_layers):
        super().__init__()
        self.cfg = cfg
        self.cross_entropy = nn.CrossEntropyLoss()
        self.kl_divergence = nn.KLDivLoss(reduction="batchmean")
        self.layer_distillation = LayerWiseDistillationLoss(
            student_hidden_size=student_hidden_size,
            teacher_hidden_size=teacher_hidden_size,
            student_num_layers=student_num_layers,
        )
        self.contrastive = SupervisedContrastiveEmotionLoss(
            temperature=cfg.contrastive_temperature
        )


    def forward(self, student_outputs, teacher_outputs, labels, attention_mask):
        student_logits = student_outputs["logits"]
        emotion_logits = student_outputs["emotion_logits"]
        teacher_logits = teacher_outputs.logits.detach()


        hard_loss = self.cross_entropy(student_logits, labels)
        emotion_loss = self.cross_entropy(emotion_logits, labels)


        temperatures = calculate_dynamic_temperature(
            student_logits,
            self.cfg.min_temperature,
            self.cfg.max_temperature,
        ).to(student_logits.device)
        temperature_column = temperatures.unsqueeze(1)


        soft_student = F.log_softmax(student_logits / temperature_column, dim=-1)
        soft_teacher = F.softmax(teacher_logits / temperature_column, dim=-1)
        soft_loss = self.kl_divergence(soft_student, soft_teacher)
        soft_loss = soft_loss * torch.mean(temperatures ** 2)


        layer_loss = self.layer_distillation(
            student_outputs["hidden_states"],
            teacher_outputs.hidden_states,
            attention_mask=attention_mask,
        )


        contrastive_loss = self.contrastive(student_outputs["embeddings"], labels)


        total_loss = (
            self.cfg.alpha_hard * hard_loss
            + self.cfg.alpha_soft * soft_loss
            + self.cfg.alpha_layer * layer_loss
            + self.cfg.alpha_emotion * emotion_loss
            + self.cfg.alpha_contrast * contrastive_loss
        )


        return total_loss, {
            "hard": hard_loss.detach().item(),
            "soft": soft_loss.detach().item(),
            "layer": layer_loss.detach().item(),
            "emotion": emotion_loss.detach().item(),
            "contrastive": contrastive_loss.detach().item(),
            "temperature": temperatures.mean().detach().item(),
        }




# =============================================================================
# Dataset and DataLoader
# =============================================================================


class EmotionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_length = max_length


    def __len__(self):
        return len(self.texts)


    def __getitem__(self, index):
        text = str(self.texts[index])
        label = int(self.labels[index])


        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        emotion_mask = build_emotion_mask(text, self.tokenizer, self.max_length)


        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "emotion_mask": torch.tensor(emotion_mask, dtype=torch.long),
            "labels": torch.tensor(label, dtype=torch.long),
        }




tokenizer = AutoTokenizer.from_pretrained(config.teacher_model, use_fast=True)
if not tokenizer.is_fast:
    raise ValueError("A fast tokenizer is required for emotion mask offset mapping.")


train_dataset = EmotionDataset(
    train_df["text"].values,
    train_df["label"].values,
    tokenizer,
    config.max_length,
)
test_dataset = EmotionDataset(
    test_df["text"].values,
    test_df["label"].values,
    tokenizer,
    config.max_length,
)


pin_memory = torch.cuda.is_available()
train_loader = DataLoader(
    train_dataset,
    batch_size=config.batch_size,
    shuffle=True,
    num_workers=config.num_workers,
    pin_memory=pin_memory,
)
test_loader = DataLoader(
    test_dataset,
    batch_size=config.batch_size,
    shuffle=False,
    num_workers=config.num_workers,
    pin_memory=pin_memory,
)




# =============================================================================
# Utilities
# =============================================================================


def move_batch_to_device(batch, target_device):
    return {key: value.to(target_device) for key, value in batch.items()}




def autocast_context():
    if use_amp:
        return torch.cuda.amp.autocast()
    return nullcontext()




def optimizer_step_count(loader_length, epochs, accumulation_steps):
    steps_per_epoch = int(np.ceil(loader_length / accumulation_steps))
    return max(1, steps_per_epoch * epochs)




def count_parameters(model):
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return total, trainable




def model_size_mb(model):
    total_bytes = 0
    for tensor in model.state_dict().values():
        total_bytes += tensor.numel() * tensor.element_size()
    return total_bytes / (1024 ** 2)




def format_params(parameter_count):
    return f"{parameter_count / 1_000_000:.2f}M"




def format_size(size_mb):
    return f"{size_mb:.2f}MB"




def evaluate_teacher(model, loader):
    model.eval()
    all_preds = []
    all_labels = []


    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating teacher", leave=False):
            batch = move_batch_to_device(batch, device)
            with autocast_context():
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    return_dict=True,
                )
            preds = torch.argmax(outputs.logits, dim=-1)
            all_preds.extend(preds.detach().cpu().numpy())
            all_labels.extend(batch["labels"].detach().cpu().numpy())


    accuracy = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return accuracy, macro_f1




def evaluate_student(model, loader):
    model.eval()
    all_preds = []
    all_labels = []


    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating student", leave=False):
            batch = move_batch_to_device(batch, device)
            with autocast_context():
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    emotion_mask=batch["emotion_mask"],
                    output_hidden_states=False,
                )
                logits = (outputs["logits"] + outputs["emotion_logits"]) / 2.0


            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.detach().cpu().numpy())
            all_labels.extend(batch["labels"].detach().cpu().numpy())


    accuracy = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    precision = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    recall = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "precision": precision,
        "recall": recall,
        "predictions": all_preds,
        "labels": all_labels,
    }




# =============================================================================
# Teacher Fine-tuning
# =============================================================================


def train_teacher(model, train_data_loader, eval_data_loader):
    print("\n" + "=" * 80)
    print("Step 1: Fine-tuning mBERT Teacher")
    print("=" * 80)


    model.to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    total_steps = optimizer_step_count(
        len(train_data_loader),
        config.teacher_epochs,
        config.gradient_accumulation_steps,
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * config.warmup_ratio),
        num_training_steps=total_steps,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)


    best_f1 = -1.0
    best_state = None


    for epoch in range(config.teacher_epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0


        progress = tqdm(
            enumerate(train_data_loader),
            total=len(train_data_loader),
            desc=f"Teacher epoch {epoch + 1}/{config.teacher_epochs}",
        )
        for step, batch in progress:
            batch = move_batch_to_device(batch, device)


            with autocast_context():
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                    return_dict=True,
                )
                loss = outputs.loss / config.gradient_accumulation_steps


            scaler.scale(loss).backward()
            running_loss += loss.detach().item() * config.gradient_accumulation_steps


            should_step = (
                (step + 1) % config.gradient_accumulation_steps == 0
                or (step + 1) == len(train_data_loader)
            )
            if should_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)


            progress.set_postfix(loss=running_loss / (step + 1))


        eval_acc, eval_f1 = evaluate_teacher(model, eval_data_loader)
        print(
            f"Teacher epoch {epoch + 1}: "
            f"eval accuracy={eval_acc * 100:.2f}, macro_f1={eval_f1 * 100:.2f}"
        )


        if eval_f1 > best_f1:
            best_f1 = eval_f1
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            torch.save(best_state, config.teacher_save_path)
            print(f"Saved best teacher to {config.teacher_save_path}")


        if torch.cuda.is_available():
            torch.cuda.empty_cache()


    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)


    print(f"Best teacher macro F1: {best_f1 * 100:.2f}")
    return model




# =============================================================================
# Student Training with Full Proposed Method
# =============================================================================


def train_student(student, teacher, train_data_loader, eval_data_loader):
    print("\n" + "=" * 80)
    print("Step 2: Training DistilBERT Student with Full Proposed Method")
    print("=" * 80)


    student.to(device)
    teacher.to(device)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad = False


    teacher_hidden_size = teacher.config.hidden_size
    criterion = ProposedDistillationLoss(
        cfg=config,
        student_hidden_size=student.hidden_size,
        teacher_hidden_size=teacher_hidden_size,
        student_num_layers=student.num_hidden_layers,
    ).to(device)


    optimizer = AdamW(
        student.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    total_steps = optimizer_step_count(
        len(train_data_loader),
        config.student_epochs,
        config.gradient_accumulation_steps,
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * config.warmup_ratio),
        num_training_steps=total_steps,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)


    best_f1 = -1.0
    best_state = None
    best_results = None


    for epoch in range(config.student_epochs):
        student.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        running_temp = 0.0


        progress = tqdm(
            enumerate(train_data_loader),
            total=len(train_data_loader),
            desc=f"Student epoch {epoch + 1}/{config.student_epochs}",
        )
        for step, batch in progress:
            batch = move_batch_to_device(batch, device)


            with autocast_context():
                student_outputs = student(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    emotion_mask=batch["emotion_mask"],
                    output_hidden_states=True,
                )


                with torch.no_grad():
                    teacher_outputs = teacher(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        output_hidden_states=True,
                        return_dict=True,
                    )


                loss, loss_items = criterion(
                    student_outputs=student_outputs,
                    teacher_outputs=teacher_outputs,
                    labels=batch["labels"],
                    attention_mask=batch["attention_mask"],
                )
                loss = loss / config.gradient_accumulation_steps


            scaler.scale(loss).backward()
            running_loss += loss.detach().item() * config.gradient_accumulation_steps
            running_temp += loss_items["temperature"]


            should_step = (
                (step + 1) % config.gradient_accumulation_steps == 0
                or (step + 1) == len(train_data_loader)
            )
            if should_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(student.parameters(), config.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)


            progress.set_postfix(
                loss=running_loss / (step + 1),
                temp=running_temp / (step + 1),
            )


        results = evaluate_student(student, eval_data_loader)
        print(
            f"Student epoch {epoch + 1}: "
            f"accuracy={results['accuracy'] * 100:.2f}, "
            f"macro_f1={results['macro_f1'] * 100:.2f}, "
            f"precision={results['precision'] * 100:.2f}, "
            f"recall={results['recall'] * 100:.2f}"
        )


        if results["macro_f1"] > best_f1:
            best_f1 = results["macro_f1"]
            best_results = results
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in student.state_dict().items()
            }
            torch.save(best_state, config.student_save_path)
            print(f"Saved best student to {config.student_save_path}")


        if torch.cuda.is_available():
            torch.cuda.empty_cache()


    if best_state is not None:
        student.load_state_dict(best_state)
        student.to(device)


    if best_results is None:
        best_results = evaluate_student(student, eval_data_loader)


    print(f"Best student macro F1: {best_f1 * 100:.2f}")
    return student, best_results




# =============================================================================
# Main Execution
# =============================================================================


teacher_model = AutoModelForSequenceClassification.from_pretrained(
    config.teacher_model,
    num_labels=len(LABELS),
    id2label=ID2LABEL,
    label2id=LABEL2ID,
)


student_model = DistilBertProposedStudent(
    model_name=config.student_model,
    num_labels=len(LABELS),
)


teacher_total_params, teacher_trainable_params = count_parameters(teacher_model)
student_total_params, student_trainable_params = count_parameters(student_model)


print("\nInitial model information:")
print(
    f"Teacher total parameters: {format_params(teacher_total_params)} | "
    f"trainable: {format_params(teacher_trainable_params)} | "
    f"size: {format_size(model_size_mb(teacher_model))}"
)
print(
    f"Student total parameters: {format_params(student_total_params)} | "
    f"trainable: {format_params(student_trainable_params)} | "
    f"size: {format_size(model_size_mb(student_model))}"
)


teacher_model = train_teacher(teacher_model, train_loader, test_loader)
student_model, final_results = train_student(
    student_model,
    teacher_model,
    train_loader,
    test_loader,
)




# =============================================================================
# Final Metrics, Reports, and Saved Outputs
# =============================================================================


final_labels = final_results["labels"]
final_predictions = final_results["predictions"]


final_accuracy = accuracy_score(final_labels, final_predictions)
final_macro_f1 = f1_score(final_labels, final_predictions, average="macro", zero_division=0)
final_precision = precision_score(
    final_labels,
    final_predictions,
    average="macro",
    zero_division=0,
)
final_recall = recall_score(
    final_labels,
    final_predictions,
    average="macro",
    zero_division=0,
)


student_total_params, student_trainable_params = count_parameters(student_model)
student_size = model_size_mb(student_model)


print("\n" + "=" * 80)
print("Final Metrics: DistilBERT + Our Proposed Method")
print("=" * 80)
print(f"Accuracy: {final_accuracy * 100:.2f}")
print(f"Macro F1: {final_macro_f1 * 100:.2f}")
print(f"Precision: {final_precision * 100:.2f}")
print(f"Recall: {final_recall * 100:.2f}")


print("\nClassification Report:")
print(
    classification_report(
        final_labels,
        final_predictions,
        labels=list(range(len(LABELS))),
        target_names=LABELS,
        digits=4,
        zero_division=0,
    )
)


cm = confusion_matrix(
    final_labels,
    final_predictions,
    labels=list(range(len(LABELS))),
)
print("Confusion Matrix:")
print(cm)


print("\nModel Information:")
print(f"Total parameters: {student_total_params:,} ({format_params(student_total_params)})")
print(
    f"Trainable parameters: "
    f"{student_trainable_params:,} ({format_params(student_trainable_params)})"
)
print(f"Model size in MB: {student_size:.2f}")


plt.figure(figsize=(9, 7))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=LABELS,
    yticklabels=LABELS,
)
plt.title("Confusion Matrix - DistilBERT Proposed Method")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig(config.confusion_matrix_path, dpi=300, bbox_inches="tight")
plt.show()


comparison_rows = [
    {
        "Student Model": "DistilBERT",
        "Method": "Base Paper KD",
        "Accuracy": 46.50,
        "Macro F1": 40.99,
        "Params": "66M",
        "Size": "254MB",
    },
    {
        "Student Model": "DistilBERT",
        "Method": "Our Proposed Method",
        "Accuracy": round(final_accuracy * 100, 2),
        "Macro F1": round(final_macro_f1 * 100, 2),
        "Params": format_params(student_total_params),
        "Size": format_size(student_size),
    },
]
comparison_df = pd.DataFrame(comparison_rows)


per_class_report = classification_report(
    final_labels,
    final_predictions,
    labels=list(range(len(LABELS))),
    target_names=LABELS,
    digits=4,
    zero_division=0,
    output_dict=True,
)


results_summary_df = pd.DataFrame(
    [
        {
            "experiment": config.experiment_name,
            "accuracy": final_accuracy,
            "macro_f1": final_macro_f1,
            "precision_macro": final_precision,
            "recall_macro": final_recall,
            "total_params": student_total_params,
            "trainable_params": student_trainable_params,
            "model_size_mb": student_size,
            "teacher_model": config.teacher_model,
            "student_model": config.student_model,
            "train_samples": len(train_df),
            "test_samples": len(test_df),
        }
    ]
)


results_summary_df.to_csv(config.results_csv_path, index=False)
pd.DataFrame(per_class_report).transpose().to_csv(
    "results/distilbert_proposed_classification_report.csv",
    index=True,
)


print("\n" + "=" * 80)
print("Final Comparison Table")
print("=" * 80)
if config.run_mode != "live":
    print(
        "NOTE: Current run is not LIVE mode, so Accuracy and Macro F1 are only "
        "for code/pipeline checking and should not be compared with the base paper."
    )
if student_total_params > 100_000_000:
    print(
        "NOTE: distilbert-base-multilingual-cased has about 135M parameters. "
        "The base-paper 66M DistilBERT row is a smaller DistilBERT variant, "
        "so the size column is not an exact same-size comparison."
    )
print(comparison_df.to_string(index=False))


print("\nSaved outputs:")
print(f"- {config.teacher_save_path}")
print(f"- {config.student_save_path}")
print(f"- {config.results_csv_path}")
print(f"- {config.confusion_matrix_path}")
print("- distilbert_proposed_classification_report.csv")
