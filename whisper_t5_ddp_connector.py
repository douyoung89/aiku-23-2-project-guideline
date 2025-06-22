import os 
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
import pandas as pd 
from transformers import WhisperModel, T5ForConditionalGeneration, WhisperFeatureExtractor, get_linear_schedule_with_warmup
import matplotlib.pyplot as plt
import jiwer
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import time 
from transformers import T5TokenizerFast 
import json 

from qformer import QFormer
from ste import SpeechEncoder
# 사용 가능한 GPU 설정
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4"

# ============================================
# DDP 설정 함수
# ============================================
def setup_ddp(rank, world_size):
    """
    DDP 모드 초기화
    """
    # 모든 프로세스가 보는 포트 설정
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12357'  # 이전과 다른 포트 사용
    
    # 프로세스 그룹 초기화
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    
    # 각 프로세스를 특정 GPU에 할당
    torch.cuda.set_device(rank)
    
    print(f"DDP 설정 완료: rank {rank}/{world_size}")

def cleanup_ddp():
    """
    DDP 모드 정리
    """
    dist.destroy_process_group()

# ============================================
# Dataset Class Definition
# ============================================
class WhisperT5Dataset(Dataset):
    def __init__(self, df, sampling_rate=16000, max_length=480000):
        self.df = df.reset_index(drop=True)
        self.audio_path = df["audio_path"].tolist()
        self.audio_id = df["id"].tolist()
        self.text = df["standard_form"].tolist()
        self.start = df["start"].tolist()
        self.end = df["end"].tolist()
        self.sampling_rate = sampling_rate
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        path = self.audio_path[idx]
        start_sec = self.start[idx]
        end_sec = self.end[idx]

        # 초 -> 샘플 단위
        start_sample = int(start_sec * self.sampling_rate)
        num_samples = int((end_sec - start_sec) * self.sampling_rate)
        num_samples = min(num_samples, self.max_length)
        
        if num_samples <= 0:
            print(f"[Warning] Invalid duration: start={start_sec}, end={end_sec} at index {idx}")
            waveform = torch.zeros(self.sampling_rate, dtype=torch.float32)
            sr = self.sampling_rate
        else : 
            try:
                waveform, sr = torchaudio.load(path, frame_offset=start_sample, num_frames=num_samples)
                if sr != self.sampling_rate:
                    resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sampling_rate)
                    waveform = resampler(waveform)

                # 모노 채널로 변환
                if waveform.shape[0] > 1:
                    waveform = waveform.mean(dim=0)
                waveform = waveform.squeeze(0)

                # 빈 오디오 처리
                if waveform.numel() == 0:
                    print(f"[Warning] Empty audio at index {idx}")
                    waveform = torch.zeros(self.sampling_rate, dtype=torch.float32)

            except Exception as e:
                print(f"[Error] Failed to load audio at index {idx}: {e}")
                waveform = torch.zeros(self.sampling_rate, dtype=torch.float32)

        return {
            "audio": waveform,  # (T,)
            "labels": self.text[idx],
            "audio_id": self.audio_id[idx],
            "start": start_sec,
            "end": end_sec
        }
        
# ============================================
# Collate Function for DataLoader
# ============================================
def collate_fn(batch):
    """
    batch: list of items, 각 아이템은 __getitem__에서 반환된 dict
    """
    # 각 배치의 오디오 텐서를 모음
    audios = [item["audio"] for item in batch]
    labels = [item["labels"] for item in batch]
    audio_ids = [item["audio_id"] for item in batch]
    starts = [item["start"] for item in batch]
    ends = [item["end"] for item in batch]
    
    padded_tensor = pad_sequence(audios, 
                                 batch_first=True)

    return {
        "audio": padded_tensor,   # [batch, max_seq_len]
        "label": labels, 
        "audio_id": audio_ids,    # [batch]
        "start": torch.tensor(starts, dtype=torch.float),
        "end": torch.tensor(ends, dtype=torch.float)
    }
        

# ============================================
# Model Definition
# ============================================
class WhisperT5Model(nn.Module):
    def __init__(self, whisper_model_name="openai/whisper-base",    pretrained_model_name="t5-base", device="cuda", connector="qformer", hidden_size=512):
        super().__init__()
        self.device = device
        
        self.whisper = WhisperModel.from_pretrained(whisper_model_name)
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(whisper_model_name)
        
        self.t5 = T5ForConditionalGeneration.from_pretrained(pretrained_model_name)
        from transformers import T5Tokenizer, T5TokenizerFast 
        self.tokenizer = T5TokenizerFast.from_pretrained(pretrained_model_name, use_fast=False)
        
        # whisper output dim 
        whisper_dim = self.whisper.config.d_model
        t5_dim = self.t5.config.d_model
        
        if connector == 'mlp':
            self.connector = nn.Sequential(
                nn.Linear(whisper_dim, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_size, t5_dim)
            )
        if connector == 'qformer':
            self.connector = nn.Sequential(
                nn.Linear(whisper_dim, 256),
                QFormer(num_layers=6, num_queries=100, hidden_dim=256, num_heads=4, ffn_dim=256),
                nn.Linear(256, t5_dim),
            )
        if connector == 'ste':
            self.connector = SpeechEncoder(speech_embedding_dim=whisper_dim, hidden_dim=256, sampling_output_dim=256, output_dim=t5_dim, num_layers=6, num_heads=4)

        # # === FREEZE–FREEZE 설정 ===
        # 1) Whisper 파라미터 전부 동결
        # for param in self.whisper.parameters():
        #     param.requires_grad = False

        # 2) T5 파라미터 전부 동결
        # for param in self.t5.parameters():
        #     param.requires_grad = False
            
        # self.transformer = nn.TransformerEncoder(
        #     nn.TransformerEncoderLayer(d_model=whisper_dim, nhead=4),
        #     num_layers=1
        # )
        
        # # Projection layer (trainable)
        # self.projection = nn.Linear(whisper_dim, t5_dim)
       
        # for param in self.t5.parameters():
        #     param.requires_grad = False
        
        # # Freeze T5 decoder parameters except for layernorms
        # for name, param in self.t5.named_parameters():
        #     if "layer_norm" not in name:
        #         param.requires_grad = False
                
        # for param in self.t5.lm_head.parameters():
        #     param.requires_grad = True
                
        # # for param in self.t5.parameters():
        # #     param.requires_grad = False
                
    
    def forward(self, audio, labels=None):
        """
        audio : tensor (batch, max_len) 
        labels : text string 
        """
        
        if isinstance(audio, torch.Tensor):
            # GPU에 있는 텐서를 CPU로 옮기고 numpy로 변환
            audio = audio.cpu().numpy()
        
        # 오디오 특성 추출
        input_features = self.feature_extractor(
            audio,
            sampling_rate=16000,
            return_tensors="pt",
            ).input_features.to(self.device)
        
        # Whisper encoder output
        encoder_outputs = self.whisper.encoder(input_features)
        encoder_hidden_states = encoder_outputs.last_hidden_state
        
        # Trainable projection layer
        # encoder_hidden_states = self.transformer(encoder_hidden_states)
        projected_features = self.connector(encoder_hidden_states)
        
        if labels is not None:
            # 텍스트 레이블을 토큰 ID로 변환
            if isinstance(labels[0], str):
                # 문자열 입력인 경우 토큰화
                # print(f"label : {labels}") 
                label_ids = self.tokenizer(labels, padding=True, return_tensors="pt").input_ids.to(self.device)
            else:
                # 이미 토큰화된 경우
                label_ids = labels
                
            # 학습 모드 (손실 계산)
            outputs = self.t5(
                inputs_embeds=projected_features,
                labels=label_ids,
                return_dict=True
            )
            return outputs  # outputs.loss, outputs.logits 포함
        else:
            # 추론 모드 (예측)
            outputs = self.t5(
                inputs_embeds=projected_features,
                return_dict=True
            )
            return outputs.logits
        
    def generate(self, audio, max_length=100, num_beams=4):
        """텍스트 생성 메서드"""
        if isinstance(audio, torch.Tensor):
            audio = audio.cpu().numpy()
        
        # 오디오 특성 추출
        input_features = self.feature_extractor(
            audio,
            sampling_rate=16000,
            return_tensors="pt",
        ).input_features.to(self.device)
        
        # Whisper 인코더
        encoder_outputs = self.whisper.encoder(input_features)
        encoder_hidden_states = encoder_outputs.last_hidden_state
        
        # 프로젝션
        # encoder_hidden_states = self.transformer(encoder_hidden_states)
        projected_features = self.connector(encoder_hidden_states)
        
        # T5 생성
        generated_ids = self.t5.generate(
            inputs_embeds=projected_features,
            max_length=max_length,
            num_beams=num_beams,
            early_stopping=True,
            repetition_penalty=2.0, 
            return_dict_in_generate=False
        )
        
        # 결과 디코딩
        generated_texts = self.tokenizer.batch_decode(
            generated_ids, 
            skip_special_tokens=True
        )
        
        return generated_texts
        
# ============================================
# Training and Validation Functions for DDP
# ============================================
def train_epoch(model, dataloader, optimizer, scheduler, device, rank):
    model.train()
    total_loss = 0
    
    if rank == 0:
        progress_bar = tqdm(dataloader, desc=f"Training on GPU {rank}")
    else:
        progress_bar = dataloader
        
    for i, batch in enumerate(progress_bar):
        audio, labels = batch["audio"].to(device), batch["label"]
        
        # Forward pass
        outputs = model(audio=audio, labels=labels)
        
        # loss 
        loss = outputs.loss
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        
        # tqdm - 랭크 0만 표시 
        if rank == 0 and isinstance(progress_bar, tqdm):
            progress_bar.set_postfix({"loss": loss.item()})
            
        # 매 배치마다 명시적으로 메모리 해제
        # del audio, labels, loss, outputs 
        # torch.cuda.empty_cache()
        # torch.cuda.ipc_collect()  # CUDA 이벤트도 정리
    
    # 모든 프로세스에서의 평균 손실 계산
    avg_loss = total_loss / len(dataloader)
    dist.all_reduce(torch.tensor(avg_loss).to(device), op=dist.ReduceOp.SUM)
    avg_loss = avg_loss / dist.get_world_size()
    
    return avg_loss

def clear_gpu_cache():
    """GPU 캐시를 명시적으로 비웁니다."""
    torch.cuda.empty_cache()
    # CUDA 이벤트 동기화
    torch.cuda.synchronize()
    
def validate(model, dataloader, device, rank):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        if rank == 0:
            progress_bar = tqdm(dataloader, desc=f"Validating on GPU {rank}")
        else:
            progress_bar = dataloader
            
        for batch in progress_bar:
            # 데이터 준비
            audio, labels = batch["audio"].to(device), batch["label"]
            outputs = model(audio=audio, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
            
            # tqdm - 랭크 0만 표시
            if rank == 0 and isinstance(progress_bar, tqdm):
                progress_bar.set_postfix({"loss": loss.item()})
    
    # 모든 프로세스에서의 평균 손실 계산
    avg_loss = total_loss / len(dataloader)
    dist.all_reduce(torch.tensor(avg_loss).to(device), op=dist.ReduceOp.SUM)
    avg_loss = avg_loss / dist.get_world_size()
    
    return avg_loss

def compute_metrics(predictions, references):
    """WER과 CER 평가 지표 계산"""
    metrics = {}
    
    # WER (Word Error Rate)
    wer_score = jiwer.wer(references, predictions)
    metrics["wer"] = wer_score
    
    # CER (Character Error Rate)
    cer_score = jiwer.cer(references, predictions)
    metrics["cer"] = cer_score

    print(predictions, references)
    input()
    
    return metrics

def evaluate_model(model, dataloader, device, rank):
    """모델 평가 함수 - rank 0만 실행"""
    model.eval()
    all_predictions = []
    all_references = []
    
    # if rank == 0 : 
    clear_gpu_cache()
    
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(dataloader, desc=f"Evaluating on GPU {rank}")):
            # 데이터 준비
            audio = batch["audio"].to(device)
            labels = batch["label"]
            
            # 예측
            pred_texts = model.module.generate(audio=audio) 
            # 예측 및 참조 저장
            all_predictions.extend(pred_texts)
            all_references.extend(labels)
            
            if (idx + 1) % 5 == 0 and rank == 0:
                clear_gpu_cache()
    
    # 평가 지표 계산
    metrics = compute_metrics(all_predictions, all_references)
    
    # 결과 출력 (rank 0만)
    if rank == 0:
        clear_gpu_cache()
        print("\n===== 평가 결과 =====")
        print(f"WER: {metrics['wer']:.4f} (낮을수록 좋음)")
        print(f"CER: {metrics['cer']:.4f} (낮을수록 좋음)")
        
        # 샘플 출력
        print("\n===== 예측 결과 샘플 =====")
        for i in range(min(5, len(all_predictions))):
            print(f"정답: {all_references[i]}")
            print(f"예측: {all_predictions[i]}")
            print(f"WER: {jiwer.wer([all_references[i]], [all_predictions[i]]):.4f}")
            print(f"CER: {jiwer.cer([all_references[i]], [all_predictions[i]]):.4f}")
            print("---")
    
    return all_predictions, all_references, metrics


def plot_losses(num_epochs, train_losses, val_losses, val_metrics, model_save_path):
    """
    학습 및 검증 손실을 그래프로 시각화
    """
    # 길이 동기화 (차이가 있을 경우 0으로 채움)
    max_len = max(len(train_losses), len(val_losses))
    train_losses += [0] * (max_len - len(train_losses))
    val_losses += [0] * (max_len - len(val_losses))

    plt.figure(figsize=(15, 5))

    # 1. 손실 그래프
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)

    # 2. WER & CER 그래프
    plt.subplot(1, 2, 2)
    if len(val_metrics['wer']) > 0:
        plt.plot(range(1, len(val_metrics['wer']) + 1), val_metrics['wer'], label='WER', marker='o')
        plt.plot(range(1, len(val_metrics['cer']) + 1), val_metrics['cer'], label='CER', marker='x')
    plt.xlabel('Epochs')
    plt.ylabel('Error Rate')
    plt.title('WER and CER (lower is better)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(model_save_path, 'training_metrics_ddp.png'))
    print("Loss visualization saved successfully.")
    
# ============================================
# DDP 훈련 함수
# ============================================
def train_ddp(rank, world_size, args):
    print(f"Process starting with rank {rank}, world_size {world_size}")
    
    # DDP 설정
    try:
        setup_ddp(rank, world_size)
        print(f"DDP setup completed for rank {rank}")
    except Exception as e:
        print(f"Error in DDP setup for rank {rank}: {e}")
        return  # 오류 발생 시 즉시 종료
    
    # DDP 설정
    #setup_ddp(rank, world_size)
    
    # 설정 불러오기
    train_csv = args.train_csv
    valid_csv = args.valid_csv
    test_csv = args.test_csv
    num_epochs = args.num_epochs
    learning_rate = args.learning_rate
    warmup_ratio = args.warmup_ratio
    weight_decay = args.weight_decay
    batch_size = args.batch_size
    model_save_path = args.model_save_path
    connector = args.connector
    
    if rank == 0 and not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    
    # 각 프로세스는 자신의 GPU에 할당
    device = torch.device(f"cuda:{rank}")
    
    # 데이터셋 로드
    if rank == 0:
        print("데이터셋 로드 중...")
        
    train_df = pd.read_csv(train_csv)
    train_df = train_df[train_df["isDialect"] == True]
    
    valid_df = pd.read_csv(valid_csv)
    valid_df = valid_df[valid_df["isDialect"] == True]
    
    test_df = pd.read_csv(test_csv)
    test_df = test_df[test_df["isDialect"] == True]
    
    if rank == 0:
        print(f"Train Dataset: {len(train_df)}, Valid Dataset: {len(valid_df)}, Test Dataset: {len(test_df)}")
    
    # 데이터셋 생성
    train_dataset = WhisperT5Dataset(df=train_df)
    val_dataset = WhisperT5Dataset(df=valid_df)
    test_dataset = WhisperT5Dataset(df=test_df)
    
    # DDP용 DistributedSampler 생성
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )
    
    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False
    )
    
    test_sampler = DistributedSampler(
        test_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False  # 평가시에는 셔플하지 않음
    )
    
    # DataLoader 생성
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        sampler=train_sampler,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        sampler=val_sampler,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=8, # test 시 단일 gpu 사용 
        sampler=test_sampler,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )
    
    # 모델 초기화
    t5_model_name = "paust/pko-t5-base" # google-t5/t5-base, KETI-AIR/ke-t5-bas
    model = WhisperT5Model(
        whisper_model_name="openai/whisper-base", 
        pretrained_model_name=t5_model_name, 
        device=device,
        connector=connector
    )
    model = model.to(device)
    
    # DDP 래핑
    model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)
    
    # 옵티마이저 및 스케줄러
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # 스케줄러
    total_steps = len(train_loader) * num_epochs
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    
    # 학습 지표 초기화
    train_losses = []
    val_losses = []
    val_metrics = {
        'wer': [],
        'cer': []
    }
    best_val_loss = float('inf')
    
    # 훈련 시작
    for epoch in range(num_epochs):
        # Sampler의 에폭 설정 (중요)
        train_sampler.set_epoch(epoch)
        
        if rank == 0:
            print(f"Epoch {epoch+1}/{num_epochs}")
        
        # 훈련
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device, rank)
        
        # 검증
        val_loss = validate(model, val_loader, device, rank)
        
        # 랭크 0만 출력 및 저장
        if rank == 0:
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            print(f"Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")
            
            # 평가 지표 계산 (랭크 0만)
            if (epoch + 1) % 100 == 0 or epoch == num_epochs - 1:
                _, _, metrics = evaluate_model(model, val_loader, device, rank)
                for k, v in metrics.items():
                    if epoch >= len(val_metrics[k]):
                        val_metrics[k].append(v)
                    else:
                        val_metrics[k][epoch] = v
                
                print(f"WER: {metrics['wer']:.4f}, CER: {metrics['cer']:.4f}")
            else:
                # 평가하지 않는 에폭에는 이전 값 유지
                if val_metrics['wer'] and len(val_metrics['wer']) > 0:
                    for k in val_metrics.keys():
                        if epoch >= len(val_metrics[k]):
                            val_metrics[k].append(val_metrics[k][-1])
                else:
                    # 첫 에폭이라면 0으로 초기화
                    for k in val_metrics.keys():
                        val_metrics[k].append(0)
            
            # 최고 모델 저장
            print(f"Current val loss: {val_loss:.4} \nBest val loss : {best_val_loss}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                
                # DDP 모델의 경우 model.module로 접근하여 state_dict 저장
                torch.save({
                    'epoch': epoch+1,
                    'model_state_dict': model.module.state_dict(),  # DDP에서는 .module로 접근
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'metrics': {k: v[-1] for k, v in val_metrics.items() if len(v) > 0}
                }, os.path.join(model_save_path, f"whisper_t5_best_loss_ddp.pt"))
                
                print(f"Best model (loss) saved with validation loss: {val_loss:.4f}")
                
    
    
            # 시각화
            print(f"Epoch {epoch+1} : Saving loss visualization ===========")
            plot_losses(num_epochs, train_losses, val_losses, val_metrics, model_save_path)
            print(f"Epoch {epoch+1} : Saved loss visualization!! ===========")
    
    
    # 학습 종료 
    # 모든 프로세스에서 DDP 정리
    dist.barrier()  # 모든 프로세스 동기화
    cleanup_ddp()
    
    # DDP 정리 후 랭크 0에서만 단일 GPU 평가 수행
    if rank == 0:
        print("\n최종 모델 평가 중 (단일 GPU)...")
        best_model_path = os.path.join(model_save_path, "whisper_t5_best_loss_ddp.pt")
        checkpoint = torch.load(best_model_path)
        
        # 단일 GPU 모델 생성 및 가중치 로드
        device = torch.device("cuda:0")  # 첫 번째 GPU 사용
        eval_model = WhisperT5Model(
            whisper_model_name="openai/whisper-base", 
            pretrained_model_name=t5_model_name, 
            device=device
        )
        eval_model.load_state_dict(checkpoint['model_state_dict'])
        eval_model = eval_model.to(device)
        
        # 단일 GPU 평가 함수
        def evaluate_single_gpu(model, dataloader, device, save_path):
            model.eval()
            predictions = []
            references = []
            
            with torch.no_grad():
                for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
                    # 디버깅 출력
                    # print(f"배치 {batch_idx+1}/{len(dataloader)} 처리 중...")
                    
                    # 데이터 준비
                    sub_audio = batch["audio"].to(device)
                    sub_labels = batch["label"]
                    
                    # 더 작은 서브배치로 나누어 처리
                    # 디버깅 출력
                    # print(f"  서브배치 {i//sub_batch_size+1}/{(audio.size(0)+sub_batch_size-1)//sub_batch_size} 처리 중...")
                    
                    try:
                        # 생성 시간 제한
                        # start_time = time.time()
                        pred_texts = model.generate(
                            audio=sub_audio,
                            max_length=100,
                            num_beams=4
                        )
                        # print(f"  생성 시간: {time.time() - start_time:.2f}초")
                        
                        predictions.extend(pred_texts)
                        references.extend(sub_labels)
                        
                        # 중간 결과 출력
                        # for idx, (pred, ref) in enumerate(zip(pred_texts, sub_labels)):
                        #     print(f"    샘플 {idx+1}:")
                        #     print(f"      참조: {ref}")
                        #     print(f"      예측: {pred}")
                        #     print(f"      WER: {jiwer.wer([ref], [pred]):.4f}")
                    except Exception as e:
                        print(f"  생성 오류: {e}")
                        # 오류 발생 시 빈 문자열 추가
                        predictions.extend([""] * len(sub_labels))
                        references.extend(sub_labels)
            
            # 평가 지표 계산
            metrics = compute_metrics(predictions, references)
            
            print("\n===== 평가 결과 =====")
            print(f"WER: {metrics['wer']:.4f} (낮을수록 좋음)")
            print(f"CER: {metrics['cer']:.4f} (낮을수록 좋음)")
            
            # 결과 출력
            print("\n===== 예측 결과 샘플 =====")
            for i in range(min(5, len(predictions))):
                print(f"정답: {references[i]}")
                print(f"예측: {predictions[i]}")
                print(f"WER: {jiwer.wer([references[i]], [predictions[i]]):.4f}")
                print(f"CER: {jiwer.cer([references[i]], [predictions[i]]):.4f}")
                print("---")
            
            # 평가 결과 저장 디렉토리 생성
            os.makedirs(save_path, exist_ok=True)

            # CSV로 결과 저장
            results_file = os.path.join(save_path, "evaluation_results.csv")
            with open(results_file, "w", encoding="utf-8") as f:
                f.write("reference,prediction\n")
                for ref, pred in zip(references, predictions):
                    f.write(f"{ref},{pred}\n")
            print(f"Prediction and Reference saved to: {results_file}")

            # 메트릭 저장 (JSON 형식)
            metrics_file = os.path.join(save_path, "evaluation_metrics.json")
            with open(metrics_file, "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=4)
            print(f"Metrics saved to: {metrics_file}")
            
            return predictions, references, metrics
        
        # 테스트 세트에서 모델 평가
        print("\n=============== For Test Set ====================")
        predictions, references, metrics = evaluate_single_gpu(eval_model, test_loader, device, save_path=model_save_path)
        

# ============================================
# Main Function
# ============================================
def main() : 
    import argparse
    
    parser = argparse.ArgumentParser(description="WhisperT5 모델 DDP 학습")
    parser.add_argument("--train_csv", type=str, default="/home/aikusrv02/dialect/data/train_valid.csv", help="훈련 데이터 CSV 경로")
    parser.add_argument("--valid_csv", type=str, default="/home/aikusrv02/dialect/data/valid_valid.csv", help="검증 데이터 CSV 경로")
    parser.add_argument("--test_csv", type=str, default="/home/aikusrv02/dialect/data/test_valid.csv", help="테스트 데이터 CSV 경로")
    parser.add_argument("--num_epochs", type=int, default=5, help="학습 에폭 수")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="학습률")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="워밍업 비율")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="가중치 감쇠")
    parser.add_argument("--batch_size", type=int, default=1, help="배치 크기 (GPU당)")
    parser.add_argument("--model_save_path", type=str, default="./saved_models_ddp_all/", help="모델 저장 경로")
    parser.add_argument("--connector", type=str, default="qformer", help="connector_type")
    
    
    
    args = parser.parse_args()
    
    # 사용 가능한 GPU 수 확인
    world_size = torch.cuda.device_count()
    print(f"사용 가능한 GPU 수: {world_size}")
    
    if world_size <= 1:
        print("DDP 학습을 위해서는 최소 2개 이상의 GPU가 필요합니다.")
        return
    
    # 멀티프로세싱 시작
    mp.spawn(
        train_ddp,
        args=(world_size, args),
        nprocs=world_size,
        join=True
    )

if __name__ == "__main__":
    main()