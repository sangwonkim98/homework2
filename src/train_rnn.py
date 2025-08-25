# train_rnn.py
import os, sys, time, re, pickle, random, yaml
from pathlib import Path

import numpy as np
from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from gensim.models import Word2Vec, KeyedVectors

# ---- 프로젝트 코드 임포트 (src-layout) ----
from src.models.nmt_rnn import EncoderLSTM_Att, DecoderLSTM_Att
from src.utils.text_prepro import (
    buildVocab, load_nmt_data, load_nmt_pair_data, text_to_indices
)
from src.utils.data_prepro import TranslateDataset, collate


# =========================
# 0) 환경/경로 유틸
# =========================
PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", "/content/homework2"))

def resolve_path(p: str) -> Path:
    p = Path(p)
    return p if p.is_absolute() else (PROJECT_ROOT / p)

def load_kv(path: str):
    """ .kv / .model 모두 지원 """
    s = str(path)
    if s.endswith(".kv"):
        return KeyedVectors.load(s, mmap="r")
    elif s.endswith(".model"):
        return Word2Vec.load(s).wv
    else:
        raise ValueError("embedding_file must end with .kv or .model")


# =========================
# 메인
# =========================
def main():
    # 1) 설정파일 경로 결정
    default_cfg = PROJECT_ROOT / "config" / "nmt_rnn.yaml"
    cfg_path = resolve_path(sys.argv[1]) if len(sys.argv) >= 2 else default_cfg

    # 2) YAML 로드
    with open(cfg_path, "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)

    # 3) YAML 내부 경로 보정
    if params.get("embedding_file"):
        params["embedding_file"] = str(resolve_path(params["embedding_file"]))

    data_params = params["data_files"][params["task"]]

    # 4) 시드/디바이스
    if "random_seed" in params:
        seed = params["random_seed"]
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = (device.type == "cuda")

    # 5) 데이터 로드 (task 분기)
    if params["task"] == "EN2DE":
        train_source_text, train_target_text = load_nmt_data(data_params["train_file"])
        dev_source_text,   dev_target_text   = load_nmt_data(data_params["dev_file"])
    elif params["task"] == "EN2FR":
        train_source_text, train_target_text = load_nmt_pair_data(data_params["train_file"])
        dev_source_text,   dev_target_text   = load_nmt_pair_data(data_params["dev_file"])
    else:
        raise ValueError(f"Unknown task {params['task']}")

    # 6) Vocab 구성
    word2id_source, _ = buildVocab(train_source_text, params["vocab_size"])
    word2id_target, _ = buildVocab(train_target_text, params["vocab_size"])

    vocab_size_source = len(word2id_source) + 4
    vocab_size_target = len(word2id_target) + 4
    print("vocabulary size of the source language:", vocab_size_source)
    print("vocabulary size of the target language:", vocab_size_target)

    # 스페셜 토큰 밀기 & 추가 (source/target 각각)
    for w in list(word2id_source.keys()):
        word2id_source[w] += 4
    word2id_source.update({'<pad>':0,'<unk>':1,'<s>':2,'</s>':3})
    id2word_source = {i:w for w,i in word2id_source.items()}

    for w in list(word2id_target.keys()):
        word2id_target[w] += 4
    word2id_target.update({'<pad>':0,'<unk>':1,'<s>':2,'</s>':3})
    id2word_target = {i:w for w,i in word2id_target.items()}

    # 7) 임베딩 초기화 (항상 랜덤으로 만들고 → 있으면 덮어쓰기)
    embed_dim = params["embedding_dim"]
    initW = np.random.uniform(-0.25, 0.25, (vocab_size_source, embed_dim)).astype(np.float32)

    if params.get("embedding_file"):
        print("Loading W2V data...")
        kv = load_kv(params["embedding_file"])
        print("loaded word2vec len", len(kv.key_to_index))

        for w, idx in word2id_source.items():
            if w in ('<pad>','<unk>','<s>','</s>'):
                continue
            s = re.sub('[^0-9a-zA-Z]+', '', w)
            vec = None
            if w in kv: vec = kv[w]
            elif w.lower() in kv: vec = kv[w.lower()]
            elif s in kv: vec = kv[s]
            elif s.isdigit() and ('1' in kv): vec = kv['1']
            if vec is not None:
                initW[idx] = vec.astype(np.float32)

    # pad 임베딩은 0으로
    initW[0] = np.zeros(embed_dim, dtype=np.float32)

    # 8) 텍스트 → 인덱스
    train_x = text_to_indices(train_source_text, word2id_source, params["use_unk_for_oov"])
    train_y = text_to_indices(train_target_text, word2id_target, params["use_unk_for_oov"])
    dev_x   = text_to_indices(dev_source_text,   word2id_source, params["use_unk_for_oov"])
    dev_y   = text_to_indices(dev_target_text,   word2id_target, params["use_unk_for_oov"])

    # 쌍 단위 셔플
    pairs = list(zip(train_x, train_y))
    random.shuffle(pairs)
    train_x, train_y = zip(*pairs) if pairs else ([], [])

    # 9) Dataloader
    train_loader = DataLoader(
        TranslateDataset(train_x, train_y, device),
        batch_size=params["batch_size"],
        shuffle=True,
        collate_fn=collate,
        drop_last=True
    )
    dev_loader = DataLoader(
        TranslateDataset(dev_x, dev_y, device),
        batch_size=params["batch_size"],
        shuffle=False,
        collate_fn=collate,
        drop_last=True
    )

    # 10) 모델/옵티마/손실
    encoder = EncoderLSTM_Att(vocab_size_source, embed_dim, params["model_params_rnn"]["hidden_size"], initW).to(device)
    decoder = DecoderLSTM_Att(embed_dim, params["model_params_rnn"]["hidden_size"], vocab_size_target).to(device)

    criterion = nn.NLLLoss()
    lr = params.get("optimizer_params", {}).get(params.get("optimizer", "adam"), {}).get("lr", 1e-3)
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=lr)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=lr)

    # 11) 로그/체크포인트 경로
    out_dir = (PROJECT_ROOT / "runs" / str(int(time.time()))).resolve()
    ckpt_dir = out_dir / "checkpoints"
    sum_dir  = out_dir / "summaries"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    sum_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(sum_dir))

    # =========================
    # 학습 루프
    # =========================
    start_time = time.time()
    lowest_val_loss = float("inf")
    global_steps = 0

    print("========================================")
    print("Start training...")
    for epoch in range(params["max_epochs"]):
        encoder.train(); decoder.train()
        train_loss_sum, train_batches = 0.0, 0

        for i, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
            input_tensor, target_tensor, input_length, output_length = batch
            B = input_tensor.size(0)   # 실제 배치 크기

            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()

            enc_out, enc_h, enc_c = encoder(input_tensor, input_length)
            dec_in  = torch.full((B,), fill_value=2, dtype=torch.long, device=device)  # <s>=2
            dec_h   = enc_h.unsqueeze(0)
            dec_c   = enc_c
            loss    = 0.0
            T_max   = torch.max(output_length)

            # Teacher forcing 여부
            tf = (random.random() < params["model_params_rnn"]["teacher_forcing_ratio"])

            if tf:
                for t in range(T_max):
                    dec_out, dec_h, dec_c = decoder(enc_out, dec_in, dec_h, dec_c)
                    loss += criterion(dec_out, target_tensor[:, t])
                    dec_in = target_tensor[:, t]
            else:
                for t in range(T_max):
                    dec_out, dec_h, dec_c = decoder(enc_out, dec_in, dec_h, dec_c)
                    topv, topi = dec_out.topk(k=1, dim=1)
                    loss += criterion(dec_out, target_tensor[:, t])
                    dec_in = topi.squeeze(1).detach()

            loss.backward()
            encoder_optimizer.step()
            decoder_optimizer.step()

            global_steps += 1
            train_loss_sum += (loss.item() / B)
            train_batches  += 1

            if global_steps % 100 == 0:
                print(f"\nEpoch [{epoch+1}], Step [{global_steps}], Loss: {loss.item():.4f}")
            writer.add_scalar("Batch/Loss", loss.item() / B, global_steps)

        train_loss = train_loss_sum / max(1, train_batches)
        print(f"({epoch+1} epochs) loss:{train_loss:.4f}")
        print(f"training_time: {(time.time()-start_time)/60:.2f} minutes")

        # =========================
        # 검증
        # =========================
        encoder.eval(); decoder.eval()
        val_loss_sum, val_batches = 0.0, 0
        target_sentences, inference_sentences = [], []

        with torch.no_grad():
            for i, batch in tqdm(enumerate(dev_loader), total=len(dev_loader)):
                input_tensor, target_tensor, input_length, output_length = batch
                B = input_tensor.size(0)

                enc_out, enc_h, enc_c = encoder(input_tensor, input_length)
                dec_in  = torch.full((B,), fill_value=2, dtype=torch.long, device=device)  # <s>
                dec_h   = enc_h.unsqueeze(0)
                dec_c   = enc_c
                loss    = 0.0
                T_max   = torch.max(output_length)

                infer_words = [[] for _ in range(B)]
                for t in range(T_max):
                    dec_out, dec_h, dec_c = decoder(enc_out, dec_in, dec_h, dec_c)
                    topv, topi = dec_out.topk(k=1, dim=1)
                    loss += criterion(dec_out, target_tensor[:, t])
                    dec_in = topi.squeeze(1).detach()
                    for q in range(B):
                        token_id = topi[q].item()
                        infer_words[q].append(id2word_target.get(token_id, "<unk>"))

                val_loss_sum += (loss.item() / B)
                val_batches  += 1

                # 정답 문장 기록
                for q in range(B):
                    sent = []
                    for token in target_tensor[q]:
                        w = id2word_target.get(int(token.item()), "<unk>")
                        if w != "<pad>": sent.append(w)
                    target_sentences.append([sent])

                # 추론 문장 기록
                for each in infer_words:
                    inference_sentences.append([w for w in each if w != "<pad>"])

        val_loss = val_loss_sum / max(1, val_batches)
        print("=" * 20)
        for i in range(min(10, len(target_sentences))):
            print("target_sentence:  ", " ".join(target_sentences[i][0]))
            print("inference_sentence:", " ".join(inference_sentences[i]))
            print("-" * 20)

        bleu = corpus_bleu(target_sentences, inference_sentences, weights=(0.25,0.25,0.25,0.25))
        print(f"val_loss: {val_loss:.4f} | bleu_score: {bleu:.4f}")
        writer.add_scalar("Train/Loss", train_loss, epoch+1)
        writer.add_scalar("Validation/Loss", val_loss, epoch+1)
        writer.add_scalar("BLEU", bleu, epoch+1)

        # 체크포인트
        if val_loss < lowest_val_loss:
            lowest_val_loss = val_loss
            best_path = ckpt_dir / "best.pth"
            torch.save({
                "epoch": epoch + 1,
                "encoder_state_dict": encoder.state_dict(),
                "decoder_state_dict": decoder.state_dict(),
            }, str(best_path))

            # 에폭별 저장도 원하면:
            epoch_path = ckpt_dir / f"epoch_{epoch+1}.pth"
            torch.save({
                "epoch": epoch + 1,
                "encoder_state_dict": encoder.state_dict(),
                "decoder_state_dict": decoder.state_dict(),
            }, str(epoch_path))

    # 최종 vocab/임베딩 저장
    source_vocab_path = (ckpt_dir / "source_vocab").resolve()
    target_vocab_path = (ckpt_dir / "target_vocab").resolve()
    emb_path          = (ckpt_dir / "emb").resolve()

    with open(source_vocab_path, "wb") as f:
        pickle.dump(word2id_source, f)
    with open(target_vocab_path, "wb") as f:
        pickle.dump(word2id_target, f)
    with open(emb_path, "wb") as f:
        pickle.dump(initW, f)

    print("✅ done. checkpoints:", str(ckpt_dir))


if __name__ == "__main__":
    main()