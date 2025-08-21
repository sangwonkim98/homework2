import numpy as np
import time
import re
import os
import random
import sys
import yaml
import smart_open
import pickle

from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu

import torch
import torch.nn as nn
import torch.optim as optim

from gensim.models.keyedvectors import KeyedVectors
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from homework2.models.nmt_rnn import EncoderLSTM_Att, DecoderLSTM_Att
from homework2.utils.text_prepro import buildVocab, load_nmt_data, load_nmt_pair_data, text_to_indices
from homework2.utils.data_prepro import TranslateDataset, collate
def main():

    if len(sys.argv) >= 2:
        params_filename = sys.argv[1]
        print(sys.argv)
    else:
        params_filename = '../config/nmt_rnn.yaml'

    with open(params_filename, 'r', encoding="UTF8") as f:
        params = yaml.safe_load(f)

    data_params = params['data_files'][params['task']]

    # 랜덤 시드 세팅
    if 'random_seed' in params:
        seed = params['random_seed']
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    torch.backends.cudnn.benchmark = True

    # 데이터 로드
    if params['task'] == "EN2DE":
        pass

    elif params['task'] == "EN2FR":
        train_source_text, train_target_text = load_nmt_pair_data(data_params['train_file'])

    word2id_source, _ = buildVocab(train_source_text, params['vocab_size'])  # training corpus를 토대로 단어사전 구축
    vocab_size_source = len(word2id_source) + 4  # e.g., 30000 + 4
    print("vocabulary size of the source language: ", vocab_size_source)

    word2id_target, _ = buildVocab(train_target_text, params['vocab_size'])  # training corpus를 토대로 단어사전 구축
    vocab_size_target = len(word2id_target) + 4  # e.g., 30000 + 4
    print("vocabulary size of the target language: ", vocab_size_target)

    for word in word2id_source.keys():
        word2id_source[word] += 4  # <pad>: 0, <unk>: 1, <s>: 2 (a: 0 -> 4)
    word2id_source['<pad>'] = 0  # zero padding을 위한 토큰
    word2id_source['<unk>'] = 1  # OOV word를 위한 토큰
    word2id_source['<s>'] = 2  # 문장 시작을 알리는 start 토큰
    word2id_source['</s>'] = 3  # 문장 마침을 알리는 end 토큰

    id2word_source = {} # 번역 대상 (source)을 확인하기 위한 index to word dictionary
    for k, v in word2id_source.items():
        id2word_source[v] = k

    for word in word2id_target.keys():
        word2id_target[word] += 4  # <pad>: 0, <unk>: 1, <s>: 2 (a: 0 -> 4)
    word2id_target['<pad>'] = 0  # zero padding을 위한 토큰
    word2id_target['<unk>'] = 1  # OOV word를 위한 토큰
    word2id_target['<s>'] = 2  # 문장 시작을 알리는 start 토큰
    word2id_target['</s>'] = 3  # 문장 마침을 알리는 end 토큰

    id2word_target = {} # 번역 결과 (target)를 확인하기 위한 index to word dictionary
    for k, v in word2id_target.items():
        id2word_target[v] = k

    if params['embedding_file']:  # word2vec 활용 시 (영어가 source language일 경우, 다른언어 word embedding 확보시 추가 필요)
        print("Loading W2V data...")
        pre_emb = KeyedVectors.load_word2vec_format(params['embedding_file'], binary=True)  # pre-trained word2vec load
        pre_emb.init_sims(replace=True)
        num_keys = len(pre_emb.key_to_index)
        print("loaded word2vec len ", num_keys)

        # initial matrix with random uniform, pretrained word2vec으로 source vocabulary 내 단어들을 초기화하기 위핸 weight matrix 초기화
        initW = np.random.uniform(-0.25, 0.25, (vocab_size_source, params['embedding_dim']))
        # load any vectors from the word2vec
        print("init initW cnn.W in FLAG")
        for w in word2id_source.keys():
            arr = []
            s = re.sub('[^0-9a-zA-Z]+', '', w)
            if w in pre_emb:  # 직접 구축한 vocab 내 단어가 google word2vec에 존재하면
                arr = pre_emb[w]  # word2vec vector를 가져옴
            elif w.lower() in pre_emb:  # 소문자로도 확인
                arr = pre_emb[w.lower()]
            elif s in pre_emb:  # 전처리 후 확인
                arr = pre_emb[s]
            elif s.isdigit():  # 숫자이면
                arr = pre_emb['1']
            if len(arr) > 0:  # 직접 구축한 vocab 내 단어가 google word2vec에 존재하면
                idx = word2id_source[w]  # 단어 index
                initW[idx] = np.asarray(arr).astype(np.float32)  # 적절한 index에 word2vec word 할당
            initW[0] = np.zeros(params['embedding_dim'])

    # - 문장을 인덱스화
    if params['task'] == "EN2DE":
        pass

    elif params['task'] == "EN2FR":
        train_x = text_to_indices(train_source_text, word2id_source, params['use_unk_for_oov'])
        train_y = text_to_indices(train_target_text, word2id_target, params['use_unk_for_oov'])

        # Shuffle pairs
        data = list(zip(train_x, train_y))
        random.shuffle(data)
        train_x, train_y = zip(*data)

        dev_sample_index = -1 * int(0.1 * float(len(train_y))) # 10% 만 dev set으로 사용
        train_x, dev_x = train_x[:dev_sample_index], train_x[dev_sample_index:]
        train_y, dev_y = train_y[:dev_sample_index], train_y[dev_sample_index:]

    # Create dataloaders
    train_dataloader = DataLoader(TranslateDataset(train_x, train_y, device), batch_size=params['batch_size'], shuffle=True, collate_fn=collate, drop_last=True)
    dev_dataloader = DataLoader(TranslateDataset(dev_x, dev_y, device), batch_size=params['batch_size'], collate_fn=collate, drop_last=True)

    # 학습 모델 생성
    # Initialize your models
    encoder = EncoderLSTM_Att(vocab_size_source, params['embedding_dim'], params['model_params_rnn']['hidden_size'], initW).to(device)
    decoder = DecoderLSTM_Att(params['embedding_dim'], params['model_params_rnn']['hidden_size'], vocab_size_target).to(device)

    criterion = nn.NLLLoss()
    # Define your optimizer
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=params['optimizer_params'][params['optimizer']]['lr'])
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=params['optimizer_params'][params['optimizer']]['lr'])

    timestamp = str(int(time.time()))
    out_dir = os.path.abspath((os.path.join(os.path.curdir, "runs", timestamp)))
    checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
    summary_dir = os.path.join(out_dir, "summaries")

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    writer = SummaryWriter(summary_dir) # TensorBoard를 위한 초기화

    # training 시작
    start_time = time.time()
    lowest_val_loss = 999
    global_steps = 0
    print('========================================')
    print("Start training...")
    for epoch in range(params['max_epochs']):
        train_loss = 0
        train_batch_cnt = 0

        encoder.train()
        decoder.train()
        for i, training_pair in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            input_tensor = training_pair[0] # source language sentence에 대한 tensor
            target_tensor = training_pair[1] # target language sentence에 대한 tensor
            input_length = training_pair[2] # source language sentence에 대한 길이
            output_length = training_pair[3] # target language sentence에 대한 길이

            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            encoder_output, encoder_hidden, encoder_cell = encoder(input_tensor, input_length) # encoder에 sentence, 길이, initial state 먹이기
            decoder_input = torch.tensor([2 for _ in range(params['batch_size'])], device=device) # start token <s>으로 decoding 준비
            decoder_hidden = encoder_hidden.unsqueeze(0)
            decoder_cell = encoder_cell # encoder의 final state를 decoder의 initial state로
            loss = 0
            max_output_length = torch.max(output_length)
            use_teacher_forcing = True if random.random() < params['model_params_rnn']['teacher_forcing_ratio'] else False
            # 학습과정에서 inference 정답과 다를 때 next token에 대한 input을 정답으로 강제하는 비율

            if use_teacher_forcing:
                for di in range(max_output_length):
                    decoder_output, decoder_hidden, decoder_cell = decoder(encoder_output, decoder_input,
                                                                           decoder_hidden, decoder_cell)
                    loss += criterion(decoder_output, target_tensor[:, di])
                    decoder_input = target_tensor[:, di] # 정답을 강제해야 하기 때문에 target_tensor에서 실제 정답을 가져옴

            else:
                for di in range(max_output_length):
                    decoder_output, decoder_hidden, decoder_cell = decoder(encoder_output, decoder_input,
                                                                           decoder_hidden, decoder_cell)
                    topv, topi = decoder_output.topk(k=1, dim=1)
                    loss += criterion(decoder_output, target_tensor[:, di])
                    decoder_input = topi.squeeze().detach() # 정답을 강제하지 않을 때는 예측 토큰을 다음 토큰으로 그대로 넣어줌


            loss.backward()# 가중치 w에 대해 loss를 미분
            encoder_optimizer.step()
            decoder_optimizer.step()# 가중치들을 업데이트

            writer.add_scalar(tag='Batch/Loss', scalar_value=loss.item() / params['batch_size'], global_step=global_steps)

            train_loss += loss.item() / params['batch_size']
            train_batch_cnt += 1

            global_steps += 1
            if (global_steps) % 100 == 0:
                print('\nEpoch [{}], Step [{}], Loss: {:.4f}'.format(epoch+1, global_steps, loss.item()))

        train_loss /= train_batch_cnt
        print('(%d epochs) (%d steps) loss:%.4f' % (epoch + 1, i, train_loss))
        training_time = (time.time() - start_time) / 60
        print("training_time: %.2f minutes" % training_time)

        encoder.eval()
        decoder.eval()

        target_sentences = []
        inference_sentences = []
        val_loss = 0
        val_cnt = 0

        for i, dev_pair in tqdm(enumerate(dev_dataloader), total=len(dev_dataloader)):
            input_tensor = dev_pair[0] # source language sentence에 대한 tensor
            target_tensor = dev_pair[1] # source language sentence에 대한 길이
            input_length = dev_pair[2] # target language sentence에 대한 tensor
            output_length = dev_pair[3] # target language sentence에 대한 길이
            with torch.no_grad():
                encoder_output, encoder_hidden, encoder_cell = encoder(input_tensor, input_length)
                decoder_input = torch.tensor([2 for _ in range(params['batch_size'])], device=device) # start token <s>으로 decoding 준비
                decoder_hidden = encoder_hidden.unsqueeze(0)
                decoder_cell = encoder_cell
                loss = 0
                max_output_length = torch.max(output_length)
                inference_words = [[] for _ in range(params['batch_size'])]

                for di in range(max_output_length):
                    decoder_output, decoder_hidden, decoder_cell = decoder(encoder_output, decoder_input,
                                                                           decoder_hidden, decoder_cell)
                    topv, topi = decoder_output.topk(k=1, dim=1)
                    loss += criterion(decoder_output, target_tensor[:, di])
                    decoder_input = topi.squeeze().detach() # inference 때는 답을 모르기 때문에 예측 토큰을 다음 토큰으로 그대로 넣어줌
                    for q in range(params['batch_size']):
                        inference_words[q].append(id2word_target[topi[q].item()]) # 추론 결과 확인을 위해 index를 word로 넣어서 list에 보관

                val_loss += loss.item() / params['batch_size']
                
                # 실제 정답 문장을 target_sentences에 기록 
                for q in range(params['batch_size']):
                    target_sentences.append([[]])
                    for word_tensor in target_tensor[q]: 
                        out_word = id2word_target[word_tensor.item()]
                        if out_word != '<pad>':
                            target_sentences[-1][-1].append(out_word)
                            
                # 모델이 추론한 문장을 inference_sentences에 기록
                for each_val in inference_words:
                    inference_sentences.append([])
                    for word in each_val:
                        if word != '<pad>':
                            inference_sentences[-1].append(word)

                val_cnt += 1

        val_loss /= val_cnt
        for i in range(10):
            print("target_sentence:", " ".join(target_sentences[i][0]))
            print("inferece_sentence:", " ".join(inference_sentences[i]))
            print('-' * 20)
        print('=' * 20)
        # BLEU-4 함수 weight는 unigram, bigram, trigram, quadgram에 대한 각각의 가중치
        bleu = corpus_bleu(target_sentences, inference_sentences, weights=(0.25, 0.25, 0.25, 0.25)) 
        print("bleu_score:", bleu)
        writer.add_scalar(tag='bleu score', scalar_value=bleu, global_step=epoch + 1)
        writer.add_scalar(tag='Train/Loss', scalar_value=train_loss, global_step=epoch + 1)
        writer.add_scalar(tag='validation/Loss', scalar_value=val_loss, global_step=epoch + 1)

        if val_loss < lowest_val_loss:# validation loss가 경신될 때
            save_path = checkpoint_dir + '/epoch_' + str(epoch + 1) + '.pth'
            torch.save({'epoch': epoch + 1,
                        'encoder_state_dict': encoder.state_dict(),
                        'decoder_state_dict': decoder.state_dict()},
                       save_path)

            save_path = checkpoint_dir + '/best.pth'
            torch.save({'epoch': epoch + 1,
                        'encoder_state_dict': encoder.state_dict(),
                        'decoder_state_dict': decoder.state_dict()},
                       save_path)  # best accuracy에 도달할 때만 모델을 저장함으로써 early stopping
            lowest_val_loss = val_loss
        epoch += 1

    source_vocab_path = os.path.abspath(os.path.join(checkpoint_dir, "source_vocab"))
    target_vocab_path = os.path.abspath(os.path.join(checkpoint_dir, "target_vocab"))
    emb_path = os.path.abspath(os.path.join(checkpoint_dir, "emb"))

    with smart_open.smart_open(source_vocab_path, 'wb') as f:
        pickle.dump(word2id_source, f)
    with smart_open.smart_open(target_vocab_path, 'wb') as f:
        pickle.dump(word2id_target, f)
    with smart_open.smart_open(emb_path, 'wb') as f:
        pickle.dump(initW, f)

if __name__ == '__main__':
    main()