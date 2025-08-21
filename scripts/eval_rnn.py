import torch
import smart_open
import pickle
import yaml
import sys
import os
from torch.utils.data import DataLoader
from homework2.models.nmt_rnn import EncoderLSTM_Att, DecoderLSTM_Att

from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu

from homework2.utils.text_prepro import buildVocab, load_nmt_data, load_nmt_pair_data, text_to_indices
from homework2.utils.data_prepro import TranslateDataset, collate

def main():
    print('RNN for machine translation evaluation')

    if len(sys.argv) >= 2:
        params_filename = sys.argv[1]
        print(sys.argv)
    else:
        params_filename = '../config/nmt_rnn.yaml'

    with open(params_filename, 'r', encoding="UTF8") as f:
        params = yaml.safe_load(f)

    data_params = params['data_files'][params['task']]

    # GPU 사용이 가능하면 사용하고, 불가능하면 CPU 활용
    print("GPU Available:", torch.cuda.is_available())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    timestamp = "1695267044"
    out_dir = os.path.abspath((os.path.join(os.path.curdir, "runs", timestamp)))
    source_vocab_path = os.path.abspath(os.path.join(out_dir, "checkpoints/source_vocab"))
    target_vocab_path = os.path.abspath(os.path.join(out_dir, "checkpoints/target_vocab"))
    emb_path = os.path.abspath(os.path.join(out_dir, "checkpoints/emb"))

    # 데이터 로드
    if params['task'] == "EN2DE":
        test_source_text, test_target_text = load_nmt_data('en', 'de', data_params['test_file'])

    elif params['task'] == "EN2FR":
        test_source_text, test_target_text = load_nmt_pair_data(data_params['test_file'])

    with smart_open.smart_open(source_vocab_path, 'rb') as f:
        word2id_source = pickle.load(f)
    with smart_open.smart_open(target_vocab_path, 'rb') as f:
        word2id_target = pickle.load(f)
    with smart_open.smart_open(emb_path, 'rb') as f:
        initW = pickle.load(f)

    vocab_size_source = len(word2id_source)
    vocab_size_target = len(word2id_target)

    id2word_source = {}  # 번역 대상 (source)을 확인하기 위한 index to word dictionary
    for k, v in word2id_source.items():
        id2word_source[v] = k

    id2word_target = {}  # 번역 결과 (target)를 확인하기 위한 index to word dictionary
    for k, v in word2id_target.items():
        id2word_target[v] = k

    # - 문장을 인덱스화
    if params['task'] == "EN2DE":
        test_x = text_to_indices(test_source_text, word2id_source, params['use_unk_for_oov'])
        test_y = text_to_indices(test_target_text, word2id_target, params['use_unk_for_oov'])

    elif params['task'] == "EN2FR":
        test_x = text_to_indices(test_source_text, word2id_source, params['use_unk_for_oov'])
        test_y = text_to_indices(test_target_text, word2id_target, params['use_unk_for_oov'])

    # data 개수 확인
    print('The number of test data: ', len(test_x))

    # Create dataloaders
    test_dataloader = DataLoader(TranslateDataset(test_x, test_y, device), batch_size=params['batch_size'], collate_fn=collate, drop_last=True)

    # 학습 모델 생성
    # Initialize your models
    encoder = EncoderLSTM_Att(vocab_size_source, params['embedding_dim'], params['model_params_rnn']['hidden_size'], initW).to(device)
    decoder = DecoderLSTM_Att(params['embedding_dim'], params['model_params_rnn']['hidden_size'], vocab_size_target).to(device)

    # test 시작
    encoder.eval()
    decoder.eval()

    # 저장된 state 불러오기
    checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints/best.pth"))

    # TODO : 세팅값 마다 save_path를 바꾸어 로드
    checkpoint = torch.load(checkpoint_dir)
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    decoder.load_state_dict(checkpoint['decoder_state_dict'])

    target_sentences = []
    inference_sentences = []
    for i, test_pair in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
        input_tensor = test_pair[0]  # source language sentence에 대한 tensor
        target_tensor = test_pair[1]  # source language sentence에 대한 길이
        input_length = test_pair[2]  # target language sentence에 대한 tensor
        output_length = test_pair[3]  # target language sentence에 대한 길이
        with torch.no_grad():
            # encoder_hidden = encoder.initHidden(device) # initial state를 위한 0벡터 초기화
            # encoder_cell = encoder.initHidden(device) # initial state를 위한 0벡터 초기화
            encoder_output, encoder_hidden, encoder_cell = encoder(input_tensor, input_length)
            decoder_input = torch.tensor([2 for _ in range(params['batch_size'])],
                                         device=device)  # start token <s>으로 decoding 준비
            decoder_hidden = encoder_hidden.unsqueeze(0)
            decoder_cell = encoder_cell
            loss = 0
            max_output_length = torch.max(output_length)
            inference_words = [[] for _ in range(params['batch_size'])]

            for di in range(max_output_length):
                decoder_output, decoder_hidden, decoder_cell = decoder(encoder_output, decoder_input,
                                                                       decoder_hidden, decoder_cell)
                topv, topi = decoder_output.topk(k=1, dim=1)
                decoder_input = topi.squeeze().detach()  # inference 때는 답을 모르기 때문에 예측 토큰을 다음 토큰으로 그대로 넣어줌
                for q in range(params['batch_size']):
                    inference_words[q].append(id2word_target[topi[q].item()])  # 추론 결과 확인을 위해 index를 word로 넣어서 list에 보관

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

    for i in range(10):
        print("target_sentence:", " ".join(target_sentences[i][0]))
        print("inference_sentence:", " ".join(inference_sentences[i]))
        print('-' * 20)
    print('=' * 20)
    # BLEU-4 함수 weight는 unigram, bigram, trigram, quadgram에 대한 각각의 가중치
    bleu = corpus_bleu(target_sentences, inference_sentences, weights=(0.25, 0.25, 0.25, 0.25))
    print("bleu_score:", bleu)

if __name__ == "__main__":
    main()