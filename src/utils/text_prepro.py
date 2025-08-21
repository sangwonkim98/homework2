import re
import collections
import torch
import unicodedata
import os
import random
## 전처리 메서드

def prepare_eng_fra_splits(
    file_path: str,
    outdir: str | None = None,
    train_ratio: float = 0.8,
    dev_ratio: float = 0.1,
    test_ratio: float = 0.1,
    min_len: int = 1,
    max_len: int = 60,
    lower: bool = True,        # clean_str(..., True)와 일치
    dedupe: bool = True,
    seed: int = 54321,
    save_format: str = "paired",   # "paired" => .tsv  /  "parallel" => .en/.fr
):
    """
    eng-fra '한 파일(영문\\t불문)'을 읽어 전처리/분할하고,
    (선택) 디스크에 저장까지 하는 '한 방' 유틸.

    return:
      {
        "train": (src_list, tgt_list),
        "dev":   (src_list, tgt_list),
        "test":  (src_list, tgt_list),
        "paths": { 저장 경로들 (outdir 지정시) }
      }
    """
    assert abs(train_ratio + dev_ratio + test_ratio - 1.0) < 1e-6, "split ratio 합은 1이어야 함"
    assert save_format in {"paired", "parallel"}

    # 1) 파일 로드 (네 기존 clean_str 로직을 그대로 사용)
    src_all, tgt_all = [], []
    with open(file_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or "\t" not in line:
                continue
            s, t = line.split("\t", 1)
            s = clean_str(s, nmt=lower)  # lower=True면 unicode normalize+소문자
            t = clean_str(t, nmt=lower)
            # 길이 필터 (아주 단순한 공백 분할 기준)
            sl, tl = len(s.split()), len(t.split())
            if (sl < min_len or tl < min_len) or (sl > max_len or tl > max_len):
                continue
            src_all.append(s)
            tgt_all.append(t)

    # 2) (선택) 중복 제거
    if dedupe:
        seen = set()
        new_src, new_tgt = [], []
        for s, t in zip(src_all, tgt_all):
            key = (s, t)
            if key in seen:
                continue
            seen.add(key)
            new_src.append(s)
            new_tgt.append(t)
        src_all, tgt_all = new_src, new_tgt

    # 3) 셔플 + 분할 (재현성 보장)
    pairs = list(zip(src_all, tgt_all))
    rnd = random.Random(seed)
    rnd.shuffle(pairs)
    n = len(pairs)
    n_train = int(n * train_ratio)
    n_dev   = int(n * dev_ratio)
    train_pairs = pairs[:n_train]
    dev_pairs   = pairs[n_train:n_train + n_dev]
    test_pairs  = pairs[n_train + n_dev:]

    def unzip(ps):
        if not ps:
            return [], []
        a, b = zip(*ps)
        return list(a), list(b)

    train_src, train_tgt = unzip(train_pairs)
    dev_src,   dev_tgt   = unzip(dev_pairs)
    test_src,  test_tgt  = unzip(test_pairs)

    result = {
        "train": (train_src, train_tgt),
        "dev":   (dev_src, dev_tgt),
        "test":  (test_src, test_tgt),
        "paths": {}
    }

    # 4) (선택) 저장
    if outdir:
        os.makedirs(outdir, exist_ok=True)
        if save_format == "paired":
            def dump_tsv(path, src, tgt):
                with open(path, "w", encoding="utf-8") as f:
                    for s, t in zip(src, tgt):
                        f.write(f"{s}\t{t}\n")

            p_train = os.path.join(outdir, "train.en-fr.tsv")
            p_dev   = os.path.join(outdir, "dev.en-fr.tsv")
            p_test  = os.path.join(outdir, "test.en-fr.tsv")
            dump_tsv(p_train, train_src, train_tgt)
            dump_tsv(p_dev,   dev_src,   dev_tgt)
            dump_tsv(p_test,  test_src,  test_tgt)
            result["paths"] = {"train": p_train, "dev": p_dev, "test": p_test}

        else:  # parallel
            def dump_parallel(prefix, src, tgt):
                with open(prefix + ".en", "w", encoding="utf-8") as fe, \
                     open(prefix + ".fr", "w", encoding="utf-8") as ff:
                    for s in src: fe.write(s + "\n")
                    for t in tgt: ff.write(t + "\n")

            p_train = os.path.join(outdir, "train")
            p_dev   = os.path.join(outdir, "dev")
            p_test  = os.path.join(outdir, "test")
            dump_parallel(p_train, train_src, train_tgt)
            dump_parallel(p_dev,   dev_src,   dev_tgt)
            dump_parallel(p_test,  test_src,  test_tgt)
            result["paths"] = {
                "train": {"src": p_train + ".en", "tgt": p_train + ".fr"},
                "dev":   {"src": p_dev   + ".en", "tgt": p_dev   + ".fr"},
                "test":  {"src": p_test  + ".en", "tgt": p_test  + ".fr"},
            }

    return result


### 


def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn')

def load_nmt_data(src_path, tgt_path):
    # 각 줄: 서로 같은 인덱스끼리 한 쌍
    with open(src_path, encoding='utf-8') as fsrc, open(tgt_path, encoding='utf-8') as ftgt:
        src_lines = [clean_str(l.strip(), True) for l in fsrc if l.strip()]
        tgt_lines = [clean_str(l.strip(), True) for l in ftgt if l.strip()]

    # 길이 맞추기(혹시 공백 줄이 달라졌을 때 방어)
    n = min(len(src_lines), len(tgt_lines))
    src_lines, tgt_lines = src_lines[:n], tgt_lines[:n]
    return src_lines, tgt_lines

def load_nmt_pair_data(file_path):
    print("Reading lines...")
    # 파일 읽고 줄 단위로 나누기
    lines = open(file_path, encoding='utf-8').read().strip().split('\n')
    #  각 줄을 pairs 단위로 나누고 normalize 함수 진행
    source_text = []
    target_text = []
    for l in lines:
        s, t = l.split('\t')
        source_text.append(clean_str(s, True))
        target_text.append(clean_str(t, True))

    return source_text, target_text
# Source,target  -> [ ]source_text = ["i am fine .", "good morning"]
## target_text = ["je vais bien .", "bonjour"]

def clean_str(string, nmt=False):
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    if nmt:
        return unicodeToAscii(string.strip().lower())
    else:
        return string.strip().lower()

def load_snips_data(file_path, label_dictionary):
    # Load data from files
    text = list(open(file_path+"/seq.in", "r", encoding='UTF-8').readlines())
    text = [clean_str(sent) for sent in text]
    labels_text = list(open(file_path+"/label", "r", encoding='UTF-8').readlines())
    labels_text = [label.strip() for label in labels_text]

    if len(label_dictionary) == 0:
        label_set = set(labels_text)
        for i, label in enumerate(label_set):
            label_dictionary[label] = i
    labels = [label_dictionary[label_text] for label_text in labels_text]
    return text, labels, label_dictionary

def load_mr_data(pos_file, neg_file):
    pos_text = list(open(pos_file, "r", encoding='latin-1').readlines()) # 긍정적인 review 읽어서 list 형태로 관리
    pos_text = [clean_str(sent) for sent in pos_text] # clean_str 함수로 전처리 (소문자, 특수 기호 제거, (), 등 분리)

    neg_text = list(open(neg_file, "r", encoding='latin-1').readlines()) # 부정적인 review 읽어서 list 형태로 관리
    neg_text = [clean_str(sent) for sent in neg_text]

    positive_labels = [1 for _ in pos_text] # 긍정 review 개수만큼 ground_truth 생성
    negative_labels = [0 for _ in neg_text] # 부정 review 개수만큼 ground_truth 생성
    y = positive_labels + negative_labels

    x_final = pos_text + neg_text
    return [x_final, y]

def buildVocab(sentences, vocab_size):
    ##: ["i am a boy", "you are a girl"] =sentence 현재 이렇게 되어있음. 
    # Build vocabulary
    words = []
    for sentence in sentences:
        words.extend(sentence.split()) # ["i","am","a","boy","you","are","a","girl"]
    print("The number of words: ", len(words))
    word_counts = collections.Counter(words) ##	•	단어별 등장 횟수를 세는 해시 맵. 예: {"a":2, "i":1, "am":1, "boy":1, "you":1, "are":1, "girl":1}
    # Mapping from index to word
    vocabulary_inv = [x[0] for x in word_counts.most_common(vocab_size)] 
    ##	•	[x[0] for ...]로 단어만 뽑아 리스트로 만듦 → “ID→단어” 리스트의 초안. [a,b,c,d, 빈도순 정렬리스트]
    
    # vocabulary_inv = list(sorted(vocabulary_inv)) >만약 이 줄을 켜면 “빈도순”이 아니라 “사전순”으로 바뀌니, 일반적으로는 그대로 두는 게 맞음(빈도순 유지).
    # Mapping from word to index
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)} # a: 0, i: 1...
    
    ## enumerate 예시
    ## fruits = ["apple", "banana", "cherry"]

    # for i, fruit in enumerate(fruits):
    #     print(i, fruit)  -> 0 apple 1 banana 2 cherry 
    # // enumerate -> 인덱스 ,요소의 튜플쌍들을 리스트에 담아줌. 그래서 위에서는x:i 를 반전시킴.[(0, "apple"), (1, "banana"), (2, "cherry")] 을만들어냄
    
    
    
    return [vocabulary, vocabulary_inv]
            #     ##
            #     vocabulary_inv=["a","i","am","boy","you","are","girl"] →
            # {"a":0,"i":1,"am":2,"boy":3,"you":4,"are":5,"girl":6} 단어→ID” 매핑 
            
def text_to_indices(x_text, word_id_dict, use_unk=False):
    text_indices = []

    for text in x_text:
        words = text.split()
        ids = [2]  # <s>
        for word in words: # i, am, a, boy
            if word in word_id_dict:
                word_id = word_id_dict[word]
            else:  # oov
                if use_unk:
                    word_id = 1 # OOV (out-of-vocabulary)
                else:
                    word_id = len(word_id_dict)
                    word_id_dict[word] = word_id
            ids.append(word_id) # 5, 8, 6, 19
        ids.append(3)  # </s>
        text_indices.append(ids)
    return text_indices
        # #x_text = ["i love you"]
        # word_id_dict = {"<pad>":0, "<unk>":1, "<s>":2, "</s>":3, "i":4, "love":5, "you":6}
        #[[2, 4, 5, 6, 3]]
        ### x_text = ["i hate you"]
        #word_id_dict = {"<pad>":0, "<unk>":1, "<s>":2, "</s>":3, "i":4, "you":6}
        #[[2, 4, 1, 6, 3]] 이게 만약에 unk false 해놓으면 그냥   word_id_dict에 새로운단어 추가됨
        #🚫 Dev/Test에서 vocab을 바꾸면 안 되는 이유
        # 	1.	모델 파라미터(임베딩 크기) 고정 문제
            # 	•	모델은 vocab_size × embedding_dim 크기의 임베딩 행렬을 학습함.
            # 	•	vocab에 새로운 단어를 추가하면 vocab_size가 달라져서, 행렬 차원이 바뀌어 버림.
            # 	•	이미 학습된 파라미터와 맞지 않아서 불일치 발생 → 모델 실행 불가.
            # 	2.	평가의 공정성/재현성
            # 	•	dev/test는 “모델이 본 적 없는 데이터”를 평가하는 용도야.
            # 	•	만약 dev/test에서 새 단어를 추가하면, 사실상 모델이 평가 단계에서 새 지식을 얻게 되는 꼴이 돼.
            # → 평가가 불공정하고 결과 재현도 안 됨.
            # 	3.	OOV 처리는 모델의 일반화 능력 확인용
            # 	•	현실 세계 데이터에는 항상 train에 없던 단어가 나타남.
            # 	•	이걸 <unk>로 묶어야 모델이 “낯선 단어도 문맥으로 대충 번역한다”는 일반화 능력을 보일 수 있어.
            # 	•	만약 새 단어를 추가해버리면, dev/test 성능이 실제보다 더 좋게 나와서 착시 효과가 생김.
        ##ex
            #         #2 → [0.2, -0.1, 0.05, ...]   # <s> 벡터
            # 4 → [0.3, 0.9, -0.7, ...]    # "i"
            # 5 → [0.8, -0.2, 0.1, ...]    # "love"
            # 6 → [-0.4, 0.3, 0.6, ...]    # "you"
            # 3 → [0.0, 0.0, 0.0, ...]     # </s>
            ## OOV (Out-Of-Vocabulary)
	# •	말 그대로 사전에 없는 단어.
	# •	우리가 학습용 단어 사전(vocabulary) 을 만들 때 Train 데이터에 등장한 단어만 모아서 dictionary(word2id)를 만들어.
	# •	그런데 Dev/Test 데이터에서는 훈련 때 본 적 없는 새로운 단어가 나올 수 있음 → 이걸 OOV라고 해.

def sequence_to_tensor(sequence_list, nb_paddings=(0, 0)):
    nb_front_pad, nb_back_pad = nb_paddings

    max_length = len(max(sequence_list, key=len)) + nb_front_pad + nb_back_pad
    sequence_tensor = torch.LongTensor(len(sequence_list), max_length).zero_()  # 0: <pad>
    print("\n max length: " + str(max_length))
    for i, sequence in enumerate(sequence_list):
        sequence_tensor[i, nb_front_pad:len(sequence) + nb_front_pad] = torch.tensor(sequence)
    return sequence_tensor