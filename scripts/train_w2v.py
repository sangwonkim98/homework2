from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import glob
import os

# 폴더 경로
data_dir = "data/utf8_news"

# utf8_news 안의 모든 파일 경로 리스트
files = glob.glob(os.path.join(data_dir, "*"))

# 여러 파일을 순차적으로 읽는 generator
def iter_sentences(files):
    for file in files:
        print(f"Loading {file} ...")
        for sentence in LineSentence(file):  # LineSentence가 자동으로 토큰화해줌
            yield sentence

sentences = iter_sentences(files)

# Word2Vec 모델 학습
model = Word2Vec(
    sentences=sentences,
    vector_size=300,   # 임베딩 차원
    window=5,          # 주변 단어 범위
    min_count=5,       # 최소 등장 횟수
    sg=1,              # 0=CBOW, 1=Skip-gram
    workers=4
)

# 모델 저장
model.save("word2vec_news.model")
print("✅ Word2Vec 학습 완료! word2vec_news.model 저장됨")