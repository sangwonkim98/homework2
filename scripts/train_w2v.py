# scripts/train_w2v.py
import os
import logging
from gensim.models import Word2Vec
from gensim.models.word2vec import PathLineSentences

def main():
    # 0) 로그(진행 상황) 보이게
    logging.basicConfig(
        format="%(asctime)s : %(levelname)s : %(message)s",
        level=logging.INFO,
    )

    # 1) 데이터 경로
    data_dir = "data/utf8_news"
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"데이터 폴더가 없습니다: {data_dir}")

    # 2) 재-이터러블 코퍼스 (폴더 내 모든 파일을 라인 단위로 읽음)
    sentences = PathLineSentences(data_dir)

    # 3) 모델 설정
    model = Word2Vec(
        vector_size=300,   # 임베딩 차원
        window=5,          # 주변 단어 윈도우
        min_count=5,       # 최소 등장 빈도
        sg=1,              # 0=CBOW, 1=Skip-gram
        workers=os.cpu_count() or 4,  # CPU 코어 수
        epochs=5,          # 에폭 수
        seed=42,           # 재현성
        compute_loss=True  # 학습 손실 추적(로그에서 확인 가능)
    )

    # 4) Vocab 빌드(진행 로그 나옴)
    logging.info("Building vocabulary...")
    model.build_vocab(sentences, progress_per=100_000)

    # 5) 학습(진행 로그/손실 보고)
    logging.info("Training Word2Vec...")
    model.train(
        sentences,
        total_examples=model.corpus_count,
        epochs=model.epochs,
        report_delay=10.0,  # 10초마다 진행 리포트
    )

    # 6) 최종 손실/저장
    logging.info(f"Final training loss: {model.get_latest_training_loss():,.2f}")
    out_path = "word2vec_news.model"
    model.save(out_path)
    print(f"✅ Word2Vec 학습 완료! {out_path} 저장됨")

if __name__ == "__main__":
    main()