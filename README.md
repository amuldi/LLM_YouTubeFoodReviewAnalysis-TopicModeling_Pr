# 맛집 소개 영상 댓글 토픽 분석

맛집 소개 영상(롱폼·쇼츠)의 YouTube 댓글을 수집해 BERTopic으로 토픽을 분석하고, 댓글에 맛에 대한 직접 평가가 얼마나 있는지 살펴보는 프로젝트입니다. (소셜 미디어 기반 데이터 수집 및 LLM 융합 토픽 모델링 과제)

## 파이프라인
1. YouTube 댓글 수집 (`youtube-comment-downloader`, 한국어)
2. 정규식 전처리, 길이·중복 필터링
3. Sentence Transformers 임베딩 → UMAP → HDBSCAN → BERTopic
4. UMAP·HDBSCAN 하이퍼파라미터 5가지 실험 비교
5. c-TF-IDF / KeyBERTInspired / 공개 LLM(Qwen2.5-1.5B) 토픽 이름 비교
6. 시각화: barchart, topics, hierarchy, 2D 군집 지도, documents, heatmap
7. 방문 경험·맛 평가 규칙 분류와 영상별 요약

## 파일
| 파일 | 설명 |
|---|---|
| `음식점_댓글_토픽모델링.ipynb` | 수집부터 시각화까지의 코드 |
| `결과보고서.md` | 주제 선정, 튜닝 비교, LLM 레이블링, 인사이트 |
| `requirements.txt` | 필요 패키지 |
| `scripts/` | 노트북 생성·검증 스크립트 |

## 실행
- Colab(T4 GPU 권장) 또는 Python 3.11~3.12 환경에서 노트북을 엽니다.
- `VIDEOS`에 영상 URL을 넣고 위에서부터 순서대로 실행합니다. 로컬 설치는 `pip install -r requirements.txt`.
- `data/`(수집 원본)와 `outputs/`(분석 결과)는 실행 시 생성되며 저장소에는 포함하지 않았습니다.

## 한계
영상 대부분이 가게 하나가 아닌 지역·여러 가게를 소개해 분석 단위가 영상입니다. 영상당 최신 500건만 수집했고, 방문·맛 분류는 규칙 기반 자동 추정이라 맛집 인증으로 해석할 수 없습니다.
