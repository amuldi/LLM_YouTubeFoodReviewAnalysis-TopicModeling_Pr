"""노트북의 경계 조건을 검사합니다. 합성 입력은 실제 분석에 사용하지 않습니다."""

import argparse
import ast
import hashlib
import json
import math
import os
import re
import sys
import tempfile
import time
import unicodedata
import html
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import parse_qs, urlparse
from unittest.mock import patch

import nbformat
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "맛집추천 유튜브 영상 토픽분석 및 모델링_20221266_전시훈.ipynb"


def check_helpers(notebook):
    namespace = dict(globals())
    namespace.update(MIN_TASTE_REVIEWS=20)
    for cell in notebook.cells:
        if cell.cell_type != "code" or "%pip" in cell.source:
            continue
        tree = ast.parse(cell.source)
        assert cell.source.startswith("#"), "모든 코드 셀에 한글 주석이 필요합니다."
        definitions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id.endswith("_PATTERN") for t in node.targets):
                definitions.append(node)
        exec(compile(ast.Module(body=definitions, type_ignores=[]), "notebook_helpers", "exec"), namespace)

    extract = namespace["extract_video_id"]
    assert extract("https://www.youtube.com/shorts/abcdefghijk") == "abcdefghijk"
    assert extract("https://youtu.be/abcdefghijk?t=1") == "abcdefghijk"
    assert extract("https://www.youtube.com/watch?v=abcdefghijk&feature=share") == "abcdefghijk"
    for invalid in ["https://example.org/abcdefghijk", "https://youtube.com/watch?v=short"]:
        try:
            extract(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError(invalid)

    videos_cell = next(cell for cell in notebook.cells if cell.cell_type == "code" and "VIDEOS =" in cell.source)
    videos_tree = ast.parse(videos_cell.source)
    videos_assignment = next(
        node for node in videos_tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "VIDEOS" for target in node.targets)
    )
    videos = namespace["validate_videos"](ast.literal_eval(videos_assignment.value))
    video_ids = {video["video_id"] for video in videos}
    assert len(videos) == 14, f"고정 영상 14개 설정, 실제 {len(videos)}개"
    assert {
        "p7kmGI7EVu8", "V6h3uI3GaAg", "hDu3qScgfvE", "2BZRoAppTsU",
        "sUTjcaiRtOA", "aOs0ymFYBRE", "6CT5XCZ4H10", "o66zq2R4Tfg",
    }.issubset(video_ids)
    columns_cell = next(cell for cell in notebook.cells if cell.cell_type == "code" and "RAW_COLUMNS =" in cell.source)
    columns_tree = ast.parse(columns_cell.source)
    columns_assignment = next(
        node for node in columns_tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "RAW_COLUMNS" for target in node.targets)
    )
    namespace["RAW_COLUMNS"] = ast.literal_eval(columns_assignment.value)
    api_key_file = Path(tempfile.mkdtemp()) / "youtube_api_key.txt"
    api_key_file.write_text("PASTE_YOUTUBE_DATA_API_KEY_HERE")
    namespace["DATA_DIR"] = api_key_file.parent
    try:
        namespace["load_youtube_api_key"]()
    except ValueError:
        pass
    else:
        raise AssertionError("API 키 placeholder를 유효한 키로 받아들였습니다.")
    api_key_file.write_text("test-key")

    class FakeRequest:
        def __init__(self, result):
            self.result = result

        def execute(self):
            return self.result

    class FakeCommentsResource:
        def __init__(self):
            self.calls = []
            self.responses = iter([
                {
                    "items": [{
                        "snippet": {"topLevelComment": {
                            "id": "top_1",
                            "snippet": {"textDisplay": "댓글 하나 맛있어요", "publishedAt": "2026-01-01T00:00:00Z",
                                        "authorChannelId": {"value": "author_channel_1"}, "likeCount": 3},
                        }}
                    }],
                    "nextPageToken": "next-page",
                },
                {
                    "items": [{
                        "snippet": {"topLevelComment": {
                            "id": "top_2",
                            "snippet": {"textDisplay": "댓글 둘 별로예요", "publishedAt": "2026-01-02T00:00:00Z",
                                        "authorChannelId": {"value": "author_channel_2"}, "likeCount": 0},
                        }}
                    }],
                },
            ])

        def list(self, **kwargs):
            self.calls.append(kwargs)
            return FakeRequest(next(self.responses))

    fake_comments = FakeCommentsResource()
    with patch("googleapiclient.discovery.build", return_value=type(
        "FakeYoutube", (), {"commentThreads": lambda self: fake_comments}
    )()), patch("time.sleep"):
        collected, collection_log = namespace["collect_comments"](
            [{"restaurant": "검증 영상", "video_id": "abcdefghijk",
              "url": "https://www.youtube.com/watch?v=abcdefghijk", "channel": "검증 채널"}],
            2,
        )
    assert len(collected) == 2 and collection_log.iloc[0]["status"] == "limit_reached"
    assert fake_comments.calls[0]["order"] == "time" and fake_comments.calls[0]["textFormat"] == "plainText"
    assert fake_comments.calls[0]["pageToken"] is None and fake_comments.calls[1]["pageToken"] == "next-page"
    assert collected["author_hash"].str.len().eq(24).all()

    class FakeSearchResource:
        def __init__(self):
            self.arguments = None

        def list(self, **kwargs):
            self.arguments = kwargs
            return FakeRequest({"items": [
                {"id": {"videoId": "searchvid01"},
                 "snippet": {"title": "&quot;검색 결과&quot; 제목", "channelTitle": "검색 채널"}},
                {"id": {"videoId": "searchvid02"},
                 "snippet": {"title": "두 번째 제목", "channelTitle": "다른 채널"}},
            ]})

    fake_search = FakeSearchResource()
    fake_search_youtube = type(
        "FakeSearchYoutube", (), {"search": lambda self: fake_search}
    )()
    with patch("googleapiclient.discovery.build", return_value=fake_search_youtube):
        search_videos, search_rows = namespace["search_top_videos"](
            fake_search_youtube, "맛집", max_results=30
        )
    assert len(search_videos) == 2 and len(search_rows) == 2
    assert fake_search.arguments == {
        "part": "snippet", "q": "맛집", "type": "video", "regionCode": "KR",
        "relevanceLanguage": "ko", "order": "relevance", "maxResults": 30,
    }
    assert search_rows["rank"].tolist() == [1, 2]
    assert search_videos[0]["video_id"] == "searchvid01"
    assert search_videos[0]["restaurant"] == '"검색 결과" 제목'

    class FakeHttpError(Exception):
        resp = type("Response", (), {"status": 403})()

        def __str__(self):
            return "request URL contains secret-test-key"

    class FailingRequest:
        def execute(self):
            raise FakeHttpError()

    class FailingCommentsResource:
        def list(self, **kwargs):
            return FailingRequest()

    api_key_file.write_text("secret-test-key")
    failing_youtube = type(
        "FailingYoutube", (), {"commentThreads": lambda self: FailingCommentsResource()}
    )()
    with patch("googleapiclient.discovery.build", return_value=failing_youtube), patch("time.sleep"):
        _, failed_log = namespace["collect_comments"](
            [{"restaurant": "검증 영상", "video_id": "abcdefghijk",
              "url": "https://www.youtube.com/watch?v=abcdefghijk", "channel": "검증 채널"}],
            2,
        )
    assert failed_log.iloc[0]["status"] == "error"
    assert "HTTP 403" in failed_log.iloc[0]["error"]
    assert "secret-test-key" not in failed_log.iloc[0]["error"]
    for filename in ["youtube_api_key.txt", "raw_comments.csv", "collection_log.csv", ".author_salt"]:
        (api_key_file.parent / filename).unlink()
    api_key_file.parent.rmdir()

    classify = namespace["classify_comment"]
    cases = [
        ("먹어봤는데 맛있어요", "yes", "positive", "positive"),
        ("직접 가봤는데 맛없어요", "yes", "negative", "negative"),
        ("맛있겠다 가보고 싶어요", "unknown", "unknown", "positive"),
        ("안 가봤지만 맛있어 보여요", "no", "unknown", "positive"),
        ("먹어봤는데 맛있지 않아요", "yes", "negative", "negative"),
        ("먹어봤는데 맛없지 않아요", "yes", "positive", "positive"),
        ("먹어봤는데 국수는 맛있고 고기는 맛없어요", "yes", "mixed", "neutral"),
        ("나 여기 맛있었음", "yes", "positive", "positive"),
        ("여기 실제로 별로임", "yes", "negative", "negative"),
        ("먹어본 적은 없는데 맛있어 보여요", "no", "unknown", "positive"),
        ("여기 맛있어 보여서 가보고 싶어요", "unknown", "unknown", "positive"),
        ("대기 시간이 길고 주차 정보가 궁금합니다", "unknown", "unknown", "neutral"),
    ]
    for text, visit, taste, sentiment in cases:
        actual = classify(text)
        assert (actual["auto_visit"], actual["auto_taste"], actual["auto_sentiment"]) == (
            visit, taste, sentiment
        ), (text, actual)
    mixed = classify("먹어봤는데 국수는 맛있고 고기는 맛없어요")
    assert mixed["auto_sentiment_mixed"] is True
    cleaned = namespace["clean_text"]("<b>맛있지 않아요</b> https://example.com @누군가 ㅋㅋㅋ")
    assert cleaned == "맛있지 않아요"
    assert all(math.isnan(x) for x in namespace["wilson_interval"](0, 0))
    low, high = namespace["wilson_interval"](5, 10)
    assert math.isclose(low, 0.2365895936, abs_tol=1e-8)
    assert math.isclose(high, 0.7634104064, abs_tol=1e-8)

    rows = pd.DataFrame([
        dict(restaurant="검증 가게", author_hash=f"a{i}", published_at=pd.Timestamp("2026-01-01", tz="UTC"),
             visit="yes", taste="positive" if i < 8 else "negative", target_ok=True,
             auto_visit="yes", channel="채널1" if i % 2 else "채널2", reviewed=True)
        for i in range(10)
    ])
    rows = pd.concat([rows, rows.iloc[[0]]], ignore_index=True)
    result = namespace["summarize_restaurant"](rows)
    assert result["all_comments"] == 11 and result["unique_authors"] == 10
    assert result["positive"] == 8 and result["negative"] == 2
    assert result["positive_rate"] == 0.8 and "판단 유보" in result["verdict"]
    assert result["automatic_positive_experiences"] == 9
    assert result["automatic_negative_experiences"] == 2
    assert math.isclose(result["net_score"], 100 * 7 / 11)
    rows["visit"] = "yes"
    rows["auto_visit"] = "yes"
    rows["reviewed"] = False
    rows["target_ok"] = False
    result = namespace["summarize_restaurant"](rows)
    assert result["taste_evidence_n"] == 0 and math.isnan(result["positive_rate"])
    assert result["auto_visit_claims"] == 11
    print("PASS: 노트북 형식·영상 URL·검색 API 조건·공식 API 페이지 수집·3분류 감성·전처리·부정문·분모·Wilson 구간")


def run_integration(notebook):
    from nbclient import NotebookClient
    from nbconvert import HTMLExporter

    directory = Path(tempfile.mkdtemp(prefix="restaurant_notebook_validation_"))
    (directory / "data").mkdir()
    templates = [
        ["먹어봤는데 국물이 맛있어요 깊고 진한 육수에 만족했습니다", "방문했는데 고기가 맛있어요 부드럽고 육즙이 풍부합니다"],
        ["먹어봤는데 음식이 맛없어요 너무 짜고 식어서 실망했어요", "가봤는데 맛이 별로예요 비린내가 심해서 남겼습니다"],
        ["대기 시간이 한 시간 넘네요 줄 서는 시스템과 예약 방법이 궁금해요", "오픈런 해야 한다는데 웨이팅 없이 입장할 수 있는 시간이 있나요"],
        ["음식 가격이 너무 비싸요 양이 적어서 가성비가 나빠 보입니다", "메뉴 가격이 올랐네요 한 접시 가격과 양을 생각하면 부담됩니다"],
        ["직원이 친절해서 좋았어요 서비스와 응대가 만족스럽습니다", "직원 응대가 불친절해요 주문을 무시해서 서비스가 아쉬웠습니다"],
        ["이 영상 협찬 광고 아닌가요 유료 광고 표시가 필요한 것 같아요", "홍보 영상처럼 보이는데 내돈내산인지 광고인지 궁금해요"],
    ]
    records = []
    for topic, variants in enumerate(templates):
        for index in range(40):
            serial = topic * 40 + index
            records.append({
                "restaurant": f"검증전용가게{index % 2}", "video_id": f"test_video_{index % 4}",
                "video_url": "검증용 합성 입력: 실제 영상 없음", "channel": f"검증채널{index % 3}",
                "comment_id": f"test_{serial}", "author_hash": f"author_{serial}",
                "text": f"{variants[index % 2]} 검증문장 {serial}",
                "published_at": "2026-01-01T00:00:00+00:00", "published_time_text": "검증용 고정시각",
                "likes_text": "0", "collected_at": "2026-01-02T00:00:00+00:00",
            })
    pd.DataFrame(records).to_csv(directory / "data/raw_comments.csv", index=False, encoding="utf-8-sig")

    test_notebook = nbformat.reads(nbformat.writes(notebook), as_version=4)
    for cell in test_notebook.cells:
        if cell.cell_type == "code" and "%pip" in cell.source:
            cell.source = "# 검증 환경에는 이미 필요한 패키지가 설치되어 있습니다."
        if cell.cell_type == "code" and "COLLECT_NEW = True" in cell.source:
            cell.source = cell.source.replace("COLLECT_NEW = True", "COLLECT_NEW = False")
        if cell.cell_type == "code" and "RUN_LLM = True" in cell.source:
            cell.source = cell.source.replace("RUN_LLM = True", "RUN_LLM = False")
    test_notebook.cells.insert(0, nbformat.v4.new_markdown_cell(
        "# 검증 전용: 합성 입력 240건\n실제 음식점 분석 결과가 아닙니다. 네트워크 수집 단계는 실행하지 않았습니다."
    ))
    kernel_dir = directory / "jupyter/kernels/validation"
    kernel_dir.mkdir(parents=True)
    (kernel_dir / "kernel.json").write_text(json.dumps({
        "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
        "display_name": "Notebook validation", "language": "python",
    }))
    os.environ["JUPYTER_PATH"] = str(directory / "jupyter")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["OMP_NUM_THREADS"] = "4"
    os.environ["MPLBACKEND"] = "Agg"
    print(f"통합 검증 경로: {directory}", flush=True)
    (ROOT / "scripts/.last_validation_path").write_text(str(directory))
    def started(cell_index, **kwargs):
        print(f"검증 셀 {cell_index} 실행", flush=True)
    client = NotebookClient(test_notebook, timeout=1200, kernel_name="validation",
                            resources={"metadata": {"path": str(directory)}}, on_cell_start=started)
    try:
        client.execute()
    finally:
        nbformat.write(test_notebook, directory / "검증전용.ipynb")
    html_body, _ = HTMLExporter().from_notebook_node(test_notebook)
    (directory / "검증전용.html").write_text(html_body)
    print("PASS: 합성 입력으로 실제 임베딩·BERTopic·감성 차트·보고서 통합 실행 (외부 LLM 다운로드 생략)", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--integration", action="store_true")
    args = parser.parse_args()
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    nbformat.validate(notebook)
    check_helpers(notebook)
    if args.integration:
        run_integration(notebook)
