import pytest
import json
import tempfile
import os
from answer_generation import (
    RetrievedDoc, build_answer_prompt,
    load_rewritten_queries, load_topk_vectors
)
from ragas_evaluation import (
    load_generated_answers, build_ragas_input,
    save_json, REFERENCE_LIST
)


class TestAnswerGeneration:
    """测试答案生成模块（只测试稳定的功能）"""

    def test_retrieved_doc_from_any(self):
        """测试RetrievedDoc的创建"""
        # 测试从字符串创建
        doc = RetrievedDoc.from_any("这是一段文本")
        assert doc.text == "这是一段文本"
        assert doc.id == -1

        # 测试从字典创建
        doc_dict = {"id": 1, "text": "测试文本", "metadata": {"source": "test"}}
        doc = RetrievedDoc.from_any(doc_dict)
        assert doc.id == 1
        assert doc.text == "测试文本"
        assert doc.metadata["source"] == "test"

    def test_build_answer_prompt(self):
        """测试提示词构建"""
        contexts = [
            {"id": 1, "text": "医学文档1", "metadata": {"department": "内科"}},
            {"id": 2, "text": "医学文档2", "metadata": {"title": "测试标题"}}
        ]

        result = build_answer_prompt("测试问题", contexts)

        assert "system_prompt" in result
        assert "user_prompt" in result
        assert "测试问题" in result["user_prompt"]

    def test_load_rewritten_queries(self):
        """测试加载改写后的问题"""
        # 创建临时测试文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            test_data = [
                {"id": 1, "rewritten_query": "问题1"},
                {"id": 2, "rewritten_query": "问题2"}
            ]
            json.dump(test_data, f)
            temp_path = f.name

        try:
            result = load_rewritten_queries(temp_path)
            assert result[1] == "问题1"
            assert result[2] == "问题2"
        finally:
            os.unlink(temp_path)

    def test_load_topk_vectors(self):
        """测试加载topk向量"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            test_data = [
                {
                    "query_id": 1,
                    "topk_docs": [
                        {"id": 1, "text": "文档1"},
                        {"id": 2, "text": "文档2"}
                    ]
                }
            ]
            json.dump(test_data, f)
            temp_path = f.name

        try:
            result = load_topk_vectors(temp_path)
            assert 1 in result
            assert len(result[1]) == 2
        finally:
            os.unlink(temp_path)


class TestRagasEvaluation:
    """测试RAGAS评估模块（只测试稳定的功能）"""

    def test_load_generated_answers(self):
        """测试加载生成的答案"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            test_data = [
                {
                    "user_input": "问题1",
                    "response": "回答1",
                    "retrieved_contexts": ["上下文1"]
                }
            ]
            json.dump(test_data, f)
            temp_path = f.name

        try:
            result = load_generated_answers(temp_path)
            assert len(result) == 1
            assert result[0]["user_input"] == "问题1"
        finally:
            os.unlink(temp_path)

    def test_build_ragas_input(self):
        """测试构建RAGAS输入"""
        generated = [
            {
                "user_input": "医学问题1",
                "response": "医学回答1",
                "retrieved_contexts": ["ctx1"]
            }
        ]

        references = [
            {
                "user_input": "医学问题1",
                "reference": "标准答案1"
            }
        ]

        result = build_ragas_input(generated, references)

        assert len(result) == 1
        assert result[0]["user_input"] == "医学问题1"
        assert "reference" in result[0]

    def test_save_json(self):
        """测试保存JSON文件"""
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            temp_path = f.name

        try:
            test_data = {"key": "value"}
            save_json(test_data, temp_path)

            with open(temp_path, 'r', encoding='utf-8') as f:
                loaded = json.load(f)
            assert loaded == test_data
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


def test_reference_data():
    """测试参考数据完整性"""
    assert len(REFERENCE_LIST) == 18
    for ref in REFERENCE_LIST:
        assert "user_input" in ref
        assert "reference" in ref
        assert len(ref["reference"]) > 0


def test_smoke():
    """冒烟测试 - 最基本的验证"""
    # 测试RetrievedDoc基础功能
    doc = RetrievedDoc.from_any("test")
    assert hasattr(doc, 'text')

    # 测试提示词构建不报错
    prompt = build_answer_prompt("test", [])
    assert isinstance(prompt, dict)

    # 测试参考数据可访问
    assert isinstance(REFERENCE_LIST, list)


if __name__ == "__main__":
    # 运行所有稳定的测试
    pytest.main([__file__, "-v"])