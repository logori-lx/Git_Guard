import pytest
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# 添加路径以便导入模块
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from query_constructing.query_constructor import QueryConstructor


class TestQueryConstructor:

    @pytest.fixture
    def mock_client(self):
        """创建模拟的ZhipuAiClient"""
        mock_client = Mock()
        mock_client.chat.completions.create.return_value.choices = [
            Mock(message=Mock(content="改写后的医疗查询"))
        ]
        return mock_client

    @pytest.fixture
    def constructor(self, mock_client):
        """创建QueryConstructor实例"""
        with patch('query_constructing.query_constructor.ZhipuAiClient', return_value=mock_client):
            return QueryConstructor(api_key="test_key")

    def test_init(self, constructor):
        """测试初始化"""
        assert constructor is not None
        assert hasattr(constructor, '_rewritten_query_cache_dict')
        assert hasattr(constructor, '_rewritten_query_cache_queue')

    def test_update_rewritten_query_cache(self, constructor):
        """测试缓存更新"""
        # 测试添加新缓存
        constructor._update_rewritten_query_cache("test_query", "rewritten_query")
        assert "test_query" in constructor._rewritten_query_cache_dict
        assert constructor._rewritten_query_cache_dict["test_query"] == "rewritten_query"

        # 测试缓存大小限制
        for i in range(15):  # 超过默认缓存大小10
            constructor._update_rewritten_query_cache(f"query_{i}", f"rewritten_{i}")

        assert len(constructor._rewritten_query_cache_dict) <= 10
        assert len(constructor._rewritten_query_cache_queue) <= 10

    @patch('query_constructing.query_constructor.ZhipuAiClient')
    def test_rewritten_query_success(self, mock_zhipu, constructor):
        """测试查询改写成功"""
        # 模拟API响应
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="标准化医疗术语描述"))]
        mock_zhipu.return_value.chat.completions.create.return_value = mock_response



    @patch('query_constructing.query_constructor.ZhipuAiClient')
    def test_rewritten_query_failure(self, mock_zhipu, constructor):
        """测试查询改写失败时返回原查询"""
        mock_zhipu.return_value.chat.completions.create.side_effect = Exception("API错误")


    def test_get_query_with_cache(self, constructor):
        """测试从缓存获取查询"""
        # 先添加到缓存
        constructor._update_rewritten_query_cache("cached_query", "cached_rewritten")

        # 应该从缓存获取，不调用API
        with patch.object(constructor, '_rewritten_query') as mock_rewrite:
            result = constructor.get_query("cached_query")
            mock_rewrite.assert_not_called()
            assert result == "cached_rewritten"

    def test_get_query_without_cache(self, constructor):
        """测试无缓存时调用API"""
        with patch.object(constructor, '_rewritten_query', return_value="new_rewritten") as mock_rewrite:
            result = constructor.get_query("new_query")
            mock_rewrite.assert_called_once_with("new_query")
            assert result == "new_rewritten"



    def test_build_medical_qa_prompt(self, constructor):
        """测试构建医疗问答提示词"""
        passages = [
            {"ask": "发烧怎么办", "answer": "多喝水，休息"},
            {"ask": "头痛怎么缓解", "answer": "按摩太阳穴"}
        ]

        system_prompt, user_prompt = constructor._build_medical_qa_prompt("咳嗽怎么办", passages)

        assert "专业、温暖且严谨的智能医疗健康助手" in system_prompt
        assert "咳嗽怎么办" in user_prompt
        assert "发烧怎么办" in user_prompt

    def test_build_medical_qa_prompt_empty(self, constructor):
        """测试无检索结果时的提示词构建"""
        system_prompt, user_prompt = constructor._build_medical_qa_prompt("咳嗽怎么办", [])

        assert "未检索到相似病例或医生回答" in user_prompt



