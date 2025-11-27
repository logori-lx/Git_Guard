import unittest
from unittest.mock import patch, MagicMock, mock_open
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
server_path = os.path.join(project_root, 'server')
if server_path not in sys.path:
    sys.path.insert(0, server_path)

# Mock external dependencies before importing
sys.modules['chromadb'] = MagicMock()
sys.modules['git'] = MagicMock()
sys.modules['zhipuai'] = MagicMock()

import analyzer_template

class TestAnalyzer(unittest.TestCase):
    def test_clean_markdown(self):
        text = "**Bold** and `code` and *italic*"
        cleaned = analyzer_template.clean_markdown(text)
        self.assertEqual(cleaned, "Bold and code and italic")

    @patch('analyzer_template.requests.get')
    def test_fetch_dynamic_rules(self, mock_get):
        # Test success case
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {"template_format": "Test"}
        
        rules = analyzer_template.fetch_dynamic_rules()
        self.assertEqual(rules['template_format'], "Test")
        
        # Test failure case
        mock_get.side_effect = Exception("Connection error")
        rules = analyzer_template.fetch_dynamic_rules()
        self.assertEqual(rules['template_format'], "Standard")

    @patch('analyzer_template.requests.post')
    @patch('analyzer_template.getpass.getuser', return_value="test_user")
    def test_report_to_cloud(self, mock_user, mock_post):
        analyzer_template.report_to_cloud("msg", "High", "Summary")
        mock_post.assert_called_once()
        args = mock_post.call_args[1]['json']
        self.assertEqual(args['developer_id'], "test_user")
        self.assertEqual(args['risk_level'], "High")

    @patch('analyzer_template.process_changes_with_rag')
    @patch('analyzer_template.fetch_dynamic_rules')
    @patch('analyzer_template.ZhipuAI')
    @patch('builtins.open', new_callable=mock_open, read_data="fix bug")
    @patch('analyzer_template.get_console_input')
    def test_run_suggestion_mode_flow(self, mock_input, mock_open_file, MockZhipu, mock_rules, mock_process):
        # Setup Mocks
        mock_process.return_value = ({"file.py": "diff"}, "context")
        mock_rules.return_value = {"template_format": "Fmt", "custom_rules": "Rules"}
        mock_input.return_value = "1" # User selects option 1
        
        # Mock LLM response
        mock_client = MockZhipu.return_value
        mock_client.chat.completions.create.return_value.choices[0].message.content = """
        RISK: Low
        SUMMARY: Test Summary
        OPTIONS: [Opt 1] msg|||[Opt 2] msg|||[Opt 3] msg
        """

        # Execute
        # We need to prevent sys.exit in case of error, though this path shouldn't hit it
        analyzer_template.run_suggestion_mode("dummy_path")

        # Assertions
        mock_process.assert_called_once()
        MockZhipu.assert_called_once()
        # Ensure file was written (commit message updated)
        mock_open_file().write.assert_called()

if __name__ == '__main__':
    unittest.main()