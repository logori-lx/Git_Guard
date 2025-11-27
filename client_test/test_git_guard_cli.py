import unittest
from unittest.mock import patch, MagicMock, mock_open
import sys
import os
import stat

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
client_path = os.path.join(project_root, 'client')
if client_path not in sys.path:
    sys.path.insert(0, client_path)

import git_guard_cli

class TestGitGuardCLI(unittest.TestCase):

    @patch('git_guard_cli.requests.get')
    def test_download_script_success(self, mock_get):
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {"code": "print('success')"}
        
        with patch('builtins.open', new_callable=mock_open) as mock_file:
            result = git_guard_cli.download_script("analyzer", "path/to/save")
            
            self.assertTrue(result)
            mock_file.assert_called_with("path/to/save", "w", encoding="utf-8")
            mock_file().write.assert_called_with("print('success')")

    @patch('git_guard_cli.requests.get')
    def test_download_script_failure(self, mock_get):
        mock_get.return_value.status_code = 404
        result = git_guard_cli.download_script("analyzer", "path")
        self.assertFalse(result)

    @patch('git_guard_cli.os.path.exists')
    @patch('git_guard_cli.os.makedirs')
    @patch('git_guard_cli.download_script')
    @patch('git_guard_cli.os.chmod')
    @patch('git_guard_cli.os.stat')
    @patch('builtins.open', new_callable=mock_open)
    def test_install_flow(self, mock_file, mock_stat, mock_chmod, mock_download, mock_makedirs, mock_exists):
        # Simulate .git existing
        mock_exists.side_effect = lambda p: p == ".git" or p == ".git/hooks"
        
        # Simulate successful downloads
        mock_download.return_value = True
        
        # Run install
        git_guard_cli.install()
        
        # Assertions
        # 1. Check downloads trigger
        self.assertEqual(mock_download.call_count, 2) # analyzer + indexer
        
        # 2. Check hooks are written (commit-msg, pre-push, pre-commit)
        # Note: Depending on implementation details, open might be called more times
        self.assertTrue(mock_file.called)
        
        # 3. Check chmod execution (making scripts executable)
        self.assertTrue(mock_chmod.called)

    @patch('git_guard_cli.os.path.exists', return_value=False)
    def test_install_no_git_repo(self, mock_exists):
        # Should return early if .git doesn't exist
        with patch('builtins.print') as mock_print:
            git_guard_cli.install()
            mock_print.assert_any_call("❌ Error: Not a git repo.")

if __name__ == '__main__':
    unittest.main()