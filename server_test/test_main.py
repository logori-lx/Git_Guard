import unittest
from unittest.mock import patch, MagicMock, mock_open
import sys
import os
import json

# Add server directory to path to import main
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
server_path = os.path.join(project_root, 'server')
if server_path not in sys.path:
    sys.path.insert(0, server_path)


from fastapi.testclient import TestClient
# We need to mock imports that might not exist in the test environment or require API keys
with patch.dict(sys.modules, {'git': MagicMock(), 'apscheduler.schedulers.asyncio': MagicMock()}):
    import main

class TestServerMain(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(main.app)
        
    @patch('main.os.path.exists')
    @patch('builtins.open', new_callable=mock_open, read_data='{"template_format": "test"}')
    def test_get_config(self, mock_file, mock_exists):
        mock_exists.return_value = True
        response = self.client.get("/api/v1/config")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['template_format'], "test")


    @patch('main.os.path.exists')
    @patch('builtins.open', new_callable=mock_open, read_data='print("hello")')
    def test_get_script_success(self, mock_file, mock_exists):
        # Mock path exists for the script
        mock_exists.return_value = True
        response = self.client.get("/api/v1/scripts/analyzer")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['code'], 'print("hello")')

    def test_get_script_invalid_name(self):
        response = self.client.get("/api/v1/scripts/hacker_script")
        self.assertEqual(response.status_code, 404)

if __name__ == '__main__':
    unittest.main()