import unittest
from unittest.mock import patch, MagicMock
import sys
import os


current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
server_path = os.path.join(project_root, 'server')
if server_path not in sys.path:
    sys.path.insert(0, server_path)


# Mock dependencies
sys.modules['chromadb'] = MagicMock()
sys.modules['git'] = MagicMock()
sys.modules['zhipuai'] = MagicMock()
sys.modules['langchain_community.document_loaders.generic'] = MagicMock()
sys.modules['langchain_community.document_loaders.parsers'] = MagicMock()
sys.modules['langchain_text_splitters'] = MagicMock()

import indexer_template

class TestIndexer(unittest.TestCase):


    @patch('indexer_template.chromadb.PersistentClient')
    @patch('indexer_template.GenericLoader')
    @patch('indexer_template.RecursiveCharacterTextSplitter')
    def test_build_index_flow(self, MockSplitter, MockLoader, MockClient):
        # Setup
        indexer_template.API_KEY = "test_key"
        
        # Mock Loader yielding docs
        mock_doc = MagicMock()
        mock_doc.page_content = "code"
        mock_doc.metadata = {"source": "file.py"}
        MockLoader.from_filesystem.return_value.load.return_value = [mock_doc]
        
        # Mock Splitter
        MockSplitter.from_language.return_value.split_documents.return_value = [mock_doc]
        
        # Mock DB Collection
        mock_collection = MockClient.return_value.get_or_create_collection.return_value
        
        # Run
        indexer_template.build_index()
        
        # Assert
        # Check if collection.add was called (meaning data was tried to be inserted)
        mock_collection.add.assert_called()
        self.assertTrue(MockLoader.from_filesystem.called)

if __name__ == '__main__':
    unittest.main()