"""
Author : Faiz Tariq
Date : 11/22/2019
Desc : Abstractive Text Summarization using NLP - UTC
"""

import unittest
from text_summarization.text_summary import predict_summary

class TestSummaryResult(unittest.TestCase):
    def test_summary(self):
        """This is a method for running test cases"""
        
        self.assertEqual(predict_summary('This is a great thing to use.'), ' great product', 'OK')
        self.assertEqual(predict_summary('This coffee tastes delicious.'), ' great tasting', 'OK')