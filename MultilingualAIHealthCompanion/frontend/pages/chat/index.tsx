// pages/chat/index.tsx
import { useState, useRef, useEffect } from 'react';
import Head from 'next/head';
import Link from 'next/link';
import axios from 'axios';
import ReactMarkdown from 'react-markdown';

export default function ChatPage() {
  const [messages, setMessages] = useState<{id: string, text: string, sender: 'user' | 'bot', timestamp: Date, citations?: any[]}[]>([]);
  const [inputText, setInputText] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const [availableReports, setAvailableReports] = useState<any[]>([]);
  const [selectedReports, setSelectedReports] = useState<string[]>([]);
  const messagesEndRef = useRef<null | HTMLDivElement>(null);

  // Fetch available reports when the page loads
  useEffect(() => {
    fetchAvailableReports();
  }, []);

  const fetchAvailableReports = async () => {
    try {
      const response = await axios.get('http://localhost:8000/admin/reports');
      setAvailableReports(response.data.reports || []);
    } catch (err) {
      console.error('Error fetching reports:', err);
      // Continue without reports if there's an error
    }
  };

  const handleReportSelection = (jobId: string) => {
    if (selectedReports.includes(jobId)) {
      setSelectedReports(selectedReports.filter(id => id !== jobId));
    } else {
      setSelectedReports([...selectedReports, jobId]);
    }
  };

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSendMessage = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!inputText.trim() || isLoading) {
      return;
    }

    try {
      setIsLoading(true);
      setError('');
      
      // Add user message
      const userMessage = {
        id: Date.now().toString(),
        text: inputText,
        sender: 'user' as const,
        timestamp: new Date()
      };
      
      setMessages(prev => [...prev, userMessage]);
      
      // Prepare request to backend
      const chatRequest = {
        query: inputText,
        top_k: 5,
        job_ids: selectedReports.length > 0 ? selectedReports : undefined
      };
      
      // Call the RAG chatbot API
      const response = await axios.post('http://localhost:8000/chat/query', chatRequest);
      
      // Add bot response
      const botMessage = {
        id: (Date.now() + 1).toString(),
        text: response.data.answer,
        sender: 'bot' as const,
        timestamp: new Date(),
        citations: response.data.citations || []
      };
      
      setMessages(prev => [...prev, botMessage]);
      setInputText('');
      
    } catch (err: any) {
      console.error('Chat error:', err);
      
      if (err.response?.status === 500) {
        setError('The RAG chatbot service is not available. Please ensure the backend server is running and the RAG service is properly initialized.');
      } else if (err.response?.status === 422) {
        setError('Invalid request. Please check your query and try again.');
      } else if (err.request) {
        setError('Network error. Please check if the backend server is running on http://localhost:8000');
      } else {
        setError('Failed to send message. Please try again.');
      }
    } finally {
      setIsLoading(false);
    }
  };

  const formatCitations = (citations: any[] = []) => {
    if (!citations || citations.length === 0) return null;
    
    return (
      <div className="mt-2 text-xs text-gray-500">
        <p className="font-medium">Sources:</p>
        <ul className="list-disc list-inside">
          {citations.slice(0, 3).map((citation, index) => (
            <li key={index}>
              {citation.source_document} (confidence: {(citation.confidence * 100).toFixed(1)}%)
            </li>
          ))}
          {citations.length > 3 && <li>... and {citations.length - 3} more sources</li>}
        </ul>
      </div>
    );
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
      <Head>
        <title>Medical RAG Chatbot - AI Health Companion</title>
        <meta name="description" content="Chat with your health reports using AI" />
      </Head>

      <header className="bg-white shadow-sm">
        <div className="max-w-7xl mx-auto px-4 py-4 sm:px-6 lg:px-8 flex justify-between items-center">
          <Link href="/" className="flex items-center space-x-3">
            <div className="bg-blue-600 p-2 rounded-lg">
              <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z" />
              </svg>
            </div>
            <h1 className="text-xl font-bold text-gray-900">Medical RAG Chatbot</h1>
          </Link>
          <Link href="/" className="text-blue-600 hover:text-blue-800 font-medium">
            ← Back to Upload
          </Link>
        </div>
      </header>

      <main className="max-w-4xl mx-auto px-4 py-6 sm:px-6 lg:px-8">
        <div className="bg-white rounded-2xl shadow-xl overflow-hidden flex flex-col h-[70vh]">
          {/* Chat header */}
          <div className="bg-gray-50 px-6 py-4 border-b border-gray-200">
            <h2 className="text-lg font-semibold text-gray-800">Chat with your health reports</h2>
            <p className="text-sm text-gray-600 mt-1">Ask questions about your medical data and get AI-powered answers</p>
          </div>

          {/* Report selection panel */}
          <div className="bg-blue-50 px-6 py-3 border-b border-gray-200">
            <h3 className="text-sm font-medium text-gray-700 mb-2">Select reports to search:</h3>
            <div className="flex flex-wrap gap-2">
              {availableReports.length > 0 ? (
                availableReports.map(report => (
                  <button
                    key={report.job_id}
                    onClick={() => handleReportSelection(report.job_id)}
                    className={`px-3 py-1 text-xs rounded-full border ${
                      selectedReports.includes(report.job_id)
                        ? 'bg-blue-100 border-blue-300 text-blue-800'
                        : 'bg-white border-gray-300 text-gray-700 hover:bg-gray-50'
                    }`}
                  >
                    Report {report.job_id.substring(0, 8)}...
                    {report.created_at && ` (${new Date(report.created_at).toLocaleDateString()})`}
                  </button>
                ))
              ) : (
                <p className="text-sm text-gray-500">No reports available. Process some health reports first.</p>
              )}
            </div>
            {selectedReports.length > 0 && (
              <p className="text-xs text-gray-500 mt-1">
                {selectedReports.length} report{selectedReports.length !== 1 ? 's' : ''} selected for search
              </p>
            )}
          </div>

          {/* Chat messages container */}
          <div className="flex-1 overflow-y-auto p-4 bg-gray-50">
            {messages.length === 0 ? (
              <div className="flex flex-col items-center justify-center h-full text-center">
                <div className="bg-blue-100 p-4 rounded-full mb-4">
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-12 w-12 text-blue-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z" />
                  </svg>
                </div>
                <h3 className="text-lg font-medium text-gray-900 mb-1">Medical RAG Chatbot</h3>
                <p className="text-gray-500 max-w-md">
                  Ask questions about your health reports and get AI-generated answers based on your medical data.
                  Select specific reports above to limit the search scope.
                </p>
                <div className="mt-6 grid grid-cols-1 md:grid-cols-2 gap-4 max-w-lg">
                  <div className="bg-white p-4 rounded-lg shadow-sm border border-gray-200">
                    <h4 className="font-medium text-gray-900 mb-2">Examples:</h4>
                    <ul className="text-sm text-gray-600 space-y-1">
                      <li className="flex items-start">
                        <span className="text-blue-500 mr-2">•</span>
                        <span>What are my cholesterol levels?</span>
                      </li>
                      <li className="flex items-start">
                        <span className="text-blue-500 mr-2">•</span>
                        <span>Is my blood sugar normal?</span>
                      </li>
                    </ul>
                  </div>
                  <div className="bg-white p-4 rounded-lg shadow-sm border border-gray-200">
                    <h4 className="font-medium text-gray-900 mb-2">Capabilities:</h4>
                    <ul className="text-sm text-gray-600 space-y-1">
                      <li className="flex items-start">
                        <span className="text-green-500 mr-2">✓</span>
                        <span>Medical terminology</span>
                      </li>
                      <li className="flex items-start">
                        <span className="text-green-500 mr-2">✓</span>
                        <span>Reference ranges</span>
                      </li>
                    </ul>
                  </div>
                </div>
              </div>
            ) : (
              <div className="space-y-4">
                {messages.map((message) => (
                  <div 
                    key={message.id} 
                    className={`flex ${message.sender === 'user' ? 'justify-end' : 'justify-start'}`}
                  >
                    <div 
                      className={`max-w-[80%] rounded-2xl px-4 py-3 ${
                        message.sender === 'user' 
                          ? 'bg-blue-500 text-white rounded-br-none' 
                          : 'bg-gray-200 text-gray-800 rounded-bl-none'
                      }`}
                    >
                      {message.sender === 'bot' ? (
                        <div className="prose prose-sm max-w-none">
                          <ReactMarkdown>{message.text}</ReactMarkdown>
                        </div>
                      ) : (
                        <div className="whitespace-pre-wrap">{message.text}</div>
                      )}
                      {message.citations && formatCitations(message.citations)}
                      <div className={`text-xs mt-1 ${message.sender === 'user' ? 'text-blue-100' : 'text-gray-500'}`}>
                        {message.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                      </div>
                    </div>
                  </div>
                ))}
                {isLoading && (
                  <div className="flex justify-start">
                    <div className="bg-gray-200 text-gray-800 rounded-2xl rounded-bl-none px-4 py-3 max-w-[80%]">
                      <div className="flex items-center">
                        <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-gray-600 mr-2"></div>
                        <span>Thinking...</span>
                      </div>
                    </div>
                  </div>
                )}
                <div ref={messagesEndRef} />
              </div>
            )}
          </div>

          {/* Chat input */}
          <div className="border-t border-gray-200 bg-white p-4">
            {error && (
              <div className="mb-3 p-3 bg-red-50 text-red-700 rounded-lg border border-red-200 text-sm">
                {error}
              </div>
            )}
            
            <form onSubmit={handleSendMessage} className="flex gap-2">
              <input
                type="text"
                value={inputText}
                onChange={(e) => setInputText(e.target.value)}
                placeholder="Ask about your health reports..."
                className="flex-1 border border-gray-300 rounded-lg px-4 py-3 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 text-black"
                disabled={isLoading}
              />
              <button
                type="submit"
                disabled={isLoading || !inputText.trim()}
                className={`px-6 py-3 rounded-lg text-white font-medium ${
                  isLoading || !inputText.trim()
                    ? 'bg-gray-400 cursor-not-allowed'
                    : 'bg-blue-600 hover:bg-blue-700'
                }`}
              >
                Send
              </button>
            </form>
            <p className="text-xs text-gray-500 mt-2">
              Important: This is an AI-generated response based on health records. It is not a substitute for professional medical advice.
            </p>
          </div>
        </div>

        {/* Info panel */}
        <div className="mt-6 bg-white rounded-2xl shadow-md p-6">
          <h3 className="font-semibold text-lg text-gray-800 mb-3">How the Medical RAG Chatbot Works</h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="flex items-start">
              <div className="bg-blue-100 p-2 rounded-lg mr-3">
                <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 text-blue-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z" />
                </svg>
              </div>
              <div>
                <h4 className="font-medium text-gray-800">Retrieval</h4>
                <p className="text-sm text-gray-600">Finds relevant information in your health reports using advanced techniques</p>
              </div>
            </div>
            <div className="flex items-start">
              <div className="bg-green-100 p-2 rounded-lg mr-3">
                <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 text-green-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
              </div>
              <div>
                <h4 className="font-medium text-gray-800">Generation</h4>
                <p className="text-sm text-gray-600">Creates answers based on retrieved information and medical knowledge</p>
              </div>
            </div>
            <div className="flex items-start">
              <div className="bg-purple-100 p-2 rounded-lg mr-3">
                <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 text-purple-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
              </div>
              <div>
                <h4 className="font-medium text-gray-800">Citations</h4>
                <p className="text-sm text-gray-600">Provides source information and confidence scores for transparency</p>
              </div>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}