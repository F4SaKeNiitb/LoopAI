// pages/report/[jobId].tsx
import { useState, useEffect } from 'react';
import { useRouter } from 'next/router';
import axios from 'axios';
import Head from 'next/head';
import ReactMarkdown from 'react-markdown';

export default function ReportPage() {
  const router = useRouter();
  const { jobId } = router.query;
  const [reportData, setReportData] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [selectedLanguage, setSelectedLanguage] = useState('en');
  const [isPlaying, setIsPlaying] = useState(false);
  const [activeSection, setActiveSection] = useState('summary');
  const [showAbnormalOnly, setShowAbnormalOnly] = useState(false);

  // Poll for report status until completed
  useEffect(() => {
    if (!jobId) return;

    const fetchReport = async () => {
      try {
        const response = await axios.get(`http://localhost:8000/report/${jobId}`);
        
        const { status, result } = response.data;

        if (status === 'COMPLETED') {
          setReportData(result);
          setLoading(false);
        } else if (status === 'FAILED') {
          setError(result?.error || 'Processing failed. Please try again.');
          setLoading(false);
        } else {
          // Still processing, poll again after 2 seconds
          setTimeout(fetchReport, 2000);
        }
      } catch (err: any) {
        console.error('Fetch report error:', err);
        
        if (err.response) {
          // Server responded with error status
          setError(`Server error: ${err.response.status} - ${err.response.data.detail || 'Failed to fetch report'}`);
        } else if (err.request) {
          // Request was made but no response received
          setError('Network error. Please check if the backend server is running on http://localhost:8000');
        } else {
          // Other error
          setError('Failed to fetch report. Please try again.');
        }
        setLoading(false);
      }
    };

    fetchReport();
  }, [jobId]);

  const handlePlayAudio = () => {
    if (reportData && reportData.translated_audio_urls[selectedLanguage]) {
      const audioUrl = reportData.translated_audio_urls[selectedLanguage].audio_url;
      if (audioUrl) {
        const audio = new Audio(audioUrl);
        audio.play();
        setIsPlaying(true);
        
        audio.onended = () => setIsPlaying(false);
        audio.onerror = () => {
          console.error('Error playing audio:', audioUrl);
          setIsPlaying(false);
        };
      }
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-16 w-16 border-t-4 border-b-4 border-blue-600 mx-auto"></div>
          <p className="mt-4 text-xl font-medium text-gray-700">Analyzing your health report...</p>
          <p className="text-gray-500 mt-2">This may take a few moments</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 flex items-center justify-center">
        <div className="bg-white p-10 rounded-2xl shadow-xl max-w-md text-center border border-gray-100">
          <div className="bg-red-100 w-16 h-16 rounded-full flex items-center justify-center mx-auto mb-6">
            <svg xmlns="http://www.w3.org/2000/svg" className="h-10 w-10 text-red-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
            </svg>
          </div>
          <h2 className="text-2xl font-bold text-gray-800 mb-3">Something went wrong</h2>
          <p className="text-gray-600 mb-6">{error}</p>
          <button 
            onClick={() => router.push('/')}
            className="px-6 py-3 bg-gradient-to-r from-blue-600 to-indigo-700 text-white rounded-lg font-medium shadow-md hover:from-blue-700 hover:to-indigo-800 transition duration-200"
          >
            Go Back Home
          </button>
        </div>
      </div>
    );
  }

  if (!reportData) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 flex items-center justify-center">
        <p className="text-lg text-gray-700">Failed to load report data</p>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
      <Head>
        <title>Health Report Analysis - Multilingual AI Health Companion</title>
        <meta name="description" content="Your analyzed health report" />
      </Head>

      <header className="bg-white shadow-sm border-b border-gray-100">
        <div className="max-w-7xl mx-auto px-4 py-6 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center mb-4">
            <div className="flex items-center space-x-3">
              <div className="bg-blue-600 p-2 rounded-lg">
                <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                </svg>
              </div>
              <h1 className="text-xl font-bold text-gray-900">Health Report Analysis</h1>
            </div>
            <div className="flex items-center space-x-4">
              <label htmlFor="language" className="text-sm font-medium text-gray-700 hidden sm:block">
                Language:
              </label>
              <select
                id="language"
                value={selectedLanguage}
                onChange={(e) => setSelectedLanguage(e.target.value)}
                className="border border-gray-300 rounded-lg px-4 py-2 bg-white text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 shadow-sm text-black"
              >
                <option value="en">English</option>
                <option value="hi">हिंदी (Hindi)</option>
                <option value="mr">मराठी (Marathi)</option>
              </select>
            </div>
          </div>
          
          {/* Patient Information Section */}
          {reportData.patient_info && (
            <div className="bg-gradient-to-r from-gray-50 to-gray-100 rounded-xl p-4 border border-gray-200">
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                {reportData.patient_info.name && (
                  <div className="bg-white rounded-lg p-3 shadow-sm">
                    <p className="text-xs text-gray-500">Patient Name</p>
                    <p className="font-medium text-gray-900">{reportData.patient_info.name}</p>
                  </div>
                )}
                {reportData.patient_info.age && (
                  <div className="bg-white rounded-lg p-3 shadow-sm">
                    <p className="text-xs text-gray-500">Age</p>
                    <p className="font-medium text-gray-900">{reportData.patient_info.age}</p>
                  </div>
                )}
                {reportData.patient_info.gender && (
                  <div className="bg-white rounded-lg p-3 shadow-sm">
                    <p className="text-xs text-gray-500">Gender</p>
                    <p className="font-medium text-gray-900">{reportData.patient_info.gender}</p>
                  </div>
                )}
                {reportData.patient_info.date && (
                  <div className="bg-white rounded-lg p-3 shadow-sm">
                    <p className="text-xs text-gray-500">Report Date</p>
                    <p className="font-medium text-gray-900">{reportData.patient_info.date}</p>
                  </div>
                )}
              </div>
              <div className="mt-2 grid grid-cols-1 md:grid-cols-2 gap-4">
                {reportData.patient_info.report_type && (
                  <div className="bg-white rounded-lg p-3 shadow-sm">
                    <p className="text-xs text-gray-500">Report Type</p>
                    <p className="font-medium text-gray-900">{reportData.patient_info.report_type}</p>
                  </div>
                )}
                {reportData.patient_info.medical_record_number && (
                  <div className="bg-white rounded-lg p-3 shadow-sm">
                    <p className="text-xs text-gray-500">MRN</p>
                    <p className="font-medium text-gray-900">{reportData.patient_info.medical_record_number}</p>
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      </header>

      <main className="max-w-6xl mx-auto px-4 py-8 sm:px-6 lg:px-8">
        {/* Warnings section - shown at top if present */}
        {reportData.warnings && reportData.warnings.length > 0 && (
          <div className="mb-8">
            {reportData.warnings.map((warning: string, index: number) => (
              <div 
                key={index} 
                className={`p-5 rounded-xl mb-4 ${
                  warning.toLowerCase().includes('urgent') 
                    ? 'bg-gradient-to-r from-red-50 to-red-100 border border-red-200 text-red-800' 
                    : 'bg-gradient-to-r from-yellow-50 to-yellow-100 border border-yellow-200 text-yellow-800'
                }`}
              >
                <div className="flex items-start">
                  <svg 
                    xmlns="http://www.w3.org/2000/svg" 
                    className={`h-6 w-6 mr-3 flex-shrink-0 ${
                      warning.toLowerCase().includes('urgent') ? 'text-red-600' : 'text-yellow-600'
                    }`} 
                    fill="none" 
                    viewBox="0 0 24 24" 
                    stroke="currentColor"
                  >
                    <path 
                      strokeLinecap="round" 
                      strokeLinejoin="round" 
                      strokeWidth={2} 
                      d={warning.toLowerCase().includes('urgent') 
                        ? "M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" 
                        : "M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"}
                    />
                  </svg>
                  <p className="text-sm font-medium">{warning}</p>
                </div>
              </div>
            ))}
          </div>
        )}

        {/* Audio Player for all languages */}
        {reportData.translated_audio_urls[selectedLanguage] && (
          <div className="bg-gradient-to-r from-blue-50 to-indigo-50 shadow-lg rounded-2xl p-6 mb-8 border border-blue-100">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between">
              <div className="mb-4 sm:mb-0">
                <h3 className="text-lg font-semibold text-gray-800 flex items-center">
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-2 text-blue-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15.536 8.464a5 5 0 010 7.072m2.828-9.9a9 9 0 010 12.728M5.586 15.536a5 5 0 001.414 1.414m1.414-4.242a5 5 0 010-7.07m-2.828 9.9a9 9 0 010-12.728" />
                  </svg>
                  Listen to Summary in {selectedLanguage === 'hi' ? 'Hindi' : selectedLanguage === 'mr' ? 'Marathi' : 'English'}
                </h3>
                <p className="text-sm text-gray-600 mt-1">Click the play button to hear the summary</p>
              </div>
              <button
                onClick={handlePlayAudio}
                disabled={isPlaying}
                className={`p-4 rounded-full flex items-center justify-center ${
                  isPlaying 
                    ? 'bg-gray-300 text-gray-500' 
                    : 'bg-gradient-to-r from-blue-600 to-indigo-700 text-white hover:from-blue-700 hover:to-indigo-800 shadow-md hover:shadow-lg transform hover:scale-105 transition duration-200'
                }`}
              >
                {isPlaying ? (
                  <div className="flex items-center">
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 10a1 1 0 011-1h4a1 1 0 011 1v4a1 1 0 01-1 1h-4a1 1 0 01-1-1v-4z" />
                    </svg>
                    <span className="ml-2">Playing...</span>
                  </div>
                ) : (
                  <div className="flex items-center">
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z" />
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                    </svg>
                    <span className="ml-2">Play Audio</span>
                  </div>
                )}
              </button>
            </div>
          </div>
        )}

        {/* Main Summary */}
        <div className="bg-white shadow-xl rounded-2xl p-8 mb-8 border border-gray-100">
          <div className="flex items-center mb-6">
            <div className="bg-blue-100 p-3 rounded-lg mr-4">
              <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 text-blue-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 8h10M7 12h4m1 8l-4-4H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-3l-4 4z" />
              </svg>
            </div>
            <h2 className="text-2xl font-bold text-gray-800">Your Health Summary</h2>
          </div>
          <div className="prose max-w-none text-gray-700 leading-relaxed">
            <div className="text-lg">
              <ReactMarkdown>
                {selectedLanguage === 'hi' 
                  ? reportData.translated_audio_urls['hi']?.text || reportData.summary
                  : selectedLanguage === 'mr'
                  ? reportData.translated_audio_urls['mr']?.text || reportData.summary
                  : reportData.translated_audio_urls['en']?.text || reportData.summary
                }
              </ReactMarkdown>
            </div>
          </div>
        </div>

        {/* Always Expanded Sections */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mb-8">
          {/* For Your Doctor - Always Expanded */}
          <div className="bg-white shadow-xl rounded-2xl border border-gray-100 overflow-hidden">
            <div className="w-full px-6 py-5 text-left font-semibold text-gray-800 flex items-center bg-gradient-to-r from-blue-50 to-indigo-50">
              <div className="flex items-center">
                <div className="bg-blue-100 p-2 rounded-lg mr-3">
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 text-blue-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0zm6 3a2 2 0 11-4 0 2 2 0 014 0zM7 10a2 2 0 11-4 0 2 2 0 014 0z" />
                  </svg>
                </div>
                <span>For Your Doctor</span>
              </div>
            </div>
            <div className="px-6 py-5 bg-white">
              <div className="text-gray-700">
                <ReactMarkdown>
                  {reportData.doctor_summary || "No doctor summary available"}
                </ReactMarkdown>
              </div>
            </div>
          </div>

          {/* Questions to Ask - Always Expanded */}
          <div className="bg-white shadow-xl rounded-2xl border border-gray-100 overflow-hidden">
            <div className="w-full px-6 py-5 text-left font-semibold text-gray-800 flex items-center bg-gradient-to-r from-purple-50 to-pink-50">
              <div className="flex items-center">
                <div className="bg-purple-100 p-2 rounded-lg mr-3">
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 text-purple-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8.228 9c.549-1.165 2.03-2 3.772-2 2.21 0 4 1.343 4 3 0 1.4-1.278 2.575-3.006 2.907-.542.104-.994.54-.994 1.093m0 3h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                </div>
                <span>Questions to Ask</span>
              </div>
            </div>
            <div className="px-6 py-5 bg-white">
              <ul className="space-y-3 text-gray-700">
                {reportData.questions && reportData.questions.length > 0 ? (
                  reportData.questions.map((question: string, index: number) => (
                    <li key={index} className="flex items-start">
                      <span className="text-purple-600 mr-2">•</span>
                      <span>{question}</span>
                    </li>
                  ))
                ) : (
                  <li>No suggested questions available</li>
                )}
              </ul>
            </div>
          </div>
        </div>

        {/* Lab Results Section with Filter */}
        <div className="bg-white shadow-xl rounded-2xl p-8 border border-gray-100">
          <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center mb-6">
            <div className="flex items-center mb-4 sm:mb-0">
              <div className="bg-green-100 p-3 rounded-lg mr-4">
                <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 text-green-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
                </svg>
              </div>
              <h2 className="text-2xl font-bold text-gray-800">Lab Results</h2>
            </div>
            
            {/* Filter button for abnormal results */}
            <div className="flex items-center space-x-4">
              <label className="flex items-center text-sm text-gray-700">
                <input
                  type="checkbox"
                  checked={showAbnormalOnly}
                  onChange={(e) => setShowAbnormalOnly(e.target.checked)}
                  className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                />
                <span className="ml-2">Show only abnormal results</span>
              </label>
            </div>
          </div>
          
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200 rounded-lg overflow-hidden">
              <thead className="bg-gradient-to-r from-gray-50 to-gray-100">
                <tr>
                  <th scope="col" className="px-6 py-4 text-left text-sm font-semibold text-gray-700 uppercase tracking-wider">Test</th>
                  <th scope="col" className="px-6 py-4 text-left text-sm font-semibold text-gray-700 uppercase tracking-wider">Value</th>
                  <th scope="col" className="px-6 py-4 text-left text-sm font-semibold text-gray-700 uppercase tracking-wider">Units</th>
                  <th scope="col" className="px-6 py-4 text-left text-sm font-semibold text-gray-700 uppercase tracking-wider">Normal Range</th>
                  <th scope="col" className="px-6 py-4 text-left text-sm font-semibold text-gray-700 uppercase tracking-wider">Status</th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {reportData.extracted_parameters && reportData.extracted_parameters.length > 0 ? (
                  reportData.extracted_parameters
                    .filter((param: any) => !showAbnormalOnly || param.is_abnormal)
                    .map((param: any, index: number) => (
                      <tr 
                        key={index} 
                        className={param.is_abnormal ? 'bg-red-50 hover:bg-red-100 transition-colors' : 'hover:bg-gray-50 transition-colors'}
                      >
                        <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">{param.field}</td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-700 font-medium">{param.value}</td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{param.units}</td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{param.reference_range}</td>
                        <td className="px-6 py-4 whitespace-nowrap">
                          <span className={`px-3 py-1 inline-flex text-xs leading-5 font-semibold rounded-full ${
                            param.is_abnormal 
                              ? param.is_critical 
                                ? 'bg-red-100 text-red-800' 
                                : 'bg-yellow-100 text-yellow-800'
                              : 'bg-green-100 text-green-800'
                          }`}>
                            {param.is_critical ? 'Critical' : param.is_abnormal ? 'Abnormal' : 'Normal'}
                          </span>
                        </td>
                      </tr>
                    ))
                ) : (
                  <tr>
                    <td colSpan={5} className="px-6 py-8 text-center text-sm text-gray-500">
                      <div className="flex flex-col items-center justify-center">
                        <svg xmlns="http://www.w3.org/2000/svg" className="h-12 w-12 text-gray-400 mb-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                        </svg>
                        No lab results found
                      </div>
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
            
            {showAbnormalOnly && reportData.extracted_parameters && 
             reportData.extracted_parameters.filter((p: any) => p.is_abnormal).length === 0 && (
              <div className="text-center py-4 text-gray-500">
                No abnormal results found
              </div>
            )}
          </div>
        </div>

        {/* Disclaimer */}
        <div className="mt-8 bg-gradient-to-r from-yellow-50 to-amber-50 border border-yellow-200 rounded-2xl p-6">
          <div className="flex items-start">
            <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 text-yellow-600 mr-3 flex-shrink-0 mt-0.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
            </svg>
            <p className="text-sm text-yellow-800">
              {reportData.disclaimer || 
                "This is an AI-generated summary and not a substitute for professional medical advice. Please consult your doctor."}
            </p>
          </div>
        </div>
      </main>

      <footer className="bg-white border-t border-gray-200 mt-12">
        <div className="max-w-7xl mx-auto px-4 py-8 sm:px-6 lg:px-8">
          <div className="flex flex-col md:flex-row justify-between items-center">
            <div className="flex items-center mb-4 md:mb-0">
              <div className="bg-blue-600 p-2 rounded-lg mr-3">
                <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                </svg>
              </div>
              <span className="text-lg font-semibold text-gray-900">AI Health Companion</span>
            </div>
            <p className="text-center text-gray-500 text-sm max-w-md">
              This is an AI-generated summary and not a substitute for professional medical advice. Please consult your doctor.
            </p>
          </div>
        </div>
      </footer>
    </div>
  );
}