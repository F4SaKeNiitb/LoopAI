# Multilingual AI Health Companion

A modern web application that processes health reports (lab results, medical documents) and provides easy-to-understand summaries in multiple languages with text-to-speech capabilities.

## Features

- **OCR and Text Extraction**: Process PDFs, images (JPG, PNG) and raw text
- **Parameter Extraction**: Identify and extract key health metrics from reports
- **AI-Powered Summaries**: Generate plain language and doctor-style summaries
- **Multilingual Support**: English, Hindi, and Marathi
- **Text-to-Speech**: Audio summaries for Hindi and Marathi
- **Clinician Review Dashboard**: Admin interface for healthcare professionals
- **Safety Warnings**: Critical result alerts and disclaimers
- **Modern UI**: Responsive, mobile-friendly design with intuitive user experience

## Tech Stack

### Backend
- **Framework**: FastAPI (Python)
- **OCR**: Pytesseract with OpenCV preprocessing
- **AI Processing**: Google Gemini API for summary generation
- **Translation & TTS**: Placeholder for ElevenLabs API
- **Database**: In-memory storage (can be extended with PostgreSQL/MongoDB)

### Frontend
- **Framework**: Next.js (React) with TypeScript
- **Styling**: Tailwind CSS
- **API Client**: Axios
- **UI Features**: Responsive design, animations, gradient effects, interactive components

## Installation

### Backend Setup

1. Navigate to the backend directory:
```bash
cd /path/to/MultilingualAIHealthCompanion/backend
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
# Create a .env file with your API keys
GEMINI_API_KEY=your_google_gemini_api_key
ELEVENLABS_API_KEY=your_elevenlabs_api_key
```

5. Run the backend server:
```bash
python main.py
```

The backend will start at `http://localhost:8000`

### Frontend Setup

1. Navigate to the frontend directory:
```bash
cd /path/to/MultilingualAIHealthCompanion/frontend
```

2. Install dependencies:
```bash
npm install
```

3. Run the development server:
```bash
npm run dev
```

The frontend will start at `http://localhost:3000`

## API Endpoints

### User Endpoints

- `POST /upload`: Upload a file (PDF, JPG, PNG) or text for analysis
  - Form data parameters:
    - `file` (optional): Uploaded file
    - `text` (optional): Raw text input
    - `language` (optional): Language code (default: "en")
  - Returns: `{"job_id": "string"}`

- `GET /report/{job_id}`: Get the status/result of a processing job
  - Returns: Status and results when completed

### Admin/Clinician Endpoints

- `GET /admin/reports`: List all processed reports
- `GET /admin/report/{job_id}`: Get specific report details
- `PUT /admin/report/{job_id}`: Update generated summary (authenticated)

## Application Structure

```
MultilingualAIHealthCompanion/
├── backend/
│   ├── main.py                 # FastAPI application and endpoints
│   ├── ocr_module.py           # OCR and file processing
│   ├── data_processing.py      # Parameter extraction and normalization
│   ├── lab_data.py             # Lab test definitions and reference ranges
│   ├── summary_generation.py   # AI-powered summaries
│   ├── translation_tts.py      # Translation and text-to-speech
│   └── requirements.txt        # Python dependencies
└── frontend/
    ├── pages/
    │   ├── index.tsx           # Home page with upload form
    │   └── report/[jobId].tsx  # Report display page
    ├── styles/
    │   └── globals.css         # Global styles
    ├── package.json            # Node.js dependencies
    └── tsconfig.json           # TypeScript configuration
```

## Key Components

### Backend Components

1. **OCR Module**: Handles file processing and text extraction using Pytesseract and OpenCV
2. **Data Processing**: Maps extracted terms to known lab tests using `lab_data.py`
3. **Summary Generation**: Creates both plain-language and doctor-style summaries using Google Gemini
4. **Translation & TTS**: Translates summaries and generates audio files

### Frontend Components

1. **File Upload Form**: Modern drag-and-drop interface with file type validation
2. **Language Toggle**: Switch between English, Hindi, and Marathi
3. **Report Display**: Shows analysis results with collapsible sections
4. **Audio Player**: Plays translated summaries when available
5. **Responsive Design**: Fully responsive with mobile-first approach
6. **UI Enhancements**: Gradient backgrounds, smooth animations, interactive elements

## UI Improvements

The frontend has been enhanced with:

- **Modern Design**: Clean, professional interface with consistent spacing and typography
- **Interactive Elements**: Hover effects, transitions, and animations for better user experience
- **Responsive Layouts**: Mobile-friendly design that works on all screen sizes
- **Visual Hierarchy**: Improved information architecture with clear sections and headings
- **Accessibility**: Proper contrast, focus states, and semantic HTML
- **Loading States**: Visual feedback during processing with spinners and progress indicators
- **Error Handling**: Clear error messages with appropriate styling
- **Card-based Layouts**: Organized content in visually appealing cards
- **Color Coding**: Consistent color scheme for different types of information

## Lab Data Configuration

The `lab_data.py` file contains mappings for common lab tests with:
- Multiple synonyms for each test
- Standard units of measurement
- Reference ranges
- Critical value thresholds

Currently supports 20+ common lab tests including:
- Hemoglobin, WBC, Creatinine
- Cholesterol (Total, LDL, HDL), Triglycerides
- Glucose, A1C
- Liver enzymes (ALT, AST)
- Electrolytes (Sodium, Potassium, Chloride)

## Translation and TTS

The application supports translation to Hindi and Marathi:
1. Plain language summary is translated by the backend
2. Translated text is converted to speech using ElevenLabs API
3. Frontend provides audio playback controls for supported languages

## Safety Features

- Hard-coded disclaimer on all reports
- Critical value detection with prominent warnings
- Red-flag system for urgent medical attention
- Clinician review interface for verification

## Extending the Application

### Adding New Lab Tests

Update `backend/lab_data.py` with new test definitions following the existing format.

### Adding Languages

1. Update language options in frontend components
2. Enhance translation functions in `backend/translation_tts.py`
3. Add appropriate text-to-speech voice selection

### Adding File Types

Extend the `extract_text_from_file` function in `backend/ocr_module.py` to handle new file formats.

## Environment Variables

Create a `.env` file in the backend directory:

```
GEMINI_API_KEY=your_google_gemini_api_key
ELEVENLABS_API_KEY=your_elevenlabs_api_key
```

## Development

### Backend Development

- Run with auto-reload: `uvicorn main:app --reload`
- API documentation available at: `http://localhost:8000/docs`

### Frontend Development

- Development server: `npm run dev`
- Build for production: `npm run build`

## Testing

The application includes comprehensive error handling and validation for:
- File type validation
- API response handling
- Network request failures
- Invalid data processing

## Production Deployment

### Backend (example using Gunicorn)

```bash
pip install gunicorn
gunicorn -w 4 -k uvicorn.workers.UvicornWorker main:app
```

### Frontend (build and serve)

```bash
npm run build
npm run start  # Runs the production server
```

## Security Considerations

- All API requests should be authenticated for admin endpoints
- File uploads are validated for type and size
- API keys are stored in environment variables
- Input sanitization is performed on text extraction

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License.