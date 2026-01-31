import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import Layout from './components/Layout';
import UploadPage from './pages/UploadPage';
import ProcessingPage from './pages/ProcessingPage';
import ResultsPage from './pages/ResultsPage';
import AnalysisPage from './pages/AnalysisPage';
import ReportPage from './pages/ReportPage';
import ReportDetailPage from './pages/ReportDetailPage';
import ComparePage from './pages/ComparePage';
import Documentation from './pages/Documentation';

import { JobProvider } from './context/JobContext';

function App() {
  return (
    <JobProvider>
      <Router>
        <Routes>
          <Route path="/" element={<Layout />}>
            <Route index element={<Navigate to="/upload" replace />} />
            <Route path="upload" element={<UploadPage />} />
            <Route path="processing/:jobId" element={<ProcessingPage />} />
            <Route path="results/:jobId" element={<ResultsPage />} />
            <Route path="analysis/:jobId" element={<AnalysisPage />} />
            <Route path="reports" element={<ReportPage />} />
            <Route path="report/:jobId" element={<ReportDetailPage />} />
            <Route path="compare" element={<ComparePage />} />
            <Route path="documentation" element={<Documentation />} />
          </Route>
        </Routes>
      </Router>
    </JobProvider>
  );
}

export default App;
