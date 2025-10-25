import React from 'react';
import ReactDOM from 'react-dom/client';
import LegalAIPro from '../LegalAIPro.jsx';
import './index.css';

// Telegram Web App интеграция
window.Telegram?.WebApp?.expand();

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <LegalAIPro />
  </React.StrictMode>
);
