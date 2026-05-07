// Get current page info (works for any website)
async function getCurrentPage() {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  
  if (!tab.url || tab.url.startsWith('chrome://') || tab.url.startsWith('chrome-extension://')) {
    return { error: 'Cannot capture this page' };
  }
  
  // Detect page type
  let pageType = 'article';
  let videoId = null;
  
  // Check if YouTube
  if (tab.url.includes('youtube.com') || tab.url.includes('youtu.be')) {
    pageType = 'youtube';
    
    // Extract video ID
    const patterns = [
      /v=([a-zA-Z0-9_-]{11})/,
      /youtu\.be\/([a-zA-Z0-9_-]{11})/,
      /embed\/([a-zA-Z0-9_-]{11})/,
      /v\/([a-zA-Z0-9_-]{11})/
    ];
    
    for (const pattern of patterns) {
      const match = tab.url.match(pattern);
      if (match) {
        videoId = match[1];
        break;
      }
    }
  }
  
  return {
    pageType,
    videoId,
    url: tab.url,
    title: tab.title
  };
}

// Send capture request to backend
async function captureVideo(videoData) {
  const backendUrl = 'http://localhost:8087/capture';
  
  try {
    const response = await fetch(backendUrl, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(videoData)
    });
    
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }
    
    const result = await response.json();
    return result;
  } catch (error) {
    console.error('Capture failed:', error);
    throw error;
  }
}

// Update UI status
function setStatus(element, message, type = 'info') {
  const statusEl = document.getElementById(element);
  statusEl.className = `status ${type}`;
  statusEl.innerHTML = message;
}

// Main click handler
document.getElementById('captureBtn').addEventListener('click', async () => {
  const btn = document.getElementById('captureBtn');
  const videoInfo = document.getElementById('videoInfo');
  const videoTitle = document.getElementById('videoTitle');
  
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span>Processing...';
  
  try {
    // Get current page info
    const pageData = await getCurrentPage();
    
    if (pageData.error) {
      setStatus('status', pageData.error, 'error');
      btn.disabled = false;
      btn.textContent = 'Capture to Vault';
      return;
    }
    
    // Show page info
    videoTitle.textContent = `${pageData.pageType === 'youtube' ? '🎥 ' : '📄 '}${pageData.title}`;
    videoInfo.style.display = 'block';
    
    setStatus('status', `Capturing ${pageData.pageType}...`, 'info');
    
    // Send to backend
    const result = await captureVideo(pageData);
    
    if (result.success) {
      setStatus('status', `✅ ${result.message}`, 'success');
    } else {
      setStatus('status', `❌ ${result.error}`, 'error');
    }
    
  } catch (error) {
    setStatus('status', `❌ ${error.message}`, 'error');
  } finally {
    btn.disabled = false;
    btn.textContent = 'Capture to Vault';
  }
});

// Load page info on open
getCurrentPage().then(pageData => {
  if (!pageData.error) {
    const videoTitle = document.getElementById('videoTitle');
    const videoInfo = document.getElementById('videoInfo');
    const icon = pageData.pageType === 'youtube' ? '🎥 ' : '📄 ';
    videoTitle.textContent = `${icon}${pageData.title}`;
    videoInfo.style.display = 'block';
  }
});
