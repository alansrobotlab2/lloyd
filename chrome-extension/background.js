// Background service worker for Chrome extension

chrome.runtime.onInstalled.addListener(() => {
  console.log('Lloyd YouTube Capture extension installed');
});

// Handle messages from popup
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.action === 'getVideoInfo') {
    chrome.tabs.query({ active: true, currentWindow: true }, ([tab]) => {
      if (!tab || (!tab.url.includes('youtube.com') && !tab.url.includes('youtu.be'))) {
        sendResponse({ error: 'Not on a YouTube page' });
        return;
      }
      
      const url = tab.url;
      let videoId = null;
      
      const patterns = [
        /v=([a-zA-Z0-9_-]{11})/,
        /youtu\.be\/([a-zA-Z0-9_-]{11})/,
        /embed\/([a-zA-Z0-9_-]{11})/,
        /v\/([a-zA-Z0-9_-]{11})/
      ];
      
      for (const pattern of patterns) {
        const match = url.match(pattern);
        if (match) {
          videoId = match[1];
          break;
        }
      }
      
      if (!videoId) {
        sendResponse({ error: 'Could not extract video ID' });
        return;
      }
      
      sendResponse({
        videoId,
        url: tab.url,
        title: tab.title
      });
    });
    
    return true; // Keep channel open for async response
  }
});
