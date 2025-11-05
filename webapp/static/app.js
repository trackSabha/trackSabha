// tracksabha Chat Application with Independent Graph Visualizations
function generateUUID() {
    return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
        var r = Math.random() * 16 | 0;
        var v = c == 'x' ? r : (r & 0x3 | 0x8);
        return v.toString(16);
    });
}

// Global function to toggle cards
function toggleCard(cardId) {
    var card = document.querySelector('[data-card-id="' + cardId + '"]');
    var details = document.getElementById(cardId + '-details');
    
    if (!card || !details) {
        console.error('Card elements not found:', cardId);
        return;
    }
    
    if (details.classList.contains('collapsed')) {
        // Expand
        details.classList.remove('collapsed');
        details.classList.add('expanded');
        card.classList.add('expanded');
        console.log('📂 Expanded card:', cardId);
    } else {
        // Collapse
        details.classList.remove('expanded');
        details.classList.add('collapsed');
        card.classList.remove('expanded');
        console.log('📁 Collapsed card:', cardId);
    }
}

// Global function to send suggestion
function sendSuggestion(suggestion) {
    console.log('💡 Sending suggestion:', suggestion);
    if (chatApp && chatApp.elements && chatApp.elements.queryInput) {
        chatApp.elements.queryInput.value = suggestion;
        chatApp.sendQuery();
    }
}

var chatApp = {
    // Configuration
    apiBase: window.location.origin,
    userId: null,
    sessionId: null,
    isProcessing: false,
    currentThinkingDots: null,
    messageCounter: 0,
    skipLocalSave: false,
    
    // DOM elements
    elements: {
        chatContainer: null,
        queryInput: null,
        sendButton: null,
        inputStatus: null,
        connectionStatus: null,
        clearButton: null,
        showGraphButton: null,
        refreshGraphButton: null,
        hideGraphButton: null,
        graphSection: null,
        graphPanel: null,
        graphMeta: null
    },
    
    // --- Local persistence helpers ---
    localKey: function() {
        return 'tracksabha_chat_' + this.sessionId;
    },
    loadLocalChat: function() {
        try {
            var raw = localStorage.getItem(this.localKey());
            if (!raw) return;
            var items = JSON.parse(raw);
            if (!Array.isArray(items) || !items.length) return;
            console.log('📥 Restoring', items.length, 'messages from local cache');
            this.skipLocalSave = true;
            for (var i=0; i<items.length; i++) {
                var m = items[i] || {};
                if (m.role === 'user') {
                    this.addMessage('user', m.content || '');
                } else if (m.role === 'assistant') {
                    this.addMessage('assistant', m.content || '');
                }
            }
            this.skipLocalSave = false;
        } catch (e) {
            console.warn('Local chat restore failed:', e);
        }
    },
    saveMessageToLocal: function(msg) {
        try {
            var key = this.localKey();
            var raw = localStorage.getItem(key);
            var arr = [];
            if (raw) { try { arr = JSON.parse(raw) || []; } catch(_){} }
            arr.push({ role: msg.role, content: msg.content, at: msg.at || Date.now() });
            localStorage.setItem(key, JSON.stringify(arr));
        } catch (e) {
            console.warn('Local chat save failed:', e);
        }
    },
    rewriteLocalFromServer: function(serverMessages) {
        try {
            var arr = [];
            for (var i=0; i<serverMessages.length; i++) {
                var m = serverMessages[i];
                arr.push({ role: m.role, content: m.content, at: Date.parse(m.timestamp) || Date.now() });
            }
            localStorage.setItem(this.localKey(), JSON.stringify(arr));
        } catch (e) {
            console.warn('Local chat rewrite failed:', e);
        }
    },

    init: function() {
        console.log('🔧 Initializing Tracksabha chat with expandable cards and independent graph visualization...');
        console.log('📊 D3.js version:', typeof d3 !== 'undefined' ? d3.version : 'Not loaded');
        
        this.initializeSession();
        this.cacheElements();
        this.displaySessionInfo();
        this.testConnection();
        this.setupEventListeners();
        // Show graph section by default
        this.showGraphPanel();
        // Fast UX: render any locally cached chat immediately
        this.loadLocalChat();
    },
    
    cacheElements: function() {
        this.elements.chatContainer = document.getElementById('chatContainer');
        this.elements.queryInput = document.getElementById('queryInput');
        this.elements.sendButton = document.getElementById('sendButton');
        this.elements.inputStatus = document.getElementById('inputStatus');
        this.elements.connectionStatus = document.getElementById('connectionStatus');
        this.elements.clearButton = document.getElementById('clearChat');
        this.elements.showGraphButton = document.getElementById('showGraph');
        this.elements.refreshGraphButton = document.getElementById('refreshGraph');
        this.elements.hideGraphButton = document.getElementById('hideGraph');
        this.elements.graphSection = document.getElementById('graphSection');
        this.elements.graphPanel = document.getElementById('graphPanel');
        this.elements.graphMeta = document.getElementById('graphMeta');
    },
    
    initializeSession: function() {
        var existingUserId = sessionStorage.getItem('tracksabha_user_id');
        var existingSessionId = sessionStorage.getItem('tracksabha_session_id');
        
        if (existingUserId && existingSessionId) {
            this.userId = existingUserId;
            this.sessionId = existingSessionId;
            this.setSessionStatus('Session Restored', 'success');
        } else {
            this.userId = generateUUID();
            this.sessionId = generateUUID();
            
            sessionStorage.setItem('tracksabha_user_id', this.userId);
            sessionStorage.setItem('tracksabha_session_id', this.sessionId);
            this.setSessionStatus('New Session', 'new');
        }
    },
    
    displaySessionInfo: function() {
        document.getElementById('sessionId').textContent = this.sessionId.substring(0, 8) + '...';
        document.getElementById('userId').textContent = this.userId.substring(0, 8) + '...';
    },
    
    setSessionStatus: function(text, type) {
        var sessionStatus = document.getElementById('sessionStatus');
        sessionStatus.textContent = text;
        
        var className = 'ml-2 px-2 py-1 rounded text-xs ';
        switch(type) {
            case 'success':
                className += 'bg-green-100 text-green-800';
                break;
            case 'new':
                className += 'bg-blue-100 text-blue-800';
                break;
            case 'error':
                className += 'bg-red-100 text-red-800';
                break;
            default:
                className += 'bg-gray-100 text-gray-800';
        }
        sessionStatus.className = className;
    },
    
    testConnection: function() {
        var self = this;
        fetch(this.apiBase + '/health')
            .then(function(response) {
                if (!response.ok) {
                    throw new Error('Health check failed: ' + response.status);
                }
                return response.json();
            })
            .then(function(health) {
                if (health.status === 'healthy') {
                    self.setConnectionStatus(true);
                    self.enableInput();
                    self.elements.inputStatus.textContent = 'Ready! Ask me about parliamentary discussions.';
                    // After connection, load previous messages for this session
                    self.loadSessionHistory();
                    // And load graph immediately
                    self.refreshGraphPanel();
                } else {
                    throw new Error('Service not healthy: ' + health.status);
                }
            })
            .catch(function(error) {
                console.error('❌ Connection failed:', error);
                self.setConnectionStatus(false);
                self.elements.inputStatus.textContent = 'Connection failed: ' + error.message;
            });
    },
    
    setupEventListeners: function() {
        var self = this;
        
        this.elements.sendButton.addEventListener('click', function() {
            self.sendQuery();
        });
        
        this.elements.queryInput.addEventListener('keypress', function(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                self.sendQuery();
            }
        });
        
        this.elements.clearButton.addEventListener('click', function() {
            self.clearChat();
        });

        // Graph controls
        if (this.elements.showGraphButton) {
            // Hide the button; graph is always visible now
            this.elements.showGraphButton.classList.add('hidden');
        }
        if (this.elements.refreshGraphButton) {
            // Hide the header refresh button; panel manages itself
            this.elements.refreshGraphButton.classList.add('hidden');
        }
        if (this.elements.hideGraphButton) {
            this.elements.hideGraphButton.addEventListener('click', function() {
                self.hideGraphPanel();
            });
        }
    },
    
    setConnectionStatus: function(connected) {
        var statusDot = this.elements.connectionStatus.querySelector('div');
        var statusText = this.elements.connectionStatus.querySelector('span');
        
        if (connected) {
            statusDot.className = 'w-3 h-3 rounded-full bg-green-400 mr-2 connection-status-connected';
            statusText.textContent = 'Connected';
        } else {
            statusDot.className = 'w-3 h-3 rounded-full bg-red-400 mr-2 connection-status-disconnected';
            statusText.textContent = 'Disconnected';
        }
    },
    
    enableInput: function() {
        this.elements.queryInput.disabled = false;
        this.elements.sendButton.disabled = false;
        this.elements.queryInput.focus();
    },
    
    disableInput: function() {
        this.elements.queryInput.disabled = true;
        this.elements.sendButton.disabled = true;
    },
    
    showThinkingDots: function() {
        this.hideThinkingDots();
        
        console.log('🎭 Showing thinking dots');
        
        var dotsContainer = document.createElement('div');
        dotsContainer.className = 'thinking-dots-container fade-in';
        dotsContainer.innerHTML = 
            '<div class="thinking-dots">' +
                '<div class="thinking-dot"></div>' +
                '<div class="thinking-dot"></div>' +
                '<div class="thinking-dot"></div>' +
            '</div>';
        
        this.elements.chatContainer.appendChild(dotsContainer);
        this.currentThinkingDots = dotsContainer;
        this.scrollToBottom();
        
        return dotsContainer;
    },
    
    hideThinkingDots: function() {
        if (this.currentThinkingDots) {
            console.log('🎭 Hiding thinking dots');
            var dots = this.currentThinkingDots;
            dots.classList.add('fade-out');
            
            setTimeout(function() {
                if (dots.parentNode) {
                    dots.parentNode.removeChild(dots);
                }
            }, 300);
            
            this.currentThinkingDots = null;
        }
    },
    
    sendQuery: function() {
        var query = this.elements.queryInput.value.trim();
        if (!query || this.isProcessing) return;
        
        console.log('🚀 Sending query:', query);
        
        // Handle slash commands locally
        if (query.startsWith('/graph')) {
            console.log('📊 Handling graph command locally');
            this.handleGraphCommand(query);
            this.elements.queryInput.value = '';
            return;
        }
        
        this.isProcessing = true;
        this.disableInput();
        
        // Add user message
        this.addMessage('user', query);
        this.elements.queryInput.value = '';
        
        // Show thinking dots immediately
        this.showThinkingDots();
        
        // Start stream
        this.streamQuery(query);
    },
    
    // FIXED: Independent graph command handler with unique IDs
    handleGraphCommand: function(command) {
        console.log('📊 Processing graph command:', command);
        
        // Add user message showing the command
        this.addMessage('user', command);
        
        // Show thinking dots
        this.showThinkingDots();
        
        // Call the direct graph visualization endpoint
        var self = this;
        var graphUrl = this.apiBase + '/session/' + this.sessionId + '/graph/visualize';
        
        fetch(graphUrl, {
            method: 'GET',
            headers: { 'Accept': 'text/html' }
        })
        .then(function(response) {
            if (!response.ok) {
                throw new Error('HTTP ' + response.status);
            }
            return response.text();
        })
        .then(function(htmlContent) {
            self.hideThinkingDots();
            
            console.log('📊 Got direct graph HTML, length:', htmlContent.length);
            
            if (htmlContent && htmlContent.includes('graph-visualization')) {
                // FIXED: Generate unique IDs for each graph to prevent conflicts
                var timestamp = Date.now();
                var uniqueGraphId = 'knowledge-graph-' + timestamp;
                var uniqueContainerId = 'graph-container-' + timestamp;
                
                // Replace the default IDs with unique ones
                htmlContent = htmlContent.replace(/id="knowledge-graph"/g, 'id="' + uniqueGraphId + '"');
                htmlContent = htmlContent.replace(/id="graph-container"/g, 'id="' + uniqueContainerId + '"');
                htmlContent = htmlContent.replace(/#knowledge-graph/g, '#' + uniqueGraphId);
                htmlContent = htmlContent.replace(/#graph-container/g, '#' + uniqueContainerId);
                
                console.log('📊 Rendering independent graph visualization with ID:', uniqueGraphId);
                self.addMessage('assistant', htmlContent);
            } else if (htmlContent.includes('Error') || htmlContent.includes('empty')) {
                self.addMessage('assistant', 'The knowledge graph is empty. Ask me some questions about Barbados Parliament first to build up the graph!');
            } else {
                self.addMessage('assistant', 'Sorry, I could not generate the graph visualization.');
            }
        })
        .catch(function(error) {
            console.error('❌ Graph command failed:', error);
            self.hideThinkingDots();
            self.addMessage('assistant', 'Error generating graph: ' + error.message);
        });
    },
    
    streamQuery: function(query) {
        var self = this;
        
        fetch(this.apiBase + '/query/stream', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                query: query,
                user_id: this.userId,
                session_id: this.sessionId
            })
        })
        .then(function(response) {
            if (!response.ok) {
                throw new Error('HTTP ' + response.status);
            }
            return response.body.getReader();
        })
        .then(function(reader) {
            var decoder = new TextDecoder();
            var buffer = '';
            
            function readStream() {
                return reader.read().then(function(result) {
                    if (result.done) {
                        console.log('🏁 Stream finished');
                        self.hideThinkingDots();
                        self.isProcessing = false;
                        self.enableInput();
                        return;
                    }
                    
                    buffer += decoder.decode(result.value, { stream: true });
                    var lines = buffer.split('\n');
                    buffer = lines.pop();
                    
                    for (var i = 0; i < lines.length; i++) {
                        var line = lines[i];
                        if (line.startsWith('data: ')) {
                            try {
                                var eventData = JSON.parse(line.substring(6));
                                console.log('📨 Event:', eventData.type);
                                
                                // Handle final response
                                if (eventData.type === 'response_ready' && eventData.data && eventData.data.response) {
                                    console.log('✅ Got final response, type:', eventData.data.response_type);
                                    // Sync sessionId with server-assigned value so graph endpoints work
                                    if (eventData.data.session_id && eventData.data.session_id !== self.sessionId) {
                                        self.sessionId = eventData.data.session_id;
                                        sessionStorage.setItem('tracksabha_session_id', self.sessionId);
                                        self.displaySessionInfo();
                                        console.log('🔗 Updated sessionId from server:', self.sessionId);
                                    }
                                    self.hideThinkingDots();
                                    
                                    // Check if we have structured response
                                    if (eventData.data.structured_response) {
                                        console.log('📋 Processing structured response with cards');
                                        self.addStructuredMessage(eventData.data.structured_response, eventData.data.response);
                                    } else {
                                        console.log('📄 Processing fallback response');
                                        self.addMessage('assistant', eventData.data.response);
                                    }

                                    // Auto-refresh graph panel after each answer
                                    self.refreshGraphPanel();

                                    // Also insert the latest graph inline as a message
                                    self.insertGraphInlineMessage();
                                    
                                } else if (eventData.type === 'error') {
                                    console.log('❌ Got error');
                                    self.hideThinkingDots();
                                    self.addMessage('assistant', 'Error: ' + eventData.message);
                                } else {
                                    console.log('🔄 Ignoring event type:', eventData.type);
                                }
                                
                            } catch (e) {
                                console.error('❌ Parse error:', e);
                            }
                        }
                    }
                    
                    return readStream();
                });
            }
            
            return readStream();
        })
        .catch(function(error) {
            console.error('❌ Stream failed:', error);
            self.hideThinkingDots();
            self.addMessage('assistant', 'Sorry, I encountered an error: ' + error.message);
            self.isProcessing = false;
            self.enableInput();
        });
    },

    // Load previous messages from the server for this session
    loadSessionHistory: function() {
        var self = this;
        var url = this.apiBase + '/session/' + this.sessionId + '/messages?limit=50';
        console.log('📜 Loading session history from', url);
        fetch(url)
            .then(function(r){ if(!r.ok) throw new Error('HTTP '+r.status); return r.json(); })
            .then(function(data){
                if (!data || !Array.isArray(data.messages)) return;
                // Replace UI with server truth
                self.elements.chatContainer.innerHTML = '';
                self.messageCounter = 0;
                var ordered = data.messages.slice();
                ordered.forEach(function(m){
                    if (m.role === 'user') {
                        self.addMessage('user', m.content || '');
                    } else if (m.role === 'assistant') {
                        var rendered = self.renderAssistantContent(m.content, m.metadata || {});
                        self.addMessage('assistant', rendered);
                    }
                });
                // Rewrite local cache based on server data
                self.rewriteLocalFromServer(ordered);
                self.scrollToBottom();
            })
            .catch(function(err){ console.warn('History load failed:', err); });
    },

    // Render assistant content: try structured JSON else markdown
    renderAssistantContent: function(content, metadata) {
        // If server provided rendered HTML, prefer it
        if (metadata && metadata.rendered_html) {
            return metadata.rendered_html;
        }
        try {
            var trimmed = (content || '').trim();
            if (trimmed.startsWith('{')) {
                var parsed = JSON.parse(trimmed);
                if (parsed && parsed.intro_message && Array.isArray(parsed.response_cards)) {
                    return this.renderStructuredHtml(parsed);
                }
            }
        } catch (e) {
            // fall through to markdown
        }
        try {
            marked.setOptions({ breaks: true, gfm: true, sanitize: false });
            return marked.parse(content || '');
        } catch (e) {
            return this.escapeHtml(content || '').replace(/\n/g,'<br>');
        }
    },

    // Minimal client-side structured renderer
    renderStructuredHtml: function(sr) {
        var html = '';
        var intro = this.escapeHtml(sr.intro_message || '');
        html += '<div class="intro-message">' + intro + '</div>';
        html += '<div class="response-cards">';
        (sr.response_cards || []).forEach(function(card, i){
            var summary = (card && card.summary) ? card.summary : 'Details';
            var detailsMd = (card && card.details) ? card.details : '';
            var detailsHtml;
            try { detailsHtml = marked.parse(detailsMd); } catch(e){ detailsHtml = detailsMd; }
            var cardId = 'hist-card-' + Date.now() + '-' + i;
            html += '\n<div class="response-card" data-card-id="' + cardId + '">\n' +
                    '  <div class="card-header" onclick="toggleCard(\'' + cardId + '\')">\n' +
                    '    <div class="card-summary">' + summary + '</div>\n' +
                    '    <div class="card-toggle"><span class="toggle-icon">▼</span></div>\n' +
                    '  </div>\n' +
                    '  <div class="card-details collapsed" id="' + cardId + '-details">\n' +
                    '    <div class="card-content">' + detailsHtml + '</div>\n' +
                    '  </div>\n' +
                    '</div>';
        });
        html += '</div>';
        if (Array.isArray(sr.follow_up_suggestions) && sr.follow_up_suggestions.length) {
            html += '<div class="follow-up-suggestions"><h4>Follow-up questions:</h4><ul class="suggestions-list">';
            sr.follow_up_suggestions.forEach(function(s){
                var text = (''+s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;').replace(/'/g,'&#39;');
                html += '<li class="suggestion-item" data-suggestion="' + text + '" onclick="sendSuggestion(this.dataset.suggestion)">' + text + '</li>';
            });
            html += '</ul></div>';
        }
        return html;
    },

    // Fetch graph HTML and insert below the last assistant message
    insertGraphInlineMessage: function() {
        var self = this;
        var graphUrl = this.apiBase + '/session/' + this.sessionId + '/graph/visualize';
        console.log('🧩 Inserting inline graph from', graphUrl);
        fetch(graphUrl, { method:'GET', headers:{ 'Accept':'text/html' } })
            .then(function(r){ return r.text().then(function(t){ return { status:r.status, ok:r.ok, text:t }; }); })
            .then(function(htmlContent){
                if (!htmlContent || !htmlContent.text) return;
                var ts = Date.now();
                var uniqueGraphId = 'knowledge-graph-inline-' + ts;
                var uniqueContainerId = 'graph-container-inline-' + ts;
                var updated = htmlContent.text
                    .replace(/id="knowledge-graph"/g, 'id="' + uniqueGraphId + '"')
                    .replace(/id="graph-container"/g, 'id="' + uniqueContainerId + '"')
                    .replace(/#knowledge-graph/g, '#' + uniqueGraphId)
                    .replace(/#graph-container/g, '#' + uniqueContainerId);
                self.addMessage('assistant', updated);
            })
            .catch(function(err){ console.warn('Inline graph insert failed:', err.message); });
    },
    
    addStructuredMessage: function(structuredResponse, htmlContent) {
        console.log('💬 Adding structured message with cards');
        
        // Generate unique message ID
        this.messageCounter++;
        var messageId = 'msg-' + this.messageCounter;
        
        var messageDiv = document.createElement('div');
        messageDiv.className = 'message-bubble';
        messageDiv.setAttribute('data-message-id', messageId);
        
        var avatarBg = 'bg-blue-500';
        var bubbleBg = 'bg-blue-50 border border-blue-200';
        var avatarText = 'YH';
        
        messageDiv.innerHTML = 
            '<div class="rounded-lg p-6 ' + bubbleBg + '">' +
                '<div class="flex items-start space-x-3">' +
                    '<div class="flex-1">' +
                        '<div class="text-gray-800">' + htmlContent + '</div>' +
                        '<div class="text-xs text-gray-500 mt-3">' + new Date().toLocaleTimeString() + '</div>' +
                    '</div>' +
                '</div>' +
            '</div>';
        
    this.elements.chatContainer.appendChild(messageDiv);
        
        // Make YouTube links open in new window
        var youtubeLinks = messageDiv.querySelectorAll('a[href*="youtube.com"], a[href*="youtu.be"]');
        for (var i = 0; i < youtubeLinks.length; i++) {
            youtubeLinks[i].target = '_blank';
            youtubeLinks[i].rel = 'noopener noreferrer';
        }
        
        this.scrollToBottom();
        
        return messageDiv;
    },

    // --- Graph Panel helpers ---
    showGraphPanel: function() {
        if (this.elements.graphSection) {
            this.elements.graphSection.classList.remove('hidden');
        }
        if (this.elements.refreshGraphButton) {
            this.elements.refreshGraphButton.classList.remove('hidden');
        }
    },

    hideGraphPanel: function() {
        if (this.elements.graphSection) {
            this.elements.graphSection.classList.add('hidden');
        }
    },

    refreshGraphPanel: function() {
        var self = this;
        if (!this.elements.graphPanel) return;

        // Ensure the section is visible when refreshing
        this.showGraphPanel();

        var graphUrl = this.apiBase + '/session/' + this.sessionId + '/graph/visualize';
        console.log('📡 Refreshing graph panel for session:', this.sessionId, 'URL:', graphUrl);
        this.elements.graphMeta.textContent = ' • loading…';

        fetch(graphUrl, { method: 'GET', headers: { 'Accept': 'text/html' } })
            .then(function(response) {
                console.log('📡 Graph fetch status:', response.status);
                if (!response.ok) {
                    // Provide clearer UX when the server has no graph for this session yet
                    if (response.status === 404) {
                        throw new Error('No graph yet for this session. Ask a question first, then refresh.');
                    }
                    throw new Error('HTTP ' + response.status);
                }
                return response.text();
            })
            .then(function(htmlContent) {
                console.log('📡 Graph HTML length:', htmlContent ? htmlContent.length : 0);
                // Generate unique IDs to avoid collisions
                var ts = Date.now();
                var uniqueGraphId = 'knowledge-graph-panel-' + ts;
                var uniqueContainerId = 'graph-container-panel-' + ts;
                var updated = htmlContent
                    .replace(/id="knowledge-graph"/g, 'id="' + uniqueGraphId + '"')
                    .replace(/id="graph-container"/g, 'id="' + uniqueContainerId + '"')
                    .replace(/#knowledge-graph/g, '#' + uniqueGraphId)
                    .replace(/#graph-container/g, '#' + uniqueContainerId);

                self.elements.graphPanel.innerHTML = updated;

                // Execute any inline scripts within the panel content
                setTimeout(function() {
                    try {
                        if (typeof d3 === 'undefined') {
                            console.error('❌ D3.js not loaded for panel rendering');
                            return;
                        }
                        var scripts = self.elements.graphPanel.querySelectorAll('script');
                        for (var i = 0; i < scripts.length; i++) {
                            var code = scripts[i].textContent;
                            if (code && !code.includes('d3.min.js')) {
                                try { eval(code); } catch (e) { console.error('Panel script error:', e); }
                            }
                        }
                        // Update meta with basic stats if present in DOM
                        var label = self.elements.graphPanel.querySelector('.graph-visualization h4');
                        self.elements.graphMeta.textContent = '';
                    } catch (e) {
                        console.error('Graph panel render error:', e);
                        self.elements.graphMeta.textContent = ' • error rendering';
                    }
                }, 50);
            })
            .catch(function(err) {
                console.error('Graph panel fetch failed:', err);
                self.elements.graphPanel.innerHTML = '<div class="text-sm text-red-600">Failed to load graph: ' + self.escapeHtml(err.message) + '</div>';
                self.elements.graphMeta.textContent = ' • failed';
            });
    },
    
    // Enhanced addMessage function with proper script execution
    addMessage: function(sender, content) {
        console.log('💬 Adding message:', sender, 'Content length:', content.length);
        
        var messageDiv = document.createElement('div');
        messageDiv.className = 'message-bubble';
        
        var isUser = sender === 'user';
        var avatarBg = isUser ? 'bg-green-500' : 'bg-blue-500';
        var bubbleBg = isUser ? 'bg-green-50 border border-green-200' : 'bg-blue-50 border border-blue-200';
        var avatarText = isUser ? 'U' : 'YH';
        
        var htmlContent;
        if (isUser) {
            htmlContent = this.escapeHtml(content).replace(/\n/g, '<br>');
        } else {
            // Check if content contains graph HTML
            if (content.includes('<div class="graph-visualization"') || 
                content.includes('<svg') || 
                content.includes('knowledge-graph')) {
                
                console.log('📊 Rendering graph content directly');
                htmlContent = content;
                
            } else if (content.includes('<div class="intro-message">') || 
                       content.includes('<div class="response-cards">')) {
                
                console.log('📋 Rendering structured response HTML');
                htmlContent = content;
                
            } else {
                // Regular markdown parsing
                try {
                    marked.setOptions({
                        breaks: true,
                        gfm: true,
                        sanitize: false
                    });
                    htmlContent = marked.parse(content);
                } catch (e) {
                    console.error('Markdown parsing error:', e);
                    htmlContent = this.escapeHtml(content).replace(/\n/g, '<br>');
                }
            }
        }
        
        var reverseClass = isUser ? 'flex-row-reverse space-x-reverse' : '';
        
        messageDiv.innerHTML = 
            '<div class="rounded-lg p-6 ' + bubbleBg + '">' +
                '<div class="flex items-start space-x-3 ' + reverseClass + '">' +
                    '<div class="flex-1">' +
                        '<div class="text-gray-800">' + htmlContent + '</div>' +
                        '<div class="text-xs text-gray-500 mt-3">' + new Date().toLocaleTimeString() + '</div>' +
                    '</div>' +
                '</div>' +
            '</div>';
        
        this.elements.chatContainer.appendChild(messageDiv);
        
        // Enhanced script execution for graph visualization
        if (!isUser && (content.includes('graph-visualization') || content.includes('knowledge-graph'))) {
            console.log('📊 Graph message detected, executing scripts...');
            
            // Wait for DOM to be ready
            setTimeout(function() {
                // Check if D3 is available
                if (typeof d3 === 'undefined') {
                    console.error('❌ D3.js not loaded globally!');
                    return;
                }
                
                console.log('📊 D3.js available, executing graph script');
                
                // Find and execute all scripts in the message
                var scripts = messageDiv.querySelectorAll('script');
                console.log('📊 Found', scripts.length, 'scripts to execute');
                
                for (var i = 0; i < scripts.length; i++) {
                    try {
                        console.log('📊 Executing script', i + 1);
                        
                        // Skip the D3 loading script since we already have it
                        var scriptContent = scripts[i].textContent;
                        if (scriptContent.includes('cdnjs.cloudflare.com/ajax/libs/d3')) {
                            console.log('📊 Skipping D3 loading script (already loaded)');
                            continue;
                        }
                        
                        // Execute the script content directly
                        eval(scriptContent);
                        
                        console.log('📊 Script', i + 1, 'executed successfully');
                    } catch (e) {
                        console.error('❌ Script execution failed:', e);
                    }
                }
                
                // Verify the graph was created
                setTimeout(function() {
                    var svgs = messageDiv.querySelectorAll('svg[id*="knowledge-graph"]');
                    console.log('📊 Found', svgs.length, 'graph SVGs');
                    
                    for (var j = 0; j < svgs.length; j++) {
                        var svg = svgs[j];
                        var circles = svg.querySelectorAll('circle');
                        console.log('📊 Graph', j + 1, 'has', circles.length, 'nodes');
                        
                        if (circles.length === 0) {
                            console.warn('⚠️ Graph', j + 1, 'has no visible nodes');
                        } else {
                            console.log('✅ Graph', j + 1, 'visualization successful!');
                        }
                    }
                }, 500);
            }, 100);
        }
        
        // Rest of function...
        if (!isUser) {
            var youtubeLinks = messageDiv.querySelectorAll('a[href*="youtube.com"], a[href*="youtu.be"]');
            for (var i = 0; i < youtubeLinks.length; i++) {
                youtubeLinks[i].target = '_blank';
                youtubeLinks[i].rel = 'noopener noreferrer';
            }
        }
        
        // Persist to local cache unless we're replaying
        if (!this.skipLocalSave) {
            this.saveMessageToLocal({ role: sender, content: content, at: Date.now() });
        }

        this.scrollToBottom();
        return messageDiv;
    },
    
    escapeHtml: function(text) {
        var div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    },
    
    scrollToBottom: function() {
        window.scrollTo({
            top: document.body.scrollHeight,
            behavior: 'smooth'
        });
    },
    
    clearChat: function() {
        // Clear UI and local cache but preserve session IDs
        this.elements.chatContainer.innerHTML = '';
        this.messageCounter = 0;
        try { localStorage.removeItem(this.localKey()); } catch(_){ }
        this.displaySessionInfo();
        this.setSessionStatus('Cleared', 'new');
        this.elements.queryInput.focus();
    }
};

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', function() {
    chatApp.init();
});

// Expose functions globally
window.chatApp = chatApp;
window.toggleCard = toggleCard;
window.sendSuggestion = sendSuggestion;