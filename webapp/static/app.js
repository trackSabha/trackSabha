// YuhHearDem Chat Application with Independent Graph Visualizations
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
    
    // DOM elements
    elements: {
        chatContainer: null,
        queryInput: null,
        sendButton: null,
        inputStatus: null,
        connectionStatus: null,
        clearButton: null
    },
    
    init: function() {
        console.log('🔧 Initializing YuhHearDem chat with expandable cards and independent graph visualization...');
        console.log('📊 D3.js version:', typeof d3 !== 'undefined' ? d3.version : 'Not loaded');
        
        this.initializeSession();
        this.cacheElements();
        this.displaySessionInfo();
        this.testConnection();
        this.setupEventListeners();
    },
    
    cacheElements: function() {
        this.elements.chatContainer = document.getElementById('chatContainer');
        this.elements.queryInput = document.getElementById('queryInput');
        this.elements.sendButton = document.getElementById('sendButton');
        this.elements.inputStatus = document.getElementById('inputStatus');
        this.elements.connectionStatus = document.getElementById('connectionStatus');
        this.elements.clearButton = document.getElementById('clearChat');
    },
    
    initializeSession: function() {
        var existingUserId = sessionStorage.getItem('yuhheardem_user_id');
        var existingSessionId = sessionStorage.getItem('yuhheardem_session_id');
        
        if (existingUserId && existingSessionId) {
            this.userId = existingUserId;
            this.sessionId = existingSessionId;
            this.setSessionStatus('Session Restored', 'success');
        } else {
            this.userId = generateUUID();
            this.sessionId = generateUUID();
            
            sessionStorage.setItem('yuhheardem_user_id', this.userId);
            sessionStorage.setItem('yuhheardem_session_id', this.sessionId);
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
                                    self.hideThinkingDots();
                                    
                                    // Check if we have structured response
                                    if (eventData.data.structured_response) {
                                        console.log('📋 Processing structured response with cards');
                                        self.addStructuredMessage(eventData.data.structured_response, eventData.data.response);
                                    } else {
                                        console.log('📄 Processing fallback response');
                                        self.addMessage('assistant', eventData.data.response);
                                    }
                                    
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
        // Clear all messages
        this.elements.chatContainer.innerHTML = '';
        
        // Reset message counter
        this.messageCounter = 0;
        
        this.userId = generateUUID();
        this.sessionId = generateUUID();
        
        sessionStorage.setItem('yuhheardem_user_id', this.userId);
        sessionStorage.setItem('yuhheardem_session_id', this.sessionId);
        
        this.displaySessionInfo();
        this.setSessionStatus('New Session', 'new');
        
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