document.addEventListener('DOMContentLoaded', () => {
    const chatForm = document.getElementById('chat-form');
    const chatInput = document.getElementById('chat-input');
    const chatWindow = document.getElementById('chat-window');
    const typingIndicator = document.getElementById('typing-indicator');
    const fileUpload = document.getElementById('file-upload');
    const progressBar = document.getElementById('upload-progress-bar');
    const progressContainer = document.getElementById('upload-progress-container');
    const progressPercentage = document.getElementById('upload-percentage');
    const vectorCount = document.getElementById('vector-count');
    const quickQueryBtns = document.querySelectorAll('.quick-query-btn');

    // Quick Query Handler
    quickQueryBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            chatInput.value = btn.textContent;
            chatForm.dispatchEvent(new Event('submit'));
        });
    });

    // Auto-resize textarea
    chatInput.addEventListener('input', () => {
        chatInput.style.height = 'auto';
        chatInput.style.height = (chatInput.scrollHeight) + 'px';
    });

    // Fetch Initial Status
    const updateStatus = async () => {
        try {
            const res = await fetch('/api/status');
            const data = await res.json();
            vectorCount.textContent = data.vector_count;
        } catch (e) {
            console.error('Status fetch failed');
        }
    };
    updateStatus();
    setInterval(updateStatus, 5000);

    // Chat Handler
    chatForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const message = chatInput.value.trim();
        if (!message) return;

        // Clear input
        chatInput.value = '';
        chatInput.style.height = 'auto';

        // Add User Message
        addMessage(message, 'user');

        // Show Typing Indicator
        typingIndicator.classList.remove('hidden');
        scrollToBottom();

        try {
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ message })
            });
            const data = await response.json();
            
            // Hide Typing Indicator
            typingIndicator.classList.add('hidden');

            if (data.error) {
                addMessage(`Error: ${data.error}`, 'bot');
            } else {
                addMessage(data.response, 'bot');
            }
        } catch (error) {
            typingIndicator.classList.add('hidden');
            addMessage('System error. Please try again.', 'bot');
        }
        scrollToBottom();
    });

    // File Upload Handler
    fileUpload.addEventListener('change', async () => {
        const file = fileUpload.files[0];
        if (!file) return;

        const formData = new FormData();
        formData.append('file', file);

        progressContainer.classList.remove('hidden');
        progressBar.style.width = '0%';
        progressPercentage.textContent = '0%';

        try {
            // Simulated progress because fetch doesn't support upload progress natively without XHR
            // But we'll use XHR for better UX as requested
            const xhr = new XMLHttpRequest();
            xhr.open('POST', '/api/upload', true);

            xhr.upload.onprogress = (e) => {
                if (e.lengthComputable) {
                    const percent = Math.round((e.loaded / e.total) * 100);
                    progressBar.style.width = percent + '%';
                    progressPercentage.textContent = percent + '%';
                }
            };

            xhr.onload = () => {
                if (xhr.status === 200) {
                    const data = JSON.parse(xhr.responseText);
                    addMessage(`📁 **${file.name}** uploaded successfully. Background ingestion started.`, 'bot');
                    setTimeout(() => {
                        progressContainer.classList.add('hidden');
                    }, 2000);
                } else {
                    addMessage('Upload failed.', 'bot');
                }
            };

            xhr.send(formData);
        } catch (error) {
            addMessage('Upload system error.', 'bot');
        }
    });

    function addMessage(text, role) {
        const div = document.createElement('div');
        div.className = `flex items-start ${role === 'user' ? 'justify-end' : ''} message-enter`;
        
        // Basic markdown-like bolding for visual pop
        const formattedText = text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');

        div.innerHTML = `
            <div class="${role === 'user' ? 'user-message rounded-tr-none' : 'bot-message rounded-tl-none'} border border-slate-800 rounded-2xl p-4 max-w-[80%] shadow-lg">
                <p class="text-slate-200 text-sm leading-relaxed chat-content">${formattedText}</p>
            </div>
        `;

        chatWindow.appendChild(div);
        
        // Trigger animation
        setTimeout(() => div.classList.add('message-enter-active'), 10);
        scrollToBottom();
    }

    function scrollToBottom() {
        chatWindow.scrollTop = chatWindow.scrollHeight;
    }

    // Handle Mobile Menu (Placeholder for logic)
    document.getElementById('mobile-menu-btn').onclick = () => {
        alert('AlphaPulse Mobile Menu: Sidebar toggled (Implementation pending for mobile view)');
    };
});
