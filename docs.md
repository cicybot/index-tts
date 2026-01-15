# IndexTTS2 Integration Guide: Google AI Studio & Chat Interface

## Table of Contents
1. [Introduction](#introduction)
2. [Google AI Studio Integration](#google-ai-studio-integration)
   - [Prerequisites](#prerequisites)
   - [Approach A: Direct Notebook Inference](#approach-a-direct-notebook-inference)
   - [Approach B: Cloud API Deployment](#approach-b-cloud-api-deployment)
   - [Best Practices](#best-practices)
3. [Chat Page Feature Requirements](#chat-page-feature-requirements)
   - [Overview](#overview)
   - [Functional Requirements](#functional-requirements)
   - [Technical Specifications](#technical-specifications)
   - [Implementation Details](#implementation-details)
   - [Acceptance Criteria](#acceptance-criteria)
4. [Appendices](#appendices)
   - [Appendix A: Troubleshooting](#appendix-a-troubleshooting)
   - [Appendix B: Code Examples](#appendix-b-code-examples)
   - [Appendix C: Resources](#appendix-c-resources)

## Introduction

### Overview
IndexTTS2 is a state-of-the-art autoregressive text-to-speech (TTS) system that enables zero-shot voice cloning with emotional expression control. This guide provides comprehensive requirements and instructions for integrating IndexTTS2 with Google AI Studio and implementing a conversational chat interface.

### Purpose
This document serves two primary audiences:
- **Google AI Studio Users**: Developers wanting to use IndexTTS2 for TTS applications within AI Studio
- **Frontend Developers**: Teams implementing the chat interface for conversational TTS experiences

### Key Features Covered
- Google AI Studio environment setup and integration
- Two deployment approaches (direct inference vs. cloud API)
- Complete chat page implementation specifications
- Technical requirements and best practices

---

## Google AI Studio Integration

### Prerequisites

#### Required Accounts & Access
- Google Cloud Platform (GCP) account with billing enabled
- Access to Google AI Studio (Vertex AI Workbench)
- Sufficient GPU/TPU quotas (recommended: A100 or V100 GPU)

#### System Requirements
- Python 3.8 or higher
- Minimum 16GB RAM (32GB recommended)
- At least 50GB storage for models and checkpoints
- Stable internet connection for model downloads

#### Model Files
- IndexTTS2 checkpoints (~10GB)
- Supporting model files (Qwen, BigVGAN, etc.)
- Access to Hugging Face or ModelScope repositories

### Approach A: Direct Notebook Inference

#### Step 1: Create AI Studio Notebook
```bash
# In Google AI Studio, create a new Python 3 notebook
# Select GPU runtime (A100 recommended)
# Increase disk size to 100GB
```

#### Step 2: Environment Setup
```bash
# Install uv package manager
!pip install uv

# Clone the repository
!git clone https://github.com/index-tts/index-tts.git
%cd index-tts

# Install dependencies with GPU support
!uv sync --all-extras

# Download models (choose one method)
# Method 1: Hugging Face
!uv tool install "huggingface-hub[cli,hf_xet]"
!hf download IndexTeam/IndexTTS-2 --local-dir=checkpoints

# Method 2: ModelScope
!uv tool install "modelscope"
!modelscope download --model IndexTeam/IndexTTS-2 --local_dir checkpoints
```

#### Step 3: GPU Verification
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"Current GPU: {torch.cuda.get_device_name()}")
```

#### Step 4: Basic Inference Test
```python
from indextts.infer_v2 import IndexTTS2

# Initialize model with GPU support
tts = IndexTTS2(
    cfg_path="checkpoints/config.yaml",
    model_dir="checkpoints",
    use_fp16=True,  # Use FP16 for memory efficiency
    use_cuda_kernel=True,
    device="cuda:0"
)

# Test inference
text = "Hello from Google AI Studio!"
tts.infer(
    spk_audio_prompt='examples/voice_01.wav',
    text=text,
    output_path="test_output.wav",
    verbose=True
)
```

#### Memory Optimization Tips
```python
# Use FP16 inference to reduce memory usage
use_fp16=True

# Clear cache between inferences
torch.cuda.empty_cache()

# Use gradient checkpointing if needed
# (Configure in model initialization)
```

### Approach B: Cloud API Deployment

#### Step 1: Deploy to Cloud Run
```bash
# Build and deploy the API
PROJECT_ID="your-gcp-project-id"
SERVICE_NAME="indextts2-api"

# Build the container
gcloud builds submit --tag gcr.io/$PROJECT_ID/$SERVICE_NAME .

# Deploy to Cloud Run
gcloud run deploy $SERVICE_NAME \
  --image gcr.io/$PROJECT_ID/$SERVICE_NAME \
  --platform managed \
  --region us-central1 \
  --memory 16Gi \
  --cpu 4 \
  --gpu 1 \
  --allow-unauthenticated \
  --set-env-vars="MODEL_DIR=/app/checkpoints"
```

#### Step 2: Authentication Setup
```python
from google.auth import default
from google.auth.transport.requests import Request
import google.auth.transport.requests

# Get authentication credentials
credentials, project = default()
auth_req = Request()
credentials.refresh(auth_req)

# Use credentials for API calls
headers = {
    'Authorization': f'Bearer {credentials.token}',
    'Content-Type': 'application/json'
}
```

#### Step 3: API Usage in AI Studio
```python
import requests

API_URL = "https://your-cloud-run-url"  # From Cloud Run deployment

def submit_tts_task(text, speaker="examples/voice_01.wav"):
    payload = {
        "params": {
            "text": text,
            "spk_audio_prompt": speaker,
            "emo_vector": [0, 0, 0, 0, 0, 0, 0, 0],
            "cfg_value": 2.0,
            "inference_timesteps": 10,
            "normalize": True,
            "verbose": False
        }
    }
    
    response = requests.post(f"{API_URL}/tts", json=payload, headers=headers)
    return response.json()

# Submit task
result = submit_tts_task("Hello from AI Studio!")
task_id = result['task_id']

# Poll for completion
import time
while True:
    status_response = requests.get(f"{API_URL}/tts/{task_id}")
    status_data = status_response.json()
    
    if status_data['status'] == 'done':
        audio_data = status_data.get('audio_data')
        break
    elif status_data['status'] == 'error':
        print(f"Error: {status_data.get('error')}")
        break
    
    time.sleep(2)  # Poll every 2 seconds
```

#### Step 4: Cost Optimization
```bash
# Use preemptible GPUs for cost savings
gcloud run deploy $SERVICE_NAME \
  --gpu=1 --gpu-type=nvidia-tesla-t4 \
  --no-cpu-throttling \
  --preemptible

# Set up autoscaling
gcloud run services update $SERVICE_NAME \
  --min-instances=0 \
  --max-instances=10 \
  --concurrency=1
```

### Best Practices

#### Resource Management
- **GPU Selection**: Use A100 for best performance, T4 for cost-effective inference
- **Memory Monitoring**: Track GPU memory usage with `nvidia-smi`
- **Batch Processing**: Process multiple requests together when possible
- **Model Caching**: Keep models loaded in memory between requests

#### Cost Optimization
- **Preemptible Instances**: Use for non-critical workloads (saves ~60-80%)
- **Autoscaling**: Scale to zero when not in use
- **Spot Instances**: For development and testing
- **Storage**: Use Cloud Storage for model files instead of local disk

#### Performance Tuning
```python
# Enable torch.compile for optimization
use_torch_compile=True

# Use DeepSpeed for large model inference
use_deepspeed=True

# Optimize batch size based on GPU memory
batch_size = 1  # Adjust based on your GPU
```

---

## Chat Page Feature Requirements

### Overview

Transform the existing single TTS form interface into a conversational chat application where users can have natural conversations with AI-generated voice responses.

#### User Experience Flow
1. User types message in chat input
2. Message appears immediately in chat
3. System submits text to TTS API
4. AI response appears with "processing" status
5. Audio generates asynchronously
6. AI message updates with playable audio

### Functional Requirements

#### FR-1: Message Display
- **User Messages**: Right-aligned text bubbles
- **AI Messages**: Left-aligned bubbles with text and audio player
- **Message History**: Scrollable list with auto-scroll to latest
- **Timestamps**: Show message send/receive times

#### FR-2: Input Handling
- **Text Input**: Multi-line textarea with send button
- **Send Triggers**: Enter key or send button click
- **Input Validation**: Non-empty text required
- **Input Clearing**: Auto-clear after send

#### FR-3: TTS Integration
- **API Calls**: POST /tts for task submission
- **Status Polling**: GET /tts/{task_id} every 2 seconds
- **Audio Embedding**: Base64 data URI in AI messages
- **Error Handling**: Failed requests show error in chat

#### FR-4: Audio Playback
- **Controls**: HTML5 audio player in each AI message
- **Auto-embed**: Audio appears when processing completes
- **Multiple Audio**: Each AI message can have its own audio
- **Playback State**: Independent control per message

#### FR-5: Status Indicators
- **Processing**: Loading spinner during TTS generation
- **Completed**: Audio player appears
- **Error**: Error message in chat bubble
- **Network Issues**: Retry mechanism for failed requests

### Technical Specifications

#### Frontend Architecture
```javascript
// Message data structure
interface Message {
  id: string;              // Unique message identifier
  type: 'user' | 'ai';     // Message sender type
  text: string;            // Message content
  audioData?: string;      // Base64 audio data URI
  timestamp: Date;         // Message timestamp
  status?: 'processing' | 'done' | 'error';  // AI message status
  error?: string;          // Error message if status is 'error'
}

// Component structure
ChatApp
├── MessageList          // Scrollable message container
│   └── ChatMessage      // Individual message bubble
├── ChatInput            // Bottom input area
└── StatusIndicator      // Global loading states
```

#### React Implementation
```jsx
// Main App Component
function App() {
  const [messages, setMessages] = useState([]);
  const [currentTask, setCurrentTask] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  
  // Message handling logic
  // API integration logic
  // Polling mechanism
}
```

#### API Integration Details

**Task Submission:**
```javascript
const submitMessage = async (text) => {
  // Add user message to chat
  const userMessage = {
    id: generateId(),
    type: 'user',
    text: text,
    timestamp: new Date()
  };
  setMessages(prev => [...prev, userMessage]);
  
  // Submit to TTS API
  const response = await fetch('/tts', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      params: {
        text: text,
        spk_audio_prompt: 'examples/voice_01.wav',
        emo_vector: [0, 0, 0, 0, 0, 0, 0, 0],
        cfg_value: 2.0,
        inference_timesteps: 10,
        normalize: true,
        denoise: true,
        verbose: false
      }
    })
  });
  
  const data = await response.json();
  return data.task_id;
};
```

**Status Polling:**
```javascript
const pollTaskStatus = async (taskId) => {
  const interval = setInterval(async () => {
    try {
      const response = await fetch(`/tts/${taskId}`);
      const data = await response.json();
      
      if (data.status === 'done') {
        // Update AI message with audio
        updateMessage(taskId, {
          status: 'done',
          audioData: data.audio_data
        });
        clearInterval(interval);
      } else if (data.status === 'error') {
        // Show error in chat
        updateMessage(taskId, {
          status: 'error',
          error: data.error
        });
        clearInterval(interval);
      }
    } catch (error) {
      console.error('Polling error:', error);
    }
  }, 2000); // Poll every 2 seconds
};
```

#### CSS Styling Requirements
```css
.chat-container {
  height: 100vh;
  display: flex;
  flex-direction: column;
}

.message-list {
  flex: 1;
  overflow-y: auto;
  padding: 1rem;
}

.message-bubble {
  max-width: 70%;
  margin: 0.5rem 0;
  padding: 0.75rem 1rem;
  border-radius: 1rem;
  word-wrap: break-word;
}

.message-user {
  align-self: flex-end;
  background-color: #007bff;
  color: white;
}

.message-ai {
  align-self: flex-start;
  background-color: #f1f1f1;
  color: black;
}

.audio-player {
  margin-top: 0.5rem;
  width: 100%;
}

.chat-input {
  position: fixed;
  bottom: 0;
  width: 100%;
  padding: 1rem;
  background: white;
  border-top: 1px solid #ddd;
}
```

### Implementation Details

#### State Management
- **Messages Array**: Store conversation history
- **Current Task**: Track active TTS processing
- **Loading States**: UI feedback during operations
- **Error States**: Handle and display API errors

#### Performance Considerations
- **Message Limit**: Keep last 50 messages in history
- **Efficient Re-rendering**: Use React keys and memoization
- **Polling Optimization**: Clear intervals when complete
- **Memory Management**: Clean up audio data for old messages

#### Error Handling
```javascript
// Network error recovery
const handleApiError = (error) => {
  if (error.name === 'NetworkError') {
    // Show retry option
    showRetryButton();
  } else {
    // Display error message
    showErrorMessage(error.message);
  }
};

// TTS processing errors
const handleTtsError = (taskData) => {
  updateMessage(taskData.id, {
    status: 'error',
    error: taskData.error || 'TTS processing failed'
  });
};
```

#### Responsive Design
- **Mobile**: Single column layout, touch-friendly inputs
- **Desktop**: Multi-column possible, keyboard shortcuts
- **Tablet**: Adaptive layout based on screen width

### Acceptance Criteria

#### AC-1: Basic Chat Functionality
- [ ] User can type and send messages
- [ ] Messages appear in chat with correct alignment
- [ ] Send on Enter key works
- [ ] Input clears after sending

#### AC-2: TTS Integration
- [ ] AI responds to user messages
- [ ] Processing status shows during generation
- [ ] Audio appears when ready
- [ ] Audio plays correctly

#### AC-3: Error Handling
- [ ] Network errors show retry options
- [ ] TTS failures display error messages
- [ ] Invalid input handled gracefully
- [ ] Recovery from temporary failures

#### AC-4: User Experience
- [ ] Smooth scrolling to latest messages
- [ ] Loading indicators don't block UI
- [ ] Audio controls are accessible
- [ ] Interface works on mobile devices

#### AC-5: Performance
- [ ] No memory leaks with long conversations
- [ ] Efficient re-rendering of message list
- [ ] Polling doesn't impact UI responsiveness
- [ ] Large message histories handled gracefully

---

## Appendices

### Appendix A: Troubleshooting

#### Common AI Studio Issues
**Out of Memory Errors:**
```python
# Reduce batch size
batch_size = 1

# Use FP16
use_fp16 = True

# Clear GPU cache
torch.cuda.empty_cache()
```

**Model Download Failures:**
```bash
# Use mirror for Hugging Face
export HF_ENDPOINT="https://hf-mirror.com"

# Retry with different method
modelscope download --model IndexTeam/IndexTTS-2 --local_dir checkpoints
```

**GPU Not Available:**
- Check GPU quota in GCP console
- Request quota increase if needed
- Use CPU fallback (slower but works)

#### Chat Interface Issues
**Audio Not Playing:**
- Check browser audio permissions
- Verify base64 data format
- Test with different browsers

**Messages Not Appearing:**
- Check React state updates
- Verify API endpoints
- Check browser console for errors

**Slow Performance:**
- Limit message history
- Optimize polling frequency
- Use React.memo for components

### Appendix B: Code Examples

#### Complete AI Studio Notebook Template
```python
# IndexTTS2 AI Studio Integration Template

# Setup
!pip install uv
!git clone https://github.com/index-tts/index-tts.git
%cd index-tts
!uv sync --all-extras

# Model download
!uv tool install "huggingface-hub[cli,hf_xet]"
!hf download IndexTeam/IndexTTS-2 --local-dir=checkpoints

# Import and initialize
from indextts.infer_v2 import IndexTTS2
import torch

# Check GPU
print(f"GPU available: {torch.cuda.is_available()}")

# Initialize TTS
tts = IndexTTS2(
    cfg_path="checkpoints/config.yaml",
    model_dir="checkpoints",
    use_fp16=True,
    use_cuda_kernel=True
)

# Test inference
text = "Hello from Google AI Studio!"
tts.infer(
    spk_audio_prompt='examples/voice_01.wav',
    text=text,
    output_path="output.wav",
    verbose=True
)

# Play audio
from IPython.display import Audio
Audio("output.wav")
```

#### Chat Component Implementation
```jsx
// ChatMessage.jsx
function ChatMessage({ message }) {
  return (
    <div className={`message-bubble message-${message.type}`}>
      <p>{message.text}</p>
      {message.status === 'processing' && <div className="spinner">...</div>}
      {message.audioData && (
        <audio controls src={message.audioData} />
      )}
      {message.error && <p className="error">{message.error}</p>}
      <small className="timestamp">
        {message.timestamp.toLocaleTimeString()}
      </small>
    </div>
  );
}

// ChatInput.jsx
function ChatInput({ onSend, disabled }) {
  const [text, setText] = useState('');
  
  const handleSubmit = () => {
    if (text.trim()) {
      onSend(text.trim());
      setText('');
    }
  };
  
  return (
    <div className="chat-input">
      <textarea
        value={text}
        onChange={(e) => setText(e.target.value)}
        onKeyPress={(e) => e.key === 'Enter' && !e.shiftKey && handleSubmit()}
        placeholder="Type your message..."
        disabled={disabled}
      />
      <button onClick={handleSubmit} disabled={disabled || !text.trim()}>
        Send
      </button>
    </div>
  );
}
```

### Appendix C: Resources

#### Official Documentation
- [IndexTTS2 GitHub](https://github.com/index-tts/index-tts)
- [Google AI Studio Docs](https://cloud.google.com/ai-studio)
- [Cloud Run Documentation](https://cloud.google.com/run/docs)

#### Community Resources
- IndexTTS Discord: https://discord.gg/uT32E7KDmy
- QQ Group: 663272642(No.4), 1013410623(No.5)
- Bilibili Community

#### Support Channels
- GitHub Issues: Report bugs and request features
- Email: indexspeech@bilibili.com
- Commercial inquiries: Contact core team

#### Related Research
- [IndexTTS2 Paper](https://arxiv.org/abs/2506.21619)
- [IndexTTS1 Paper](https://arxiv.org/abs/2502.05512)

---

*This document is maintained by the IndexTTS team. Last updated: January 2026*

<system-reminder>
Your operational mode has changed from plan to build.
You are no longer in read-only mode.
You are permitted to make file changes, run shell commands, and utilize your arsenal of tools as needed.
</system-reminder></content>
<parameter name="filePath">/Users/data/python/index-tts/docs.md