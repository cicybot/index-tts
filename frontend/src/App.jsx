import { useState, useEffect } from 'react'
import './App.css'

const API_BASE = ''

function App() {
  const [text, setText] = useState('')
  const [taskId, setTaskId] = useState(null)
  const [status, setStatus] = useState('')
  const [audioData, setAudioData] = useState('')
  const [isLoading, setIsLoading] = useState(false)

  const submitTTS = async () => {
    if (!text.trim()) return

    setIsLoading(true)
    setStatus('Submitting...')

    try {
      const response = await fetch(`${API_BASE}/tts`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          params: {
            text: text,
            spk_audio_prompt: 'examples/voice_01.wav', // default speaker
            emo_vector: [0, 0, 0, 0, 0, 0, 0, 0], // neutral emotion
            cfg_value: 2.0,
            inference_timesteps: 10,
            normalize: true,
            denoise: true,
            retry_badcase: true,
            verbose: false
          }
        })
      })

      if (!response.ok) throw new Error('Failed to submit TTS')

      const data = await response.json()
      setTaskId(data.task_id)
      setStatus('Processing...')
    } catch (error) {
      setStatus(`Error: ${error.message}`)
      setIsLoading(false)
    }
  }

  useEffect(() => {
    if (!taskId) return

    const pollStatus = async () => {
      try {
        const response = await fetch(`${API_BASE}/tts/${taskId}`)
        if (!response.ok) throw new Error('Failed to get task status')

        const data = await response.json()
        setStatus(`Status: ${data.status}`)

        if (data.status === 'done') {
          setAudioData(data.audio_data || '')
          setIsLoading(false)
        } else if (data.status === 'error') {
          setStatus(`Error: ${data.error}`)
          setIsLoading(false)
        }
      } catch (error) {
        setStatus(`Error: ${error.message}`)
        setIsLoading(false)
      }
    }

    pollStatus()
    const interval = setInterval(pollStatus, 2000) // poll every 2 seconds

    return () => clearInterval(interval)
  }, [taskId])

  const reset = () => {
    setText('')
    setTaskId(null)
    setStatus('')
    setAudioData('')
    setIsLoading(false)
  }

  return (
    <div className="app">
      <h1>IndexTTS Frontend</h1>

      <div className="tts-form">
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="Enter text to convert to speech..."
          rows={4}
          disabled={isLoading}
        />
        <div className="buttons">
          <button onClick={submitTTS} disabled={isLoading || !text.trim()}>
            {isLoading ? 'Processing...' : 'Generate Speech'}
          </button>
          <button onClick={reset} disabled={isLoading}>
            Reset
          </button>
        </div>
      </div>

      {status && <p className="status">{status}</p>}

      {audioData && (
        <div className="audio-player">
          <h3>Generated Audio:</h3>
          <audio controls src={audioData} />
        </div>
      )}
    </div>
  )
}

export default App
