import { useState, useRef, useEffect, useCallback } from 'react';
import { GoogleGenAI, LiveServerMessage, Modality } from '@google/genai';

// Audio Utils
const floatTo16BitPCM = (float32Array: Float32Array) => {
  const buffer = new ArrayBuffer(float32Array.length * 2);
  const view = new DataView(buffer);
  let offset = 0;
  for (let i = 0; i < float32Array.length; i++, offset += 2) {
    let s = Math.max(-1, Math.min(1, float32Array[i]));
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
  return buffer;
};

const base64ToUint8Array = (base64: string) => {
  const binaryString = atob(base64);
  const bytes = new Uint8Array(binaryString.length);
  for (let i = 0; i < binaryString.length; i++) {
    bytes[i] = binaryString.charCodeAt(i);
  }
  return bytes;
};

export const useLiveInterview = (role: string, videoRef: React.RefObject<HTMLVideoElement | null>) => {
  const [isConnected, setIsConnected] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [logs, setLogs] = useState<string[]>([]);

  const audioContextRef = useRef<AudioContext | null>(null);
  const inputSourceRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const processorRef = useRef<ScriptProcessorNode | null>(null);
  const sessionRef = useRef<any>(null);
  const nextStartTimeRef = useRef<number>(0);
  const videoIntervalRef = useRef<number | null>(null);

  const addLog = (msg: string) => setLogs(prev => [...prev, msg]);

  const connect = useCallback(async () => {
    try {
      setError(null);
      const apiKey = process.env.API_KEY;
      if (!apiKey) throw new Error("API Key missing");

      const ai = new GoogleGenAI({ apiKey });
      
      const AudioContextClass = window.AudioContext || (window as any).webkitAudioContext;
      audioContextRef.current = new AudioContextClass({ sampleRate: 24000 });
      
      // Get User Media (Audio + Video)
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true, video: true });
      
      // Setup Audio Input
      const inputContext = new AudioContextClass({ sampleRate: 16000 });
      const source = inputContext.createMediaStreamSource(stream);
      const processor = inputContext.createScriptProcessor(4096, 1, 1);
      
      inputSourceRef.current = source;
      processorRef.current = processor;

      source.connect(processor);
      processor.connect(inputContext.destination);

      // Setup Video Preview
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.play();
      }

      const sessionPromise = ai.live.connect({
        model: 'gemini-2.5-flash-native-audio-preview-09-2025',
        callbacks: {
          onopen: () => {
            setIsConnected(true);
            addLog("Connected to Interviewer");
            
            // 1. Stream Audio
            processor.onaudioprocess = (e) => {
              const inputData = e.inputBuffer.getChannelData(0);
              const pcmData = floatTo16BitPCM(inputData);
              let binary = '';
              const bytes = new Uint8Array(pcmData);
              const len = bytes.byteLength;
              for (let i = 0; i < len; i++) {
                binary += String.fromCharCode(bytes[i]);
              }
              const b64 = btoa(binary);

              sessionPromise.then(session => {
                  session.sendRealtimeInput({
                      media: { mimeType: 'audio/pcm;rate=16000', data: b64 }
                  });
              });
            };

            // 2. Stream Video Frames (approx 1 FPS is sufficient for interview presence)
            const canvas = document.createElement('canvas');
            const ctx = canvas.getContext('2d');
            
            videoIntervalRef.current = window.setInterval(() => {
                if (videoRef.current && ctx) {
                    canvas.width = videoRef.current.videoWidth || 640;
                    canvas.height = videoRef.current.videoHeight || 480;
                    ctx.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height);
                    
                    const base64Image = canvas.toDataURL('image/jpeg', 0.5).split(',')[1];
                    
                    sessionPromise.then(session => {
                        session.sendRealtimeInput({
                            media: { mimeType: 'image/jpeg', data: base64Image }
                        });
                    });
                }
            }, 1000); // 1 FPS
          },
          onmessage: async (msg: LiveServerMessage) => {
            const audioData = msg.serverContent?.modelTurn?.parts?.[0]?.inlineData?.data;
            if (audioData) {
              setIsSpeaking(true);
              const audioCtx = audioContextRef.current;
              if (!audioCtx) return;

              const uint8Array = base64ToUint8Array(audioData);
              const float32Array = new Float32Array(uint8Array.length / 2);
              const dataView = new DataView(uint8Array.buffer);

              for (let i = 0; i < float32Array.length; i++) {
                float32Array[i] = dataView.getInt16(i * 2, true) / 32768.0;
              }

              const buffer = audioCtx.createBuffer(1, float32Array.length, 24000);
              buffer.getChannelData(0).set(float32Array);

              const source = audioCtx.createBufferSource();
              source.buffer = buffer;
              source.connect(audioCtx.destination);
              
              const now = audioCtx.currentTime;
              const startTime = Math.max(now, nextStartTimeRef.current);
              source.start(startTime);
              nextStartTimeRef.current = startTime + buffer.duration;
              
              source.onended = () => {
                 if (audioCtx.currentTime >= nextStartTimeRef.current) {
                     setIsSpeaking(false);
                 }
              };
            }
          },
          onclose: () => {
            setIsConnected(false);
            addLog("Connection closed");
          },
          onerror: (err) => {
            console.error(err);
            setError("Connection error occurred");
            setIsConnected(false);
          }
        },
        config: {
          responseModalities: [Modality.AUDIO],
          speechConfig: {
            voiceConfig: { prebuiltVoiceConfig: { voiceName: 'Fenrir' } }
          },
          systemInstruction: `You are a professional HR manager conducting a video interview for a candidate applying for the position of "${role}". 
          You can see the candidate via their camera feed. 
          Start by greeting them warmly and asking them to introduce themselves. 
          Ask one question at a time. 
          Focus on soft skills, past experience, and situational questions related to ${role}.`
        }
      });
      
      sessionRef.current = sessionPromise;

    } catch (err: any) {
      setError(err.message);
      setIsConnected(false);
    }
  }, [role, videoRef]);

  const disconnect = useCallback(() => {
    if (inputSourceRef.current) inputSourceRef.current.disconnect();
    if (processorRef.current) processorRef.current.disconnect();
    if (audioContextRef.current) audioContextRef.current.close();
    if (videoIntervalRef.current) window.clearInterval(videoIntervalRef.current);
    
    // Stop tracks
    if (videoRef.current && videoRef.current.srcObject) {
        const stream = videoRef.current.srcObject as MediaStream;
        stream.getTracks().forEach(track => track.stop());
        videoRef.current.srcObject = null;
    }

    setIsConnected(false);
    setIsSpeaking(false);
    nextStartTimeRef.current = 0;
  }, [videoRef]);

  useEffect(() => {
    return () => disconnect();
  }, [disconnect]);

  return { connect, disconnect, isConnected, isSpeaking, error, logs };
};