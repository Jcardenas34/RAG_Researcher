'use client';

import { useState, FormEvent, useRef, useEffect, ChangeEvent } from 'react';
import { Bot, User, SendHorizonal, Cpu, Flag } from 'lucide-react';

// Define the structure of a message object
interface Message {
  text: string;
  isUser: boolean;
}

export default function ChatPage() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [prompt, setPrompt] = useState<string>('');
  const [isLoading, setIsLoading] = useState<boolean>(false);
  
  // --- State for the selected LLM ---
  const [selectedLlm, setSelectedLlm] = useState<string>('gemini-2.5-flash');
  // ---------------------------------

  // --- State for the selected Subject ---
  const [selectedSubject, setSelectedSubject] = useState<string>('ML');
  // ---------------------------------


  const chatContainerRef = useRef<HTMLDivElement>(null);

  // --- Effect to Fetch Initial LLM ---
  useEffect(() => {
    const fetchInitialLlm = async () => {
      try {
        const response = await fetch('http://127.0.0.1:5000/api/llm');
        if (response.ok) {
          const data = await response.json();
          setSelectedLlm(data.llm || 'gemini-2.5-flash');
        }
      } catch (error) {
        console.error("Could not fetch initial LLM:", error);
      }
    };
    fetchInitialLlm();
  }, []);
  // ------------------------------------

  // --- Effect to Fetch Initial Subject---
  useEffect(() => {
    const fetchInitialSubject = async () => {
      try {
        const response = await fetch('http://127.0.0.1:5000/api/subject');
        if (response.ok) {
          const data = await response.json();
          setSelectedSubject(data.subject || 'ML');
        }
      } catch (error) {
        console.error("Could not fetch initial Subject:", error);
      }
    };
    fetchInitialSubject();
  }, []);
  // ------------------------------------


  useEffect(() => {
    if (chatContainerRef.current) {
      chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight;
    }
  }, [messages]);

  // --- Function to Handle LLM Change from Dropdown ---
  const handleLlmChange = async (e: ChangeEvent<HTMLSelectElement>) => {
    const newLlm = e.target.value;
    console.log(`Changing LLM to: ${newLlm}`);
    try {
      const response = await fetch('http://127.0.0.1:5000/api/llm', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ llm: newLlm }),
      });

      if (response.ok) {
        setSelectedLlm(newLlm);
      } else {
        console.error("Failed to set LLM on backend.");
      }
    } catch (error) {
      console.error("Error changing LLM:", error);
    }
  };
  // ----------------------------------------------------

  // --- Function to Handle LLM Change from Dropdown ---
  const handlesubjectChange = async (e: ChangeEvent<HTMLSelectElement>) => {
    const newSubject = e.target.value;
    console.log(`Changing subject to: ${newSubject}`);
    try {
      const response = await fetch('http://127.0.0.1:5000/api/subject', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ subject: newSubject }),
      });

      if (response.ok) {
        setSelectedSubject(newSubject);
      } else {
        console.error("Failed to set subject on backend.");
      }
    } catch (error) {
      console.error("Error changing subject:", error);
    }
  };
  // ----------------------------------------------------


  // Function to handle form submission
  const handleSend = async (e: FormEvent) => {
    e.preventDefault();
    if (!prompt.trim() || isLoading) return;

    const userMessage: Message = { text: prompt, isUser: true };
    setMessages((prevMessages) => [...prevMessages, userMessage]);
    
    setPrompt('');
    setIsLoading(true);

    try {
      const response = await fetch('http://127.0.0.1:5000/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt: prompt }),
      });

      if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);

      const data = await response.json();
      const botMessage: Message = { text: data.response, isUser: false };
      setMessages((prevMessages) => [...prevMessages, botMessage]);

    } catch (error) {
      console.error("Failed to fetch from backend:", error);
      const errorMessage: Message = { text: "Sorry, I couldn't connect to the backend.", isUser: false };
      setMessages((prevMessages) => [...prevMessages, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const llmOptions = ['gemini-2.5-flash', 'gemini-2.5-pro', 
                      'gpt-4.1', 'gpt-o4-mini', 
                      'tinyllama',
                      'claude-sonnet-4-20250514','claude-opus-4-20250514','claude-3-5-haiku-20241022'];

  const subjectOptions = ["test","ML", "Physics", "Math", "Chemistry"]

  return (
    <div className="flex h-screen bg-gray-900 text-white font-sans">
      {/* Sidebar */}
      <div className="w-72 bg-gray-950 border-r border-gray-700 p-4 flex flex-col">
        <h1 className="text-xl font-bold mb-6">RAG Researcher</h1>
        <button className="w-full bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2 px-4 rounded-lg transition-colors">
          + New Chat
        </button>
        <div className="mt-auto">
          {/* --- New Subject Selector Dropdown --- */}
          <div className="mt-6 p-4 bg-gray-900 rounded-lg">
            <label htmlFor="subject-select" className="text-sm font-semibold text-gray-400 flex items-center gap-2 mb-2">
              <Flag size={16} />
              Subject
            </label>
            <select
              id="subject-select"
              value={selectedSubject}
              onChange={handlesubjectChange}
              className="w-full p-2 rounded-md bg-gray-700 text-white border border-gray-600 focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              {subjectOptions.map((option) => (
                <option key={option} value={option}>
                  {option}
                </option>
              ))}
            </select>
          </div>
          {/* ---------------------------------- */}
        </div>
        <div className="mt-auto">
          {/* --- New LLM Selector Dropdown --- */}
          <div className="mt-6 p-4 bg-gray-900 rounded-lg">
            <label htmlFor="llm-select" className="text-sm font-semibold text-gray-400 flex items-center gap-2 mb-2">
              <Cpu size={16} />
              Language Model
            </label>
            <select
              id="llm-select"
              value={selectedLlm}
              onChange={handleLlmChange}
              className="w-full p-2 rounded-md bg-gray-700 text-white border border-gray-600 focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              {llmOptions.map((option) => (
                <option key={option} value={option}>
                  {option}
                </option>
              ))}
            </select>
          </div>
          {/* ---------------------------------- */}
        </div>
      </div>

      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col">
        <div ref={chatContainerRef} className="flex-1 p-6 overflow-y-auto">
          {messages.length === 0 ? (
            <div className="text-center text-gray-400">
              Select a model and start a conversation.
            </div>
          ) : (
            messages.map((message, index) => (
              <div key={index} className={`my-4 flex items-start gap-4 ${message.isUser ? 'justify-end' : ''}`}>
                {!message.isUser && <div className="p-2 bg-gray-700 rounded-full"><Bot size={20} /></div>}
                <div className={`p-4 rounded-lg max-w-2xl ${message.isUser ? 'bg-blue-600' : 'bg-gray-800'}`}>
                  <p>{message.text}</p>
                </div>
                {message.isUser && <div className="p-2 bg-gray-700 rounded-full"><User size={20} /></div>}
              </div>
            ))
          )}
          {isLoading && (
            <div className="my-4 flex items-start gap-4">
              <div className="p-2 bg-gray-700 rounded-full"><Bot size={20} /></div>
              <div className="p-4 rounded-lg bg-gray-800">
                <div className="flex items-center gap-2">
                  <div className="w-2 h-2 bg-blue-400 rounded-full animate-pulse"></div>
                  <div className="w-2 h-2 bg-blue-400 rounded-full animate-pulse [animation-delay:0.2s]"></div>
                  <div className="w-2 h-2 bg-blue-400 rounded-full animate-pulse [animation-delay:0.4s]"></div>
                </div>
              </div>
            </div>
          )}
        </div>
        <div className="p-4 bg-gray-950 border-t border-gray-700">
          <form onSubmit={handleSend} className="max-w-3xl mx-auto">
            <div className="relative">
              <textarea
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                onKeyDown={(e) => { if (e.key === 'Enter' && !e.shiftKey) { handleSend(e); } }}
                placeholder="Ask anything about the selected paper..."
                className="w-full bg-gray-800 border border-gray-600 rounded-lg p-4 pr-16 resize-none focus:outline-none focus:ring-2 focus:ring-blue-500"
                rows={1}
              />
              <button
                type="submit"
                disabled={isLoading || !prompt.trim()}
                className="absolute right-4 top-1/2 -translate-y-1/2 p-2 rounded-lg bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 disabled:cursor-not-allowed transition-colors"
              >
                <SendHorizonal size={20} />
              </button>
            </div>
          </form>
        </div>
      </div>
    </div>
  );
}
