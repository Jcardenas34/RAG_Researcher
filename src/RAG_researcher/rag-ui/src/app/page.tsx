'use client';

import { useState, FormEvent, useRef, useEffect, ChangeEvent } from 'react';
import { Bot, User, SendHorizonal, Cpu, Flag } from 'lucide-react';

// Define the structure of a message object
interface Message {
  text: string;
  isUser: boolean;
}

// Simple ReactMarkdown-like renderer (since we can't import ReactMarkdown in this environment)
const ReactMarkdown = ({ children, components }: { children: string, components?: any }) => {
  const renderContent = () => {
    let html = children;
    
    // Handle LaTeX-style math (inline: $...$, block: $$...$$)
    html = html.replace(/\$\$([\s\S]*?)\$\$/g, '<div class="katex-display bg-gray-700 p-3 rounded-lg my-3 text-center overflow-x-auto">$1</div>');
    html = html.replace(/\$([^$\n]+)\$/g, '<span class="katex-inline bg-gray-700 px-2 py-1 rounded text-blue-300">$1</span>');
    
    // Handle code blocks first (```language\ncode```)
    html = html.replace(/```(\w+)?\n([\s\S]*?)```/g, '<pre class="bg-gray-900 rounded-lg p-4 mb-3 overflow-x-auto border border-gray-600"><code class="text-gray-100 font-mono text-sm">$2</code></pre>');
    html = html.replace(/```([\s\S]*?)```/g, '<pre class="bg-gray-900 rounded-lg p-4 mb-3 overflow-x-auto border border-gray-600"><code class="text-gray-100 font-mono text-sm">$1</code></pre>');
    
    // Handle inline code (`code`)
    html = html.replace(/`([^`]+)`/g, '<code class="bg-gray-900 px-2 py-1 rounded text-blue-300 font-mono text-sm">$1</code>');
    
    // Handle headers (must be at start of line)
    html = html.replace(/^### (.*$)/gim, '<h3 class="text-lg font-bold mb-2 mt-4 text-white">$1</h3>');
    html = html.replace(/^## (.*$)/gim, '<h2 class="text-xl font-bold mb-3 mt-5 text-white">$1</h2>');
    html = html.replace(/^# (.*$)/gim, '<h1 class="text-2xl font-bold mb-4 mt-6 text-white">$1</h1>');
    
    // Handle bold text (**text** or __text__)
    html = html.replace(/\*\*(.*?)\*\*/g, '<strong class="font-bold text-white">$1</strong>');
    html = html.replace(/__(.*?)__/g, '<strong class="font-bold text-white">$1</strong>');
    
    // Handle italic text (*text* or _text_)
    html = html.replace(/\*(.*?)\*/g, '<em class="italic text-gray-300">$1</em>');
    html = html.replace(/_(.*?)_/g, '<em class="italic text-gray-300">$1</em>');
    
    // Handle strikethrough (~~text~~)
    html = html.replace(/~~(.*?)~~/g, '<del class="line-through text-gray-400">$1</del>');
    
    // Handle links [text](url)
    html = html.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" class="text-blue-400 hover:text-blue-300 underline" target="_blank" rel="noopener noreferrer">$1</a>');
    
    // Handle unordered lists (- or * at start of line)
    html = html.replace(/^[\*\-] (.*$)/gim, '<li class="text-white ml-4 mb-1">• $1</li>');
    
    // Handle ordered lists (1. at start of line)
    let olCounter = 1;
    html = html.replace(/^\d+\. (.*$)/gim, (match, content) => {
      const result = `<li class="text-white ml-4 mb-1">${olCounter}. ${content}</li>`;
      olCounter++;
      return result;
    });
    
    // Handle blockquotes (> at start of line)
    html = html.replace(/^> (.*$)/gim, '<blockquote class="border-l-4 border-blue-500 pl-4 py-2 mb-3 text-gray-300 italic bg-gray-800 rounded-r-lg">$1</blockquote>');
    
    // Handle horizontal rules (--- or ***)
    html = html.replace(/^(---|\*\*\*)$/gim, '<hr class="border-gray-600 my-4">');
    
    // Handle tables (basic support for |col1|col2|)
    html = html.replace(/\|(.+)\|/g, (match, content) => {
      const cells = content.split('|').map((cell: string) => cell.trim());
      const cellsHtml = cells.map((cell: string) => `<td class="border border-gray-600 px-3 py-2">${cell}</td>`).join('');
      return `<tr>${cellsHtml}</tr>`;
    });
    
    // Wrap table rows in table
    html = html.replace(/(<tr>.*<\/tr>)/s, '<div class="overflow-x-auto mb-3"><table class="min-w-full border border-gray-600 rounded-lg">$1</table></div>');
    
    // Handle line breaks - convert double newlines to paragraph breaks
    html = html.replace(/\n\n+/g, '</p><p class="mb-3 leading-relaxed text-white">');
    html = html.replace(/\n/g, '<br />');
    
    // Wrap in paragraph tags if not already wrapped in block elements
    if (!html.includes('<p') && !html.includes('<h') && !html.includes('<pre') && !html.includes('<blockquote') && !html.includes('<li')) {
      html = `<p class="mb-3 leading-relaxed text-white">${html}</p>`;
    } else if (!html.startsWith('<')) {
      html = `<p class="mb-3 leading-relaxed text-white">${html}`;
    }
    
    // Clean up any unclosed paragraph tags
    if (html.includes('<p class="mb-3 leading-relaxed text-white">') && !html.includes('</p>')) {
      html += '</p>';
    }
    
    return html;
  };

  return (
    <div 
      className="markdown-content text-white prose prose-invert max-w-none" 
      dangerouslySetInnerHTML={{ __html: renderContent() }}
    />
  );
};

export default function ChatPage() {
  const [messages, setMessages] = useState<Message[]>([
    // Demo message showing various markdown features
    {
      text: `# Welcome to RAG Researcher!

Here's a demonstration of **markdown formatting**:

## Code Examples
Here's some Python code:
\`\`\`python
def calculate_accuracy(predictions, labels):
    correct = sum(p == l for p, l in zip(predictions, labels))
    return correct / len(labels)
\`\`\`

And some inline code: \`numpy.array([1, 2, 3])\`

## Mathematical Expressions
You can write inline math like $E = mc^2$ and display math:
$$\\frac{d}{dx}\\int_{a}^{x} f(t) dt = f(x)$$

## Lists and Formatting
- **Bold text** and *italic text*
- ~~Strikethrough text~~
- [External links](https://example.com)

### Numbered Lists
1. First item with **bold**
2. Second item with *emphasis*
3. Third item with \`code\`

> This is a blockquote with important information about the research paper.

---

Ready to start asking questions about your documents!`,
      isUser: false
    }
  ]);
  const [prompt, setPrompt] = useState<string>('');
  const [isLoading, setIsLoading] = useState<boolean>(false);
  
  // --- State for the selected LLM ---
  const [selectedLlm, setSelectedLlm] = useState<string>('gemma3:270m');
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
          setSelectedLlm(data.llm || 'gemma3:270m');
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

  // --- Function to Handle Subject Change from Dropdown ---
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
  const handleSend = async (e?: FormEvent) => {
    if (e) e.preventDefault();
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

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const llmOptions = ['gemini-2.5-flash', 'gemini-2.5-pro', 
                      'gpt-4.1', 'gpt-o4-mini', 
                      'tinyllama',
                      'gemma3:270m',
                      'claude-sonnet-4-20250514','claude-opus-4-20250514','claude-3-5-haiku-20241022'];

  const subjectOptions = ["test", "ML", "Physics", "Math", "Chemistry"]

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
              className="w-full p-2 rounded-md bg-gray-700 text-white border border-gray-600 focus:outline-none focus:ring-2 focus:ring-2 focus:ring-blue-500"
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
                  {message.isUser ? (
                    <div className="whitespace-pre-wrap text-white">{message.text}</div>
                  ) : (
                    <ReactMarkdown>{message.text}</ReactMarkdown>
                  )}
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
          <div className="max-w-3xl mx-auto">
            <div className="relative">
              <textarea
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Ask anything about the selected paper..."
                className="w-full bg-gray-800 border border-gray-600 rounded-lg p-4 pr-16 resize-none focus:outline-none focus:ring-2 focus:ring-blue-500"
                rows={1}
              />
              <button
                onClick={handleSend}
                disabled={isLoading || !prompt.trim()}
                className="absolute right-4 top-1/2 -translate-y-1/2 p-2 rounded-lg bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 disabled:cursor-not-allowed transition-colors"
              >
                <SendHorizonal size={20} />
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}