import React, { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import { Send, Image as ImageIcon, X, History, ChevronRight, Loader2, Sparkles, CheckCircle2, Plus, Mic, ArrowUp, ArrowDown, ArrowLeft, Edit2, Search } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { Routes, Route, Link, useNavigate } from 'react-router-dom';
import { generateSuggestions, executeAction, getEmbedding } from './lib/gemini';
import { LogEntry, Rule } from './types';

// Utility for cosine similarity
function cosineSimilarity(a: number[], b: number[]) {
  let dotProduct = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < a.length; i++) {
    dotProduct += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  if (normA === 0 || normB === 0) return 0;
  return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
}

export default function App() {
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [rules, setRules] = useState<Rule[]>([]);

  // Load logs from local storage on mount
  useEffect(() => {
    const savedLogs = localStorage.getItem('proactive-ai-logs');
    if (savedLogs) {
      try {
        setLogs(JSON.parse(savedLogs));
      } catch (e) {
        console.error("Failed to parse logs", e);
      }
    }
    const savedRules = localStorage.getItem('proactive-ai-rules');
    if (savedRules) {
      try {
        setRules(JSON.parse(savedRules));
      } catch (e) {
        console.error("Failed to parse rules", e);
      }
    }
  }, []);

  // Save logs to local storage when updated
  useEffect(() => {
    localStorage.setItem('proactive-ai-logs', JSON.stringify(logs));
  }, [logs]);

  useEffect(() => {
    localStorage.setItem('proactive-ai-rules', JSON.stringify(rules));
  }, [rules]);

  return (
    <Routes>
      <Route path="/" element={<MainView logs={logs} setLogs={setLogs} rules={rules} />} />
      <Route path="/trace" element={<TraceLogPage logs={logs} setLogs={setLogs} />} />
      <Route path="/taxonomy" element={<TaxonomyPage logs={logs} setLogs={setLogs} rules={rules} setRules={setRules} />} />
    </Routes>
  );
}

function MainView({ logs, setLogs, rules }: { logs: LogEntry[], setLogs: React.Dispatch<React.SetStateAction<LogEntry[]>>, rules: Rule[] }) {
  const [inputText, setInputText] = useState('');
  const [inputImage, setInputImage] = useState<string | null>(null);
  const [showLogs, setShowLogs] = useState(false);
  
  const [appState, setAppState] = useState<'idle' | 'analyzing' | 'suggestions' | 'executing' | 'result'>('idle');
  const [isRefining, setIsRefining] = useState(false);
  
  const [currentClassification, setCurrentClassification] = useState<{entities: string[], category: string} | null>(null);
  const [currentSuggestions, setCurrentSuggestions] = useState<string[]>([]);
  const [currentResult, setCurrentResult] = useState<string | null>(null);
  const [currentEmbedding, setCurrentEmbedding] = useState<number[]>([]);
  
  const [customAction, setCustomAction] = useState('');
  const [showCustomInput, setShowCustomInput] = useState(false);
  const [selectedAction, setSelectedAction] = useState<string | null>(null);
  
  const [refiningCountdown, setRefiningCountdown] = useState(10);
  const [excludedTags, setExcludedTags] = useState<string[]>([]);
  const [isExecuting, setIsExecuting] = useState(false);
  
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onloadend = () => {
      setInputImage(reader.result as string);
    };
    reader.readAsDataURL(file);
  };

  const removeImage = () => {
    setInputImage(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    const file = e.dataTransfer.files?.[0];
    if (file && file.type.startsWith('image/')) {
      const reader = new FileReader();
      reader.onloadend = () => {
        setInputImage(reader.result as string);
      };
      reader.readAsDataURL(file);
    }
  };

  const handleAnalyze = async () => {
    if (!inputText.trim() && !inputImage) return;

    setAppState('analyzing');
    setShowCustomInput(false);
    setSelectedAction(null);
    setCurrentResult(null);
    
    try {
      const textToEmbed = inputText + (inputImage ? " [Image attached]" : "");
      const embedding = await getEmbedding(textToEmbed);
      setCurrentEmbedding(embedding);

      // --- PASS 1: Pre-LLM Rule Check ---
      let predefinedSuggestions: string[] = [];
      let matchedCategory: string | null = null;
      let matchedEntities: string[] = [];
      const matchedRuleIds = new Set<string>();

      const checkRulePass1 = (rule: Rule) => {
        const exactMatch = inputText.toLowerCase().includes(rule.term.toLowerCase());
        if (rule.isExactMatch) {
          return exactMatch;
        } else {
          // Semantic mode: match if exact substring OR semantic threshold met
          let sim = 0;
          if (rule.termEmbedding) {
            sim = cosineSimilarity(embedding, rule.termEmbedding);
          }
          return exactMatch || (sim >= (rule.threshold / 100));
        }
      };

      for (const rule of rules) {
        if (checkRulePass1(rule)) {
          matchedRuleIds.add(rule.id);
          predefinedSuggestions.push(...rule.suggestions);
          if (rule.taxonomy === 'category') matchedCategory = rule.term;
          if (rule.taxonomy === 'entity') matchedEntities.push(rule.term);
        }
      }

      predefinedSuggestions = [...new Set(predefinedSuggestions)]; // deduplicate
      const DEFAULT_SUGGESTION_COUNT = 3;
      const suggestionsNeeded = Math.max(0, DEFAULT_SUGGESTION_COUNT - predefinedSuggestions.length);

      let similarLogsContext = "[]";
      if (logs.length > 0 && embedding.length > 0) {
        const now = Date.now();
        const scoredLogs = logs.map(log => {
          const rawSimilarity = cosineSimilarity(embedding, log.embedding);
          // Decay factor: 5% decay per day
          const ageDays = (now - log.timestamp) / (1000 * 60 * 60 * 24);
          const decay = Math.pow(0.95, ageDays);
          return {
            ...log,
            similarity: rawSimilarity * decay
          };
        });
        
        scoredLogs.sort((a, b) => b.similarity - a.similarity);
        
        const topLogs = scoredLogs.slice(0, 3).map(l => ({
          input: l.input.text,
          hasImage: !!l.input.image,
          chosenAction: l.generalizedAction || l.actionTaken,
          scores: l.scores
        }));
        
        similarLogsContext = JSON.stringify(topLogs, null, 2);
      }

      const response = await generateSuggestions(inputText, inputImage, similarLogsContext, suggestionsNeeded);
      
      // --- PASS 2: Post-LLM Rule Check ---
      const extractedCategory = response.classification?.category || 'Unknown';
      const extractedEntities = response.classification?.entities || [];

      for (const rule of rules) {
        if (matchedRuleIds.has(rule.id)) continue;

        const termLower = rule.term.toLowerCase();
        const matchesExtracted = 
          extractedCategory.toLowerCase() === termLower ||
          extractedEntities.some((e: string) => e.toLowerCase() === termLower);

        if (matchesExtracted) {
          matchedRuleIds.add(rule.id);
          predefinedSuggestions.push(...rule.suggestions);
          if (rule.taxonomy === 'category') matchedCategory = rule.term;
          if (rule.taxonomy === 'entity') matchedEntities.push(rule.term);
        }
      }

      predefinedSuggestions = [...new Set(predefinedSuggestions)];

      let finalSuggestions = [...predefinedSuggestions];
      if (response.suggestions && response.suggestions.length > 0) {
        const remainingSlots = Math.max(0, DEFAULT_SUGGESTION_COUNT - finalSuggestions.length);
        finalSuggestions.push(...response.suggestions.slice(0, remainingSlots));
      }

      const finalClassification = {
        category: matchedCategory || extractedCategory,
        entities: [...new Set([...matchedEntities, ...extractedEntities])]
      };

      setCurrentClassification(finalClassification);
      setCurrentSuggestions(finalSuggestions);
      setAppState('suggestions');
      
    } catch (error: any) {
      console.error("Error during analysis:", error);
      if (error.message.includes("quota exhaustion")) {
        alert("We've hit our API usage limit for all available models. Please wait a few minutes and try again.");
      } else if (error.message.includes("PERMISSION_DENIED")) {
        alert("Permission denied. Please check your API key settings in the project configuration.");
      } else {
        alert("An error occurred during analysis. Please try again.");
      }
      setAppState('idle');
    }
  };

  const handleActionSelect = async (action: string, isCustom: boolean) => {
    setAppState('executing');
    setSelectedAction(action);
    setExcludedTags([]);
    setRefiningCountdown(20);
    setIsRefining(true);
    setIsExecuting(true);
    
    try {
      const result = await executeAction(action, inputText, inputImage);
      setCurrentResult(result);
      setAppState('result');
    } catch (error: any) {
      console.error("Error executing action:", error);
      if (error.message.includes("quota exhaustion")) {
        setCurrentResult("We've hit our API usage limit for all available models. Please wait a few minutes and try again.");
      } else if (error.message.includes("PERMISSION_DENIED")) {
        setCurrentResult("Permission denied. Please check your API key settings in the project configuration.");
      } else {
        setCurrentResult("An error occurred while executing the action.");
      }
      setAppState('result');
    } finally {
      setIsExecuting(false);
    }
  };

  const finalizeLogAndComplete = () => {
    const scores: Record<string, number> = {};
    const isCustom = !!(customAction && selectedAction === customAction);
    
    if (isCustom) {
      currentSuggestions.forEach(s => { scores[s] = -1; });
      scores[selectedAction!] = 1;
    } else {
      currentSuggestions.forEach(s => {
        scores[s] = (s === selectedAction) ? 1 : -1;
      });
    }

    // Programmatic sanitization
    let generalizedAction = selectedAction;
    if (generalizedAction && currentClassification?.entities) {
      currentClassification.entities.forEach(entity => {
        // Case-insensitive replacement of the entity with "{entity}"
        const regex = new RegExp(entity, 'gi');
        generalizedAction = generalizedAction!.replace(regex, '{entity}');
      });
    }

    const newLog: LogEntry = {
      id: Date.now().toString(),
      timestamp: Date.now(),
      input: {
        text: inputText,
        image: inputImage
      },
      classification: currentClassification || { entities: [], category: 'Unknown' },
      suggestionsOffered: currentSuggestions,
      actionTaken: selectedAction!,
      generalizedAction,
      isCustomAction: isCustom,
      scores,
      result: currentResult || "",
      embedding: currentEmbedding,
      excludedTags
    };
    
    setLogs(prev => [newLog, ...prev]);
    setIsRefining(false);
  };

  useEffect(() => {
    if (isRefining) {
      if (refiningCountdown > 0) {
        const timer = setTimeout(() => setRefiningCountdown(c => c - 1), 1000);
        return () => clearTimeout(timer);
      } else if (!isExecuting && appState === 'result') {
        finalizeLogAndComplete();
      }
    }
  }, [isRefining, refiningCountdown, isExecuting, appState]);

  const reset = () => {
    setInputText('');
    setInputImage(null);
    setCurrentClassification(null);
    setCurrentSuggestions([]);
    setCurrentResult(null);
    setCustomAction('');
    setShowCustomInput(false);
    setSelectedAction(null);
    setAppState('idle');
  };

  const isIdle = appState === 'idle' || appState === 'analyzing';

  return (
    <div className="min-h-screen bg-neutral-50 text-neutral-900 font-sans selection:bg-neutral-200 flex overflow-hidden">
      
      {/* Main Content Area */}
      <div className={`flex-1 flex flex-col transition-all duration-500 ease-in-out ${showLogs ? 'md:mr-80' : 'mr-0'}`}>
        
        {/* Header */}
        <header className="p-4 md:p-6 flex justify-between items-center bg-white/80 backdrop-blur-md border-b border-neutral-200 sticky top-0 z-10">
          <div className="flex items-center gap-2">
            <Sparkles className="w-5 h-5 text-neutral-700" />
            <h1 className="font-medium tracking-tight text-lg hidden sm:block">Proactive AI</h1>
          </div>
          <div className="flex items-center gap-2 md:gap-4">
            <Link 
              to="/taxonomy"
              className="text-sm font-medium text-neutral-500 hover:text-neutral-900 transition-colors px-2 py-1"
            >
              Taxonomy
            </Link>
            <button 
              onClick={reset}
              className="text-sm font-medium text-neutral-500 hover:text-neutral-900 transition-colors px-2 py-1"
            >
              New Chat
            </button>
            <button 
              onClick={() => setShowLogs(!showLogs)}
              className="flex items-center gap-2 px-3 py-1.5 rounded-full hover:bg-neutral-100 transition-colors text-sm font-medium text-neutral-600"
            >
              <History className="w-4 h-4" />
              <span className="hidden sm:inline">{showLogs ? 'Hide Trace' : 'View Trace'}</span>
            </button>
          </div>
        </header>

        {/* Main Viewport */}
        <main className="flex-1 overflow-y-auto p-4 md:p-8 flex justify-center relative">
          
          <div className={`flex flex-col md:flex-row items-center md:items-start gap-8 md:gap-12 w-full max-w-6xl transition-all duration-700 ease-in-out h-full ${isIdle ? 'justify-center' : 'justify-start md:ml-16'}`}>
            
            <div className={`flex ${isIdle ? 'flex-col items-center justify-center w-full h-full' : 'flex-col md:flex-row items-center md:items-start gap-8 md:gap-12 w-full'} transition-all duration-700`}>
              {/* Input Box */}
              <motion.div 
                layout
                className={`relative bg-white border border-neutral-200 overflow-hidden flex flex-col transition-all duration-500 focus-within:border-neutral-300 focus-within:shadow-md flex-shrink-0 shadow-sm ${
                  isIdle ? 'w-full max-w-[500px] min-h-[300px] md:h-[500px] rounded-[32px] md:rounded-[40px]' : 'w-full max-w-[350px] min-h-[250px] md:h-[350px] rounded-[24px] md:rounded-[32px]'
                }`}
              onDragOver={handleDragOver}
              onDrop={handleDrop}
            >
              <textarea
                value={inputText}
                onChange={(e) => setInputText(e.target.value)}
                placeholder="Type a name, paste a link, or ask a question..."
                className={`flex-1 w-full outline-none resize-none bg-transparent ${isIdle ? 'p-6 md:p-8 text-lg md:text-xl' : 'p-4 md:p-6 text-base md:text-lg'}`}
                disabled={!isIdle}
              />
              
              {inputImage && (
                <div className={`pb-4 relative ${isIdle ? 'px-6 md:px-8' : 'px-4 md:px-6'}`}>
                  <div className="relative rounded-xl overflow-hidden border border-neutral-200 inline-block">
                    <img src={inputImage} alt="Uploaded" className={`${isIdle ? 'max-h-32 md:max-h-48' : 'max-h-24 md:max-h-32'} object-contain`} />
                    <button 
                      onClick={removeImage}
                      className="absolute top-2 right-2 p-1.5 bg-black/50 text-white rounded-full hover:bg-black/70 transition-colors backdrop-blur-sm"
                      disabled={!isIdle}
                    >
                      <X className="w-4 h-4" />
                    </button>
                  </div>
                </div>
              )}
              
              <div className={`flex justify-between items-center bg-neutral-50/50 border-t border-neutral-100 ${isIdle ? 'p-4 md:p-6' : 'p-3 md:p-4'}`}>
                <div>
                  <input 
                    type="file" 
                    accept="image/*" 
                    className="hidden" 
                    ref={fileInputRef}
                    onChange={handleImageUpload}
                    disabled={!isIdle}
                  />
                  <button 
                    onClick={() => fileInputRef.current?.click()}
                    className="p-3 bg-neutral-100 text-neutral-600 hover:text-neutral-900 hover:bg-neutral-200 rounded-full transition-colors"
                    disabled={!isIdle}
                    title="Upload Image"
                  >
                    <Plus className="w-5 h-5" />
                  </button>
                </div>
                <div className="flex items-center gap-2">
                  <button
                    className="p-3 bg-neutral-100 text-neutral-600 hover:text-neutral-900 hover:bg-neutral-200 rounded-full transition-colors"
                    disabled={!isIdle}
                    title="Voice Input"
                  >
                    <Mic className="w-5 h-5" />
                  </button>
                  <button
                    onClick={handleAnalyze}
                    disabled={(!inputText.trim() && !inputImage) || !isIdle}
                    className="p-3 bg-black text-white rounded-full hover:bg-neutral-800 disabled:opacity-50 disabled:cursor-not-allowed transition-all active:scale-95"
                  >
                    {appState === 'analyzing' ? (
                      <Loader2 className="w-5 h-5 animate-spin" />
                    ) : (
                      <ArrowUp className="w-5 h-5" />
                    )}
                  </button>
                </div>
              </div>
            </motion.div>

            {/* Suggestions & Results Area */}
            <AnimatePresence>
              {!isIdle && (
                <motion.div 
                  initial={{ opacity: 0, x: 20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: 0.3, duration: 0.5 }}
                  className="flex flex-col gap-6 w-full max-w-xl flex-1"
                >
                  {/* Suggestions List */}
                  {(appState === 'suggestions' || appState === 'executing' || appState === 'result') && (
                    <div className="flex flex-col gap-3 items-start w-full">
                      {currentSuggestions.map((suggestion, idx) => {
                        const isSelected = selectedAction === suggestion;
                        const isFaded = selectedAction && !isSelected;
                        
                        if (isFaded && (appState === 'result' || isRefining)) return null; // Hide unselected in result/refining state

                        return (
                          <motion.button
                            key={idx}
                            initial={{ opacity: 0, y: 10 }}
                            animate={{ opacity: isFaded ? 0.5 : 1, y: 0 }}
                            transition={{ delay: idx * 0.1, type: "spring", stiffness: 300, damping: 30 }}
                            onClick={() => handleActionSelect(suggestion, false)}
                            disabled={appState === 'executing' || appState === 'result'}
                            className={`text-left px-[15px] py-[10px] rounded-3xl transition-all max-w-full disabled:cursor-default ${
                              isSelected 
                                ? 'bg-blue-600 text-white' 
                                : 'bg-blue-500 text-white hover:bg-blue-600'
                            }`}
                          >
                            {suggestion}
                          </motion.button>
                        );
                      })}
                      
                      {/* Custom Action Plus Sign */}
                      {(!selectedAction || (selectedAction && customAction === selectedAction)) && (
                        <motion.div
                          initial={{ opacity: 0, y: 10 }}
                          animate={{ opacity: 1, y: 0 }}
                          transition={{ delay: currentSuggestions.length * 0.1, type: "spring", stiffness: 300, damping: 30 }}
                          className="flex items-center w-full mt-1"
                        >
                          {showCustomInput || (selectedAction && customAction === selectedAction) ? (
                            <div className={`flex items-center gap-2 rounded-3xl px-4 py-2 w-full max-w-md ${
                              selectedAction ? 'bg-blue-600 text-white' : 'bg-neutral-200 text-neutral-800'
                            }`}>
                              <input
                                autoFocus
                                type="text"
                                value={customAction}
                                onChange={(e) => setCustomAction(e.target.value)}
                                placeholder="Type custom action..."
                                className={`bg-transparent outline-none w-full ${selectedAction ? 'text-white placeholder:text-blue-200' : 'text-neutral-800 placeholder:text-neutral-500'}`}
                                disabled={appState === 'executing' || appState === 'result'}
                                onKeyDown={(e) => {
                                  if (e.key === 'Enter' && customAction.trim()) {
                                    handleActionSelect(customAction, true);
                                  }
                                }}
                              />
                              {!selectedAction && (
                                <button 
                                  onClick={() => handleActionSelect(customAction, true)} 
                                  disabled={!customAction.trim()}
                                  className="p-1.5 bg-blue-500 text-white rounded-full hover:bg-blue-600 disabled:opacity-50"
                                >
                                  <ArrowUp className="w-4 h-4" />
                                </button>
                              )}
                            </div>
                          ) : (
                            <button 
                              onClick={() => setShowCustomInput(true)}
                              className="w-[44px] h-[44px] rounded-full bg-blue-500 text-white flex items-center justify-center hover:bg-blue-600 transition-colors"
                            >
                              <Plus className="w-6 h-6" />
                            </button>
                          )}
                        </motion.div>
                      )}
                    </div>
                  )}

                  {/* Executing State */}
                  {appState === 'executing' && (
                    <motion.div 
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      className="flex items-center gap-3 text-neutral-500 py-4 px-2"
                    >
                      <Loader2 className="w-5 h-5 animate-spin" />
                      <span className="font-medium">Executing action...</span>
                    </motion.div>
                  )}

                  {/* Result Area */}
                  {appState === 'result' && (
                    <motion.div 
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ type: "spring", stiffness: 300, damping: 30 }}
                      className="bg-white rounded-3xl rounded-tr-sm p-4 md:p-6 border border-neutral-200 prose prose-neutral max-w-none mt-4"
                    >
                      <ReactMarkdown remarkPlugins={[remarkGfm]}>{currentResult || ''}</ReactMarkdown>
                    </motion.div>
                  )}
                </motion.div>
              )}
            </AnimatePresence>
            </div>
            
            {/* Parallel Refining UI */}
            <AnimatePresence>
              {isRefining && (
                <motion.div 
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: 20 }}
                  className="fixed bottom-4 left-4 right-4 md:bottom-auto md:left-auto md:top-24 md:right-8 z-50 bg-white/90 md:bg-transparent backdrop-blur-md md:backdrop-blur-none p-4 md:p-0 rounded-2xl md:rounded-none shadow-2xl md:shadow-none border border-neutral-200 md:border-none"
                >
                  <div className="w-full md:w-72 opacity-100 md:opacity-70 hover:opacity-100 focus-within:opacity-100 transition-opacity">
                    <div className="mb-4 text-left md:text-right">
                      <h3 className="font-medium text-neutral-800">
                        Appending positive scoring to:
                      </h3>
                    </div>
                    
                    <div className="flex flex-row md:flex-col flex-wrap items-start md:items-end gap-2 mb-6">
                      {currentClassification && (
                        <>
                          <div className="group flex items-center gap-1">
                            <div 
                              className="w-5 h-5 rounded-full bg-neutral-400 hover:bg-black text-white flex items-center justify-center cursor-pointer opacity-0 group-hover:opacity-100 transition-all"
                              onClick={() => setExcludedTags(prev => prev.includes(currentClassification.category) ? prev.filter(t => t !== currentClassification.category) : [...prev, currentClassification.category])}
                            >
                              <X className="w-3 h-3" />
                            </div>
                            <button
                              onClick={() => setExcludedTags(prev => prev.includes(currentClassification.category) ? prev.filter(t => t !== currentClassification.category) : [...prev, currentClassification.category])}
                              className={`px-2 py-1.5 rounded-lg text-sm transition-all border ${
                                excludedTags.includes(currentClassification.category) 
                                  ? 'bg-neutral-100 text-neutral-400 border-neutral-200 line-through' 
                                  : 'bg-blue-50 text-blue-700 border-blue-100 hover:bg-blue-100'
                              }`}
                            >
                              {currentClassification.category}
                            </button>
                          </div>
                          
                          {currentClassification.entities.map((entity, i) => (
                            <div key={i} className="group flex items-center gap-1">
                              <div 
                                className="w-5 h-5 rounded-full bg-neutral-400 hover:bg-black text-white flex items-center justify-center cursor-pointer opacity-0 group-hover:opacity-100 transition-all"
                                onClick={() => setExcludedTags(prev => prev.includes(entity) ? prev.filter(t => t !== entity) : [...prev, entity])}
                              >
                                <X className="w-3 h-3" />
                              </div>
                              <button
                                onClick={() => setExcludedTags(prev => prev.includes(entity) ? prev.filter(t => t !== entity) : [...prev, entity])}
                                className={`px-2 py-1.5 rounded-lg text-sm transition-all border ${
                                  excludedTags.includes(entity) 
                                    ? 'bg-neutral-100 text-neutral-400 border-neutral-200 line-through' 
                                    : 'bg-neutral-50 text-neutral-700 border-neutral-200 hover:bg-neutral-100'
                                }`}
                              >
                                {entity}
                              </button>
                            </div>
                          ))}
                        </>
                      )}
                    </div>
                    
                    <div className="flex justify-end items-center gap-4">
                      <div className="text-xs font-mono bg-neutral-100 text-neutral-600 px-2 py-1 rounded-md">
                        {refiningCountdown}s
                      </div>
                      <button
                        onClick={finalizeLogAndComplete}
                        className="px-4 py-2 bg-black text-white rounded-full text-sm font-medium hover:bg-neutral-800 transition-colors"
                      >
                        Save
                      </button>
                    </div>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
            
          </div>
        </main>
      </div>

      {/* Trace / Feedback Log Sidebar */}
      <AnimatePresence>
        {showLogs && (
          <>
            <motion.div 
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              onClick={() => setShowLogs(false)}
              className="fixed inset-0 bg-black/20 z-20 md:hidden backdrop-blur-sm"
            />
            <motion.aside
              initial={{ x: '100%' }}
              animate={{ x: 0 }}
              exit={{ x: '100%' }}
              transition={{ type: 'spring', damping: 25, stiffness: 200 }}
              className="fixed top-0 right-0 w-full max-w-[320px] md:w-80 h-full bg-white border-l border-neutral-200 z-30 flex flex-col shadow-2xl md:shadow-none"
            >
            <div className="p-4 border-b border-neutral-100 flex justify-between items-center bg-neutral-50/50">
              <div className="flex items-center gap-2">
                <History className="w-4 h-4 text-neutral-600" />
                <h2 className="font-semibold text-neutral-800">Trace Log</h2>
              </div>
              <button 
                onClick={() => setShowLogs(false)}
                className="p-1.5 text-neutral-400 hover:text-neutral-800 hover:bg-neutral-200/50 rounded-md transition-colors"
              >
                <X className="w-4 h-4" />
              </button>
            </div>
            
            <div className="flex-1 overflow-y-auto p-4 flex flex-col gap-4">
              {logs.length === 0 ? (
                <div className="text-center text-neutral-400 py-10 text-sm">
                  No interactions logged yet.
                </div>
              ) : (
                logs.slice(0, 10).map((log) => (
                  <div key={log.id} className="bg-neutral-50 rounded-xl p-4 border border-neutral-100 text-sm flex flex-col gap-3">
                    <div>
                      <div className="text-xs font-semibold text-neutral-400 uppercase mb-1">Input</div>
                      <div className="text-neutral-700 line-clamp-2">
                        {log.input.text || (log.input.image ? "[Image Uploaded]" : "Empty")}
                      </div>
                    </div>
                    <div>
                      <div className="text-xs font-semibold text-neutral-400 uppercase mb-1">Action Taken</div>
                      <div className="text-neutral-900 font-medium line-clamp-2">{log.actionTaken}</div>
                    </div>
                    <div className="text-[10px] text-neutral-400 text-right mt-1">
                      {new Date(log.timestamp).toLocaleString()}
                    </div>
                  </div>
                ))
              )}
            </div>
            
            <div className="p-4 border-t border-neutral-100 bg-white">
              <Link 
                to="/trace"
                className="w-full py-2.5 bg-neutral-900 text-white rounded-xl font-medium flex items-center justify-center gap-2 hover:bg-neutral-800 transition-colors"
              >
                View Full Trace Page <ChevronRight className="w-4 h-4" />
              </Link>
            </div>
          </motion.aside>
          </>
        )}
      </AnimatePresence>
      
    </div>
  );
}

function MiniRecordModal({ isOpen, onClose, title, logs }: { isOpen: boolean, onClose: () => void, title: string, logs: LogEntry[] }) {
  if (!isOpen) return null;
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4">
      <div className="bg-white rounded-2xl w-full max-w-2xl max-h-[80vh] flex flex-col">
        <div className="p-4 border-b border-neutral-100 flex justify-between items-center bg-neutral-50 rounded-t-2xl">
          <h2 className="text-lg font-semibold text-neutral-800">Records for "{title}"</h2>
          <button onClick={onClose} className="p-1.5 hover:bg-neutral-200 rounded-md transition-colors"><X className="w-5 h-5" /></button>
        </div>
        <div className="p-4 overflow-y-auto flex-1 space-y-4">
          {logs.length === 0 ? (
            <div className="text-center text-neutral-400 py-8">No records found.</div>
          ) : (
            logs.map(log => (
              <div key={log.id} className="p-4 border border-neutral-100 rounded-xl bg-white">
                <div className="text-sm text-neutral-600 mb-3 line-clamp-2 bg-neutral-50 p-2 rounded-lg">{log.input.text || "[Image Input]"}</div>
                <div className="flex flex-wrap gap-2 mb-4">
                  <span className="px-2 py-0.5 bg-blue-50 text-blue-700 rounded text-xs border border-blue-100 font-medium">{log.classification.category}</span>
                  {log.classification.entities.map((e, i) => (
                    <span key={i} className="px-2 py-0.5 bg-neutral-100 text-neutral-700 rounded text-xs border border-neutral-200">{e}</span>
                  ))}
                </div>
                <div className="space-y-2">
                  <h4 className="text-xs font-semibold text-neutral-400 uppercase tracking-wider">Scores</h4>
                  {Object.entries(log.scores).map(([action, score], i) => (
                    <div key={i} className="flex justify-between items-center text-sm p-2 bg-neutral-50 rounded-lg">
                      <span className="text-neutral-700">{action}</span>
                      <span className={`font-mono px-2 py-0.5 rounded-md text-xs ${score > 0 ? 'bg-green-100 text-green-700' : 'bg-red-50 text-red-600'}`}>{score > 0 ? '+1' : '-1'}</span>
                    </div>
                  ))}
                </div>
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  );
}

function TaxonomyLabel({ 
  type, 
  value, 
  isEditMode, 
  allLogs, 
  onEdit,
  isExcluded
}: { 
  type: 'category' | 'entity', 
  value: string, 
  isEditMode: boolean, 
  allLogs: LogEntry[], 
  onEdit: (newVal: string) => void,
  isExcluded?: boolean
}) {
  const [showModal, setShowModal] = useState(false);
  const [showEdit, setShowEdit] = useState(false);
  const [editValue, setEditValue] = useState(value);
  const editRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    function handleClickOutside(e: MouseEvent) {
      if (editRef.current && !editRef.current.contains(e.target as Node)) {
        setShowEdit(false);
      }
    }
    if (showEdit) document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, [showEdit]);

  const associatedLogs = allLogs.filter(log => {
    if (type === 'category') return log.classification.category === value;
    return log.classification.entities.includes(value);
  });

  const allOptions = Array.from(new Set(allLogs.flatMap(log => {
    if (type === 'category') return [log.classification.category];
    return log.classification.entities;
  }))).filter(Boolean);

  const filteredOptions = allOptions.filter(opt => opt.toLowerCase().includes(editValue.toLowerCase()) && opt !== editValue);

  const handleSave = (val: string) => {
    if (val.trim() && val !== value) {
      onEdit(val.trim());
    }
    setShowEdit(false);
  };

  const baseClasses = `px-2.5 py-1 rounded-md text-xs border cursor-pointer transition-all relative inline-flex items-center gap-1 ${
    isExcluded 
      ? 'bg-neutral-100 text-neutral-400 border-neutral-200 line-through hover:bg-neutral-200' 
      : type === 'category' 
        ? 'bg-blue-50 text-blue-700 border-blue-100 hover:bg-blue-100 font-medium' 
        : 'bg-neutral-100 text-neutral-700 border-neutral-200 hover:bg-neutral-200'
  } ${isEditMode ? 'ring-2 ring-blue-500/20 ring-offset-1' : ''}`;

  return (
    <>
      <div className="relative inline-block" ref={editRef}>
        <span 
          className={baseClasses}
          onClick={() => {
            if (isEditMode) {
              setEditValue(value);
              setShowEdit(true);
            } else {
              setShowModal(true);
            }
          }}
        >
          {value}
          {isEditMode && <Edit2 className="w-3 h-3 ml-1 opacity-50" />}
        </span>

        {showEdit && (
          <div className="absolute top-full left-0 mt-2 w-56 bg-white border border-neutral-200 rounded-xl z-50 overflow-hidden">
            <div className="p-2 border-b border-neutral-100 bg-neutral-50 relative">
              <Search className="w-4 h-4 absolute left-4 top-1/2 -translate-y-1/2 text-neutral-400" />
              <input 
                type="text"
                value={editValue}
                onChange={e => setEditValue(e.target.value)}
                className="w-full text-sm pl-8 pr-2 py-1.5 border border-neutral-200 rounded-lg focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
                placeholder={`Edit ${type}...`}
                autoFocus
                onKeyDown={e => {
                  if (e.key === 'Enter') handleSave(editValue);
                  if (e.key === 'Escape') setShowEdit(false);
                }}
              />
            </div>
            {filteredOptions.length > 0 && (
              <div className="max-h-40 overflow-y-auto py-1">
                {filteredOptions.map(opt => (
                  <div 
                    key={opt} 
                    className="px-3 py-2 text-sm text-neutral-700 hover:bg-blue-50 hover:text-blue-700 cursor-pointer transition-colors"
                    onClick={() => handleSave(opt)}
                  >
                    {opt}
                  </div>
                ))}
              </div>
            )}
            <div className="p-2 bg-neutral-50 border-t border-neutral-100 flex justify-between items-center">
              <span className="text-xs text-neutral-400">Press Enter to save</span>
              <button 
                onClick={() => handleSave(editValue)}
                className="px-3 py-1.5 bg-black text-white text-xs font-medium rounded-lg hover:bg-neutral-800 transition-colors"
              >
                Save
              </button>
            </div>
          </div>
        )}
      </div>

      <MiniRecordModal 
        isOpen={showModal} 
        onClose={() => setShowModal(false)} 
        title={value} 
        logs={associatedLogs} 
      />
    </>
  );
}

function TraceLogPage({ logs, setLogs }: { logs: LogEntry[], setLogs: React.Dispatch<React.SetStateAction<LogEntry[]>> }) {
  const navigate = useNavigate();
  const [isEditMode, setIsEditMode] = useState(false);

  const handleEditCategory = (logId: string, newCategory: string) => {
    setLogs(prev => prev.map(log => 
      log.id === logId 
        ? { ...log, classification: { ...log.classification, category: newCategory } } 
        : log
    ));
  };

  const handleEditEntity = (logId: string, oldEntity: string, newEntity: string) => {
    setLogs(prev => prev.map(log => {
      if (log.id !== logId) return log;
      const newEntities = log.classification.entities.map(e => e === oldEntity ? newEntity : e);
      return { ...log, classification: { ...log.classification, entities: newEntities } };
    }));
  };
  
  return (
    <div className="min-h-screen bg-neutral-50 text-neutral-900 font-sans p-4 md:p-8">
      <div className="max-w-5xl mx-auto">
        <header className="flex flex-col md:flex-row items-start md:items-center justify-between gap-4 mb-8">
          <div className="flex items-center gap-4">
            <button 
              onClick={() => navigate('/')}
              className="p-2 hover:bg-neutral-200 rounded-full transition-colors shrink-0"
            >
              <ArrowLeft className="w-6 h-6" />
            </button>
            <div>
              <h1 className="text-xl md:text-2xl font-semibold tracking-tight">Feedback & Trace Log</h1>
              <p className="text-sm md:text-base text-neutral-500">View and edit scoring to improve future suggestions.</p>
            </div>
          </div>
          <div className="flex flex-wrap items-center gap-3 w-full md:w-auto pl-14 md:pl-0">
            <button 
              onClick={() => setIsEditMode(!isEditMode)}
              className={`px-4 py-2 rounded-xl text-sm font-medium transition-colors flex items-center gap-2 ${
                isEditMode 
                  ? 'bg-blue-100 text-blue-700 hover:bg-blue-200' 
                  : 'bg-white border border-neutral-200 text-neutral-700 hover:bg-neutral-50'
              }`}
            >
              <Edit2 className="w-4 h-4" />
              {isEditMode ? 'Exit Edit Mode' : 'Edit Taxonomy'}
            </button>
            <Link 
              to="/taxonomy"
              className="px-4 py-2 bg-neutral-900 text-white rounded-xl text-sm font-medium hover:bg-neutral-800 transition-colors"
            >
              View Taxonomy
            </Link>
          </div>
        </header>
        
        <div className="bg-white rounded-3xl border border-neutral-200 overflow-hidden">
          {logs.length === 0 ? (
            <div className="p-12 text-center text-neutral-500">
              No logs available yet. Interact with the AI to generate trace data.
            </div>
          ) : (
            <div className="divide-y divide-neutral-100">
              {logs.map(log => (
                <div key={log.id} className="p-4 md:p-6 hover:bg-neutral-50 transition-colors flex flex-col md:flex-row gap-6">
                  <div className="flex-1 space-y-4">
                    <div>
                      <h3 className="text-sm font-semibold text-neutral-400 uppercase tracking-wider mb-2">Input</h3>
                      <p className="text-neutral-800 bg-neutral-100 p-3 rounded-xl">{log.input.text || (log.input.image ? "[Image Uploaded]" : "Empty")}</p>
                    </div>
                    <div>
                      <h3 className="text-sm font-semibold text-neutral-400 uppercase tracking-wider mb-2">Classification</h3>
                      <div className="flex flex-wrap gap-2">
                        <TaxonomyLabel 
                          type="category"
                          value={log.classification.category}
                          isEditMode={isEditMode}
                          allLogs={logs}
                          onEdit={(newVal) => handleEditCategory(log.id, newVal)}
                          isExcluded={log.excludedTags?.includes(log.classification.category)}
                        />
                        {log.classification.entities.map((e, i) => (
                          <TaxonomyLabel 
                            key={i}
                            type="entity"
                            value={e}
                            isEditMode={isEditMode}
                            allLogs={logs}
                            onEdit={(newVal) => handleEditEntity(log.id, e, newVal)}
                            isExcluded={log.excludedTags?.includes(e)}
                          />
                        ))}
                      </div>
                    </div>
                  </div>
                  
                  <div className="flex-1 space-y-4">
                    <div>
                      <h3 className="text-sm font-semibold text-neutral-400 uppercase tracking-wider mb-2">Scoring & Actions</h3>
                      <div className="space-y-2">
                        {Object.entries(log.scores).map(([action, score], i) => (
                          <div key={i} className={`flex justify-between items-center p-3 rounded-xl border ${score > 0 ? 'bg-green-50/50 border-green-100' : 'bg-white border-neutral-100'}`}>
                            <span className={`text-sm ${score > 0 ? 'font-medium text-neutral-900' : 'text-neutral-500 line-through'}`}>
                              {action}
                            </span>
                            <div className="flex items-center gap-2">
                              <span className={`font-mono text-xs px-2 py-1 rounded-md ${score > 0 ? 'bg-green-100 text-green-700' : 'bg-red-50 text-red-600'}`}>
                                {score > 0 ? '+1' : '-1'}
                              </span>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                    <div className="text-xs text-neutral-400 text-right">
                      {new Date(log.timestamp).toLocaleString()}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function TaxonomyPage({ logs, setLogs, rules, setRules }: { logs: LogEntry[], setLogs: React.Dispatch<React.SetStateAction<LogEntry[]>>, rules: Rule[], setRules: React.Dispatch<React.SetStateAction<Rule[]>> }) {
  const navigate = useNavigate();
  const [isEditMode, setIsEditMode] = useState(false);
  
  // Rule form state
  const [editingRuleId, setEditingRuleId] = useState<string | null>(null);
  const [newTerm, setNewTerm] = useState('');
  const [newTaxonomy, setNewTaxonomy] = useState<'category' | 'entity'>('category');
  const [newIsExact, setNewIsExact] = useState(false);
  const [newThreshold, setNewThreshold] = useState(80);
  const [newSuggestions, setNewSuggestions] = useState<string[]>(['']);
  const [isSavingRule, setIsSavingRule] = useState(false);

  const handleSaveRule = async () => {
    if (!newTerm.trim()) return;
    setIsSavingRule(true);
    try {
      let termEmbedding;
      if (!newIsExact) {
        termEmbedding = await getEmbedding(newTerm);
      }
      const filteredSuggestions = newSuggestions.filter(s => s.trim() !== '');
      
      if (editingRuleId) {
        setRules(prev => prev.map(rule => 
          rule.id === editingRuleId 
            ? {
                ...rule,
                term: newTerm.trim(),
                taxonomy: newTaxonomy,
                isExactMatch: newIsExact,
                threshold: newThreshold,
                suggestions: filteredSuggestions,
                termEmbedding
              }
            : rule
        ));
        setEditingRuleId(null);
      } else {
        const newRule: Rule = {
          id: Date.now().toString(),
          term: newTerm.trim(),
          taxonomy: newTaxonomy,
          isExactMatch: newIsExact,
          threshold: newThreshold,
          suggestions: filteredSuggestions,
          termEmbedding
        };
        setRules(prev => [...prev, newRule]);
      }
      
      // Reset form
      setNewTerm('');
      setNewSuggestions(['']);
      setNewIsExact(false);
      setNewThreshold(80);
      setNewTaxonomy('category');
    } catch (e) {
      console.error(e);
      alert("Failed to save rule. Please check your API key.");
    } finally {
      setIsSavingRule(false);
    }
  };

  const handleEditRule = (rule: Rule) => {
    setEditingRuleId(rule.id);
    setNewTerm(rule.term);
    setNewTaxonomy(rule.taxonomy);
    setNewIsExact(rule.isExactMatch);
    setNewThreshold(rule.threshold);
    setNewSuggestions(rule.suggestions.length > 0 ? rule.suggestions : ['']);
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  const handleCancelEdit = () => {
    setEditingRuleId(null);
    setNewTerm('');
    setNewSuggestions(['']);
    setNewIsExact(false);
    setNewThreshold(80);
    setNewTaxonomy('category');
  };

  const handleDeleteRule = (id: string) => {
    setRules(prev => prev.filter(o => o.id !== id));
    if (editingRuleId === id) {
      handleCancelEdit();
    }
  };

  // Group logs by category and entity independently
  const { categories, entities } = React.useMemo(() => {
    const cats: Record<string, LogEntry[]> = {};
    const ents: Record<string, LogEntry[]> = {};
    
    logs.forEach(log => {
      const cat = log.classification.category || 'Unknown';
      if (!cats[cat]) cats[cat] = [];
      cats[cat].push(log);
      
      log.classification.entities.forEach(ent => {
        if (!ents[ent]) ents[ent] = [];
        ents[ent].push(log);
      });
    });
    
    return { categories: cats, entities: ents };
  }, [logs]);

  const handleEditCategory = (logId: string, newCategory: string) => {
    setLogs(prev => prev.map(log => 
      log.id === logId 
        ? { ...log, classification: { ...log.classification, category: newCategory } } 
        : log
    ));
  };

  const handleEditEntity = (logId: string, oldEntity: string, newEntity: string) => {
    setLogs(prev => prev.map(log => {
      if (log.id !== logId) return log;
      const newEntities = log.classification.entities.map(e => e === oldEntity ? newEntity : e);
      return { ...log, classification: { ...log.classification, entities: newEntities } };
    }));
  };

  return (
    <div className="min-h-screen bg-neutral-50 text-neutral-900 font-sans p-4 md:p-8">
      <div className="max-w-5xl mx-auto">
        <header className="flex flex-col md:flex-row items-start md:items-center justify-between gap-4 mb-8">
          <div className="flex items-center gap-4">
            <button 
              onClick={() => navigate('/')}
              className="p-2 hover:bg-neutral-200 rounded-full transition-colors shrink-0"
            >
              <ArrowLeft className="w-6 h-6" />
            </button>
            <div>
              <h1 className="text-xl md:text-2xl font-semibold tracking-tight">Taxonomy & Relationships</h1>
              <p className="text-sm md:text-base text-neutral-500">Understand how the AI organizes and learns from your data.</p>
            </div>
          </div>
          <div className="w-full md:w-auto pl-14 md:pl-0">
            <button 
              onClick={() => setIsEditMode(!isEditMode)}
              className={`px-4 py-2 rounded-xl text-sm font-medium transition-colors flex items-center justify-center w-full md:w-auto gap-2 ${
                isEditMode 
                  ? 'bg-blue-100 text-blue-700 hover:bg-blue-200' 
                  : 'bg-white border border-neutral-200 text-neutral-700 hover:bg-neutral-50'
              }`}
            >
              <Edit2 className="w-4 h-4" />
              {isEditMode ? 'Exit Edit Mode' : 'Edit Taxonomy'}
            </button>
          </div>
        </header>

        {/* Rules Section */}
        <div className="bg-white rounded-3xl border border-neutral-200 overflow-hidden p-4 md:p-8 mb-8">
          <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
            <Sparkles className="w-5 h-5 text-purple-500" />
            Rules
          </h2>
          <p className="text-sm text-neutral-500 mb-6">
            Define deterministic rules to guide the AI. When an input matches a term (either exactly or semantically), the system will prioritize your pre-defined suggestions.
          </p>

          {/* Add/Edit Rule Form */}
          <div className="bg-neutral-50 p-4 md:p-6 rounded-2xl border border-neutral-200 mb-8">
            <h3 className="text-sm font-semibold mb-4 text-neutral-700">
              {editingRuleId ? 'Edit Rule' : 'Create New Rule'}
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
              <div>
                <label className="block text-xs font-medium text-neutral-500 mb-1">Term / Topic</label>
                <input 
                  type="text" 
                  value={newTerm}
                  onChange={e => setNewTerm(e.target.value)}
                  placeholder="e.g., Design, React, Billing"
                  className="w-full px-3 py-2 rounded-lg border border-neutral-300 focus:outline-none focus:ring-2 focus:ring-blue-500 text-sm"
                />
              </div>
              <div>
                <label className="block text-xs font-medium text-neutral-500 mb-1">Map to Taxonomy</label>
                <select 
                  value={newTaxonomy}
                  onChange={e => setNewTaxonomy(e.target.value as 'category' | 'entity')}
                  className="w-full px-3 py-2 rounded-lg border border-neutral-300 focus:outline-none focus:ring-2 focus:ring-blue-500 text-sm bg-white"
                >
                  <option value="category">Category</option>
                  <option value="entity">Entity</option>
                </select>
              </div>
            </div>

            <div className="flex flex-col md:flex-row md:items-center gap-4 md:gap-6 mb-6 p-4 bg-white rounded-xl border border-neutral-200">
              <label className="flex items-center gap-2 cursor-pointer">
                <input 
                  type="checkbox" 
                  checked={newIsExact}
                  onChange={e => setNewIsExact(e.target.checked)}
                  className="w-4 h-4 text-blue-600 rounded border-gray-300 focus:ring-blue-500"
                />
                <span className="text-sm font-medium text-neutral-700">Exact Match</span>
              </label>
              
              <div className={`flex-1 flex items-center gap-4 ${newIsExact ? 'opacity-50 pointer-events-none' : ''}`}>
                <label className="text-sm font-medium text-neutral-700 whitespace-nowrap">Semantic Threshold:</label>
                <input 
                  type="range" 
                  min="1" 
                  max="100" 
                  value={newThreshold}
                  onChange={e => setNewThreshold(parseInt(e.target.value))}
                  className="flex-1"
                />
                <span className="text-sm font-mono w-12 text-right">{newThreshold}%</span>
              </div>
            </div>

            <div className="mb-6">
              <label className="block text-xs font-medium text-neutral-500 mb-2">Pre-defined Suggestions (Ordered)</label>
              <div className="space-y-2">
                {newSuggestions.map((sug, idx) => (
                  <div key={idx} className="flex items-center gap-2">
                    <span className="text-xs font-mono text-neutral-400 w-4">{idx + 1}.</span>
                    <input 
                      type="text" 
                      value={sug}
                      onChange={e => {
                        const newSugs = [...newSuggestions];
                        newSugs[idx] = e.target.value;
                        setNewSuggestions(newSugs);
                      }}
                      placeholder="Enter suggestion..."
                      className="flex-1 px-3 py-2 rounded-lg border border-neutral-300 focus:outline-none focus:ring-2 focus:ring-blue-500 text-sm"
                    />
                    <div className="flex flex-col gap-0.5">
                      <button
                        onClick={() => {
                          if (idx === 0) return;
                          const newSugs = [...newSuggestions];
                          const temp = newSugs[idx - 1];
                          newSugs[idx - 1] = newSugs[idx];
                          newSugs[idx] = temp;
                          setNewSuggestions(newSugs);
                        }}
                        disabled={idx === 0}
                        className="p-0.5 text-neutral-400 hover:text-neutral-700 disabled:opacity-30 disabled:hover:text-neutral-400 transition-colors"
                      >
                        <ArrowUp className="w-3.5 h-3.5" />
                      </button>
                      <button
                        onClick={() => {
                          if (idx === newSuggestions.length - 1) return;
                          const newSugs = [...newSuggestions];
                          const temp = newSugs[idx + 1];
                          newSugs[idx + 1] = newSugs[idx];
                          newSugs[idx] = temp;
                          setNewSuggestions(newSugs);
                        }}
                        disabled={idx === newSuggestions.length - 1}
                        className="p-0.5 text-neutral-400 hover:text-neutral-700 disabled:opacity-30 disabled:hover:text-neutral-400 transition-colors"
                      >
                        <ArrowDown className="w-3.5 h-3.5" />
                      </button>
                    </div>
                    <button 
                      onClick={() => {
                        const newSugs = newSuggestions.filter((_, i) => i !== idx);
                        if (newSugs.length === 0) newSugs.push('');
                        setNewSuggestions(newSugs);
                      }}
                      className="p-2 text-neutral-400 hover:text-red-500 transition-colors"
                    >
                      <X className="w-4 h-4" />
                    </button>
                  </div>
                ))}
                <button 
                  onClick={() => setNewSuggestions([...newSuggestions, ''])}
                  className="text-xs font-medium text-blue-600 hover:text-blue-700 flex items-center gap-1 mt-2 ml-6"
                >
                  <Plus className="w-3 h-3" /> Add another suggestion
                </button>
              </div>
            </div>

            <div className="flex justify-end gap-3">
              {editingRuleId && (
                <button 
                  onClick={handleCancelEdit}
                  className="px-4 py-2 bg-white border border-neutral-300 text-neutral-700 rounded-xl text-sm font-medium hover:bg-neutral-50 transition-colors"
                >
                  Cancel
                </button>
              )}
              <button 
                onClick={handleSaveRule}
                disabled={!newTerm.trim() || isSavingRule}
                className="px-4 py-2 bg-black text-white rounded-xl text-sm font-medium hover:bg-neutral-800 transition-colors disabled:opacity-50 flex items-center gap-2"
              >
                {isSavingRule && <Loader2 className="w-4 h-4 animate-spin" />}
                {editingRuleId ? 'Update Rule' : 'Save Rule'}
              </button>
            </div>
          </div>

          {/* List of Rules */}
          {rules.length > 0 && (
            <div>
              <h3 className="text-sm font-semibold mb-4 text-neutral-700">Active Rules</h3>
              <div className="space-y-3">
                {rules.map(rule => (
                  <div key={rule.id} className={`flex items-start justify-between p-4 bg-white border rounded-xl transition-colors ${editingRuleId === rule.id ? 'border-blue-500 ring-1 ring-blue-500' : 'border-neutral-200'}`}>
                    <div>
                      <div className="flex items-center gap-2 mb-1">
                        <span className="font-semibold text-neutral-900">{rule.term}</span>
                        <span className="text-[10px] uppercase tracking-wider font-bold px-2 py-0.5 rounded bg-neutral-100 text-neutral-500">
                          {rule.taxonomy}
                        </span>
                        <span className={`text-[10px] uppercase tracking-wider font-bold px-2 py-0.5 rounded ${rule.isExactMatch ? 'bg-purple-100 text-purple-700' : 'bg-blue-100 text-blue-700'}`}>
                          {rule.isExactMatch ? 'Exact Match' : `>= ${rule.threshold}% Match`}
                        </span>
                      </div>
                      <ul className="list-disc pl-5 text-sm text-neutral-600 mt-2 space-y-1">
                        {rule.suggestions.map((s, i) => (
                          <li key={i}>{s}</li>
                        ))}
                      </ul>
                    </div>
                    <div className="flex items-center gap-1">
                      <button 
                        onClick={() => handleEditRule(rule)}
                        className="p-2 text-neutral-400 hover:text-blue-600 hover:bg-blue-50 rounded-lg transition-colors"
                        title="Edit Rule"
                      >
                        <Edit2 className="w-4 h-4" />
                      </button>
                      <button 
                        onClick={() => handleDeleteRule(rule.id)}
                        className="p-2 text-neutral-400 hover:text-red-500 hover:bg-red-50 rounded-lg transition-colors"
                        title="Delete Rule"
                      >
                        <X className="w-4 h-4" />
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Taxonomy Data Section */}
        <div className="bg-white rounded-3xl border border-neutral-200 overflow-hidden p-4 md:p-6 mb-8">
          <h2 className="text-xl font-semibold mb-6">Your Taxonomy Data</h2>
          {Object.keys(categories).length === 0 && Object.keys(entities).length === 0 ? (
            <div className="p-12 text-center text-neutral-500">
              No taxonomy data available yet. Interact with the AI to generate data.
            </div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
              {/* Categories Column */}
              <div>
                <h3 className="text-lg font-medium mb-4 text-neutral-700 border-b pb-2">Categories</h3>
                <div className="space-y-4">
                  {Object.entries(categories).map(([category, catLogs]) => (
                    <div key={category} className="bg-neutral-50 p-4 rounded-xl border border-neutral-100">
                      <div className="flex justify-between items-center mb-3">
                        <div className="flex items-center gap-2">
                          <div className="w-2 h-2 rounded-full bg-blue-500"></div>
                          {catLogs[0] && (
                            <TaxonomyLabel 
                              type="category"
                              value={category}
                              isEditMode={isEditMode}
                              allLogs={logs}
                              onEdit={(newVal) => {
                                catLogs.forEach(log => handleEditCategory(log.id, newVal));
                              }}
                            />
                          )}
                        </div>
                        <span className="text-xs text-neutral-400">{catLogs.length} uses</span>
                      </div>
                      <div className="space-y-2">
                        {catLogs.slice(0, 3).map(log => (
                          <div key={log.id} className="text-xs p-2 bg-white rounded-lg border border-neutral-100 flex justify-between items-center">
                            <span className="truncate max-w-[150px] text-neutral-600" title={log.actionTaken}>{log.actionTaken}</span>
                            <span className={`font-mono px-1.5 py-0.5 rounded ${log.scores[log.actionTaken] > 0 ? 'bg-green-100 text-green-700' : 'bg-red-50 text-red-600'}`}>
                              {log.scores[log.actionTaken] > 0 ? '+1' : '-1'}
                            </span>
                          </div>
                        ))}
                        {catLogs.length > 3 && (
                          <div className="text-xs text-center text-neutral-400 pt-1">
                            +{catLogs.length - 3} more
                          </div>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Entities Column */}
              <div>
                <h3 className="text-lg font-medium mb-4 text-neutral-700 border-b pb-2">Entities</h3>
                <div className="space-y-4">
                  {Object.entries(entities).map(([entity, entLogs]) => (
                    <div key={entity} className="bg-neutral-50 p-4 rounded-xl border border-neutral-100">
                      <div className="flex justify-between items-center mb-3">
                        <div className="flex items-center gap-2">
                          <div className="w-2 h-2 rounded-full bg-purple-500"></div>
                          {entLogs[0] && (
                            <TaxonomyLabel 
                              type="entity"
                              value={entity}
                              isEditMode={isEditMode}
                              allLogs={logs}
                              onEdit={(newVal) => {
                                entLogs.forEach(log => handleEditEntity(log.id, entity, newVal));
                              }}
                            />
                          )}
                        </div>
                        <span className="text-xs text-neutral-400">{entLogs.length} uses</span>
                      </div>
                      <div className="space-y-2">
                        {entLogs.slice(0, 3).map(log => (
                          <div key={log.id} className="text-xs p-2 bg-white rounded-lg border border-neutral-100 flex justify-between items-center">
                            <span className="truncate max-w-[150px] text-neutral-600" title={log.actionTaken}>{log.actionTaken}</span>
                            <span className={`font-mono px-1.5 py-0.5 rounded ${log.scores[log.actionTaken] > 0 ? 'bg-green-100 text-green-700' : 'bg-red-50 text-red-600'}`}>
                              {log.scores[log.actionTaken] > 0 ? '+1' : '-1'}
                            </span>
                          </div>
                        ))}
                        {entLogs.length > 3 && (
                          <div className="text-xs text-center text-neutral-400 pt-1">
                            +{entLogs.length - 3} more
                          </div>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}
        </div>

        {/* System Explanation Section */}
        <div className="bg-white rounded-3xl border border-neutral-200 overflow-hidden p-4 md:p-8">
          <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
            <Sparkles className="w-5 h-5 text-blue-500" />
            How Our System Works
          </h2>
          <div className="prose prose-neutral max-w-none text-sm space-y-4">
            <p>
              Welcome! This system is a self-learning assistant that adapts to your preferences through a specific workflow and mathematical memory.
            </p>

            <h3 className="text-lg font-medium mt-6 mb-2">The Interaction Flow</h3>
            <ol className="list-decimal pl-5 space-y-1">
              <li><strong>Initial Input:</strong> You provide a prompt or upload an image.</li>
              <li><strong>Classification & Suggestions:</strong> The AI analyzes your input, classifies it (assigning a Category and Entities), and offers suggested actions.</li>
              <li><strong>Action Taken (Second Input):</strong> You either select one of the AI's suggestions or type a custom action/message.</li>
              <li><strong>Output:</strong> The AI generates the final result based on the action taken.</li>
              <li><strong>Scoring:</strong> You score the action (+1 or -1) to teach the AI if that was the right approach.</li>
            </ol>
            
            <h3 className="text-lg font-medium mt-6 mb-2">How It Learns: Taxonomy, Embeddings, & Scoring</h3>
            <p>
              When you score an action, that score is saved to a <strong>Trace Log</strong> record. This record associates the score with <em>both</em> the Category and the Entities, but more importantly, it associates it with an <strong>Embedding</strong>.
            </p>
            <ul className="list-disc pl-5 space-y-2">
              <li>
                <strong>Taxonomy (Categories & Entities):</strong> The Category is the broad bucket (e.g., "Communication"), and Entities are specific tags (e.g., "Email", "Vendor"). These help you organize and filter your history.
              </li>
              <li>
                <strong>Embeddings (The "Brain"):</strong> Behind the scenes, the AI converts your initial input into a mathematical vector called an embedding. This allows the system to understand the <em>semantic meaning</em> of your request, not just the exact words.
              </li>
              <li>
                <strong>Scoring (The "Memory"):</strong> When you make a new request, the system calculates its embedding and finds past logs with similar embeddings. It looks at the actions you scored positively (+1) in those past similar logs and uses them to generate better, personalized suggestions for your new request.
              </li>
            </ul>

            <h3 className="text-lg font-medium mt-6 mb-2">3 Concrete Examples</h3>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-4">
              <div className="bg-neutral-50 p-4 rounded-xl border border-neutral-100">
                <h4 className="font-semibold mb-2">Example 1: Drafting an Email</h4>
                <p className="mb-1"><strong>1. Input:</strong> "I need to reply to this vendor."</p>
                <p className="mb-1"><strong>2. Action Taken:</strong> You select the suggestion: <em>"Write a polite decline."</em></p>
                <p className="mb-1"><strong>3. Output:</strong> "Dear Vendor, thank you but..."</p>
                <p className="mb-1"><strong>4. Score:</strong> +1</p>
                <p className="text-neutral-500 mt-2 text-xs"><strong>Learning:</strong> The system's embedding matches future vendor emails. It learns to suggest "polite decline" for this category/entity combination.</p>
              </div>
              <div className="bg-neutral-50 p-4 rounded-xl border border-neutral-100">
                <h4 className="font-semibold mb-2">Example 2: Code Help</h4>
                <p className="mb-1"><strong>1. Input:</strong> "Fix this React useEffect bug."</p>
                <p className="mb-1"><strong>2. Action Taken:</strong> You type a custom action: <em>"Explain the dependency array."</em></p>
                <p className="mb-1"><strong>3. Output:</strong> [Detailed explanation]</p>
                <p className="mb-1"><strong>4. Score:</strong> +1</p>
                <p className="text-neutral-500 mt-2 text-xs"><strong>Learning:</strong> The custom action is saved. For future React bug embeddings, it will suggest explaining dependencies rather than just writing code.</p>
              </div>
              <div className="bg-neutral-50 p-4 rounded-xl border border-neutral-100">
                <h4 className="font-semibold mb-2">Example 3: Image Analysis</h4>
                <p className="mb-1"><strong>1. Input:</strong> [Uploads picture of a dying plant]</p>
                <p className="mb-1"><strong>2. Action Taken:</strong> You select: <em>"Diagnose watering issues."</em></p>
                <p className="mb-1"><strong>3. Output:</strong> "It looks overwatered."</p>
                <p className="mb-1"><strong>4. Score:</strong> -1</p>
                <p className="text-neutral-500 mt-2 text-xs"><strong>Learning:</strong> The AI learns that for this specific visual embedding and botany category, jumping to watering issues is a bad suggestion.</p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
