import React, { useState, useEffect } from 'react';
import { Settings, Check, Save, Loader, ChevronDown, ChevronUp, Zap, Database } from 'lucide-react';
import { Card } from '../ui/Card';
import { Input } from '../ui/Input';
import { Select } from '../ui/Select';
import { Button } from '../ui/Button';
import { useToast } from '../../contexts/ToastContext';
import { credentialsService } from '../../services/credentialsService';
// Helper function to get chat model options for each provider
function getChatModelOptions(provider: string): Array<{value: string, label: string}> {
  switch (provider) {
    case 'openai':
      return [
        // Tier 1 models
        { value: 'gpt-5-2025-08-07', label: 'GPT-5 (Tier 1)' },
        { value: 'gpt-5-chat-latest', label: 'GPT-5 Chat Latest (Tier 1)' },
        { value: 'gpt-4.1-2025-04-14', label: 'GPT-4.1 (Tier 1)' },
        { value: 'gpt-4o-2024-11-20', label: 'GPT-4o (Tier 1)' },
        // Tier 2 models
        { value: 'gpt-5-mini-2025-08-07', label: 'GPT-5 Mini (Tier 2)' },
        { value: 'gpt-5-nano-2025-08-07', label: 'GPT-5 Nano (Tier 2)' },
        { value: 'gpt-4.1-mini-2025-04-14', label: 'GPT-4.1 Mini (Tier 2)' },
        { value: 'gpt-4.1-nano-2025-04-14', label: 'GPT-4.1 Nano (Tier 2)' },
      ];
    case 'google':
      return [
        { value: 'gemini-1.5-pro', label: 'Gemini 1.5 Pro' },
        { value: 'gemini-1.5-flash', label: 'Gemini 1.5 Flash' },
        { value: 'gemini-pro', label: 'Gemini Pro' },
        { value: 'gemini-pro-vision', label: 'Gemini Pro Vision' },
      ];
    case 'localcloudcode':
      return [
        { value: 'sonnet', label: 'Sonnet' },
        { value: 'haiku', label: 'Haiku' },
        { value: 'opus', label: 'Opus' },
      ];
    case 'ollama':
      return [
        // No predefined models - user should enter model name manually
      ];
    default:
      return [
        { value: 'gpt-4o-mini', label: 'GPT-4o Mini' },
      ];
  }
}

interface RAGSettingsProps {
  ragSettings: {
    MODEL_CHOICE: string;
    USE_CONTEXTUAL_EMBEDDINGS: boolean;
    CONTEXTUAL_EMBEDDINGS_MAX_WORKERS: number;
    USE_HYBRID_SEARCH: boolean;
    USE_AGENTIC_RAG: boolean;
    USE_RERANKING: boolean;
    LLM_PROVIDER?: string;
    LLM_BASE_URL?: string;
    EMBEDDING_PROVIDER?: string;
    EMBEDDING_BASE_URL?: string;
    // Qwen3 Reranking Settings
    QWEN3_TORCH_DTYPE?: string;
    QWEN3_DEVICE?: string;
    QWEN3_MAX_LENGTH?: number;
    QWEN3_USE_FLASH_ATTENTION?: boolean;
    QWEN3_INSTRUCTION?: string;
    EMBEDDING_MODEL?: string;
    // Crawling Performance Settings
    CRAWL_BATCH_SIZE?: number;
    CRAWL_MAX_CONCURRENT?: number;
    CRAWL_WAIT_STRATEGY?: string;
    CRAWL_PAGE_TIMEOUT?: number;
    CRAWL_DELAY_BEFORE_HTML?: number;
    // Storage Performance Settings
    DOCUMENT_STORAGE_BATCH_SIZE?: number;
    EMBEDDING_BATCH_SIZE?: number;
    DELETE_BATCH_SIZE?: number;
    ENABLE_PARALLEL_BATCHES?: boolean;
    // Advanced Settings
    MEMORY_THRESHOLD_PERCENT?: number;
    DISPATCHER_CHECK_INTERVAL?: number;
    CODE_EXTRACTION_BATCH_SIZE?: number;
    CODE_SUMMARY_MAX_WORKERS?: number;
  };
  setRagSettings: (settings: any) => void;
}

export const RAGSettings = ({
  ragSettings,
  setRagSettings
}: RAGSettingsProps) => {
  const [saving, setSaving] = useState(false);
  const [showCrawlingSettings, setShowCrawlingSettings] = useState(false);
  const [showStorageSettings, setShowStorageSettings] = useState(false);
  const [showQwen3Settings, setShowQwen3Settings] = useState(false);
  const [ollamaModels, setOllamaModels] = useState<Array<{name: string, size?: number}>>([]);
  const [loadingOllamaModels, setLoadingOllamaModels] = useState(false);
  const { showToast } = useToast();
  // Function to fetch available Ollama models
  const fetchOllamaModels = async () => {
    try {
      setLoadingOllamaModels(true);
      const baseUrl = ragSettings.EMBEDDING_BASE_URL || 'http://localhost:11434';
      const response = await fetch(`${baseUrl}/api/tags`);
      if (response.ok) {
        const data = await response.json();
        setOllamaModels(data.models || []);
      } else {
        console.warn('Failed to fetch Ollama models');
        setOllamaModels([]);
      }
    } catch (error) {
      console.warn('Error fetching Ollama models:', error);
      setOllamaModels([]);
    } finally {
      setLoadingOllamaModels(false);
    }
  };

  // Fetch Ollama models when embedding provider is Ollama
  useEffect(() => {
    const embeddingProvider = ragSettings.EMBEDDING_PROVIDER || ragSettings.LLM_PROVIDER;
    if (embeddingProvider === 'ollama') {
      fetchOllamaModels();
    }
  }, [ragSettings.EMBEDDING_PROVIDER, ragSettings.LLM_PROVIDER, ragSettings.EMBEDDING_BASE_URL]);

  // Handle save functionality
  const handleSave = async () => {
    try {
      setSaving(true);
      await credentialsService.updateRagSettings(ragSettings);
      showToast('Settings saved successfully!', 'success');
    } catch (error) {
      showToast('Failed to save settings', 'error');
      console.error('Error saving settings:', error);
    } finally {
      setSaving(false);
    }
  };
  return <Card accentColor="green" className="overflow-hidden p-8">
        {/* Description */}
        <p className="text-sm text-gray-600 dark:text-zinc-400 mb-6">
          Configure Retrieval-Augmented Generation (RAG) strategies for optimal
          knowledge retrieval.
        </p>
        
        {/* Chat Provider Section */}
        <div className="mb-6 p-4 rounded-lg border border-green-500/20 bg-gradient-to-r from-green-500/5 to-green-600/5">
          <h3 className="font-semibold text-gray-800 dark:text-white mb-3">Chat Provider Settings</h3>
          <div className="grid grid-cols-3 gap-4">
            <div>
              <Select
                label="Chat Provider"
                value={ragSettings.LLM_PROVIDER || 'openai'}
                onChange={e => handleChatProviderChange(e.target.value, ragSettings, setRagSettings)}
                accentColor="green"
                options={[
                  { value: 'openai', label: 'OpenAI' },
                  { value: 'google', label: 'Google Gemini' },
                  { value: 'ollama', label: 'Ollama' },
                  { value: 'localcloudcode', label: 'Claude Code' },
                ]}
              />
            </div>
            <div>
              {ragSettings.LLM_PROVIDER === 'ollama' ? (
                <Input
                  label="Chat Model"
                  placeholder="Enter model name (e.g., llama3.2:3b)"
                  value={ragSettings.MODEL_CHOICE || ''}
                  onChange={e => setRagSettings({
                    ...ragSettings,
                    MODEL_CHOICE: e.target.value
                  })}
                  accentColor="green"
                />
              ) : (
                <Select
                  label="Chat Model" 
                  value={ragSettings.MODEL_CHOICE || getDefaultChatModel(ragSettings.LLM_PROVIDER || 'openai')} 
                  onChange={e => setRagSettings({
                    ...ragSettings,
                    MODEL_CHOICE: e.target.value
                  })} 
                  accentColor="green"
                  options={getChatModelOptions(ragSettings.LLM_PROVIDER || 'openai')}
                />
              )}
            </div>
            <div>
              <Input
                label="Chat Base URL"
                value={ragSettings.LLM_BASE_URL || getDefaultChatBaseUrl(ragSettings.LLM_PROVIDER || 'openai')}
                onChange={e => setRagSettings({
                  ...ragSettings,
                  LLM_BASE_URL: e.target.value
                })}
                placeholder={getDefaultChatBaseUrl(ragSettings.LLM_PROVIDER || 'openai')}
                accentColor="green"
              />
            </div>
          </div>
        </div>

        {/* Embedding Provider Section */}
        <div className="mb-6 p-4 rounded-lg border border-blue-500/20 bg-gradient-to-r from-blue-500/5 to-blue-600/5">
          <h3 className="font-semibold text-gray-800 dark:text-white mb-3">Embedding Provider Settings</h3>
          <div className="grid grid-cols-3 gap-4">
            <div>
              <Select
                label="Embedding Provider"
                value={ragSettings.EMBEDDING_PROVIDER || ragSettings.LLM_PROVIDER || 'openai'}
                onChange={e => handleEmbeddingProviderChange(e.target.value, ragSettings, setRagSettings)}
                accentColor="blue"
                options={[
                  { value: 'openai', label: 'OpenAI' },
                  { value: 'google', label: 'Google Gemini' },
                  { value: 'ollama', label: 'Ollama' },
                ]}
              />
            </div>
            <div>
              {(ragSettings.EMBEDDING_PROVIDER || ragSettings.LLM_PROVIDER) === 'ollama' ? (
                <Select
                  label="Embedding Model"
                  value={ragSettings.EMBEDDING_MODEL || ''}
                  onChange={e => setRagSettings({
                    ...ragSettings,
                    EMBEDDING_MODEL: e.target.value
                  })}
                  accentColor="blue"
                  options={[
                    ...(ollamaModels.map(model => ({
                      value: model.name,
                      label: model.name + (model.size ? ` (${Math.round(model.size / 1024 / 1024 / 1024 * 10) / 10}GB)` : '')
                    }))),
                    // Default options if no models found
                    ...(ollamaModels.length === 0 ? [
                      { value: 'qwen3-embedding-4b:q5_k_m', label: 'qwen3-embedding-4b:q5_k_m (Recommended)' },
                      { value: 'nomic-embed-text', label: 'nomic-embed-text' },
                      { value: 'mxbai-embed-large', label: 'mxbai-embed-large' }
                    ] : [])
                  ]}
                  disabled={loadingOllamaModels}
                  placeholder={loadingOllamaModels ? "Loading models..." : "Select embedding model"}
                />
              ) : (
                <Input
                  label="Embedding Model"
                  value={ragSettings.EMBEDDING_MODEL || ''}
                  onChange={e => setRagSettings({
                    ...ragSettings,
                    EMBEDDING_MODEL: e.target.value
                  })}
                  placeholder={getEmbeddingPlaceholder(ragSettings.EMBEDDING_PROVIDER || ragSettings.LLM_PROVIDER || 'openai')}
                  accentColor="blue"
                />
              )}
            </div>
            {(ragSettings.EMBEDDING_PROVIDER || ragSettings.LLM_PROVIDER) === 'ollama' && (
              <div>
                <Input
                  label="Embedding Base URL"
                  value={ragSettings.EMBEDDING_BASE_URL || 'http://localhost:11434/v1'}
                  onChange={e => setRagSettings({
                    ...ragSettings,
                    EMBEDDING_BASE_URL: e.target.value
                  })}
                  placeholder="http://localhost:11434/v1"
                  accentColor="blue"
                />
              </div>
            )}
          </div>
        </div>
        {/* Second row: Contextual Embeddings, Max Workers, and description */}
        <div className="grid grid-cols-8 gap-4 mb-4 p-4 rounded-lg border border-green-500/20 shadow-[0_2px_8px_rgba(34,197,94,0.1)]">
          <div className="col-span-4">
            <CustomCheckbox 
              id="contextualEmbeddings" 
              checked={ragSettings.USE_CONTEXTUAL_EMBEDDINGS} 
              onChange={e => setRagSettings({
                ...ragSettings,
                USE_CONTEXTUAL_EMBEDDINGS: e.target.checked
              })} 
              label="Use Contextual Embeddings" 
              description="Enhances embeddings with contextual information for better retrieval" 
            />
          </div>
                      <div className="col-span-1">
              {ragSettings.USE_CONTEXTUAL_EMBEDDINGS && (
                <div className="flex flex-col items-center">
                  <div className="relative ml-2 mr-6">
                    <input
                      type="number"
                      min="1"
                      max="10"
                      value={ragSettings.CONTEXTUAL_EMBEDDINGS_MAX_WORKERS}
                      onChange={e => setRagSettings({
                        ...ragSettings,
                        CONTEXTUAL_EMBEDDINGS_MAX_WORKERS: parseInt(e.target.value, 10) || 3
                      })}
                      className="w-14 h-10 pl-1 pr-7 text-center font-medium rounded-md 
                        bg-gradient-to-b from-gray-100 to-gray-200 dark:from-gray-900 dark:to-black 
                        border border-green-500/30 
                        text-gray-900 dark:text-white
                        focus:border-green-500 focus:shadow-[0_0_15px_rgba(34,197,94,0.4)]
                        transition-all duration-200
                        [appearance:textfield] 
                        [&::-webkit-outer-spin-button]:appearance-none 
                        [&::-webkit-inner-spin-button]:appearance-none"
                    />
                    <div className="absolute right-1 top-1 bottom-1 flex flex-col">
                      <button
                        type="button"
                        onClick={() => setRagSettings({
                          ...ragSettings,
                          CONTEXTUAL_EMBEDDINGS_MAX_WORKERS: Math.min(ragSettings.CONTEXTUAL_EMBEDDINGS_MAX_WORKERS + 1, 10)
                        })}
                        className="flex-1 px-1 rounded-t-sm 
                          bg-gradient-to-b from-green-500/20 to-green-600/10
                          hover:from-green-500/30 hover:to-green-600/20
                          border border-green-500/30 border-b-0
                          transition-all duration-200 group"
                      >
                        <svg className="w-2.5 h-2.5 text-green-500 group-hover:filter group-hover:drop-shadow-[0_0_4px_rgba(34,197,94,0.8)]" 
                          viewBox="0 0 10 6" fill="none" stroke="currentColor" strokeWidth="2">
                          <path d="M1 5L5 1L9 5" />
                        </svg>
                      </button>
                      <button
                        type="button"
                        onClick={() => setRagSettings({
                          ...ragSettings,
                          CONTEXTUAL_EMBEDDINGS_MAX_WORKERS: Math.max(ragSettings.CONTEXTUAL_EMBEDDINGS_MAX_WORKERS - 1, 1)
                        })}
                        className="flex-1 px-1 rounded-b-sm 
                          bg-gradient-to-b from-green-500/20 to-green-600/10
                          hover:from-green-500/30 hover:to-green-600/20
                          border border-green-500/30 border-t-0
                          transition-all duration-200 group"
                      >
                        <svg className="w-2.5 h-2.5 text-green-500 group-hover:filter group-hover:drop-shadow-[0_0_4px_rgba(34,197,94,0.8)]" 
                          viewBox="0 0 10 6" fill="none" stroke="currentColor" strokeWidth="2">
                          <path d="M1 1L5 5L9 1" />
                        </svg>
                      </button>
                    </div>
                  </div>
                  <label className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                    Max
                  </label>
                </div>
              )}
            </div>
          <div className="col-span-3">
            {ragSettings.USE_CONTEXTUAL_EMBEDDINGS && (
              <p className="text-xs text-green-900 dark:text-blue-600 mt-2">
                Controls parallel processing for embeddings (1-10)
              </p>
            )}
          </div>
        </div>
        
        {/* Third row: Hybrid Search and Agentic RAG */}
        <div className="grid grid-cols-2 gap-4 mb-4">
          <div>
            <CustomCheckbox 
              id="hybridSearch" 
              checked={ragSettings.USE_HYBRID_SEARCH} 
              onChange={e => setRagSettings({
                ...ragSettings,
                USE_HYBRID_SEARCH: e.target.checked
              })} 
              label="Use Hybrid Search" 
              description="Combines vector similarity search with keyword search for better results" 
            />
          </div>
          <div>
            <CustomCheckbox 
              id="agenticRag" 
              checked={ragSettings.USE_AGENTIC_RAG} 
              onChange={e => setRagSettings({
                ...ragSettings,
                USE_AGENTIC_RAG: e.target.checked
              })} 
              label="Use Agentic RAG" 
              description="Enables code extraction and specialized search for technical content" 
            />
          </div>
        </div>
        
        {/* Fourth row: Use Reranking */}
        <div className="grid grid-cols-2 gap-4">
          <div>
            <CustomCheckbox 
              id="reranking" 
              checked={ragSettings.USE_RERANKING} 
              onChange={e => setRagSettings({
                ...ragSettings,
                USE_RERANKING: e.target.checked
              })} 
              label="Use Reranking" 
              description="Uses Qwen3-Reranker-4B for superior search result relevance scoring"
            />
          </div>
          <div>{/* Empty column */}</div>
        </div>
        {/* Qwen3 Reranking Settings - Only show when reranking is enabled */}
        {ragSettings.USE_RERANKING && (
          <div className="mt-4">
            <div
              className="flex items-center justify-between cursor-pointer p-3 rounded-lg border border-purple-500/20 bg-gradient-to-r from-purple-500/5 to-purple-600/5 hover:from-purple-500/10 hover:to-purple-600/10 transition-all duration-200"
              onClick={() => setShowQwen3Settings(!showQwen3Settings)}
            >
              <div className="flex items-center">
                <Settings className="mr-2 text-purple-500 filter drop-shadow-[0_0_8px_rgba(147,51,234,0.6)]" size={18} />
                <h3 className="font-semibold text-gray-800 dark:text-white">Qwen3 Reranker Optimization</h3>
              </div>
              {showQwen3Settings ? (
                <ChevronUp className="text-gray-500 dark:text-gray-400" size={20} />
              ) : (
                <ChevronDown className="text-gray-500 dark:text-gray-400" size={20} />
              )}
            </div>
            
            {showQwen3Settings && (
              <div className="mt-4 p-4 border border-purple-500/10 rounded-lg bg-purple-500/5">
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                      Model Precision
                    </label>
                    <select
                      value={ragSettings.QWEN3_TORCH_DTYPE || 'auto'}
                      onChange={e => setRagSettings({
                        ...ragSettings,
                        QWEN3_TORCH_DTYPE: e.target.value === 'auto' ? undefined : e.target.value
                      })}
                      className="w-full px-3 py-2 border border-purple-500/30 rounded-md bg-gray-50 dark:bg-gray-900 text-gray-900 dark:text-white focus:border-purple-500 focus:ring-1 focus:ring-purple-500"
                    >
                      <option value="auto">Auto (Recommended)</option>
                      <option value="float16">Half Precision (float16) - Memory Efficient</option>
                      <option value="bfloat16">Brain Float (bfloat16) - Balanced</option>
                      <option value="float32">Full Precision (float32) - CPU/Stability</option>
                    </select>
                    <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">Lower precision uses less memory but may reduce accuracy</p>
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                      Device Selection
                    </label>
                    <select
                      value={ragSettings.QWEN3_DEVICE || 'auto'}
                      onChange={e => setRagSettings({
                        ...ragSettings,
                        QWEN3_DEVICE: e.target.value === 'auto' ? undefined : e.target.value
                      })}
                      className="w-full px-3 py-2 border border-purple-500/30 rounded-md bg-gray-50 dark:bg-gray-900 text-gray-900 dark:text-white focus:border-purple-500 focus:ring-1 focus:ring-purple-500"
                    >
                      <option value="auto">Auto Detect</option>
                      <option value="cuda">CUDA GPU (Fastest)</option>
                      <option value="cpu">CPU Only (Slower)</option>
                    </select>
                    <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">GPU provides significant speed improvement</p>
                  </div>
                </div>
                
                <div className="grid grid-cols-2 gap-4 mt-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                      Max Context Length
                    </label>
                    <input
                      type="number"
                      min="1024"
                      max="16384"
                      step="1024"
                      value={ragSettings.QWEN3_MAX_LENGTH || 8192}
                      onChange={e => setRagSettings({
                        ...ragSettings,
                        QWEN3_MAX_LENGTH: parseInt(e.target.value, 10) || 8192
                      })}
                      className="w-full px-3 py-2 border border-purple-500/30 rounded-md bg-gray-50 dark:bg-gray-900 text-gray-900 dark:text-white focus:border-purple-500 focus:ring-1 focus:ring-purple-500"
                    />
                    <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">Maximum tokens for query+document pairs</p>
                  </div>
                  <div className="flex items-center mt-6">
                    <CustomCheckbox
                      id="qwen3FlashAttention"
                      checked={ragSettings.QWEN3_USE_FLASH_ATTENTION !== false}
                      onChange={e => setRagSettings({
                        ...ragSettings,
                        QWEN3_USE_FLASH_ATTENTION: e.target.checked
                      })}
                      label="Flash Attention"
                      description="Accelerates inference on compatible GPUs (RTX 3080+, A100, H100)"
                    />
                  </div>
                </div>
                
                <div className="mt-4">
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                    Custom Instruction (Optional)
                  </label>
                  <textarea
                    value={ragSettings.QWEN3_INSTRUCTION || ''}
                    onChange={e => setRagSettings({
                      ...ragSettings,
                      QWEN3_INSTRUCTION: e.target.value || undefined
                    })}
                    placeholder="Given a web search query, retrieve relevant passages that answer the query"
                    rows={2}
                    className="w-full px-3 py-2 border border-purple-500/30 rounded-md bg-gray-50 dark:bg-gray-900 text-gray-900 dark:text-white focus:border-purple-500 focus:ring-1 focus:ring-purple-500"
                  />
                  <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">Leave empty to use default instruction for web search relevance</p>
                </div>
              </div>
            )}
          </div>
        )}

        {/* Crawling Performance Settings */}
        <div className="mt-6">
          <div
            className="flex items-center justify-between cursor-pointer p-3 rounded-lg border border-green-500/20 bg-gradient-to-r from-green-500/5 to-green-600/5 hover:from-green-500/10 hover:to-green-600/10 transition-all duration-200"
            onClick={() => setShowCrawlingSettings(!showCrawlingSettings)}
          >
            <div className="flex items-center">
              <Zap className="mr-2 text-green-500 filter drop-shadow-[0_0_8px_rgba(34,197,94,0.6)]" size={18} />
              <h3 className="font-semibold text-gray-800 dark:text-white">Crawling Performance Settings</h3>
            </div>
            {showCrawlingSettings ? (
              <ChevronUp className="text-gray-500 dark:text-gray-400" size={20} />
            ) : (
              <ChevronDown className="text-gray-500 dark:text-gray-400" size={20} />
            )}
          </div>
          
          {showCrawlingSettings && (
            <div className="mt-4 p-4 border border-green-500/10 rounded-lg bg-green-500/5">
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                    Batch Size
                  </label>
                  <input
                    type="number"
                    min="10"
                    max="100"
                    value={ragSettings.CRAWL_BATCH_SIZE || 50}
                    onChange={e => setRagSettings({
                      ...ragSettings,
                      CRAWL_BATCH_SIZE: parseInt(e.target.value, 10) || 50
                    })}
                    className="w-full px-3 py-2 border border-green-500/30 rounded-md bg-gray-50 dark:bg-gray-900 text-gray-900 dark:text-white focus:border-green-500 focus:ring-1 focus:ring-green-500"
                  />
                  <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">URLs to crawl in parallel (10-100)</p>
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                    Max Concurrent
                  </label>
                  <input
                    type="number"
                    min="1"
                    max="20"
                    value={ragSettings.CRAWL_MAX_CONCURRENT || 10}
                    onChange={e => setRagSettings({
                      ...ragSettings,
                      CRAWL_MAX_CONCURRENT: parseInt(e.target.value, 10) || 10
                    })}
                    className="w-full px-3 py-2 border border-green-500/30 rounded-md bg-gray-50 dark:bg-gray-900 text-gray-900 dark:text-white focus:border-green-500 focus:ring-1 focus:ring-green-500"
                  />
                  <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">Browser sessions (1-20)</p>
                </div>
              </div>
              
              <div className="grid grid-cols-3 gap-4 mt-4">
                <div>
                  <Select
                    label="Wait Strategy"
                    value={ragSettings.CRAWL_WAIT_STRATEGY || 'domcontentloaded'}
                    onChange={e => setRagSettings({
                      ...ragSettings,
                      CRAWL_WAIT_STRATEGY: e.target.value
                    })}
                    accentColor="green"
                    options={[
                      { value: 'domcontentloaded', label: 'DOM Loaded (Fast)' },
                      { value: 'networkidle', label: 'Network Idle (Thorough)' },
                      { value: 'load', label: 'Full Load (Slowest)' }
                    ]}
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                    Page Timeout (sec)
                  </label>
                  <input
                    type="number"
                    min="5"
                    max="120"
                    value={(ragSettings.CRAWL_PAGE_TIMEOUT || 60000) / 1000}
                    onChange={e => setRagSettings({
                      ...ragSettings,
                      CRAWL_PAGE_TIMEOUT: (parseInt(e.target.value, 10) || 60) * 1000
                    })}
                    className="w-full px-3 py-2 border border-green-500/30 rounded-md bg-gray-50 dark:bg-gray-900 text-gray-900 dark:text-white focus:border-green-500 focus:ring-1 focus:ring-green-500"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                    Render Delay (sec)
                  </label>
                  <input
                    type="number"
                    min="0.1"
                    max="5"
                    step="0.1"
                    value={ragSettings.CRAWL_DELAY_BEFORE_HTML || 0.5}
                    onChange={e => setRagSettings({
                      ...ragSettings,
                      CRAWL_DELAY_BEFORE_HTML: parseFloat(e.target.value) || 0.5
                    })}
                    className="w-full px-3 py-2 border border-green-500/30 rounded-md bg-gray-50 dark:bg-gray-900 text-gray-900 dark:text-white focus:border-green-500 focus:ring-1 focus:ring-green-500"
                  />
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Storage Performance Settings */}
        <div className="mt-4">
          <div
            className="flex items-center justify-between cursor-pointer p-3 rounded-lg border border-green-500/20 bg-gradient-to-r from-green-500/5 to-green-600/5 hover:from-green-500/10 hover:to-green-600/10 transition-all duration-200"
            onClick={() => setShowStorageSettings(!showStorageSettings)}
          >
            <div className="flex items-center">
              <Database className="mr-2 text-green-500 filter drop-shadow-[0_0_8px_rgba(34,197,94,0.6)]" size={18} />
              <h3 className="font-semibold text-gray-800 dark:text-white">Storage Performance Settings</h3>
            </div>
            {showStorageSettings ? (
              <ChevronUp className="text-gray-500 dark:text-gray-400" size={20} />
            ) : (
              <ChevronDown className="text-gray-500 dark:text-gray-400" size={20} />
            )}
          </div>
          
          {showStorageSettings && (
            <div className="mt-4 p-4 border border-green-500/10 rounded-lg bg-green-500/5">
              <div className="grid grid-cols-3 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                    Document Batch Size
                  </label>
                  <input
                    type="number"
                    min="10"
                    max="100"
                    value={ragSettings.DOCUMENT_STORAGE_BATCH_SIZE || 50}
                    onChange={e => setRagSettings({
                      ...ragSettings,
                      DOCUMENT_STORAGE_BATCH_SIZE: parseInt(e.target.value, 10) || 50
                    })}
                    className="w-full px-3 py-2 border border-green-500/30 rounded-md bg-gray-50 dark:bg-gray-900 text-gray-900 dark:text-white focus:border-green-500 focus:ring-1 focus:ring-green-500"
                  />
                  <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">Chunks per batch (10-100)</p>
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                    Embedding Batch Size
                  </label>
                  <input
                    type="number"
                    min="20"
                    max="200"
                    value={ragSettings.EMBEDDING_BATCH_SIZE || 100}
                    onChange={e => setRagSettings({
                      ...ragSettings,
                      EMBEDDING_BATCH_SIZE: parseInt(e.target.value, 10) || 100
                    })}
                    className="w-full px-3 py-2 border border-green-500/30 rounded-md bg-gray-50 dark:bg-gray-900 text-gray-900 dark:text-white focus:border-green-500 focus:ring-1 focus:ring-green-500"
                  />
                  <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">Per API call (20-200)</p>
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                    Code Extraction Workers
                  </label>
                  <input
                    type="number"
                    min="1"
                    max="10"
                    value={ragSettings.CODE_SUMMARY_MAX_WORKERS || 3}
                    onChange={e => setRagSettings({
                      ...ragSettings,
                      CODE_SUMMARY_MAX_WORKERS: parseInt(e.target.value, 10) || 3
                    })}
                    className="w-full px-3 py-2 border border-green-500/30 rounded-md bg-gray-50 dark:bg-gray-900 text-gray-900 dark:text-white focus:border-green-500 focus:ring-1 focus:ring-green-500"
                  />
                  <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">Parallel workers (1-10)</p>
                </div>
              </div>
              
              <div className="mt-4 flex items-center">
                <CustomCheckbox
                  id="parallelBatches"
                  checked={ragSettings.ENABLE_PARALLEL_BATCHES !== false}
                  onChange={e => setRagSettings({
                    ...ragSettings,
                    ENABLE_PARALLEL_BATCHES: e.target.checked
                  })}
                  label="Enable Parallel Processing"
                  description="Process multiple document batches simultaneously for faster storage"
                />
              </div>
            </div>
          )}
        </div>

        {/* Save Button Section */}
        <div className="flex justify-end pt-4 border-t border-green-500/10 mt-6">
          <button
            onClick={handleSave}
            disabled={saving}
            className="px-6 py-2 bg-gradient-to-r from-green-500 to-green-600 text-white rounded-lg
              hover:from-green-600 hover:to-green-700 disabled:from-gray-400 disabled:to-gray-500
              disabled:cursor-not-allowed transition-all duration-200 font-medium
              shadow-lg shadow-green-500/25 hover:shadow-green-500/40
              disabled:shadow-none flex items-center"
          >
            {saving ? (
              <>
                <svg className="animate-spin -ml-1 mr-2 h-4 w-4 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
                Saving...
              </>
            ) : (
              <>
                <Save size={16} className="mr-2" />
                Save Settings
              </>
            )}
          </button>
        </div>
    </Card>
};

// Helper functions for model placeholders
function getModelPlaceholder(provider: string): string {
  switch (provider) {
    case 'openai':
      return 'e.g., gpt-4o-mini';
    case 'ollama':
      return 'e.g., llama2, mistral';
    case 'google':
      return 'e.g., gemini-1.5-flash';
    case 'localcloudcode':
      return 'e.g., sonnet, haiku, opus';
    default:
      return 'e.g., gpt-4o-mini';
  }
}
function getEmbeddingPlaceholder(provider: string): string {
  switch (provider) {
    case 'openai':
      return 'Default: text-embedding-3-small';
    case 'ollama':
      return 'e.g., qwen3-embedding-4b:q5_k_m, nomic-embed-text';
    case 'google':
      return 'e.g., text-embedding-004';
    default:
      return 'Default: text-embedding-3-small';
  }
}
// Helper functions for default model values
function getDefaultChatModel(provider: string): string {
  switch (provider) {
    case 'openai':
      return 'gpt-4o-mini';
    case 'ollama':
      return 'llama3.1:8b';
    case 'google':
      return 'gemini-1.5-flash';
    case 'localcloudcode':
      return 'sonnet';
    default:
      return 'gpt-4o-mini';
  }
}

function getDefaultEmbeddingModel(provider: string): string {
  switch (provider) {
    case 'openai':
      return 'text-embedding-3-small';
    case 'ollama':
      return 'qwen3-embedding-4b:q5_k_m';
    case 'google':
      return 'text-embedding-004';
    default:
      return 'text-embedding-3-small';
  }
}

// Handler for provider changes that updates model defaults
// Handler for chat provider changes
function handleChatProviderChange(provider: string, ragSettings: any, setRagSettings: any): void {
  setRagSettings({
    ...ragSettings,
    LLM_PROVIDER: provider,
    MODEL_CHOICE: getDefaultChatModel(provider),
    LLM_BASE_URL: getDefaultBaseUrl(provider)
  });
}

// Handler for embedding provider changes
function handleEmbeddingProviderChange(provider: string, ragSettings: any, setRagSettings: any): void {
  setRagSettings({
    ...ragSettings,
    EMBEDDING_PROVIDER: provider,
    EMBEDDING_MODEL: getDefaultEmbeddingModel(provider),
    EMBEDDING_BASE_URL: getDefaultBaseUrl(provider)
  });
}

// Helper to get default base URLs
function getDefaultBaseUrl(provider: string): string {
  switch (provider) {
    case 'ollama':
      return 'http://localhost:11434/v1';
    case 'localcloudcode':
      return 'http://localhost:11222';
    case 'openai':
    case 'google':
    default:
      return '';
  }
}

// Helper to get default chat base URLs  
function getDefaultChatBaseUrl(provider: string): string {
  switch (provider) {
    case 'ollama':
      return 'http://localhost:11434/v1';
    case 'localcloudcode':
      return 'http://localhost:11222';
    case 'openai':
    case 'google':
    default:
      return '';
  }
}
interface CustomCheckboxProps {
  id: string;
  checked: boolean;
  onChange: (e: React.ChangeEvent<HTMLInputElement>) => void;
  label: string;
  description: string;
}

const CustomCheckbox = ({
  id,
  checked,
  onChange,
  label,
  description
}: CustomCheckboxProps) => {
  return (
    <div className="flex items-start group">
      <div className="relative flex items-center h-5 mt-1">
        <input 
          type="checkbox" 
          id={id} 
          checked={checked} 
          onChange={onChange} 
          className="sr-only peer" 
        />
        <label 
          htmlFor={id}
          className="relative w-5 h-5 rounded-md transition-all duration-200 cursor-pointer
            bg-gradient-to-b from-white/80 to-white/60 dark:from-white/5 dark:to-black/40
            border border-gray-300 dark:border-gray-700
            peer-checked:border-green-500 dark:peer-checked:border-green-500/50
            peer-checked:bg-gradient-to-b peer-checked:from-green-500/20 peer-checked:to-green-600/20
            group-hover:border-green-500/50 dark:group-hover:border-green-500/30
            peer-checked:shadow-[0_0_10px_rgba(34,197,94,0.2)] dark:peer-checked:shadow-[0_0_15px_rgba(34,197,94,0.3)]"
        >
          <Check className={`
              w-3.5 h-3.5 absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2
              transition-all duration-200 text-green-500 pointer-events-none
              ${checked ? 'opacity-100 scale-100' : 'opacity-0 scale-50'}
            `} />
        </label>
      </div>
      <div className="ml-3 flex-1">
        <label htmlFor={id} className="text-gray-700 dark:text-zinc-300 font-medium cursor-pointer block text-sm">
          {label}
        </label>
        <p className="text-xs text-gray-600 dark:text-zinc-400 mt-0.5 leading-tight">
          {description}
        </p>
      </div>
    </div>
  );
};