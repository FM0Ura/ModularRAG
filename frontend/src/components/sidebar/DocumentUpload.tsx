import React, { useState, useRef } from 'react';
import { Upload, Loader2, AlertCircle } from 'lucide-react';
import { DocumentItem } from '../../types';
import { api } from '../../services/api';

interface DocumentUploadProps {
  onUploadSuccess: (newDoc: DocumentItem) => void;
  disabled?: boolean;
}

const ALLOWED_EXTENSIONS = ['.pdf', '.docx', '.txt', '.md'];

export const DocumentUpload: React.FC<DocumentUploadProps> = ({ onUploadSuccess, disabled }) => {
  const [isDragging, setIsDragging] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const isValidExtension = (fileName: string): boolean => {
    const lower = fileName.toLowerCase();
    return ALLOWED_EXTENSIONS.some((ext) => lower.endsWith(ext));
  };

  const handleUploadFile = async (file: File) => {
    if (!isValidExtension(file.name)) {
      setErrorMessage(`Formato não suportado. Use apenas: ${ALLOWED_EXTENSIONS.join(', ')}`);
      return;
    }

    if (file.size === 0) {
      setErrorMessage('O arquivo selecionado está vazio (0 bytes).');
      return;
    }

    setErrorMessage(null);
    setIsUploading(true);

    try {
      const doc = await api.uploadDocument(file);
      onUploadSuccess(doc);
    } catch (err: any) {
      setErrorMessage(err.message || 'Falha ao processar upload do arquivo.');
    } finally {
      setIsUploading(false);
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
    }
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    if (disabled || isUploading) return;
    setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = async (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    if (disabled || isUploading) return;

    const files = e.dataTransfer.files;
    if (files && files.length > 0) {
      await handleUploadFile(files[0]);
    }
  };

  const handleInputChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      await handleUploadFile(file);
    }
  };

  return (
    <div className="w-full">
      <div
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={() => !isUploading && !disabled && fileInputRef.current?.click()}
        className={`border-2 border-dashed rounded-xl p-3 text-center cursor-pointer transition-all duration-200 flex flex-col items-center justify-center space-y-2 ${
          isDragging
            ? 'border-indigo-500 bg-indigo-50/70 dark:bg-indigo-950/40 ring-2 ring-indigo-400'
            : 'border-gray-300 dark:border-gray-700 bg-gray-50/50 dark:bg-gray-800/40 hover:bg-gray-100/70 dark:hover:bg-gray-800/70'
        } ${isUploading || disabled ? 'opacity-60 cursor-not-allowed' : ''}`}
      >
        <input
          ref={fileInputRef}
          type="file"
          className="hidden"
          accept=".pdf,.docx,.txt,.md"
          disabled={isUploading || disabled}
          onChange={handleInputChange}
        />

        {isUploading ? (
          <div className="flex flex-col items-center py-2 space-y-1">
            <Loader2 className="w-6 h-6 text-indigo-600 animate-spin" />
            <span className="text-xs font-medium text-gray-600 dark:text-gray-300">
              Indexando documento...
            </span>
          </div>
        ) : (
          <>
            <div className="p-2 rounded-lg bg-indigo-50 dark:bg-indigo-950/60 text-indigo-600 dark:text-indigo-400">
              <Upload className="w-4 h-4" />
            </div>
            <div>
              <p className="text-xs font-semibold text-gray-800 dark:text-gray-200">
                Arraste um documento ou clique aqui
              </p>
              <p className="text-[11px] text-gray-500 dark:text-gray-400 mt-0.5">
                PDF, DOCX, TXT ou MD
              </p>
            </div>
          </>
        )}
      </div>

      {errorMessage && (
        <div className="mt-2 p-2 rounded-lg bg-red-50 dark:bg-red-950/40 border border-red-200 dark:border-red-900/50 flex items-start space-x-1.5 text-xs text-red-600 dark:text-red-400">
          <AlertCircle className="w-4 h-4 shrink-0 mt-0.5" />
          <span className="leading-snug">{errorMessage}</span>
        </div>
      )}
    </div>
  );
};
