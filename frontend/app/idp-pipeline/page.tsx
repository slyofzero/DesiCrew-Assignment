"use client";

import React, { useState, useEffect, useRef } from "react";
import Link from "next/link";
import {
  Upload,
  FileText,
  AlertCircle,
  CheckCircle2,
  Clock,
  Sparkles,
  RefreshCw,
  Download,
  Eye,
  Edit3,
  Check,
  ChevronDown,
  Info,
  ShieldCheck,
  ArrowLeft,
  FileCheck,
  AlertTriangle,
} from "lucide-react";

interface FieldResult {
  field_name: string;
  value: string | null;
  confidence: number;
  is_handwritten: boolean;
  is_flagged: boolean;
  flag_reason: string | null;
  human_verified: boolean;
}

interface DocumentProcessResponse {
  document_id: string;
  filename: string;
  preview_url: string | null;
  document_type: string;
  classification_confidence: number;
  fields: FieldResult[];
  overall_confidence: number;
  needs_review: boolean;
  flagged_count: number;
  processing_time_sec: number;
  raw_ocr_lines?: string[];
}

interface SampleDoc {
  doc_id: string;
  filename: string;
  size_bytes: number;
  file_type: string;
}

const DOCUMENT_TYPES = [
  "Aadhaar Card",
  "PAN Card",
  "Driving Licence",
  "Passport",
  "NACH / ECS Mandate",
  "FATCA Annexure Form",
  "Benefit Illustration Declaration",
  "Moral Hazard Questionnaire",
  "Multiple Policies Consent Form",
  "Suitability Profiler Declaration",
];

const API_BASE = "http://localhost:8000/api/idp";

export default function IDPPipelinePage() {
  const [samples, setSamples] = useState<SampleDoc[]>([]);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [activeFilename, setActiveFilename] = useState<string | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [processingStep, setProcessingStep] = useState("Analyzing document strokes...");
  const [docResult, setDocResult] = useState<DocumentProcessResponse | null>(null);
  const [selectedType, setSelectedType] = useState<string>("");
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [editingField, setEditingField] = useState<string | null>(null);
  const [editValue, setEditValue] = useState<string>("");
  const [showReportModal, setShowReportModal] = useState(false);
  const [reportData, setReportData] = useState<any>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);

  // Load sample documents on mount
  useEffect(() => {
    fetch(`${API_BASE}/samples`)
      .then((res) => (res.ok ? res.json() : []))
      .then((data) => setSamples(data))
      .catch((err) => console.error("Error loading samples:", err));

    fetch(`${API_BASE}/report`)
      .then((res) => (res.ok ? res.json() : null))
      .then((data) => setReportData(data))
      .catch((err) => console.error("Error loading report:", err));
  }, []);

  // Handle user file upload
  const handleFileUpload = async (file: File) => {
    setSelectedFile(file);
    setActiveFilename(file.name);
    setIsProcessing(true);
    setDocResult(null);
    setPreviewUrl(URL.createObjectURL(file));

    // Simulated progress steps for great UX
    setProcessingStep("Reading document & applying OCR engine...");
    setTimeout(() => setProcessingStep("Classifying document type & anchor keywords..."), 1200);
    setTimeout(() => setProcessingStep("Extracting target fields & validating formats..."), 2200);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch(`${API_BASE}/upload`, {
        method: "POST",
        body: formData,
      });

      if (!res.ok) throw new Error("Upload processing failed");
      const data: DocumentProcessResponse = await res.json();
      setDocResult(data);
      setSelectedType(data.document_type);
      setPreviewUrl(`${API_BASE}/preview/${data.document_id}`);
    } catch (err) {
      console.error(err);
      alert("Failed to process uploaded document. Please ensure backend is running.");
    } finally {
      setIsProcessing(false);
    }
  };

  // Handle selecting a pre-loaded sample document
  const handleSampleSelect = async (sample: SampleDoc) => {
    setSelectedFile(null);
    setActiveFilename(sample.filename);
    setIsProcessing(true);
    setDocResult(null);
    setPreviewUrl(`${API_BASE}/preview/${sample.doc_id}`);

    setProcessingStep("Reading document & applying OCR engine...");
    setTimeout(() => setProcessingStep("Classifying document type & anchor keywords..."), 1000);
    setTimeout(() => setProcessingStep("Extracting target fields & validating formats..."), 2000);

    try {
      const res = await fetch(`${API_BASE}/process-sample/${sample.filename}`, {
        method: "POST",
      });

      if (!res.ok) throw new Error("Sample processing failed");
      const data: DocumentProcessResponse = await res.json();
      setDocResult(data);
      setSelectedType(data.document_type);
      setPreviewUrl(`${API_BASE}/preview/${data.document_id}`);
    } catch (err) {
      console.error(err);
      alert("Failed to process sample document.");
    } finally {
      setIsProcessing(false);
    }
  };

  // Handle user changing the Document Type dropdown
  const handleTypeChange = async (newType: string) => {
    setSelectedType(newType);
    if (!docResult) return;

    try {
      const res = await fetch(`${API_BASE}/reclassify`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          document_id: docResult.document_id,
          chosen_type: newType,
        }),
      });

      if (res.ok) {
        const updated: DocumentProcessResponse = await res.json();
        setDocResult(updated);
      }
    } catch (err) {
      console.error("Failed to reclassify:", err);
    }
  };

  // Handle inline human reviewer edit
  const handleSaveFieldEdit = async (fieldName: string) => {
    if (!docResult) return;

    try {
      const res = await fetch(`${API_BASE}/triage/update`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          document_id: docResult.document_id,
          field_name: fieldName,
          updated_value: editValue,
          confirm_as_verified: true,
        }),
      });

      if (res.ok) {
        const updated: DocumentProcessResponse = await res.json();
        setDocResult(updated);
        setEditingField(null);
      }
    } catch (err) {
      console.error("Failed to save edit:", err);
    }
  };

  // Direct 1-click verify button for flagged fields
  const handleVerifyField = async (fieldName: string, currentValue: string) => {
    if (!docResult) return;
    try {
      const res = await fetch(`${API_BASE}/triage/update`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          document_id: docResult.document_id,
          field_name: fieldName,
          updated_value: currentValue,
          confirm_as_verified: true,
        }),
      });
      if (res.ok) {
        const updated: DocumentProcessResponse = await res.json();
        setDocResult(updated);
      }
    } catch (err) {
      console.error("Failed to verify field:", err);
    }
  };

  // Download structured extraction output as JSON
  const handleDownloadJSON = () => {
    if (!docResult) return;
    const jsonStr = JSON.stringify(docResult, null, 2);
    const blob = new Blob([jsonStr], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${docResult.document_id}_extracted_kyc.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const hasDocument = activeFilename !== null;

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 p-4 md:p-8 flex flex-col font-sans">
      {/* Top Header */}
      <header className="max-w-7xl w-full mx-auto mb-8 flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-slate-800 pb-5">
        <div>
          <div className="flex items-center gap-3 mb-1">
            <Link
              href="/"
              className="text-slate-400 hover:text-white flex items-center gap-1 text-sm transition-colors"
            >
              <ArrowLeft className="w-4 h-4" /> Home
            </Link>
            <span className="text-slate-600">/</span>
            <span className="text-xs font-semibold uppercase tracking-wider text-emerald-400 bg-emerald-950/70 px-2.5 py-0.5 rounded-full border border-emerald-800/60">
              Question 3 Pipeline
            </span>
          </div>
          <h1 className="text-2xl md:text-3xl font-extrabold tracking-tight bg-gradient-to-r from-white via-slate-200 to-slate-400 bg-clip-text text-transparent">
            Intelligent Document Processing (IDP)
          </h1>
          <p className="text-sm text-slate-400 mt-1 max-w-2xl">
            Automated classification, field extraction, handwriting analysis, and per-field confidence scoring for KYC and insurance onboarding forms.
          </p>
        </div>

        <div className="flex items-center gap-3">
          <button
            onClick={() => setShowReportModal(true)}
            className="flex items-center gap-2 bg-slate-900 hover:bg-slate-800 text-slate-300 text-sm font-medium px-4 py-2 rounded-lg border border-slate-700/80 transition shadow-sm"
          >
            <Info className="w-4 h-4 text-emerald-400" />
            Triage Methodology & Report
          </button>
        </div>
      </header>

      {/* Main Dynamic Container */}
      <main className="max-w-7xl w-full mx-auto flex-1 flex flex-col">
        {/* Dynamic Grid: Centered single card when no document; Side-by-side 2-column layout once uploaded */}
        <div
          className={`grid gap-6 transition-all duration-500 ease-in-out ${
            hasDocument
              ? "grid-cols-1 lg:grid-cols-12"
              : "grid-cols-1 max-w-3xl mx-auto w-full"
          }`}
        >
          {/* LEFT COLUMN: Upload Dialog & Document Viewer */}
          <div
            className={`transition-all duration-500 ${
              hasDocument ? "lg:col-span-5 flex flex-col gap-5" : "col-span-1"
            }`}
          >
            {/* Upload Box Card */}
            <div className="bg-slate-900/90 rounded-2xl border border-slate-800 p-6 shadow-xl backdrop-blur-sm">
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-2.5">
                  <div className="w-8 h-8 rounded-lg bg-emerald-500/10 border border-emerald-500/20 flex items-center justify-center text-emerald-400">
                    <Upload className="w-4 h-4" />
                  </div>
                  <h2 className="text-base font-semibold text-slate-200">
                    {hasDocument ? "Document Source" : "Upload Document"}
                  </h2>
                </div>
                {hasDocument && (
                  <button
                    onClick={() => {
                      setActiveFilename(null);
                      setDocResult(null);
                      setPreviewUrl(null);
                    }}
                    className="text-xs text-slate-400 hover:text-rose-400 transition flex items-center gap-1"
                  >
                    <RefreshCw className="w-3 h-3" /> Reset
                  </button>
                )}
              </div>

              {/* Drag and Drop Upload Area */}
              <div
                onClick={() => fileInputRef.current?.click()}
                className={`border-2 border-dashed rounded-xl p-5 text-center cursor-pointer transition-all ${
                  isProcessing
                    ? "border-emerald-500/40 bg-emerald-950/10"
                    : "border-slate-700/80 hover:border-emerald-500/60 bg-slate-950/50 hover:bg-slate-800/30"
                }`}
              >
                <input
                  type="file"
                  ref={fileInputRef}
                  className="hidden"
                  accept=".png,.jpg,.jpeg,.pdf"
                  onChange={(e) => {
                    const f = e.target.files?.[0];
                    if (f) handleFileUpload(f);
                  }}
                />
                <div className="flex flex-col items-center justify-center gap-2">
                  <div className="w-10 h-10 rounded-full bg-slate-800/80 flex items-center justify-center text-slate-300">
                    <Upload className="w-5 h-5 text-emerald-400" />
                  </div>
                  <div>
                    <span className="text-sm font-medium text-emerald-400 hover:underline">
                      Click to upload
                    </span>{" "}
                    <span className="text-sm text-slate-400">or drag and drop</span>
                  </div>
                  <p className="text-xs text-slate-500">
                    Supports PNG, JPG, JPEG, and PDF documents
                  </p>
                </div>
              </div>

              {/* Quick Sample Selector */}
              <div className="mt-5">
                <div className="flex items-center justify-between mb-2.5">
                  <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider">
                    Or select from test set (10 Documents)
                  </span>
                  <span className="text-xs text-emerald-400/90 font-mono">
                    {samples.length} files available
                  </span>
                </div>
                <div className="grid grid-cols-2 gap-2 max-h-48 overflow-y-auto pr-1">
                  {samples.map((s) => (
                    <button
                      key={s.doc_id}
                      onClick={() => handleSampleSelect(s)}
                      disabled={isProcessing}
                      className={`text-left p-2.5 rounded-lg border text-xs transition flex items-center gap-2 truncate ${
                        activeFilename === s.filename
                          ? "bg-emerald-950/60 border-emerald-700/80 text-emerald-300 font-semibold"
                          : "bg-slate-950/60 border-slate-800 text-slate-300 hover:bg-slate-800/60 hover:border-slate-700"
                      }`}
                    >
                      <FileText className="w-3.5 h-3.5 shrink-0 text-slate-400" />
                      <span className="truncate">{s.filename}</span>
                    </button>
                  ))}
                </div>
              </div>
            </div>

            {/* Document Preview Box (visible when file is active) */}
            {hasDocument && (
              <div className="bg-slate-900/90 rounded-2xl border border-slate-800 p-5 shadow-xl backdrop-blur-sm flex flex-col">
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center gap-2">
                    <Eye className="w-4 h-4 text-emerald-400" />
                    <span className="text-xs font-semibold text-slate-300 uppercase tracking-wider">
                      Document Preview
                    </span>
                  </div>
                  <span className="text-xs text-slate-400 truncate max-w-[200px]">
                    {activeFilename}
                  </span>
                </div>
                <div className="relative rounded-xl border border-slate-800/80 bg-slate-950 overflow-hidden min-h-[300px] max-h-[500px] flex items-center justify-center p-2">
                  {previewUrl ? (
                    <img
                      src={previewUrl}
                      alt="Document Preview"
                      className="max-h-[480px] w-auto object-contain rounded-lg shadow-md"
                    />
                  ) : (
                    <div className="text-slate-500 text-sm flex items-center gap-2">
                      <Clock className="w-4 h-4 animate-spin text-emerald-400" />
                      Generating image preview...
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>

          {/* RIGHT COLUMN: Appears when document is uploaded */}
          {hasDocument && (
            <div className="lg:col-span-7 flex flex-col">
              {/* STATE A: Document Processing Card */}
              {isProcessing && (
                <div className="bg-slate-900/90 rounded-2xl border border-emerald-500/30 p-8 shadow-2xl backdrop-blur-sm flex flex-col items-center justify-center min-h-[420px] text-center">
                  <div className="relative mb-6">
                    <div className="w-16 h-16 rounded-2xl bg-emerald-500/10 border border-emerald-500/30 flex items-center justify-center text-emerald-400 animate-pulse">
                      <Sparkles className="w-8 h-8" />
                    </div>
                    <div className="absolute -top-1 -right-1 w-4 h-4 rounded-full bg-emerald-500 animate-ping" />
                  </div>
                  <h3 className="text-xl font-bold text-white mb-2">
                    Document Processing...
                  </h3>
                  <p className="text-sm text-emerald-400 font-medium mb-4 animate-pulse">
                    {processingStep}
                  </p>
                  <p className="text-xs text-slate-400 max-w-md">
                    Running RapidOCR engine on CPU, classifying document type anchors, isolating handwritten strokes, and evaluating confidence scores.
                  </p>
                </div>
              )}

              {/* STATE B: Extracted Information & Triage Review Card */}
              {!isProcessing && docResult && (
                <div className="bg-slate-900/90 rounded-2xl border border-slate-800 p-6 shadow-2xl backdrop-blur-sm flex flex-col gap-6">
                  {/* Top Bar: Dropdown Menu to Select Document Type */}
                  <div className="bg-slate-950/80 rounded-xl p-4 border border-slate-800 flex flex-col md:flex-row md:items-center justify-between gap-4">
                    <div className="flex-1">
                      <label className="block text-xs font-semibold text-slate-400 uppercase tracking-wider mb-1.5">
                        Classified Document Type (Editable)
                      </label>
                      <div className="relative">
                        <select
                          value={selectedType}
                          onChange={(e) => handleTypeChange(e.target.value)}
                          className="w-full bg-slate-900 text-white font-medium text-sm rounded-lg border border-slate-700/80 py-2.5 pl-3 pr-10 appearance-none focus:outline-none focus:border-emerald-500 transition"
                        >
                          {DOCUMENT_TYPES.map((t) => (
                            <option key={t} value={t}>
                              {t}
                            </option>
                          ))}
                        </select>
                        <ChevronDown className="w-4 h-4 text-slate-400 absolute right-3 top-3.5 pointer-events-none" />
                      </div>
                    </div>

                    {/* Classification Status Badge */}
                    <div className="flex md:flex-col items-center md:items-end justify-between gap-1 pt-1">
                      <span className="text-xs text-slate-400">Model Confidence</span>
                      <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold bg-emerald-950/80 text-emerald-300 border border-emerald-800">
                        <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />
                        {Math.round(docResult.classification_confidence * 100)}% Match
                      </span>
                    </div>
                  </div>

                  {/* Extraction Metrics Bar */}
                  <div className="grid grid-cols-3 gap-3 text-center">
                    <div className="bg-slate-950/50 p-3 rounded-xl border border-slate-800/80">
                      <span className="text-xs text-slate-400 block mb-0.5">Total Fields</span>
                      <span className="text-lg font-bold text-slate-200">
                        {docResult.fields.length}
                      </span>
                    </div>

                    <div className="bg-slate-950/50 p-3 rounded-xl border border-slate-800/80">
                      <span className="text-xs text-slate-400 block mb-0.5">Overall Conf.</span>
                      <span
                        className={`text-lg font-bold ${
                          docResult.overall_confidence >= 0.85
                            ? "text-emerald-400"
                            : "text-amber-400"
                        }`}
                      >
                        {Math.round(docResult.overall_confidence * 100)}%
                      </span>
                    </div>

                    <div className="bg-slate-950/50 p-3 rounded-xl border border-slate-800/80">
                      <span className="text-xs text-slate-400 block mb-0.5">Review Queue</span>
                      <span
                        className={`text-lg font-bold ${
                          docResult.flagged_count > 0 ? "text-amber-400" : "text-emerald-400"
                        }`}
                      >
                        {docResult.flagged_count > 0
                          ? `${docResult.flagged_count} Flagged`
                          : "0 Flagged"}
                      </span>
                    </div>
                  </div>

                  {/* Extracted Fields Table */}
                  <div className="overflow-x-auto rounded-xl border border-slate-800">
                    <table className="w-full text-left text-xs">
                      <thead className="bg-slate-950 text-slate-400 font-semibold uppercase tracking-wider border-b border-slate-800">
                        <tr>
                          <th className="py-3 px-3.5">Target Field</th>
                          <th className="py-3 px-3.5">Extracted Value</th>
                          <th className="py-3 px-3.5">Type</th>
                          <th className="py-3 px-3.5">Confidence</th>
                          <th className="py-3 px-3.5">Status & Action</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-slate-800/80">
                        {docResult.fields.map((f) => {
                          const isLowConfidence = f.confidence < 0.85 || f.is_flagged;
                          const isBeingEdited = editingField === f.field_name;

                          return (
                            <tr
                              key={f.field_name}
                              className={`transition-colors ${
                                isLowConfidence
                                  ? "bg-amber-950/20 hover:bg-amber-950/30"
                                  : "hover:bg-slate-800/30"
                              }`}
                            >
                              {/* Field Name */}
                              <td className="py-3 px-3.5 font-medium text-slate-200">
                                {f.field_name}
                              </td>

                              {/* Value (Editable) */}
                              <td className="py-3 px-3.5 max-w-[220px]">
                                {isBeingEdited ? (
                                  <div className="flex items-center gap-1.5">
                                    <input
                                      type="text"
                                      value={editValue}
                                      onChange={(e) => setEditValue(e.target.value)}
                                      className="bg-slate-950 text-white border border-emerald-500 rounded px-2 py-1 text-xs w-full focus:outline-none"
                                      autoFocus
                                    />
                                    <button
                                      onClick={() => handleSaveFieldEdit(f.field_name)}
                                      className="p-1 bg-emerald-600 hover:bg-emerald-500 text-white rounded transition"
                                      title="Save"
                                    >
                                      <Check className="w-3.5 h-3.5" />
                                    </button>
                                  </div>
                                ) : (
                                  <div className="flex items-center justify-between gap-1 group">
                                    <span
                                      className={`truncate font-mono ${
                                        f.value ? "text-slate-200" : "text-slate-500 italic"
                                      }`}
                                    >
                                      {f.value || "Not Detected"}
                                    </span>
                                    <button
                                      onClick={() => {
                                        setEditingField(f.field_name);
                                        setEditValue(f.value || "");
                                      }}
                                      className="opacity-0 group-hover:opacity-100 p-1 text-slate-400 hover:text-white transition"
                                      title="Edit field value"
                                    >
                                      <Edit3 className="w-3.5 h-3.5" />
                                    </button>
                                  </div>
                                )}

                                {/* Flag reason note */}
                                {f.flag_reason && (
                                  <span className="text-[10px] text-amber-400/90 block mt-0.5">
                                    ⚠️ {f.flag_reason}
                                  </span>
                                )}
                              </td>

                              {/* Nature Badge (Printed vs Handwritten) */}
                              <td className="py-3 px-3.5">
                                <span
                                  className={`px-2 py-0.5 rounded text-[10px] font-medium border ${
                                    f.is_handwritten
                                      ? "bg-purple-950/70 text-purple-300 border-purple-800"
                                      : "bg-slate-800 text-slate-300 border-slate-700"
                                  }`}
                                >
                                  {f.is_handwritten ? "Handwritten" : "Printed"}
                                </span>
                              </td>

                              {/* Confidence Score */}
                              <td className="py-3 px-3.5">
                                <div className="flex items-center gap-2">
                                  <div className="w-12 bg-slate-800 rounded-full h-1.5 overflow-hidden">
                                    <div
                                      className={`h-full rounded-full ${
                                        f.confidence >= 0.85
                                          ? "bg-emerald-400"
                                          : f.confidence >= 0.70
                                          ? "bg-amber-400"
                                          : "bg-rose-500"
                                      }`}
                                      style={{ width: `${Math.round(f.confidence * 100)}%` }}
                                    />
                                  </div>
                                  <span
                                    className={`font-mono text-xs font-semibold ${
                                      f.confidence >= 0.85
                                        ? "text-emerald-400"
                                        : f.confidence >= 0.70
                                        ? "text-amber-400"
                                        : "text-rose-400"
                                    }`}
                                  >
                                    {Math.round(f.confidence * 100)}%
                                  </span>
                                </div>
                              </td>

                              {/* Status & Review Action */}
                              <td className="py-3 px-3.5">
                                {f.human_verified ? (
                                  <span className="inline-flex items-center gap-1 text-[11px] font-medium text-emerald-400">
                                    <ShieldCheck className="w-3.5 h-3.5" /> Verified
                                  </span>
                                ) : isLowConfidence ? (
                                  <div className="flex items-center gap-2">
                                    <span className="inline-flex items-center gap-1 text-[11px] font-medium text-amber-400">
                                      <AlertTriangle className="w-3.5 h-3.5" /> Review Req.
                                    </span>
                                    <button
                                      onClick={() => handleVerifyField(f.field_name, f.value || "")}
                                      className="px-2 py-0.5 bg-amber-500/20 hover:bg-amber-500/30 text-amber-300 border border-amber-500/40 rounded text-[10px] font-semibold transition"
                                    >
                                      Approve
                                    </button>
                                  </div>
                                ) : (
                                  <span className="inline-flex items-center gap-1 text-[11px] font-medium text-slate-400">
                                    <CheckCircle2 className="w-3.5 h-3.5 text-emerald-500" /> Auto-Passed
                                  </span>
                                )}
                              </td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>

                  {/* Bottom Action Footer */}
                  <div className="flex items-center justify-between pt-2">
                    <span className="text-xs text-slate-500">
                      Processed in {docResult.processing_time_sec}s via Local RapidOCR
                    </span>
                    <button
                      onClick={handleDownloadJSON}
                      className="flex items-center gap-2 bg-emerald-600 hover:bg-emerald-500 text-white text-xs font-semibold px-4 py-2 rounded-lg shadow transition"
                    >
                      <Download className="w-4 h-4" /> Download Structured JSON
                    </button>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      </main>

      {/* Methodology & Technical Report Modal */}
      {showReportModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm p-4">
          <div className="bg-slate-900 border border-slate-800 rounded-2xl max-w-2xl w-full p-6 shadow-2xl overflow-y-auto max-h-[90vh]">
            <div className="flex items-center justify-between border-b border-slate-800 pb-4 mb-4">
              <h3 className="text-lg font-bold text-white flex items-center gap-2">
                <FileCheck className="w-5 h-5 text-emerald-400" />
                Question 3: Triage Report & Methodology
              </h3>
              <button
                onClick={() => setShowReportModal(false)}
                className="text-slate-400 hover:text-white text-sm"
              >
                ✕ Close
              </button>
            </div>

            <div className="space-y-4 text-xs text-slate-300 leading-relaxed">
              <div>
                <h4 className="font-semibold text-emerald-400 mb-1">
                  1. Confidence Threshold Value & Rationale
                </h4>
                <p className="bg-slate-950 p-3 rounded-lg border border-slate-800 text-slate-300">
                  <strong>Threshold chosen: \(\tau = 0.85\) (85%)</strong>. In banking KYC and life insurance onboarding, false positives on core financial identifiers (wrong IFSC, garbled PAN, incorrect account numbers) lead to transaction failures or policy rejection. Printed cards score consistently \(\ge 0.90\), while ambiguous handwritten fields score between \(0.60 - 0.84\), cleanly routing only questionable fields to human review without overloading the review queue.
                </p>
              </div>

              <div>
                <h4 className="font-semibold text-emerald-400 mb-1">
                  2. Handling Handwritten Text Differently from Printed Text
                </h4>
                <p className="bg-slate-950 p-3 rounded-lg border border-slate-800 text-slate-300">
                  Printed text has fixed font geometries and uniform spacing. Handwritten text (such as in ECS Mandates, FATCA declarations, Moral Hazard questionnaires, and Suitability profilers) has variable stroke widths, cursive ligatures, and ambiguous characters (e.g. <code>O</code> vs <code>0</code>, <code>I</code> vs <code>1</code>, <code>SBJNO</code> vs <code>SBIN0</code>). The pipeline applies regex-guided character disambiguation, assigns an inherent handwriting scrutiny penalty, and flags any field with uncertain strokes for human triage.
                </p>
              </div>

              <div>
                <h4 className="font-semibold text-emerald-400 mb-1">
                  3. Observed Failure Cases & Mitigation
                </h4>
                <ul className="list-disc list-inside bg-slate-950 p-3 rounded-lg border border-slate-800 space-y-1 text-slate-300">
                  <li><strong>Handwritten IFSC on ECS form:</strong> Read as <code>JFscSB1N0221</code> due to cursive strokes. Mitigated by auto-correcting to <code>SBIN</code> and flagging at 50% confidence for approval.</li>
                  <li><strong>Handwritten Location:</strong> &quot;West Bihar&quot; occasionally scanned as &quot;Wes 4 Bihar&quot; due to fragmented crossbars. Flagged with low confidence.</li>
                  <li><strong>Nominee Relationship:</strong> Cursive &quot;Nephew&quot; handwriting variance flagged for reviewer confirmation.</li>
                </ul>
              </div>
            </div>

            <div className="mt-6 flex justify-end">
              <button
                onClick={() => setShowReportModal(false)}
                className="bg-slate-800 hover:bg-slate-700 text-white text-xs font-medium px-4 py-2 rounded-lg transition"
              >
                Close Report
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
