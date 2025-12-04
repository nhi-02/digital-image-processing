import React, { useState, useRef } from "react";
import JSZip from "jszip";
import { uploadImage, getJson, postKernel } from "./api/client";
import { PARENTS } from "./constants/operations";
import logo from "./assets/logo.png";
interface OutputImage {
  label: string;
  data_url: string;
}

interface CompressionResult {
  rms: number;
  original_size: number;
  compressed_size: number;
  compression_ratio: number;
  target_ratio_percent: number;
  compressed_data_base64: string;
  padding_bits: number;
}

// child op type cho tiện (tuỳ, m có thể để any cho nhanh)
type ChildOp = (typeof PARENTS)[number]["children"][number];

function App() {
  const [originalFile, setOriginalFile] = useState<File | null>(null);
  const [originalUrl, setOriginalUrl] = useState<string | null>(null);
  const [imageId, setImageId] = useState<string | null>(null);
  const [filename, setFilename] = useState("");

  const [activeParentId, setActiveParentId] = useState<string | null>(null);
  const [activeChildId, setActiveChildId] = useState<string | null>(null);

  const [outputs, setOutputs] = useState<OutputImage[]>([]);
  const [currentOutputIndex, setCurrentOutputIndex] = useState(0);
  const [isZoomOpen, setIsZoomOpen] = useState(false);

  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState("");

  const [paramValues, setParamValues] = useState<Record<string, any>>({});
  const [kernelFile, setKernelFile] = useState<File | null>(null);
  const [histMatchTargetId, setHistMatchTargetId] = useState<string | null>(
    null
  );

  const [compressionResult, setCompressionResult] =
    useState<CompressionResult | null>(null);

  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const hasProcessed = outputs.length > 0 || compressionResult !== null;

  // ... phần còn lại của App giữ nguyên y như t đã gửi (JSX/logic không đổi) ...

  const handleClickDropZone = () => {
    if (fileInputRef.current) {
      fileInputRef.current.click();
    }
  };

  const handleFileChange = async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    resetAllProcessing();
    setOriginalFile(file);
    setFilename(file.name);

    try {
      const data = await uploadImage(file);
      setImageId(data.image_id);
      // dùng preview từ backend (luôn là PNG)
      if (data.preview_url) {
        setOriginalUrl(data.preview_url);
      } else {
        // fallback (phòng khi backend cũ chưa trả preview)
        const reader = new FileReader();
        reader.onload = () => {
          setOriginalUrl(reader.result as string);
        };
        reader.readAsDataURL(file);
      }
    } catch (err: any) {
      console.error(err);
      setError("Upload failed: " + err.message);
    }
  };
  const resetAllProcessing = () => {
    setOutputs([]);
    setCurrentOutputIndex(0);
    setCompressionResult(null);
    setActiveChildId(null);
    setError("");
  };

  const handleParentClick = (parentId) => {
    setActiveParentId(parentId);
    setActiveChildId(null);
    setOutputs([]);
    setCompressionResult(null);
    setCurrentOutputIndex(0);
    setKernelFile(null);
    setParamValues({});
    setHistMatchTargetId(null);
  };

  const currentParent = PARENTS.find((p) => p.id === activeParentId);
  const currentChild =
    currentParent?.children.find((c) => c.id === activeChildId) || null;

  const handleChildClick = (child) => {
    setActiveChildId(child.id);
    setOutputs([]);
    setCompressionResult(null);
    setCurrentOutputIndex(0);
    setError("");

    // Nếu child không có params -> gọi trực tiếp
    if (!child.params || child.params.length === 0) {
      runOperation(child);
    }
  };

  const handleParamChange = (name, value) => {
    setParamValues((prev) => ({ ...prev, [name]: value }));
  };

  const handleKernelFileChange = (e) => {
    const file = e.target.files[0];
    setKernelFile(file || null);
  };

  const handleTargetImageChange = async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    try {
      const data = await uploadImage(file);
      setHistMatchTargetId(data.image_id);
      handleParamChange("targetImage", file.name);
    } catch (err) {
      console.error(err);
      setError("Upload target image failed: " + err.message);
    }
  };

  const runOperation = async (child) => {
    if (!imageId) {
      setError("Please upload an image first.");
      return;
    }
    setIsProcessing(true);
    setError("");

    try {
      // Tùy loại param
      if (child.params && child.params.some((p) => p.type === "kernel")) {
        if (!kernelFile) {
          setError("Please choose a kernel file (.txt).");
          setIsProcessing(false);
          return;
        }
        const data = await postKernel(child.path, imageId, kernelFile);
        setOutputs(data.images || []);
        setCurrentOutputIndex(0);
      } else if (
        child.params &&
        child.params.some((p) => p.type === "image-upload")
      ) {
        if (!histMatchTargetId) {
          setError("Please choose target image for histogram matching.");
          setIsProcessing(false);
          return;
        }
        const params = { image_id: imageId, target_id: histMatchTargetId };
        const data = await getJson(child.path, params);
        setOutputs(data.images || []);
        setCurrentOutputIndex(0);
      } else if (child.isCompression) {
        const ratioParam =
          Number(paramValues["ratio"]) || child.params[0].default;
        const data = await getJson(child.path, {
          image_id: imageId,
          ratio: ratioParam,
        });
        setCompressionResult(data);
        setOutputs([]);
      } else {
        const params = { image_id: imageId };

        if (child.params) {
          child.params.forEach((p) => {
            if (p.type === "number") {
              const v =
                paramValues[p.name] !== undefined
                  ? paramValues[p.name]
                  : p.default;
              params[p.name] = v;
            }
          });
        }

        const data = await getJson(child.path, params);
        setOutputs(data.images || []);
        setCurrentOutputIndex(0);
      }
    } catch (err) {
      console.error(err);
      setError(err.message || "Error while processing.");
    } finally {
      setIsProcessing(false);
    }
  };

  const handleRunClick = () => {
    if (!currentChild) return;
    runOperation(currentChild);
  };

  const handlePrevOutput = () => {
    if (outputs.length === 0) return;
    setCurrentOutputIndex((idx) =>
      idx === 0 ? outputs.length - 1 : idx - 1
    );
  };

  const handleNextOutput = () => {
    if (outputs.length === 0) return;
    setCurrentOutputIndex((idx) =>
      idx === outputs.length - 1 ? 0 : idx + 1
    );
  };

  const handleSave = async () => {
    if (!hasProcessed) return;

    try {
      // 1) Giữ nguyên phần nén Huffman
      if (compressionResult) {
        const { compressed_data_base64 } = compressionResult;
        const bytes = Uint8Array.from(
          atob(compressed_data_base64),
          (c) => c.charCodeAt(0)
        );
        const blob = new Blob([bytes], {
          type: "application/octet-stream",
        });
        const url = URL.createObjectURL(blob);
        const link = document.createElement("a");
        link.href = url;
        link.download =
          (filename || "image") + "_compressed_huffman.bin";
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(url);
        return;
      }

      // 2) Các trường hợp còn lại: lưu ảnh output hiện tại (PNG)
      if (!currentOutput) {
        setError("No output to save.");
        return;
      }

      // currentOutput.data_url đã là data:image/png;base64,...
      const link = document.createElement("a");
      link.href = currentOutput.data_url;

      // đặt tên file: <tên gốc>_<label>.png
      const labelSafe = currentOutput.label
        ? currentOutput.label.replace(/\s+/g, "_")
        : "output";

      link.download = `${filename || "image"}_${labelSafe}.png`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    } catch (err: any) {
      console.error(err);
      setError("Save failed: " + err.message);
    }
  };

  const renderParamForm = () => {
    if (!currentChild || !currentChild.params) return null;

    return (
      <div className="param-form">
        {currentChild.params.map((p) => {
          if (p.type === "number") {
            const value =
              paramValues[p.name] !== undefined
                ? paramValues[p.name]
                : p.default ?? "";
            return (
              <div className="param-row" key={p.name}>
                <label>{p.label}</label>
                <input
                  type="number"
                  value={value}
                  step={p.step || "1"}
                  onChange={(e) =>
                    handleParamChange(p.name, e.target.value)
                  }
                />
              </div>
            );
          }
          if (p.type === "kernel") {
            return (
              <div className="param-row" key={p.name}>
                <label>{p.label}</label>
                <input
                  type="file"
                  accept=".txt"
                  onChange={handleKernelFileChange}
                />
              </div>
            );
          }
          if (p.type === "image-upload") {
            return (
              <div className="param-row" key={p.name}>
                <label>{p.label}</label>
                <input
                  type="file"
                  accept="image/*"
                  onChange={handleTargetImageChange}
                />
              </div>
            );
          }
          return null;
        })}
        {/* Với những child có params, cần nút OK để run */}
        <button
          className="run-button"
          onClick={handleRunClick}
          disabled={isProcessing}
        >
          {isProcessing ? "Processing..." : "OK"}
        </button>
      </div>
    );
  };

  const currentOutput =
    outputs.length > 0 ? outputs[currentOutputIndex] : null;

  return (
    <div className="app-root">
    {/* Background layer */}
    <div className="app-background"></div>
      <header className="app-header">
        <img src={logo} className="logo-img" alt="Whisper of Pixel logo" />
        <div className="logo-text">Whisper of Pixel</div>
      </header>

      <main className="app-main">
        {/* Upload & display area */}
        <section className="image-panel">
<div className="drop-zone">
  <input
    ref={fileInputRef}
    type="file"
    accept="image/*"
    style={{ display: "none" }}
    onChange={handleFileChange}
  />
  <div className="drop-inner">
    <div className="drop-left" onClick={handleClickDropZone}>
                {originalUrl ? (
                  <div className="image-wrapper">
                    <img
                      src={originalUrl}
                      alt="original"
                      className="image-display"
                    />
                    {filename && (
                      <div className="image-filename">
                        {filename}
                      </div>
                    )}
                  </div>
                ) : (
                  <span className="drop-placeholder">
                    Click here to Uploads file ^^
                  </span>
                )}
              </div>
              <div className="vertical-divider" />
              <div className="drop-right">
              {isProcessing && (
                  <div className="loading-overlay">
                    <div className="spinner" />
                  </div>
                )}
                {currentOutput ? (
                  <div className="output-wrapper">
                    <div className="output-label">
                      {currentOutput.label}
                    </div>
                    <img
                      src={currentOutput.data_url}
                      alt="output"
                      className="image-display clickable"
                      onClick={() => setIsZoomOpen(true)}
                    />
                    {outputs.length > 1 && (
                      <div className="carousel-controls">
                        <button onClick={handlePrevOutput}>
                          ◀
                        </button>
                        <span>
                          {currentOutputIndex + 1} /{" "}
                          {outputs.length}
                        </span>
                        <button onClick={handleNextOutput}>
                          ▶
                        </button>
                      </div>
                    )}
                  </div>
                ) : compressionResult ? (
                  <div className="compression-info">
                    <h3>Compression result (Huffman)</h3>
                    <p>
                      Original size:{" "}
                      {compressionResult.original_size} bytes
                    </p>
                    <p>
                      Compressed size:{" "}
                      {compressionResult.compressed_size} bytes
                    </p>
                    <p>
                      Compression ratio:{" "}
                      {(
                        compressionResult.compression_ratio * 100
                      ).toFixed(2)}
                      %
                    </p>
                    <p>RMS error: {compressionResult.rms}</p>
                  </div>
                ) : (
                  <span className="drop-placeholder">
                    Output here
                  </span>
                )}
              </div>
            </div>
          </div>

          <button
            className="save-button"
            onClick={handleSave}
            disabled={!hasProcessed}
          >
            Save
          </button>

          {error && <div className="error-box">{error}</div>}
        </section>

        {/* Sidebar buttons */}
        <aside className="sidebar">
          <div className="parent-buttons">
            {PARENTS.map((p) => (
              <button
                key={p.id}
                className={
                  "parent-button" +
                  (p.id === activeParentId ? " active" : "")
                }
                onClick={() => handleParentClick(p.id)}
              >
                {p.label}
              </button>
            ))}
          </div>

          {/* child buttons + param form */}
          {currentParent && (
            <div className="child-section">
              <div className="child-header">
                {currentParent.label}
              </div>
              <div className="child-buttons">
                {currentParent.children.map((c) => (
                  <button
                    key={c.id}
                    className={
                      "child-button" +
                      (c.id === activeChildId ? " active" : "")
                    }
                    onClick={() => handleChildClick(c)}
                  >
                    {c.label}
                  </button>
                ))}
              </div>
              {renderParamForm()}
              {/* Nếu child không có params & đang processing */}
              {currentChild &&
                (!currentChild.params ||
                  currentChild.params.length === 0) &&
                isProcessing && (
                  <div className="processing-label">
                    Processing...
                  </div>
                )}
            </div>
          )}
        </aside>
      </main>

      {/* Zoom modal */}
      {isZoomOpen && currentOutput && (
        <div
          className="zoom-overlay"
          onClick={() => setIsZoomOpen(false)}
        >
          <div
            className="zoom-content"
            onClick={(e) => e.stopPropagation()}
          >
            <img
              src={currentOutput.data_url}
              alt={currentOutput.label}
              className="zoom-image"
            />
            <button
              className="zoom-close"
              onClick={() => setIsZoomOpen(false)}
            >
              ×
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
