export const PARENTS = [
  {
    id: "color",
    label: "Color representation",
    children: [
      { id: "color-rgb", label: "RGB", path: "/color/rgb" },
      { id: "color-hsv", label: "HSV", path: "/color/hsv" },
      { id: "color-gray", label: "Gray", path: "/color/gray" },
      { id: "color-yuv", label: "YUV", path: "/color/yuv" },
    ],
  },
  {
    id: "point",
    label: "Point processing",
    children: [
      { id: "point-negative", label: "Negative", path: "/point/negative" },
      {
        id: "point-threshold",
        label: "Threshoding",
        path: "/point/threshold",
        params: [{ name: "T", label: "Threshold (r)", type: "number", default: 128 }],
      },
      {
        id: "point-log",
        label: "Logarithmic trans",
        path: "/point/log",
        params: [{ name: "c", label: "c", type: "number", step: "0.1", default: 1 }],
      },
      {
        id: "point-gamma",
        label: "Gamma",
        path: "/point/gamma",
        params: [
          { name: "gamma", label: "gamma (y)", type: "number", step: "0.1", default: 1 },
          { name: "c", label: "c", type: "number", step: "0.1", default: 1 },
        ],
      },
      {
        id: "point-fuzzy",
        label: "Fuzzy rule-based",
        path: "/point/fuzzy-contrast",
      },
      {
        id: "point-bitplane",
        label: "Bit plane slicing",
        path: "/point/bit-plane",
      },
    ],
  },
  {
    id: "histogram",
    label: "Histogram Equalization",
    children: [
      { id: "hist-show", label: "Histogram", path: "/histogram/show" },
      { id: "hist-eq", label: "Equalization", path: "/histogram/equalization" },
      {
        id: "hist-match",
        label: "Matching",
        path: "/histogram/matching",
        params: [
          {
            name: "targetImage",
            label: "Target image",
            type: "image-upload",
          },
        ],
      },
    ],
  },
  {
    id: "noise",
    label: "Add Noise",
    children: [
      { id: "noise-gauss", label: "Gaussian", path: "/noise/gaussian" },
      {
        id: "noise-snp",
        label: "Salt and pepper",
        path: "/noise/salt-pepper",
      },
      { id: "noise-uniform", label: "Uniform", path: "/noise/uniform" },
      { id: "noise-impulse", label: "Impulse", path: "/noise/impulse" },
    ],
  },
  {
    id: "spatial",
    label: "Spatial filter",
    children: [
      {
        id: "spatial-corr",
        label: "Correlation",
        path: "/spatial/correlation",
        params: [{ name: "kernel", label: "Kernel file (.txt)", type: "kernel" }],
      },
      {
        id: "spatial-conv",
        label: "Convolution",
        path: "/spatial/convolution",
        params: [{ name: "kernel", label: "Kernel file (.txt)", type: "kernel" }],
      },
      {
        id: "spatial-mean",
        label: "Mean filter",
        path: "/spatial/mean",
        params: [{ name: "ksize", label: "Kernel size", type: "number", default: 3 }],
      },
{
  id: "gaussian",
  label: "Gaussian filter",
  path: "/spatial/gaussian",
  params: [
    {
      name: "ksize",
      label: "Kernel size",
      type: "number",
      default: 3,
    },
    {
      name: "sigma",
      label: "Sigma",
      type: "number",
      default: 1.0,
    },
  ],
},
      {
        id: "spatial-weighted",
        label: "Weighted filter",
        path: "/spatial/weighted-mean",
      },
      {
        id: "spatial-median",
        label: "Median filter",
        path: "/spatial/median",
        params: [{ name: "ksize", label: "Kernel size", type: "number", default: 3 }],
      },
      {
        id: "spatial-max",
        label: "Max filter",
        path: "/spatial/max",
        params: [{ name: "ksize", label: "Kernel size", type: "number", default: 3 }],
      },
      {
        id: "spatial-min",
        label: "Min filter",
        path: "/spatial/min",
        params: [{ name: "ksize", label: "Kernel size", type: "number", default: 3 }],
      },
      { id: "spatial-sharp", label: "Sharpening", path: "/spatial/sharpen" },
      {
        id: "spatial-alpha",
        label: "Alphar-trimmed mean filter",
        path: "/spatial/alpha-trimmed",
        params: [
          { name: "ksize", label: "Kernel size", type: "number", default: 3 },
          { name: "d", label: "d", type: "number", default: 2 },
        ],
      },
      {
        id: "spatial-mid",
        label: "Mid-point filter",
        path: "/spatial/midpoint",
        params: [{ name: "ksize", label: "Kernel size", type: "number", default: 3 }],
      },
    ],
  },
  {
    id: "frequency",
    label: "Frequency filter",
    children: [
      { id: "freq-fft2", label: "FFT-2D", path: "/frequency/fft2" },
      { id: "freq-ilpf", label: "ILPF", path: "/frequency/ilpf" },
      { id: "freq-ihpf", label: "IHPF", path: "/frequency/ihpf" },
      { id: "freq-glpf", label: "GLPF", path: "/frequency/glpf" },
      { id: "freq-ghpf", label: "GHPF", path: "/frequency/ghpf" },
      { id: "freq-blpf", label: "BLPF", path: "/frequency/blpf" },
      { id: "freq-bhpf", label: "BHPF", path: "/frequency/bhpf" },
      {
        id: "freq-bandreject",
        label: "Bandreject",
        path: "/frequency/bandreject",
      },
      {
        id: "freq-bandpass",
        label: "Bandpass",
        path: "/frequency/bandpass",
      },
      {
        id: "freq-notch",
        label: "Notch",
        path: "/frequency/notch",
      },
      {
        id: "freq-inverse",
        label: "Lọc nghịch đảo",
        path: "/frequency/inverse",
      },
    ],
  },
  {
    id: "pca",
    label: "PCA",
    children: [
      {
        id: "pca-compress",
        label: "PCA compress",
        path: "/pca/compress",
        params: [{ name: "k", label: "Num components", type: "number", default: 50 }],
      },
    ],
  },
  {
    id: "morph",
    label: "Morphological",
    children: [
      { id: "morph-erosion", label: "Erosion", path: "/morph/erosion" },
      { id: "morph-dilation", label: "Dilation", path: "/morph/dilation" },
      { id: "morph-opening", label: "Opening", path: "/morph/opening" },
      { id: "morph-closing", label: "Closing", path: "/morph/closing" },
    ],
  },
  {
    id: "seg",
    label: "Segmentation",
    children: [
      {
        id: "seg-basic",
        label: "Basic global T",
        path: "/seg/basic-global",
        params: [{ name: "T", label: "Threshold", type: "number", default: 128 }],
      },
      { id: "seg-otsu", label: "Otsu", path: "/seg/otsu" },
      {
        id: "seg-kmeans",
        label: "Clustering",
        path: "/seg/kmeans",
        params: [{ name: "k", label: "k (clusters)", type: "number", default: 3 }],
      },
      {
        id: "seg-yolo",
        label: "Yolov8",
        path: "/detect/yolov8",
      },
    ],
  },
  // Compression để riêng, không có child trong mockup,
  // nhưng để tiện code ta cho 1 child.
  {
    id: "compression",
    label: "Compression",
    children: [
      {
        id: "compression-huffman",
        label: "Huffman",
        path: "/compression/huffman",
        params: [
          {
            name: "ratio",
            label: "Target ratio (%)",
            type: "number",
            default: 50,
          },
        ],
        isCompression: true,
      },
    ],
  },
];
