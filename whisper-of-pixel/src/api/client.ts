const API_BASE =
  import.meta.env.VITE_API_BASE_URL || "http://localhost:8500";

async function handleResponse(res: Response): Promise<any> {
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || `HTTP error ${res.status}`);
  }
  const ct = res.headers.get("content-type") || "";
  if (ct.includes("application/json")) {
    return res.json();
  }
  return res.text();
}

export async function uploadImage(file: File) {
  const formData = new FormData();
  formData.append("file", file);

  const res = await fetch(`${API_BASE}/upload`, {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    const text = await res.text();
    throw new Error(text);
  }

  return res.json(); // { image_id }
}

export async function getJson(
  path: string,
  params: Record<string, any> = {}
) {
  const url = new URL(`${API_BASE}${path}`);
  Object.entries(params).forEach(([key, value]) => {
    if (value !== undefined && value !== null && value !== "") {
      url.searchParams.set(key, String(value));
    }
  });
  const res = await fetch(url.toString(), { method: "GET" });
  return handleResponse(res);
}

export async function postKernel(
  path: string,
  imageId: string,
  kernelFile: File
) {
  const formData = new FormData();
  formData.append("image_id", imageId);
  formData.append("kernel", kernelFile);
  const res = await fetch(
    `${API_BASE}${path}?image_id=${encodeURIComponent(imageId)}`,
    {
      method: "POST",
      body: formData,
    }
  );
  return handleResponse(res);
}
