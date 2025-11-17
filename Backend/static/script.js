const uploadArea = document.getElementById("uploadArea");
const fileInput = document.getElementById("fileInput");
const preview = document.getElementById("preview");
const runBtn = document.getElementById("runBtn");
const loader = document.getElementById("loader");
const result = document.getElementById("result");

let uploadedFile = null;

// Open file browser on click
uploadArea.addEventListener("click", () => fileInput.click());

// Handle file input
fileInput.addEventListener("change", (e) => {
  uploadedFile = e.target.files[0];
  showPreview(uploadedFile);
});

// Drag & drop
uploadArea.addEventListener("dragover", (e) => {
  e.preventDefault();
  uploadArea.style.background = "rgba(79,195,247,0.2)";
});

uploadArea.addEventListener("dragleave", () => {
  uploadArea.style.background = "rgba(255,255,255,0.05)";
});

uploadArea.addEventListener("drop", (e) => {
  e.preventDefault();
  uploadedFile = e.dataTransfer.files[0];
  showPreview(uploadedFile);
});

function showPreview(file) {
  if (!file) return;
  const reader = new FileReader();
  reader.onload = (e) => {
    preview.innerHTML = `<img src="${e.target.result}" alt="Preview">`;
  };
  reader.readAsDataURL(file);
}

runBtn.addEventListener("click", async () => {
  if (!uploadedFile) {
    alert("Please upload an image first!");
    return;
  }

  loader.style.display = "block";
  result.innerHTML = "";

  const formData = new FormData();
  formData.append("file", uploadedFile);

  try {
    const response = await fetch("http://127.0.0.1:5000/predict", {
      method: "POST",
      body: formData,
    });

    if (!response.ok) throw new Error("Failed to fetch result from backend.");

    const blob = await response.blob();
    const url = URL.createObjectURL(blob);

    loader.style.display = "none";
    result.innerHTML = `
      <div>
        <h3>Original</h3>
        <img src="${URL.createObjectURL(uploadedFile)}" alt="Original">
      </div>
      <div>
        <h3>Denoised Image</h3>
        <img src="${url}" alt="Denoised Image">
      </div>
    `;

  } catch (err) {
    loader.style.display = "none";
    alert(err.message);
  }
});
