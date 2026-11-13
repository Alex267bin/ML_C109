const form = document.getElementById("predict-form");
const button = document.getElementById("submit-btn");
const emptyState = document.getElementById("empty-state");
const result = document.getElementById("result");
const errorBox = document.getElementById("error-box");

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  errorBox.hidden = true;
  result.hidden = true;
  emptyState.hidden = false;
  button.disabled = true;
  button.querySelector("span").textContent = "Đang tính toán...";
  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({
        store_nbr: document.getElementById("store").value,
        family: document.getElementById("family").value,
        date: document.getElementById("date").value,
        onpromotion: document.getElementById("promotion").value
      })
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Không thể dự đoán.");
    document.getElementById("prediction-value").textContent =
      new Intl.NumberFormat("vi-VN", {maximumFractionDigits: 2}).format(data.prediction);
    document.getElementById("model-name").textContent = data.model;
    document.getElementById("result-date").textContent = document.getElementById("date").value;
    document.getElementById("result-family").textContent = document.getElementById("family").value;
    emptyState.hidden = true;
    result.hidden = false;
  } catch (error) {
    emptyState.hidden = true;
    errorBox.textContent = error.message;
    errorBox.hidden = false;
  } finally {
    button.disabled = false;
    button.querySelector("span").textContent = "Dự đoán doanh số";
  }
});
