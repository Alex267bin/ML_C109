// The dashboard is rendered from the real CSV/JSON outputs produced by the Python scripts.
document.addEventListener("DOMContentLoaded", () => {
  const images = document.querySelectorAll(".figure, .segmentation-image");
  images.forEach((image) => {
    image.addEventListener("error", () => {
      image.hidden = true;
      const fallback = image.nextElementSibling;
      if (fallback && fallback.classList.contains("figure-empty")) fallback.hidden = false;
    }, { once: true });
  });
});
