document.querySelectorAll(".slider-container").forEach(container => {
    const slider = container.querySelector(".slider");
    const prev = container.querySelector(".prev");
    const next = container.querySelector(".next");

    next.addEventListener("click", () => {
        slider.scrollBy({ left: 300, behavior: "smooth" });
    });

    prev.addEventListener("click", () => {
        slider.scrollBy({ left: -300, behavior: "smooth" });
    });
});

