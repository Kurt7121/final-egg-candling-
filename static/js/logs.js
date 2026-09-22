const dayFilter = document.getElementById("dayFilter");
const dateFilter = document.getElementById("dateFilter");
const logsBody = document.getElementById("logsBody");
const pageInfo = document.getElementById("pageInfo");
const prevPage = document.getElementById("prevPage");
const nextPage = document.getElementById("nextPage");
const modal = document.getElementById("imageModal");
const modalImage = document.getElementById("modalImage");
const modalCaption = document.getElementById("modalCaption");

let page = 1;
const perPage = 10;

for (let day = 1; day <= 21; day += 1) {
  const option = document.createElement("option");
  option.value = `Day ${day}`;
  option.textContent = `Day ${day}`;
  dayFilter.appendChild(option);
}

function closeModal() {
  modal.classList.remove("show");
}

function openImage(item) {
  modalImage.src = item.image_url;
  modalCaption.textContent = `ID ${item.id} · ${item.predicted_day || "Uncertain"} · ${item.date} ${item.time}`;
  modal.classList.add("show");
}

async function loadLogs() {
  const params = new URLSearchParams({
    page: String(page),
    per_page: String(perPage),
  });
  if (dayFilter.value) params.set("day", dayFilter.value);
  if (dateFilter.value) params.set("date", dateFilter.value);

  logsBody.innerHTML = '<tr><td colspan="8">Loading...</td></tr>';

  try {
    const response = await fetch("/api/logs?" + params.toString());
    const data = await response.json();
    if (!data.success) {
      throw new Error("Could not load logs");
    }

    if (!data.logs.length) {
      logsBody.innerHTML = '<tr><td colspan="8">No detection records yet.</td></tr>';
    } else {
      logsBody.innerHTML = "";
      data.logs.forEach((item) => {
        const row = document.createElement("tr");
        const confidence =
          typeof item.confidence === "number" ? `${Number(item.confidence).toFixed(2)}%` : "—";
        row.innerHTML = `
          <td>${item.id}</td>
          <td><img class="thumb" src="${item.image_url}" alt="egg ${item.id}"></td>
          <td>${item.predicted_day || "—"}</td>
          <td>${item.development_stage || "—"}</td>
          <td>${confidence}</td>
          <td>${item.date}</td>
          <td>${item.time}</td>
          <td><button class="primary view-btn" type="button">VIEW IMAGE</button></td>
        `;
        row.querySelector(".view-btn").addEventListener("click", () => openImage(item));
        row.querySelector(".thumb").addEventListener("click", () => openImage(item));
        logsBody.appendChild(row);
      });
    }

    pageInfo.textContent = `Page ${data.page} of ${data.pages} (${data.total} records)`;
    prevPage.disabled = data.page <= 1;
    nextPage.disabled = data.page >= data.pages;
  } catch (error) {
    logsBody.innerHTML = `<tr><td colspan="8">${error.message}</td></tr>`;
  }
}

document.getElementById("applyFilters").addEventListener("click", () => {
  page = 1;
  loadLogs();
});
document.getElementById("resetFilters").addEventListener("click", () => {
  dayFilter.value = "";
  dateFilter.value = "";
  page = 1;
  loadLogs();
});
prevPage.addEventListener("click", () => {
  page = Math.max(1, page - 1);
  loadLogs();
});
nextPage.addEventListener("click", () => {
  page += 1;
  loadLogs();
});
document.getElementById("closeModal").addEventListener("click", closeModal);
modal.addEventListener("click", (event) => {
  if (event.target.id === "imageModal") closeModal();
});

loadLogs();
