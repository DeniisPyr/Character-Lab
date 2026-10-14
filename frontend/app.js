// Fills the page from data.json, which frontend/build.py makes from examples/diana.

const VIEWS = {
  turnaround: "Turnaround",
  portrait: "Portrait",
  expressions: "Expressions",
  scenes: "Scenes",
};
const VARIANTS = { three_quarter: "three-quarter", over_shoulder: "over the shoulder" };

const variantName = (variant) => VARIANTS[variant] ?? variant.replaceAll("_", " ");
const title = (image) => `${VIEWS[image.view] ?? image.view} · ${variantName(image.variant)}`;

function element(tag, attributes = {}, ...children) {
  const node = document.createElement(tag);
  Object.assign(node, attributes);
  node.append(...children);
  return node;
}

function renderStats(data) {
  const summary = data.summary;
  const images = summary.images;
  const passed = data.images.filter((image) => image.passed && image.tries === 1).length;
  const stats = [
    [images, "images"],
    [`${passed}/${images}`, "passed the quality gate first try"],
    [`${summary.identity.same_character}/${images}`, "seen as the same character"],
    [summary.captions.trigger, "the trigger word"],
  ];
  document
    .getElementById("stats")
    .append(...stats.map(([value, label]) => element("li", {}, element("strong", { textContent: value }), label)));
}

function renderGallery(data, open) {
  const gallery = document.getElementById("gallery");
  for (const view of Object.keys(VIEWS)) {
    const shots = data.images.filter((image) => image.view === view);
    if (!shots.length) continue;
    const [width, height] = shots[0].size;
    const grid = element("div", { className: `grid grid-${view}` });
    for (const image of shots) {
      const index = data.images.indexOf(image);
      const button = element(
        "button",
        { className: "card", type: "button", onclick: () => open(index) },
        element("img", { src: image.src, alt: title(image), loading: "lazy", width, height }),
        element("span", { className: "card-label", textContent: variantName(image.variant) }),
      );
      grid.append(button);
    }
    gallery.append(
      element(
        "div",
        { className: "view-group" },
        element(
          "h3",
          {},
          VIEWS[view],
          element("small", { textContent: ` ${shots.length} images · ${width}×${height}` }),
        ),
        grid,
      ),
    );
  }
}

function renderScores(data) {
  const rows = document.getElementById("score-rows");
  for (const score of data.scores) {
    rows.append(
      element(
        "tr",
        {},
        element(
          "th",
          { scope: "row" },
          element("strong", { textContent: score.name }),
          element("small", { textContent: score.description }),
        ),
        element("td", { textContent: score.mean }),
        element("td", { textContent: score.worst }),
        element("td", { className: "file", textContent: score.where }),
      ),
    );
  }
  document.getElementById("identity-note").textContent =
    `CCIP counts a difference below ${data.same_character_below} as the same character. For ` +
    "semi-realistic characters like Diana that limit is loose; a better identity check is in progress.";
}

function setUpViewer(data) {
  const dialog = document.getElementById("viewer");
  const field = (id) => document.getElementById(id);
  let current = 0;

  function show(index) {
    current = (index + data.images.length) % data.images.length;
    const image = data.images[current];
    const [width, height] = image.size;
    field("viewer-image").src = image.src;
    field("viewer-image").alt = title(image);
    field("viewer-title").textContent = title(image);
    field("viewer-meta").textContent =
      `${image.file} · ${width}×${height} · seed ${image.seed} · ` +
      (image.passed ? `passed on try ${image.tries}` : "kept the best of the tries");
    const [trigger, ...rest] = image.caption.split(",");
    field("viewer-caption").replaceChildren(element("mark", { textContent: trigger }), `,${rest.join(",")}`);
    field("viewer-prompt").textContent = image.prompt;
    const scores = [`Identity ${image.identity}`];
    if (image.face) scores.push(`face ${image.face} px`);
    if (image.sharpness !== null) scores.push(`sharpness ${image.sharpness}×`);
    if (!image.face) scores.push("no face: a shot from behind");
    field("viewer-scores").textContent = scores.join(" · ");
  }

  field("viewer-close").onclick = () => dialog.close();
  field("viewer-previous").onclick = () => show(current - 1);
  field("viewer-next").onclick = () => show(current + 1);
  dialog.addEventListener("click", (event) => {
    if (event.target === dialog) dialog.close();
  });
  dialog.addEventListener("keydown", (event) => {
    if (event.key === "ArrowLeft") show(current - 1);
    if (event.key === "ArrowRight") show(current + 1);
  });
  return (index) => {
    show(index);
    dialog.showModal();
  };
}

fetch("data.json")
  .then((response) => response.json())
  .then((data) => {
    renderStats(data);
    renderGallery(data, setUpViewer(data));
    renderScores(data);
  })
  .catch(() => {
    document.getElementById("gallery").textContent = "The dataset could not be loaded.";
  });
