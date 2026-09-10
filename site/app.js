"use strict";
const allAudio = new Set();
const synchronizedTracks = new Set(["mixture", "estimate", "target"]);
const trackLabels = {enrollment: "Enrollment", mixture: "Mixture", estimate: "APSR-P", target: "Clean target"};

function makeTrack(id) {
  const figure = document.createElement("figure");
  figure.className = "track";
  figure.dataset.track = id;
  const heading = document.createElement("h4");
  heading.textContent = trackLabels[id];
  const description = document.createElement("span");
  description.className = "track-description";
  description.hidden = true;
  const image = document.createElement("img");
  image.loading = "lazy";
  image.alt = `${trackLabels[id]} spectrogram; 0–4 kHz from bottom to top`;
  const axis = document.createElement("div");
  axis.className = "track-axis";
  const start = document.createElement("span"); start.textContent = "0 s";
  const end = document.createElement("span"); axis.append(start, end);
  const audio = document.createElement("audio");
  audio.controls = true; audio.preload = "metadata";
  const download = document.createElement("a");
  download.className = "download"; download.textContent = "Download";
  figure.append(heading, description, image, axis, audio, download);
  allAudio.add(audio);
  return {figure, description, image, end, audio, download};
}

function makeExample(example, index) {
  const article = document.getElementById("example-template").content.firstElementChild.cloneNode(true);
  article.id = example.id;
  let selectedTarget = "a";
  let sharedTime = 0;
  let activeSynchronizedAudio = null;
  let revising = false;
  const tracks = Object.fromEntries(Object.keys(trackLabels).map(id => [id, makeTrack(id)]));
  article.querySelector(".track-grid").append(...Object.values(tracks).map(t => t.figure));
  const buttons = article.querySelectorAll("[data-target]");
  article.querySelector(".target-switch").setAttribute("aria-label", `Target for example ${index + 1}`);
  function reset() {
    sharedTime = 0; activeSynchronizedAudio = null;
    Object.values(tracks).forEach(({audio}) => { audio.pause(); if (audio.readyState) audio.currentTime = 0; });
  }
  Object.entries(tracks).forEach(([id, {audio}]) => {
    let pendingPosition = null;
    function alignPosition() {
      if (pendingPosition !== null && audio.readyState >= 2 && Number.isFinite(audio.duration)) {
        const position = Math.min(pendingPosition, Math.max(0, audio.duration - .05));
        pendingPosition = null;
        if (Math.abs(audio.currentTime - position) > .08) audio.currentTime = position;
      }
    }
    audio.addEventListener("play", () => {
      if (synchronizedTracks.has(id)) {
        if (activeSynchronizedAudio && activeSynchronizedAudio !== audio && !activeSynchronizedAudio.ended) sharedTime = activeSynchronizedAudio.currentTime;
        pendingPosition = sharedTime;
        activeSynchronizedAudio = audio;
      }
      allAudio.forEach(other => { if (other !== audio) other.pause(); });
      alignPosition();
    });
    ["loadeddata", "canplay"].forEach(event => audio.addEventListener(event, () => { if (!audio.paused && !revising) alignPosition(); }));
    audio.addEventListener("pause", () => { pendingPosition = null; });
    audio.addEventListener("timeupdate", () => { if (!revising && !audio.paused && audio === activeSynchronizedAudio && pendingPosition === null) sharedTime = audio.currentTime; });
    audio.addEventListener("seeked", () => { if (!revising && audio === activeSynchronizedAudio && pendingPosition === null) sharedTime = audio.currentTime; });
    audio.addEventListener("ended", () => { if (synchronizedTracks.has(id)) sharedTime = 0; });
    audio.addEventListener("error", () => {
      tracks[id].description.textContent = "Audio unavailable; please reload the page.";
      tracks[id].description.hidden = false;
      tracks[id].figure.classList.add("unavailable");
    });
  });
  function render() {
    revising = true; reset();
    const query = example.queries.find(q => q.target === selectedTarget);
    const targetName = selectedTarget.toUpperCase();
    buttons.forEach(button => button.setAttribute("aria-pressed", String(button.dataset.target === selectedTarget)));
    article.querySelector(".example-title").textContent = `Example ${String(index + 1).padStart(2, "0")}`;
    Object.entries(tracks).forEach(([id, track]) => {
      const url = id === "mixture" ? example.mixture_audio : query[`${id}_audio`];
      const seconds = id === "enrollment" ? query.enrollment_excerpt_seconds : example.excerpt_seconds;
      track.description.hidden = true;
      track.image.src = url.replace(/^audio\//, "assets/tracks/").replace(/\.wav$/, ".png");
      track.end.textContent = `${seconds.toFixed(2)} s`;
      track.audio.setAttribute("aria-label", `Example ${index + 1}, target ${targetName}: ${trackLabels[id]}`);
      track.audio.src = url; track.audio.load();
      track.download.href = url; track.download.download = url.split("/").pop();
      track.download.setAttribute("aria-label", `Download example ${index + 1} ${trackLabels[id]} excerpt`);
      track.figure.classList.remove("unavailable");
    });
    revising = false;
  }
  buttons.forEach(button => button.addEventListener("click", () => {
    if (selectedTarget !== button.dataset.target) { selectedTarget = button.dataset.target; render(); }
  }));
  article.querySelector(".restart").addEventListener("click", reset);
  render();
  return article;
}

fetch("data.json", {cache: "no-store"}).then(response => {
  if (!response.ok) throw new Error("Missing project manifest");
  return response.json();
}).then(data => {
  document.getElementById("load-status").hidden = true;
  if (data.audio_publication?.status === "pending") {
    document.getElementById("audio-pending").hidden = false;
    return;
  }
  if (!Array.isArray(data.examples) || data.examples.length !== 8) throw new Error("Expected eight verified mixtures");
  data.examples.forEach((example, index) => {
    document.getElementById("examples").append(makeExample(example, index));
    const link = document.createElement("a"); link.href = `#${example.id}`; link.textContent = `Example ${String(index + 1).padStart(2, "0")}`;
    document.getElementById("example-index").append(link);
  });
  document.getElementById("local-status").hidden = data.audio_publication?.status !== "local-preview";
  document.getElementById("listening-content").hidden = false;
}).catch(error => {
  const status = document.getElementById("load-status"); status.hidden = false;
  status.textContent = "Listening-example information could not be loaded. Please reload this page or serve it with a local HTTP server.";
  console.error(error);
});
