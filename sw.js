/*
 * Service worker: lets the chart be installed as an app.
 * It only handles this site's own files, network first, so
 * a new version is picked up as soon as it is published;
 * the saved copy is used when the phone is offline.
 * Binance requests are never touched.
 */
const CACHE = "btcfdusd-app-v1";
const SHELL = [
  "./btcfdusd-volume-profile.html",
  "./manifest.webmanifest",
  "./icon-192.png",
  "./icon-512.png",
];
self.addEventListener("install", (e) => {
  e.waitUntil(caches.open(CACHE).then((c) => c.addAll(SHELL)));
  self.skipWaiting();
});
self.addEventListener("activate", (e) => {
  e.waitUntil(
    caches
      .keys()
      .then((ks) =>
        Promise.all(
          ks.filter((k) => k !== CACHE).map((k) => caches.delete(k)),
        ),
      )
      .then(() => self.clients.claim()),
  );
});
self.addEventListener("fetch", (e) => {
  const u = new URL(e.request.url);
  const own = u.origin === self.location.origin;
  if (!own || e.request.method !== "GET") return;
  /* the orders file is private and local: never cached */
  if (u.pathname.endsWith("/orders.js")) return;
  e.respondWith(
    fetch(e.request)
      .then((r) => {
        if (r.ok) {
          const copy = r.clone();
          caches.open(CACHE).then((c) => c.put(e.request, copy));
        }
        return r;
      })
      .catch(() =>
        caches
          .match(e.request, { ignoreSearch: true })
          .then((r) => r || Response.error()),
      ),
  );
});
