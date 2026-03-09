self.addEventListener('push', event => {
  const data = event.data ? event.data.json() : { title: 'PotholeAI', body: 'New alert!' };
  
  event.waitUntil(
    self.registration.showNotification(data.title, {
      body: data.body,
      icon: '/static/detection.png',
      badge: '/static/logo.png',
      vibrate: [100, 50, 100],
      actions: [{ action: 'view', title: 'View Pothole' }]
    })
  );
});

self.addEventListener('notificationclick', event => {
  event.notification.close();
  event.waitUntil(clients.openWindow('/dashboard'));
});
