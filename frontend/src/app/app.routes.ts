import { Routes } from '@angular/router';

import { Shell } from './shell/shell';

export const routes: Routes = [
  {
    path: '',
    component: Shell,
    children: [
      { path: '', redirectTo: 'review', pathMatch: 'full' },
      {
        path: 'review',
        loadComponent: () => import('./pages/review/review-page').then((m) => m.ReviewPage),
      },
      {
        path: 'live',
        loadComponent: () => import('./pages/live/live-page').then((m) => m.LivePage),
      },
      {
        path: 'events',
        loadComponent: () => import('./pages/events/events-page').then((m) => m.EventsPage),
      },
      {
        path: 'recordings',
        loadComponent: () => import('./pages/recordings/recordings-page').then((m) => m.RecordingsPage),
      },
      {
        path: 'add-camera',
        loadComponent: () => import('./pages/add-camera/add-camera-page').then((m) => m.AddCameraPage),
      },
      {
        path: 'cameras',
        loadComponent: () => import('./pages/manage-cameras/manage-cameras-page').then((m) => m.ManageCamerasPage),
      },
      {
        path: 'detectors',
        loadComponent: () => import('./pages/manage-detectors/manage-detectors-page').then((m) => m.ManageDetectorsPage),
      },
      {
        path: 'config',
        loadComponent: () => import('./pages/config/config-page').then((m) => m.ConfigPage),
      },
      {
        path: 'logs',
        loadComponent: () => import('./pages/logs/logs-page').then((m) => m.LogsPage),
      },
    ],
  },
];
