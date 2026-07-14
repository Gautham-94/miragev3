import { Component, OnInit, signal } from '@angular/core';

import { ApiService } from '../../core/services/api.service';
import { Camera } from '../../core/models/api.models';
import { CameraTile } from './camera-tile/camera-tile';

@Component({
  selector: 'app-live-page',
  standalone: true,
  imports: [CameraTile],
  templateUrl: './live-page.html',
  styleUrl: './live-page.scss',
})
export class LivePage implements OnInit {
  protected readonly cameras = signal<Camera[]>([]);
  protected readonly loading = signal(true);

  constructor(private readonly api: ApiService) {}

  ngOnInit(): void {
    this.api.listCameras().subscribe({
      next: (cameras) => {
        this.cameras.set(cameras);
        this.loading.set(false);
      },
      error: () => this.loading.set(false),
    });
  }
}
