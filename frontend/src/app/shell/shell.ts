import { Component } from '@angular/core';
import { RouterLink, RouterLinkActive, RouterOutlet } from '@angular/router';

import { Icon } from '../shared/icon/icon';

@Component({
  selector: 'app-shell',
  standalone: true,
  imports: [RouterOutlet, RouterLink, RouterLinkActive, Icon],
  templateUrl: './shell.html',
  styleUrl: './shell.scss',
})
export class Shell {
  protected readonly navItems = [
    { path: '/review', label: 'Review', icon: 'grid' },
    { path: '/live', label: 'Live', icon: 'video' },
    { path: '/events', label: 'Events', icon: 'target' },
    { path: '/recordings', label: 'Recordings', icon: 'film' },
    { path: '/cameras', label: 'Cameras', icon: 'settings' },
    { path: '/detectors', label: 'Detectors', icon: 'cpu' },
    { path: '/queries', label: 'Queries', icon: 'search' },
  ];
}
