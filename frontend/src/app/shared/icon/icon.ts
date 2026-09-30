import { Component, input } from '@angular/core';

// A small, self-contained icon set (no external icon font/library dependency) --
// just the handful of glyphs this app actually uses, as inline SVG paths.
const ICONS: Record<string, string> = {
  grid: 'M4 4h7v7H4zM13 4h7v7h-7zM4 13h7v7H4zM13 13h7v7h-7z',
  video: 'M3 6.5A2.5 2.5 0 0 1 5.5 4h7A2.5 2.5 0 0 1 15 6.5v11A2.5 2.5 0 0 1 12.5 20h-7A2.5 2.5 0 0 1 3 17.5zM15 9.5l6-3.2v11.4l-6-3.2',
  target: 'M12 21a9 9 0 1 0 0-18 9 9 0 0 0 0 18ZM12 16a4 4 0 1 0 0-8 4 4 0 0 0 0 8ZM12 12h.01',
  film: 'M3 4h18v16H3zM7 4v16M17 4v16M3 8h4M3 16h4M17 8h4M17 16h4',
  alert: 'M12 3 2 20h20zM12 9v5M12 17h.01',
  camera: 'M3 8.5A2.5 2.5 0 0 1 5.5 6h1.6l1-1.6A1 1 0 0 1 9 4h6a1 1 0 0 1 .9.4L17 6h1.5A2.5 2.5 0 0 1 21 8.5v9A2.5 2.5 0 0 1 18.5 20h-13A2.5 2.5 0 0 1 3 17.5ZM12 16a3.5 3.5 0 1 0 0-7 3.5 3.5 0 0 0 0 7Z',
  clock: 'M12 21a9 9 0 1 0 0-18 9 9 0 0 0 0 18ZM12 7v5l3.5 2',
  play: 'M7 5v14l12-7z',
  x: 'M18 6 6 18M6 6l12 12',
  chevronRight: 'M9 6l6 6-6 6',
  chevronDown: 'M6 9l6 6 6-6',
  wifiOff: 'M3 3l18 18M8.5 16.5a5 5 0 0 1 6 0M5 12a10 10 0 0 1 4-2.3M19 12a10 10 0 0 0-3-2M12 20h.01',
  imageOff: 'M3 3l18 18M8 8l-1 .5A2 2 0 0 0 6 10.3V17a2 2 0 0 0 2 2h9M15 9a2 2 0 0 1 2 2v6M11 5h6a2 2 0 0 1 2 2v6M9.5 13.5 12 11l2 2',
  search: 'M11 19a8 8 0 1 0 0-16 8 8 0 0 0 0 16ZM21 21l-4.35-4.35',
  checkCircle: 'M12 21a9 9 0 1 0 0-18 9 9 0 0 0 0 18ZM8 12l3 3 5-6',
  wifi: 'M2 8.5a16 16 0 0 1 20 0M5.5 12a11 11 0 0 1 13 0M9 15.5a6 6 0 0 1 6 0M12 19h.01',
  keyboard: 'M3 6.5A1.5 1.5 0 0 1 4.5 5h15A1.5 1.5 0 0 1 21 6.5v11a1.5 1.5 0 0 1-1.5 1.5h-15A1.5 1.5 0 0 1 3 17.5ZM6.5 9h.01M9.5 9h.01M12.5 9h.01M15.5 9h.01M17.5 9h.01M6.5 12h.01M9.5 12h.01M12.5 12h.01M15.5 12h.01M17.5 12h.01M7 15.5h10',
  plus: 'M12 5v14M5 12h14',
  arrowLeft: 'M19 12H5M12 19l-7-7 7-7',
  settings: 'M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 0 0 2.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 0 0 1.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 0 0-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 0 0-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 0 0-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 0 0-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 0 0 1.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065ZM12 15a3 3 0 1 0 0-6 3 3 0 0 0 0 6Z',
  trash: 'M3 6h18M8 6V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2m3 0-1 14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2L4 6h16Z',
  pencil: 'M12 20h9M16.5 3.5a2.121 2.121 0 0 1 3 3L7 19l-4 1 1-4Z',
  download: 'M12 3v12m0 0 4.5-4.5M12 15l-4.5-4.5M4 17v2a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-2',
  cpu: 'M9 3v3M15 3v3M9 18v3M15 18v3M3 9h3M3 15h3M18 9h3M18 15h3M7.5 7.5h9v9h-9zM10 10h4v4h-4z',
  pause: 'M7 5h4v14H7zM13 5h4v14h-4z',
  list: 'M8 6h13M8 12h13M8 18h13M3 6h.01M3 12h.01M3 18h.01',
  power: 'M12 2v10M18.36 6.64a9 9 0 1 1-12.73 0',
  maximize: 'M8 3H5a2 2 0 0 0-2 2v3M16 3h3a2 2 0 0 1 2 2v3M21 16v3a2 2 0 0 1-2 2h-3M3 16v3a2 2 0 0 0 2 2h3',
};

@Component({
  selector: 'app-icon',
  standalone: true,
  template: `
    <svg
      [attr.width]="size()"
      [attr.height]="size()"
      viewBox="0 0 24 24"
      fill="none"
      [attr.stroke]="'currentColor'"
      stroke-width="1.75"
      stroke-linecap="round"
      stroke-linejoin="round"
    >
      <path [attr.d]="path()" />
    </svg>
  `,
  styles: `
    :host {
      display: inline-flex;
      line-height: 0;
    }
  `,
})
export class Icon {
  readonly name = input.required<string>();
  readonly size = input<number>(18);

  protected path(): string {
    return ICONS[this.name()] ?? '';
  }
}
