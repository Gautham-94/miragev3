import { Component, input, model } from '@angular/core';

import { Icon } from '../icon/icon';

export interface FilterOption {
  value: string;
  label: string;
}

@Component({
  selector: 'app-filter-select',
  standalone: true,
  imports: [Icon],
  templateUrl: './filter-select.html',
  styleUrl: './filter-select.scss',
})
export class FilterSelect {
  readonly label = input.required<string>();
  readonly options = input.required<FilterOption[]>();
  readonly value = model<string>('');

  protected onChange(event: Event): void {
    this.value.set((event.target as HTMLSelectElement).value);
  }
}
