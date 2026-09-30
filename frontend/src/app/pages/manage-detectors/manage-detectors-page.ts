import { Component, OnInit, computed, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';

import { ApiService } from '../../core/services/api.service';
import { Detector, ExecutionProvider } from '../../core/models/api.models';
import { Icon } from '../../shared/icon/icon';

const PROVIDER_LABELS: Record<ExecutionProvider, string> = {
  auto: 'Auto (prefer fastest available)',
  cpu: 'CPU',
  coreml: 'CoreML (Apple Neural Engine / GPU)',
  cuda: 'CUDA (NVIDIA GPU)',
};

@Component({
  selector: 'app-manage-detectors-page',
  standalone: true,
  imports: [FormsModule, Icon],
  templateUrl: './manage-detectors-page.html',
  styleUrl: './manage-detectors-page.scss',
})
export class ManageDetectorsPage implements OnInit {
  protected readonly detectors = signal<Detector[]>([]);
  protected readonly loading = signal(true);
  protected readonly loadError = signal<string | null>(null);

  protected readonly showAddForm = signal(false);
  protected readonly editingDetectorName = signal<string | null>(null);
  protected readonly name = signal('');
  protected readonly modelPath = signal('');
  protected readonly labelmapPath = signal('');
  protected readonly width = signal(320);
  protected readonly height = signal(320);
  protected readonly executionProvider = signal<ExecutionProvider>('cpu');
  protected readonly availableProviders = signal<ExecutionProvider[]>(['cpu']);
  protected readonly backend = signal<string>('onnx_yolov8');
  protected readonly availableBackends = signal<string[]>(['onnx_yolov8']);

  protected readonly saving = signal(false);
  protected readonly saveError = signal<string | null>(null);
  protected readonly justAdded = signal(false);

  protected readonly pendingDelete = signal<string | null>(null);
  protected readonly deleting = signal(false);
  protected readonly deleteError = signal<string | null>(null);
  protected readonly justDeleted = signal(false);

  protected readonly canSubmit = computed(
    () => this.name().trim().length > 0 && this.modelPath().trim().length > 0 && this.labelmapPath().trim().length > 0,
  );

  constructor(private readonly api: ApiService) {}

  ngOnInit(): void {
    this.load();
    this.api.listExecutionProviders().subscribe({
      next: (providers) => this.availableProviders.set(providers),
      // Non-fatal: the form still works with just the always-safe 'cpu' default if this
      // call fails for some reason (e.g. API briefly unreachable).
      error: () => {},
    });
    this.api.listDetectorBackends().subscribe({
      next: (backends) => this.availableBackends.set(backends),
      // Non-fatal: the form still works with just the always-safe 'onnx_yolov8' default
      // if this call fails for some reason (e.g. API briefly unreachable).
      error: () => {},
    });
  }

  protected providerLabel(provider: ExecutionProvider): string {
    return PROVIDER_LABELS[provider] ?? provider;
  }

  private load(): void {
    this.loading.set(true);
    this.api.listDetectors().subscribe({
      next: (detectors) => {
        this.detectors.set(detectors);
        this.loading.set(false);
      },
      error: (err) => {
        this.loadError.set(err?.error?.detail ?? 'Could not load detectors.');
        this.loading.set(false);
      },
    });
  }

  protected openAddForm(): void {
    this.showAddForm.set(true);
    this.editingDetectorName.set(null);
    this.justAdded.set(false);
    this.saveError.set(null);
  }

  protected editDetector(detector: Detector): void {
    this.editingDetectorName.set(detector.name);
    this.name.set(detector.name);
    this.modelPath.set(detector.model_path);
    this.labelmapPath.set(detector.labelmap_path);
    this.width.set(detector.model_width);
    this.height.set(detector.model_height);
    this.executionProvider.set(detector.execution_provider);
    this.backend.set(detector.device);
    this.showAddForm.set(true);
    this.justAdded.set(false);
    this.saveError.set(null);
  }

  protected cancelAddForm(): void {
    this.showAddForm.set(false);
    this.editingDetectorName.set(null);
    this.name.set('');
    this.modelPath.set('');
    this.labelmapPath.set('');
    this.width.set(320);
    this.height.set(320);
    this.executionProvider.set('cpu');
    this.backend.set('onnx_yolov8');
    this.saveError.set(null);
  }

  protected save(): void {
    if (!this.canSubmit()) return;
    this.saving.set(true);
    this.saveError.set(null);

    const payload = {
      name: this.name().trim(),
      model_path: this.modelPath().trim(),
      labelmap_path: this.labelmapPath().trim(),
      width: this.width(),
      height: this.height(),
      execution_provider: this.executionProvider(),
      device: this.backend(),
    };
    const editingName = this.editingDetectorName();
    const request = editingName ? this.api.updateDetector(editingName, payload) : this.api.createDetector(payload);

    request.subscribe({
      next: () => {
        this.saving.set(false);
        this.justAdded.set(true);
        this.cancelAddForm();
        this.showAddForm.set(false);
        this.load();
      },
      error: (err) => {
        this.saving.set(false);
        this.saveError.set(err?.error?.detail ?? `Could not ${editingName ? 'update' : 'add'} this detector.`);
      },
    });
  }

  protected confirmDelete(name: string): void {
    this.pendingDelete.set(name);
    this.deleteError.set(null);
  }

  protected cancelDelete(): void {
    this.pendingDelete.set(null);
  }

  protected deleteDetector(name: string): void {
    this.deleting.set(true);
    this.deleteError.set(null);
    this.api.deleteDetector(name).subscribe({
      next: () => {
        this.deleting.set(false);
        this.pendingDelete.set(null);
        this.justDeleted.set(true);
        this.load();
      },
      error: (err) => {
        this.deleting.set(false);
        this.deleteError.set(err?.error?.detail ?? `Could not delete ${name}.`);
      },
    });
  }
}
