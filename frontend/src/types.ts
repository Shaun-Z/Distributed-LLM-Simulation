export type Draft = {
  label: string;
  hardware: string;
  workload: string;
  placement: string;
  config: string;
};
export type Section = "hardware" | "workload" | "placement" | "config";
export type Edit = {
  section: Section;
  path: (string | number)[];
  value: string;
  kind?: "number" | "integer" | "string" | "json" | "boolean";
  unit?: string;
};
export type Resource = {
  id: string;
  kind: string;
  unit: string;
  capacity: number | string;
};
export type HardwareNode = {
  id: string;
  kind: string;
  host_id?: string;
  device_type?: string;
  capacity_resource_id?: string;
  physical_pool_id?: string;
  accessible_pool_ids: string[];
  kernel_profile_ids: string[];
};
export type Channel = {
  id: string;
  src: string;
  dst: string;
  kind: string;
  payload_Bps: number;
  latency_s: number;
  setup_s: number;
  resource_demands: [string, number][];
};
export type Operator = {
  control_predecessors: string[];
  id: string;
  op_type: string;
  inputs: string[];
  outputs: string[];
  request_id?: string;
  input_dtype: string;
  accumulate_dtype: string;
  output_dtype: string;
};
export type Tensor = {
  id: string;
  producer_id?: string;
  consumer_ids: string[];
  storage_bytes: number | string;
  dtype: string;
};
export type Activity = {
  id: string;
  kind: string;
  op_id?: string;
  tensor_id?: string;
  request_id?: string;
  predecessor_ids: string[];
  input_replica_ids: string[];
  output_replica_ids: string[];
  slot_demands: [string, number][];
  rate_demands: [string, number][];
  setup_s: number;
  total_work: number;
};
export type Specs = {
  hardware: {
    name: string;
    synthetic: boolean;
    nodes: HardwareNode[];
    channels: Channel[];
    resources: Resource[];
    kernel_profiles: {
      id: string;
      op_type: string;
      input_dtype: string;
      accumulate_dtype: string;
      output_dtype: string;
    }[];
  };
  workload: {
    name: string;
    operators: Operator[];
    tensors: Tensor[];
    requests: { id: string; arrival_s: number }[];
  };
  placement: { op_compute: [string, string][]; op_kernel: [string, string][] };
  config: {
    scheduler: string;
    network_mode: string;
    seed: number;
    max_events: number;
    max_sim_time: number;
    trace_level: string;
  };
  execution?: {
    activities: Activity[];
    replicas: { id: string; tensor_id: string; pool_id: string }[];
  };
};
export type Preview = {
  draft: Draft;
  specs: Specs;
  counts: Record<string, number>;
  hashes: Record<string, string>;
  execution: NonNullable<Specs["execution"]>;
};
export type Template = {
  id: string;
  title: string;
  description: string;
  profile: string;
  parameters: Record<string, unknown>;
};
export type Catalog = {
  profiles: {
    id: string;
    name: string;
    synthetic: boolean;
    devices: { id: string; type: string; pools: string[] }[];
  }[];
  templates: Template[];
};
export type Metrics = {
  peak_memory_bytes?: Record<string, number | string>;
  bytes_transferred?: number;
  throughput_requests_per_s?: number;
  p50_latency_s?: number;
  p95_latency_s?: number;
  p99_latency_s?: number;
  completed_requests?: number;
  unfinished_requests?: number;
  compute_transfer_overlap_s?: number;
  [key: string]: unknown;
};
export type Run = {
  id: string;
  directory: string;
  label: string;
  workload: string;
  status: string;
  makespan_s?: number;
  metrics: Metrics;
  model_level?: string;
  scheduler?: string;
  network_mode?: string;
  modified_at: number;
};
export type Job = {
  id: string;
  label: string;
  state: string;
  created_at: string;
  started_at?: string;
  finished_at?: string;
  result_status?: string;
  error?: string;
  run_directory: string;
};
export type Detail = {
  id: string;
  directory: string;
  summary: {
    status?: string;
    makespan_s?: number;
    finish_time_s?: number;
    failure_reason?: string;
    metrics: Metrics;
    assumptions?: string[];
  };
  manifest: {
    model_level?: string;
    synthetic_hardware?: boolean;
    seed?: number;
    hashes?: Record<string, string>;
    [key: string]: unknown;
  };
  critical_path?: {
    activity_ids: string[];
    edges: unknown[];
    accounted_path_s: number;
    limitations: string;
  };
  bounds?: unknown;
  llm_metrics?: {
    requests: {
      request_id: string;
      ttft_s?: number;
      tpot_s?: number;
      itl_s: number[];
      delivered_tokens: number;
    }[];
    output_tokens_per_s: number;
  };
  failure?: unknown;
  invalid?: unknown;
  resources: Record<string, string>[];
  requests: Record<string, string>[];
  artifacts: string[];
};
export type TimelineRow = {
  activity_id: string;
  kind: string;
  op_id?: string;
  tensor_id?: string;
  request_id?: string;
  start_s: number | null;
  finish_s: number | null;
  setup_s: number;
  state: string;
};
export type Series = {
  series: {
    kind: "memory" | "service" | "capacity";
    id: string;
    unit: string;
    capacity?: number | string;
    points: [number, number][];
  }[];
  memory_available: boolean;
  rates_available: boolean;
  reason?: string;
  downsampled: boolean;
};
export type TraceRow = {
  sim_time: number;
  seq: number;
  event_type: string;
  activity_id?: string;
  request_id?: string;
  reason?: unknown;
  [key: string]: unknown;
};
export type TracePage = {
  rows: TraceRow[];
  total: number;
  offset: number;
  limit: number;
};
