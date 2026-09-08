export type ChatRole = "user" | "assistant" | "error";

export interface ChatEntry {
  id: string;
  role: ChatRole;
  content: string;
  timestamp: string;
  topK?: number;
  raw?: unknown;
}

export interface TimeseriesBucket {
  bucket: string;
  transactions: number;
  fraud: number;
  normal: number;
}

export interface TimeseriesResponse {
  data: TimeseriesBucket[];
  total_transactions: number;
  total_fraud: number;
  total_normal: number;
}

export interface TransactionRecord {
  transaction_id: string;
  event_timestamp: string;
  amount: number;
  is_fraud: boolean;
  fraud_probability: number | null;
  status: string | null;
  data_source: string;
}

export interface TransactionListResponse {
  data: TransactionRecord[];
  total: number;
}

export interface StatusCountsResponse {
  received: number;
  flagged: number;
  embedding: number;
  embedded: number;
}

export interface ServiceHealth {
  name: string;
  label: string;
  status: "up" | "down";
}

export interface ServiceHealthResponse {
  services: ServiceHealth[];
}
