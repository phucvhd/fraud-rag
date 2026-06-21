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
