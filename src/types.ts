export type LogEntry = {
  id: string;
  timestamp: number;
  input: {
    text: string;
    image: string | null; // base64
  };
  classification: {
    entities: string[];
    category: string;
  };
  suggestionsOffered: string[];
  actionTaken: string;
  generalizedAction?: string; // Sanitized version of the action (e.g. "Summarize {entity}")
  isCustomAction: boolean;
  scores: Record<string, number>;
  result: string;
  embedding: number[];
  excludedTags?: string[];
};

export interface Rule {
  id: string;
  term: string;
  taxonomy: 'category' | 'entity';
  isExactMatch: boolean;
  threshold: number; // 1-100
  suggestions: string[];
  termEmbedding?: number[];
}
