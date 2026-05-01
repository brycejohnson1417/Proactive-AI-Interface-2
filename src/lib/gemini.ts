import { GoogleGenAI, Type } from '@google/genai';

// Initialize the Gemini API client
// The API key is automatically injected by the AI Studio environment
const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });

const TEXT_MODELS = [
  'gemini-3-flash-preview', 
  'gemini-3.1-flash-lite-preview',
  'gemini-2.5-flash',
  'gemini-2.5-pro'
];

async function withRetry<T>(fn: () => Promise<T>, retries = 3, delay = 1000): Promise<T> {
  try {
    return await fn();
  } catch (error: any) {
    if (retries > 0 && error?.status === 'RESOURCE_EXHAUSTED') {
      console.warn(`Rate limit exceeded, retrying in ${delay}ms... (${retries} retries left)`);
      await new Promise(resolve => setTimeout(resolve, delay));
      return withRetry(fn, retries - 1, delay * 2);
    }
    throw error;
  }
}

async function callWithFallback<T>(
  models: string[],
  fn: (model: string) => Promise<T>,
  delay = 2000
): Promise<T> {
  for (let i = 0; i < models.length; i++) {
    try {
      return await fn(models[i]);
    } catch (error: any) {
      if (error?.status === 403 || error?.message?.includes('permission denied')) {
        throw new Error("PERMISSION_DENIED");
      }
      if (error?.status === 'RESOURCE_EXHAUSTED' && i < models.length - 1) {
        console.warn(`Rate limit exceeded for ${models[i]}, retrying with ${models[i+1]} in ${delay}ms...`);
        await new Promise(resolve => setTimeout(resolve, delay));
        continue;
      }
      throw error;
    }
  }
  throw new Error("All models failed due to quota exhaustion.");
}

export async function getEmbedding(text: string): Promise<number[]> {
  try {
    const response = await withRetry(() => ai.models.embedContent({
      model: 'gemini-embedding-2-preview',
      contents: text,
    }));
    return response.embeddings[0].values;
  } catch (error) {
    console.error("Error getting embedding:", error);
    return [];
  }
}

export async function generateSuggestions(
  text: string,
  imageBase64: string | null,
  pastLogsContext: string,
  suggestionCount: number = 3
) {
  const parts: any[] = [];
  if (text) {
    parts.push({ text: `User Input Text: ${text}` });
  }
  if (imageBase64) {
    const mimeType = imageBase64.split(';')[0].split(':')[1];
    const base64Data = imageBase64.split(',')[1];
    parts.push({
      inlineData: {
        data: base64Data,
        mimeType: mimeType,
      },
    });
  }

  const prompt = `
You are an intelligent proactive assistant. Analyze the user's input (which may be text, an image, or both).
1. Classify the input (extract key entities, nouns, and determine the general category).
${suggestionCount > 0 ? `2. Generate exactly ${suggestionCount} distinct, highly relevant suggested actions the user might want to take with this input. Make them concise (e.g., "Learn more about Steve Jobs", "Transcribe video", "Summarize article").\n3. Consider the provided "Past Relevant Interactions" to improve your suggestions. If past interactions show the user preferred a specific type of action for similar inputs, lean towards that.` : ''}

Past Relevant Interactions (JSON format, showing past inputs and the actions the user actually chose):
${pastLogsContext || "[]"}
`;

  parts.push({ text: prompt });

  const properties: any = {
    classification: {
      type: Type.OBJECT,
      properties: {
        entities: { type: Type.ARRAY, items: { type: Type.STRING } },
        category: { type: Type.STRING },
      },
      required: ['entities', 'category'],
    }
  };
  const required = ['classification'];

  if (suggestionCount > 0) {
    properties.suggestions = {
      type: Type.ARRAY,
      items: { type: Type.STRING },
      description: `Exactly ${suggestionCount} concise suggested actions.`,
    };
    required.push('suggestions');
  }

  try {
    const response = await callWithFallback(TEXT_MODELS, (model) => ai.models.generateContent({
      model: model,
      contents: { parts },
      config: {
        systemInstruction: "You are a helpful assistant. Keep your response concise and strictly follow the JSON schema.",
        responseMimeType: 'application/json',
        responseSchema: {
          type: Type.OBJECT,
          properties,
          required,
        },
      },
    }));

    if (!response.text) {
      throw new Error("No response text from Gemini");
    }

    return JSON.parse(response.text);
  } catch (error) {
    console.error("Error generating suggestions:", error);
    throw error;
  }
}

export async function executeAction(
  action: string,
  text: string,
  imageBase64: string | null
) {
  const parts: any[] = [];
  if (text) {
    parts.push({ text: `User Input:\n${text}` });
  }
  if (imageBase64) {
    const mimeType = imageBase64.split(';')[0].split(':')[1];
    const base64Data = imageBase64.split(',')[1];
    parts.push({
      inlineData: {
        data: base64Data,
        mimeType: mimeType,
      },
    });
  }
  
  parts.push({ text: `\n\nAction to execute: "${action}"\n\nPlease perform this action on the user input and provide the result. Format the output nicely using Markdown.` });

  try {
    const response = await callWithFallback(TEXT_MODELS, (model) => ai.models.generateContent({
      model: model,
      contents: { parts },
      config: {
        systemInstruction: "You are a helpful assistant. Keep your response concise and within 2000 tokens.",
      },
    }));

    return response.text;
  } catch (error) {
    console.error("Error executing action:", error);
    throw error;
  }
}
