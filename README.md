# Proactive AI Interface

Proactive AI Interface is a prototype for an assistant that suggests next actions from text or image input and improves from feedback. The focus is the interaction loop: suggestion, user choice, feedback, and adaptation.

View the AI Studio prototype: https://ai.studio/apps/7c72077a-53e3-4957-b4c7-ce9890d70eac

## What It Explores

- Proactive suggestions instead of reactive chat only.
- Feedback logging as a lightweight learning loop.
- Text and image input as context for recommendations.
- Trust boundaries for systems that act before being asked directly.

## Technical Notes

- React and Vite frontend.
- Gemini API integration through `@google/genai`.
- React Router, React Markdown, and remark-gfm for richer interface flows.
- Motion, Tailwind, and lucide-react for interaction polish.

## Current Status

This is a prototype source repo. A production version would need persistent feedback storage, evaluation metrics, permission controls, privacy settings, and clear limits on what the assistant can suggest or automate.

## Run Locally

Prerequisite: Node.js.

1. Install dependencies:
   `npm install`
2. Create a local environment file based on `.env.example`.
3. Add your own Gemini API key locally.
4. Run the app:
   `npm run dev`

## API Key Boundary

Do not deploy this Vite app with a private Gemini key embedded into browser JavaScript. If deploying outside AI Studio, use a server-side API route or an explicit visitor-provided key flow.

## AI-Assisted Build Note

This prototype was built with AI assistance. The useful work is defining the product loop, identifying trust and privacy boundaries, and separating a compelling demo from a reliable assistant system.

## Related Public Notes

See the combined prototype overview repo: https://github.com/brycejohnson1417/ai-studio-prototype-overviews
