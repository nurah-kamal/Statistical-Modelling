import { fromRubric } from './rubric.js';

// An invented rubric to start from, for reviewing written replies to customers.
// The last item is missing one band's guidance on purpose, so the read-out
// shows a problem until it is filled in.
export function sampleForm() {
  return fromRubric({
    name: 'Service desk reply review',
    context: 'A written reply from a service desk agent to a customer query. Judge the reply, not the query.',
    scale: { min: 1, max: 4 },
    items: [
      {
        id: 'accuracy',
        name: 'Accuracy',
        description: 'Whether the reply answers the question that was asked, with correct facts and steps.',
        bands: {
          1: 'Answers a different question, or gives wrong facts or steps.',
          2: 'Partly answers the question; at least one fact or step is wrong or missing.',
          3: 'Answers the question correctly, with a minor gap the customer can work round.',
          4: 'Answers the question fully and correctly, with nothing left to guess.',
        },
      },
      {
        id: 'tone',
        name: 'Tone',
        description: 'Whether the reply is courteous, plain and suited to the customer.',
        bands: {
          1: 'Curt, blaming or full of internal jargon.',
          2: 'Polite but stiff, or with jargon the customer would not know.',
          3: 'Courteous and plain, with the odd stiff phrase.',
          4: 'Warm, plain and suited to this customer throughout.',
        },
      },
      {
        id: 'next_steps',
        name: 'Next steps',
        description: 'Whether the customer knows what happens next and what, if anything, they must do.',
        bands: {
          1: 'No next step, or a next step that cannot be followed.',
          2: '',
          3: 'Clear next steps, but no timeframe.',
          4: 'Clear next steps, who does each, and by when.',
        },
      },
    ],
  });
}
