# IndicPulse Intent Annotation Guide — v0.1

## Goal

Identify the communicative purpose of everyday English,
Hindi/Hinglish, and Punjabi/Punglish messages.

These are provisional categories. We will test them on a
small manually reviewed sample before building a dataset.

## Labels

### Question
The speaker asks for information, clarification, or confirmation.

Example: "Kal exam kinne baje aa?"
Example: "What time does the exam start?"

A question mark alone does not determine this label.

### Request
The speaker asks someone to perform an action.

Example: "Please notes bhej do."
Example: "Can you send me the notes?"

A request phrased as a question is still a Request.
Do not add Question merely because it uses question grammar.

### Complaint
The speaker expresses dissatisfaction about a problem,
failure, inconvenience, or unmet expectation.

Example: "Yaar website fer crash ho gayi."
Example: "My order is three days late. This is frustrating."

A neutral report of an event is not automatically a Complaint.

### Opinion
The speaker expresses an evaluation, preference, or judgment.

Example: "Eh movie bahut vadhiya si."
Example: "I think this design looks better."

An opinion does not automatically imply a particular emotion.

### Other
The message has a clear purpose outside the four categories.

Example: "Good morning!"
Example: "The exam starts at nine."

Do not use Other as a substitute for uncertainty.
Do not combine Other with another intent label.

## Overlap Rules

Assign multiple labels only when the message independently
expresses multiple purposes.

"Website crashed again. Please fix it."
Labels: Complaint, Request

"Why has my order still not arrived? This is unacceptable."
Labels: Question, Complaint

"Can you send the notes?"
Labels: Request

"This app is terrible."
Labels: Complaint, Opinion

## Complaint versus Opinion

Use Complaint when the message expresses dissatisfaction
about a problem or negative experience.

Add Opinion only when it also explicitly evaluates something,
such as calling an app terrible or a design attractive.

Examples:
"The website crashed again. Please fix it."
→ Complaint, Request

"This app is terrible."
→ Complaint, Opinion

"This app is terrible. Please fix it."
→ Complaint, Opinion, Request

## Uncertainty

If the meaning is unclear or requires missing context:
- Set review_needed to true.
- Explain the uncertainty in notes.
- Leave intents empty if no label can be assigned confidently.

## Language Rules

Assign the label by the primary language carrying the message. English is
primarily English; Hinglish is primarily Hindi, including Romanized Hindi;
Punglish is primarily Punjabi, including Romanized Punjabi. English words may
appear in either Hindi or Punjabi messages. If Hindi and Punjabi are both
substantial and neither is clearly primary, flag the language as uncertain for
review. Script alone does not determine the language.

## Pilot Record Format

text
language
intents
review_needed
notes
source

## Provenance

Record whether each message is:
- independently human-authored,
- consented contributed text, or
- synthetic.

Do not describe invented examples as naturally occurring messages.
