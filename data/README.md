# Data Policy

Raw datasets and processed CSV files are excluded from this repository.

## Sources

### IndicCMix

Used for English, Hinglish, and Punglish language identification.

- Source: `ai4bharat/IndicCMix`
- Used fields:
  - English: `english`
  - Hinglish: Hindi partition `romanized_casual`
  - Punglish: Punjabi partition `romanized_casual`

### EmoHinD-58K

Used for Hindi emotion classification.

- Source: `debadityashome/emohi58k-finegrained-hindi-emotion-datatext`
- Used fields: `text`, `labels`
- Initial baseline: 8-class single-label subset
- Original dataset contains multi-label fine-grained emotion annotations.

### EmoInHindi

Not included in this repository.

- Restricted to research purposes.
- Must not be redistributed without author permission.