# Controversy Detector — Calibration Findings

## What we tested

We tuned `UMBRAL_CONTROVERSIA`, the similarity threshold that decides when a
review contradicts the user query. The goal was to find the value that separates
contradictory reviews from the rest.

## What we found

The threshold cannot do that — and it is not a matter of finding the right value.
The signal is simply not there.

SBERT similarity measures what a review is *about* (its topic), not whether the
opinion is positive or negative. Two reviews of the same business with opposite
sentiment get almost the same score:

- "Amazing & delicious! Staff was very friendly!" → 0.2932
- "The food was far from a favorite for me."      → 0.2922

Opposite opinions, nearly identical similarity. The model groups them together
because they talk about the same thing.

We also confirmed the scores never go negative. For our test query they all fall
in a narrow positive band (about 0.05 to 0.30). There is no "near -1" region, so
the threshold can only catch reviews with *low relevance* — never reviews that
say the opposite of the query.

## Side observation: the low end is noise, not contradiction

The reviews with the lowest similarity are not contradictions. They are data
problems: corrupted text, reviews attached to the wrong business, and reviews
written in another language. At its low end, the threshold behaves like a
noise / quality filter.

## Decisions

- **For now:** `detector.py` is kept as a noise / off-topic filter. It is good at
  surfacing irrelevant, corrupted, or non-English reviews before they reach the
  recommender.
- **Future version:** real controversy detection needs sentiment, specifically
  ABSA (Aspect-Based Sentiment Analysis). That is planned for a later iteration.
