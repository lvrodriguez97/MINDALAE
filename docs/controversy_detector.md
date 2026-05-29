Controversy Detector
Problem
MINDALAE uses SBERT to find the reviews most similar to the user query. This is good for matching, but it leaves a blind spot.
A single business can have very mixed experiences about the same thing — some reviews saying it was very good, others saying the opposite. Since comparar.py only keeps the reviews most similar to the query, it takes the positive side of that pair and ignores the contradiction.
In practice that contradiction means the business is inconsistent, and the user should know about it before receiving the recommendation.
What counts as controversy
The detector does not care about negative reviews in general. A review that is simply unrelated to the query (similarity near 0) is ignored. Only reviews that talk about the same topic as the query but in the opposite direction (similarity near -1) matter.
So the difference is: irrelevant reviews are ignored, contradictory reviews trigger the alarm.
Inputs
detector.py imports:

the SBERT model
the array of review vectors

And receives:

the user query
the business_ids of the Top 10 candidates returned by SBERT

How it works

Take the business_ids of the Top 10.
For each business_id, get all of its reviews.
Vectorize those reviews with SBERT.
Compute the similarity of each review against the query.
Keep the reviews whose similarity is negative enough to pass the threshold.
If a business has at least one of those reviews, raise the alarm for that business.

Output
For each business that triggers the alarm, the detector returns:

the business_id
the specific review that triggered it

Calibrable parameter
The similarity threshold decides when a negative score is a real contradiction and not just a review talking about something else.
It starts as a tentative value and will be adjusted with real data, because reviews written in natural language almost never reach a perfect -1.
Where it lives
The detector is its own module, detector.py, separate from the existing pipeline:
cargar_datos.py, limpiar.py, vectorizar.py, comparar.py, prompt.py, app.py.