# The essays and their human scores

`sample.json` holds 30 essays from the **ELLIPSE corpus** (English Language Learner Insight, Proficiency and Skills Evaluation), with the scores two trained raters gave each one.

| | |
| --- | --- |
| **Source** | https://github.com/scrosseye/ELLIPSE-Corpus, files `ELLIPSE_Final_github_train.csv` and `ellipsis_raw_rater_scores_anon_all_essay.zip`, downloaded 25 October 2026 |
| **Licence** | [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/), as stated in the corpus README. The essays, their scores and the adapted rubric in [../rubrics/ellipse.yaml](../rubrics/ellipse.yaml) stay under that licence; the code in this repository is MIT. |
| **Citation** | Crossley, S. A., Tian, Y., Baffour, P., Franklin, A., Kim, Y., Morris, W., Benner, B., Picou, A., & Boser, U. (2023). Measuring second language proficiency using the English Language Learner Insight, Proficiency and Skills Evaluation (ELLIPSE) Corpus. *International Journal of Learner Corpus Research*, 9(2), 248–269. |
| **Changes** | A sample of 30 essays. Demographic columns dropped. Line endings normalised. The essay text is otherwise unchanged. |

## What the corpus is

About 6,500 essays written by English language learners in grades 8 to 12, each scored 1 to 5 on six analytic items: cohesion, syntax, vocabulary, phraseology, grammar and conventions. Two raters scored each essay on their own. The published score is normally the mean of the two, so it moves in half points.

## How the sample was drawn

`scripts/make-sample.js` reads the two downloaded files and keeps an essay only if:

- it is 150 to 450 words long, to keep each request small;
- it holds none of the placeholders the corpus put where a name, school or place was removed (`Generic_Name`, `STUDENT_NAME` and so on). An essay that needed one is left out in case another name slipped through;
- neither rater flagged it as holding identifying information.

That leaves 2,103 essays. From those it takes six at random from each of five overall-score bands (2 or below, 2.5, 3, 3.5, 4 or above), with a fixed seed, so the sample covers weak and strong writing rather than mostly the middle. Running the script again on the same files gives the same 30 essays.

## What was taken out

The corpus also records each writer's gender, grade, race and ethnicity, and economic status. None of that is in `sample.json`. Each essay keeps only its corpus ID, the task it answered, its text and its scores. The 30 essays were also read for names: the only ones present are public figures the essays quote (Albert Einstein, Thomas Jefferson, Ralph Waldo Emerson, Albert Schweitzer).

## One thing to know about the scores

In 4 of the 180 item scores the two raters were two points apart, and the published score is not their mean (for example raters 4 and 2, published 2). The corpus does not say how those were settled. The tests check that every published score sits between the two raters, and equals their mean whenever they were within a point of each other.
