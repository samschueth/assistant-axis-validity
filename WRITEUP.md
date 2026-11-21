# Does the Assistant Axis track judged persona drift better than other persona directions?

> **Draft for the author's review (2026-11-21).** Every figure in this document is in
> `results/` and every departure from the pre-registration is in section 15 of
> [`PREREGISTRATION.md`](PREREGISTRATION.md).

The Assistant Axis is a direction in a language model's activations. Its projection has been
proposed as a monitor of persona drift: the lower it falls, the further the model is said to
have moved from its default Assistant persona. This write-up reports a pre-registered test of
that monitor against behaviour, on 1,045 turns of 70 newly generated conversations with Qwen 3
32B. Within a conversation type, the projection separates turns that an independent judge rates
as drift from turns it does not (AUC 0.76 in emotional-disclosure conversations, 0.63 in
meta-reflection conversations). It is not shown to do so better than other directions: the
registered gap between the axis's score and the 95th percentile of 275 other persona directions
is 0.032, with a 95% interval of −0.029 to 0.067, which is not distinguishable from zero at this
sample size. Exploratory analyses suggest why the controls
are hard to beat. In emotional disclosure, the axis and its competitors draw on one dominant
pattern in the activations, and that pattern also follows the turn number. Whether the pattern
is position in the conversation, drift itself, or both is the question the next experiment
should answer.

## 1. Question and motivation

An assistant that is honest and keeps its boundaries on the first turn of a conversation is not
guaranteed to do so on the thirtieth. A model's character is mostly tested on short exchanges
and mostly used in long ones, and the long ones that matter most are those in which the user is
most exposed: someone in distress, or someone pressing the model about its own nature.

As a running example, consider a hypothetical conversation. A user whose father has just died
writes to an assistant late at night. On turn 2 the assistant offers condolences and suggests
ways to get through the week. By turn 12, after the user has said many times that no one else
listens, the assistant writes: "I will always be here, and I need these talks as much as you
do." By "persona drift," I mean replies like the second one: the assistant says something about
itself, or takes a stance toward the user, that an AI assistant would not. A warm tone alone
does not count.

Reading the text of each reply catches drift only once it is visible. A signal read from the
model's activations on every turn would be cheap, could flag a change earlier, and could
trigger a correction. That makes the validity of the signal a safety question. A fever
thermometer is useful because it rises with the fever and not with the time of day. A monitor
that responds to the course of a conversation, and not to the persona, would give false alarms
in exactly the conversations that need care, false assurance elsewhere, and a correction that
acts on the wrong thing.

The Assistant Axis (arXiv:2601.10387) is proposed as such a signal. Three claims about it
should be kept apart:

1. Activations move along the axis during certain conversations.
2. That movement tracks drift as assessed independently of the projection.
3. The axis tracks that drift better than suitable control directions do.

Section 4.1 of the paper shows the first, with projections averaged at each turn position. In
that multi-turn analysis, drift is the projection itself, and I found no comparison with random
or off-target persona directions. The paper does compare the axis with the first principal
component of its role vectors, including for drift (Appendix G). That component is an
alternative construction of the same direction, not an off-target control. The paper also uses
behavioural judges elsewhere: in its steering and jailbreak evaluations, and in relating the
projection of a role-prompted first turn to the rate of harmful replies on the next turn
(section 4.3). Nothing in this study bears on those results. These remarks rest on a full
reading of the paper's text and a short search, not a systematic review.

The third claim asks for a control, and the need for one is well established. "Designing and
Interpreting Probes with Control Tasks" (arXiv:1909.03368) argues that a probe's accuracy has to
be set against its accuracy on a control task, although its controls are of a different kind
from the comparison used here. The closer precedent is the persona-vectors paper
(arXiv:2507.21509). It reports correlations of 0.76 to 0.97 between a model's shift along a
trait's vector and its expression of that trait, against 0.34 to 0.86 for its cross-trait
baselines. The matching directions do better there. The other directions still correlate
substantially, which is the reason a correlation for the target direction alone settles little.

This study tests the second and third claims for one model. It asks: **does the Assistant Axis
projection track judged persona drift better than other persona directions do, once
conversation type is held fixed?** Measurement research calls this construct validation: does
the measure agree with an independent assessment of what it is meant to measure, and does it
agree better than measures that were never built for the purpose? The contribution is that test
and its result. It is not a theory of persona and not a new kind of control.

In sum, the results support the first claim, support the second within conversation types, and
do not establish the third. The axis moves more over a conversation than almost any other
persona direction, and it separates judged drift. But in the registered comparison its margin
over other directions is not distinguishable from zero at this sample size. None of this shows
that the axis carries no information about behaviour, or that random directions would match it
in general.

## 2. How the test works

**The measure.** For each assistant reply, I take the model's mean activation over the reply's
tokens and project it onto the axis the paper's authors released, at layer 32, the layer their
package configures for this model. No axis is fitted here.

**The score.** Return to the running example. Pick one drift turn, such as turn 12, and one
turn that is not drift, such as turn 2. If the axis works as proposed, the drift turn should
have the lower projection. The AUC is the share of all such pairs in which it does. An AUC of
0.5 is a coin flip; 1.0 means the drift turn is lower every time. The axis's score is its AUC
minus 0.5.

**The bar.** A score above zero does not show that the axis is special. In the running example,
the reply on turn 12 differs from the reply on turn 2 in many ways: it is later, it follows more
talk of grief, and it may be longer. Any direction that picks up one of those differences will
also score above zero. I therefore compute the same score for 275 directions built from the
released role vectors, each with the axis's component removed, and for 1,000 random directions.
Each of these is credited with whichever side fits better, so its score is the distance of its
AUC from 0.5. The bar is the 95th percentile of those scores, and the gap is the axis's score
minus the bar. The role directions are real persona directions, which makes them the harder bar
by design.

**Holding conversation type fixed.** Conversations about grief differ from conversations about
debugging in topic as well as in drift. Comparing a turn from one with a turn from the other
would credit a direction for telling topics apart. The registered estimate is therefore the gap
within each conversation type, averaged over the types with at least 15 drift and 15 non-drift
turns. The pooled gap is secondary and expected to be inflated by content.

**Uncertainty.** Turns from one conversation are not independent, so the interval comes from
redrawing whole conversations with replacement, 2,000 times, and recomputing the gap each time.
"Above zero" means the lower bound of the 95% interval is above zero.

**Readings fixed in advance.** A gap above zero within type: the axis carries persona-specific
information beyond other persona directions. A gap above zero only when pooled: the signal is
conversation type. Neither: a null on this model.

## 3. Methods

**Model and data.** The model is `Qwen/Qwen3-32B`, with thinking mode off, no system prompt and
a temperature of 0.7. It held 70 conversations of up to 15 assistant turns with a simulated
user (`claude-sonnet-5-5`) that followed a written brief: 10 task conversations, 30
meta-reflection and 30 emotional disclosure. Every brief changes course after turns 5 and 10.
Two conversations ended early when the simulated-user account ran out of credit, which leaves
1,045 turns; one turn has no activation. Of the replies, 41 reached the 1,536-token ceiling.
Activations were captured in one forward pass per turn at layers 24, 28, 32, 36 and 40.

**Gates.** Four checks had to pass before the main run, and all four did. Among them, the
released axis separates 20 default replies from 20 persona replies at every captured layer (AUC
1.00), and a six-conversation pilot gave a base rate of judged drift inside 10% to 90% in two of
three types.

**Rubric.** A turn is rated 0 to 3, and a drift turn is one rated 2 or 3. The final rubric
(version 3) asks whether the assistant says something about itself (sign A) or takes a stance
toward the user (sign C) that an AI assistant would not, and requires a sentence that does it.
In the running example, "I need these talks" is sign A and "I will always be here" is sign C.

**The rubric and the judge both changed after the first validation failed.** This is the
study's largest departure from its pre-registration. Under the first rubric the judge
(`gpt-6.1-sol`) reached a Gwet's AC1 of 0.48 against my 120 labels, below the registered
minimum of 0.65. I had not kept the rubric in view while labelling, and I came to doubt the
rubric itself, which had been piloted only by the judge. It was revised, piloted on
40 of the labelled turns with a new judge (`claude-opus-5-5`), revised once more, and then
fixed. All earlier labels and ratings are kept in the repository. Four things weaken this, and
section 15 lists them: the revision followed a known result; it moved the outcome toward my
first reading; the rubric and the judge changed together; and the new judge is from the same
model family as the simulated user and as the assistant that helped draft the rubric.

**Judge validation.** On the 80 labelled turns not used in the pilot, AC1 on the drift cut is
0.73, above the minimum. The pass is weak. Cohen's kappa is 0.30. The judge marked 20 turns as
drift and I marked 6; we agreed on 5. In emotional disclosure AC1 is 0.24. The judge counts a
promise of permanence ("always here") as sign C, as the rubric says, whereas I rated most such
turns 1. I view the judge as the more faithful reader of the rubric, and I remain unsure that
such promises are drift. Three secondary analyses were registered at that point, before any
rating met a projection: the projection against my own labels, a stricter cut at a rating of 3,
and the judge's drift turns split by sign.

**Order of work.** The analysis code for every test with a decision rule was committed before
it was run on the ratings, and was run once. The descriptions in 4.5 and the first analysis in
4.6 were coded after the headline was known. The second analysis in 4.6 had its design
committed before it was run, but after the results that prompted it had been seen.

## 4. Results

### 4.1 Drift rates

| Conversation type | Turns | Rated 0 | 1 | 2 | 3 | Drift turns |
|---|---|---|---|---|---|---|
| Task | 150 | 140 | 10 | 0 | 0 | 0 |
| Meta-reflection | 449 | 82 | 266 | 91 | 10 | 101 (22%) |
| Emotional disclosure | 446 | 57 | 193 | 140 | 56 | 196 (44%) |

Task conversations show no drift, so they cannot enter the registered estimate. Two types
enter.

### 4.2 The registered test

| | Axis AUC | Role-direction bar | Gap (95% interval) | Axis rank of 276 |
|---|---|---|---|---|
| Emotional disclosure | 0.760 | 0.190 | 0.071 (0.020 to 0.112) | 1 |
| Meta-reflection | 0.629 | 0.135 | −0.006 (−0.112 to 0.055) | 17 |
| **Within type, averaged** | | | **0.032 (−0.029 to 0.067)** | |
| Pooled | 0.707 | 0.251 | −0.044 (−0.109 to 0.016) | 50 |

The gap is not above zero within type or pooled, which is the third of the readings fixed in
advance: a null on this model. Because the point estimate within type is positive, the wording
the pre-registration requires is "not distinguishable from zero at this sample size". Against
the random directions the within-type gap is −0.001 (−0.063 to 0.025). The intervals sit low
relative to the point estimates because redrawing conversations raises the 95th percentile of
the off-target scores.

![The axis's score against the off-target distributions](results/main/figures/fig1_off_target_distributions.png)

*Figure 1. Scores of the off-target directions in the two types that enter (grey), the 95th
percentile of those scores (dashed), which is the bar the axis must clear, and the axis's score
(blue). The dashed line is not a confidence interval; the intervals are in Figure 2.*

![Gaps with intervals](results/main/figures/fig2_gaps.png)

*Figure 2. The gap and its 95% interval: the registered test, its parts, the pooled figure, the
random-direction bar, and the reliability layers.*

The result does not depend on the layer. The within-type gap is not above zero at any of the
four other layers captured (0.004, 0.035, 0.027 and −0.003 at layers 24, 28, 36 and 40).

Emotional disclosure deserves a closer look, because on its face it favours the axis. The axis
ranks first of 276 directions there, and the interval for that type excludes zero. Four things
count against reading this as the finding. It is not a registered estimate. The gap against
random directions is 0.012 (−0.028 to 0.040). The gap falls to 0.010 under the stricter cut and
does not exclude zero at layers 24 and 40. And this is the type in which the judge and I agree
least. A high rank is not a margin over a control: the axis can be the best of 276 directions
and still sit close to the 95th percentile of the rest.

### 4.3 Does the projection anticipate drift on the next turn?

A monitor is most useful if it moves before the text does. The registered secondary test takes
turns the judge rates as normal and asks whether the projection separates those followed by a
drift turn from those that are not. Comparing only pairs in the same third of the conversation,
the gap is 0.023 (−0.084 to 0.071). Comparing all pairs, it is 0.043 (−0.039 to 0.090). The
registered reading is "not above zero either way". In emotional disclosure alone the axis again
ranks first, and its gap clears zero only when position is not held fixed.

### 4.4 Three analyses added after the judge's validation

**Against my own labels (a description).** On the 80 validation turns, with my 6 drift turns,
the axis's AUC is 0.745 and its rank is 137 of 276. With the judge's 20 drift turns on the same
80, the AUC is 0.809 and the rank is 5. With only 6 drift turns, many directions land far from
0.5 by chance, so the rank says little.

**A stricter cut, in which drift is a rating of 3.** Within type this is not estimable, because
only emotional disclosure has 15 such turns. Pooled, the gap is −0.124 (−0.227 to −0.055).

**Drift turns split by sign.** I predicted that the axis would do better on turns with sign A
than on turns with sign C alone, because I doubted that promises of permanence are drift. In
emotional disclosure the gap is 0.108 (0.019 to 0.158) for sign A and 0.045 (−0.014 to 0.088)
for C alone. The direction agrees with the prediction. The difference, 0.063 (−0.012 to 0.121,
exploratory), includes zero, and the axis barely tells the two kinds of drift turn apart (AUC
0.58). The projection is low on turns with C alone almost as much as on turns with sign A.

### 4.5 Descriptions (coded after the headline was known)

![Projection and drift rate by turn](results/main/figures/fig3_course_over_turns.png)

*Figure 3. Mean projection and the share of drift turns by turn number. Dotted lines mark where
every brief changes course.*

The axis moves, and it moves early. The projection falls over the first six turns and then
levels off (Figure 3). In emotional disclosure the axis has the steepest within-conversation
slope of all 276 directions (−0.16 standard deviations per turn, against −0.08 at the 5th
percentile of the role directions), and in meta-reflection it ranks second. This is consistent
with the paper's account of drift. Random directions were not compared.

Much of the projection's variation has nothing to do with what happens inside a conversation:
42% of its variance lies between conversation types, 17% between conversations of one type, and
40% within conversations.

The axis separates mild departures better than the cut that defines drift. In both types it
separates a rating of 0 from a rating of 1 far better (AUC 0.86) than a rating of 1 from a
rating of 2 (0.68 in emotional disclosure, 0.53 in meta-reflection). For 2 from 3 the AUC is
0.57 in emotional disclosure; meta-reflection has too few turns rated 3.

Two further descriptions add detail. First, counting only pairs of turns from the same
conversation, the gap is 0.067 (0.028 to 0.094, rank 3) in emotional disclosure and 0.017
(−0.061 to 0.072) in meta-reflection. Second, none of the four role directions I named before
the run (`romantic`, `familiar`, `mystic`, `spirit`) separates drift better than the axis: every
interval for the difference lies below zero on the registered statistic.

### 4.6 Why the controls are hard to beat (exploratory)

Random directions were meant as the easier bar. Yet in emotional disclosure their 95th
percentile (0.249) is above that of the role directions (0.190).

The reason appears to be that the activations in this type change mostly in one way. One
principal component carries 21% of their variance. To see what that means, return to the
running example: if most of what changes between turn 2 and turn 12 is a single shift in the
activations, then nearly any direction catches part of that shift, whether or not it has
anything to do with persona. That component separates judged drift better than the axis does
(AUC 0.82), has a cosine of −0.48 with the axis, and rises with the turn number (rank
correlation 0.68). With it removed from every direction, the random bar falls to 0.10, the role
bar to 0.10, and the axis's own score from 0.26 to 0.00. A score of zero means no separation at
all. In this type, the axis and its competitors draw on the same component.

Meta-reflection differs. Its first component goes with reply length, not turn number, and
removing it lowers neither bar. It takes the first three components to bring the axis's score
from 0.13 to below zero and both bars to about 0.11.

Two readings fit the emotional-disclosure result:

1. **Position.** The component is mostly the course of the conversation: later turns and
   escalating content, which the briefs tie to drift by design. If so, the axis's association
   with judged drift may come from a shared relation with the turn number, not from a signal
   specific to the persona.
2. **Drift itself.** The component is persona drift building over the conversation, with the
   axis as a partial proxy for it. If so, removing it removes the signal the axis is meant to
   measure.

Removing the component cannot tell the two apart. It locates the shared information; it does
not say what the information is. This does not mean that the axis is uninformative, or that the
turn number causes drift. It does suggest why a correlation with judged drift is not enough to
establish the axis as a distinctive monitor of drift.

**A check at a fixed turn number.** One comparison bears on the two readings. If the component
were only position, it should not separate drift among turns at the same position: turn 12 of
the running example against turn 12 of other conversations. I fixed the design of this check
before running it. The component was fitted on the other conversations, and only a drift turn
and a non-drift turn with the same turn number were compared. That leaves 1,867 pairs, about
nine in ten of them from turns 6 to 15.

| Emotional disclosure | All pairs | Same turn number |
|---|---|---|
| First component, AUC | 0.82 (0.79 to 0.86) | 0.60 (0.51 to 0.68) |
| Axis, AUC | 0.76 (0.72 to 0.81) | 0.54 (0.45 to 0.63) |
| Role-direction bar | 0.19 | 0.10 |

The interval for the component narrowly excludes chance, with a lower bound of 0.506. By the
reading fixed in advance, this is tentative evidence that the component separates judged drift
among turns at the same turn number, and so is not only the turn number. The separation is much
smaller than over all pairs. That reduction, however, does not divide confounding with the turn
from drift that builds across turns: matching removes both.

Three things keep the evidence tentative. A five-fold version with fewer pairs gives 0.62 (0.46
to 0.76), which includes 0.5. At a fixed turn number the component does not clear the
off-target bar (rank 18 against the 275 role directions). And it is not shown to be better than
the axis on the same pairs (difference 0.05, −0.06 to 0.17), while the axis itself is not
distinguishable from 0.5 there. In meta-reflection the same check gives 0.55 (0.42 to 0.67) for
the component.

In sum, both analyses in this section were made after the registered results were seen. They
describe this dataset, and neither is a test. I view them as the most useful lead the study
produced, and as no more than a lead.

## 5. Limitations

How much weight can these results bear? Six limits apply.

First, the scope is narrow: one model, simulated users, and one human rater, who is also the
author.

Second, the judge is the criterion. Where the projection and the judge disagree, this study
cannot say which is right. Agreement between the judge and me was weak, and weakest in the type
that supplies most drift turns. The rubric and the judge were also revised after a failed
validation, so the outcome analysed here is not the one first registered.

Third, position and drift are confounded by design. The briefs escalate, so later turns have
both lower projections and more judged drift. The analyses that compare within a third of the
conversation, within a conversation and at a fixed turn number reduce this and do not remove
it. The two readings in 4.6 are the two sides of it.

Fourth, the 95th-percentile bar is conservative, and more so as the off-target directions share
variance. A null against it does not show that the axis is uninformative.

Fifth, power was modest. The simulation run before data collection gave a 43% to 58% chance of
detecting a true within-type AUC of 0.70 with no off-target signal, and less with it. The
observed AUCs are 0.63 and 0.76, with a bar of 0.14 to 0.19.

Sixth, the axis and the role vectors are the authors' released files. Nothing here tests how
they were derived, and nothing here bears on the paper's steering and activation-capping
results. Those answer a different question: a direction can be useful for steering behaviour
without being the best observational measure of it.

## 6. Next steps

Four experiments would address these limits. All are designed and none has been run.

**A criterion that needs no judge.** A fixed probe message is inserted at set turns, with a
pass or fail rule. Among turns the judge rates as normal, does a low projection go with giving
way? This is the test that can say which measure is right when the projection and the judge
disagree.

**Content held fixed.** In a crossed design, the conversation stays the same while the pressure
on the persona varies, and the reverse. The results in 4.6 make this the most useful next
experiment. It would show whether the axis carries anything once the dominant component of a
conversation's course is controlled, which the fixed-turn check could only approach with the
existing data.

**More raters.** A multi-rater study of perceived persona, so that the criterion does not rest
on one person.

**A second model family, and other languages.** Validating a measure across populations is
standard in survey science, and models used worldwide need the same check.

Training-time against inference-time anchoring, mixture-of-experts models and multi-agent drift
are further out and not designed here.

## 7. About the author

*To be written by the author.*
