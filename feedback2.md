# Submission Review: Kathak Engineering Project

Reviewed `Kathak-Engineering-Project.docx`, including its embedded diagram, against the implementation on the `apap` branch. The submission document was not modified. This is a review of the proposal, not a new experimental validation of the system.

## Overall assessment

This is a solid submission concept, but a focused revision is recommended before submitting. The title and purpose are clear, the project has substantial technical content, and the numerical design criteria give the school committee something concrete to assess. The main weaknesses are inconsistencies between the proposed capabilities, the diagram, and the tests.

Keep the title: **A Multimodal Kathak Practice Aid Using Machine Learning and Haptic Feedback**.

The goal communicates the relationship to the guru appropriately. One wording correction: **“readable haptic signals” should be “wearable haptic signals.”**

## 1. Distinguish the existing prototype from proposed enhancements

A research plan can describe capabilities intended for development. However, the software section mixes present-tense descriptions with future plans, making it difficult to tell what already works.

The current live code triggers the **thumb motor** when a mudra's rule-agreement score falls below a threshold. It does not yet select the motor corresponding to the detected error. Finger-specific correction is a reasonable project objective, but should be identified as work to complete.

A failed thumb–index contact rule identifies a relationship that needs correction; it does not necessarily establish which finger is wrong.

## 2. Correct the description of footwork and taal analysis

The submission says the taal module “evaluates footwork timing” against the rhythmic cycle. Currently, the code checks **wrist-movement timing against musical beats** and **chakkar endings against sam**, using a user-supplied taal and first-sam time.

Audio-based tatkaar detection is a separate, still-unvalidated component. Presenting footwork-to-taal analysis as a planned extension would be accurate; presenting it as the existing behavior is not.

## 3. Align latency criteria with the tests

The first criterion requires feedback in under one second for **each chunk**, but the test checks only the **average**. An average can pass while many updates exceed the limit. Either define an average target or report the proportion of updates meeting the deadline, together with a high-percentile or maximum delay.

The haptic test needs a precise starting point: when the dancer first forms the incorrect gesture, or when the software detects it. These measure different delays. Recording the glove may not reliably reveal when a small vibration starts, so explain how motor onset will be observed.

The current implementation has a five-second cooldown between buzzes, which matters when testing repeated errors.

## 4. Strengthen the mudra evaluation

The **80% accuracy target is appropriate as a target**, but the test needs a stated number of gestures, repetitions, and recording conditions. “Mudras the system hasn't been tuned on” is ambiguous: test new recordings of gesture classes included in training, rather than entirely unseen classes.

Keep test recordings separate from training and threshold tuning. Count missed holds and withheld predictions explicitly, so accuracy is not calculated only on the easy cases where the system provides an answer.

For the haptic tests, include **correctly performed holds** as well as incorrect ones. Otherwise, the test can measure successful alerts but not unwanted buzzing. Teacher-reviewed examples would provide an independent reference for whether the correction is appropriate.

## 5. Complete the hardware design and constraint checks

The form explicitly requests a hardware sketch. The flowchart includes an “Arduino + Haptic Glove” box, but does not show the glove layout or circuitry. A simple sketch showing motor placement, controller, motor drivers, power source, and connection to the computer would address that requirement.

The test plan lacks a direct check of the **$50 budget**. An itemized parts-cost total would suffice.

“Using only a standard consumer webcam” should clarify that the system also needs a computer and microphone, potentially the laptop's built-in microphone.

## 6. Adjust criteria that are stronger than their tests support

- **“Without … losing body tracking” for 30 minutes** is an absolute requirement. Brief tracking loss is plausible during occlusion or spins. Separate application reliability from tracking availability, and define acceptable tracking coverage or recovery time.
- **Glove mass below 150 g** does not by itself demonstrate unrestricted movement. The planned mudra-formation test helps; comparing bare-hand and gloved performance would also reveal whether the glove interferes with camera tracking.

## 7. Reconcile the diagram with the proposal and implementation

The diagram communicates the architecture well, but contains older or unsupported details.

| Diagram detail | Review finding |
|---|---|
| “< 50 ms total” for haptic feedback | Unsupported as an end-to-end claim with approximately 1.5-second video chunks. |
| “29 mudras · 4,350 rows” | Reflects an earlier dataset stage; the methods describe subsequent expansion. Confirm the dataset intended for this submission. |
| Threshold example of 20% | The current prototype uses 85% as a testing threshold. A generic “configured threshold” would avoid an unnecessary discrepancy. |
| Finger-specific feedback | Shows the intended enhancement beyond the current thumb-only trigger. |

## 8. Clarify expression analysis and performance scores

The rasa module estimates expression categories from MediaPipe blendshape outputs; it does not directly measure facial muscle activation.

“Overall performance percentages” should not imply validated measures of Kathak proficiency. Rasa and tatkaar are currently excluded from the overall score.

## 9. Check bibliography details and formatting

The bibliography exceeds the five-reference minimum and covers relevant areas. **Each citation has not been independently verified in this review.**

The extracted text contains literal `<u>` tags, so check their appearance in Word. Verify publication details, access dates, and statements such as “full text read.”

## Revision priorities

The submission already establishes a meaningful problem and a credible engineering approach. The highest-value revisions are to clarify prototype versus planned functionality, align tests with criteria, and reconcile the diagram's claims. These changes would make the proposal easier for a committee to trust without adding more features.
