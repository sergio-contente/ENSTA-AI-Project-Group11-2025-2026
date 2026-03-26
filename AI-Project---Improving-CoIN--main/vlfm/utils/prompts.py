LLaVa_TARGET_OBJECT_IS_DETECTED = """Describe the {target_object} in the provided image."""

LLava_REDUCE_FALSE_POSITIVE = """Is the object outlined with a red border in this image a {target_object}? You must answer only with Yes, No, or ?=I don't know."""

LLM_IS_THIS_THE_TARGET_IMAGE_HUMAN_FEEDBACK_ORACLE_V1 = """You are an intelligent embodied agent equipped with an RGB sensor, object detector, and a VQA model. Your task is to explore an indoor environment and find a specific target {target_object} based on the provided description.

Target {target_object} Description:
<start_target_object_description>
{target_image_description}
<end_target_object_description>

Detected {target_object} Attributes and its surroundings:
{list_of_self_questioner_detected_attributes}

Task: Answer the following question in YAML format, explaining the most significant difference between the target {target_object} and the detected {target_object}, as if you were a human.

Ensure your output follows the following format exactly:
First, provide your step-by-step reasoning.
Then, output the final result enclosed in the YAML tags.

<reasoning step-by-step>

YAML_START
Question: "Is this the target {target_object}?"
Answer: "No, this is not the {target_object} I'm looking for, <explain the difference using the most significant attribute>"
YAML_END"""


LLM_FACTS_UPDATER_AFTER_IS_THIS_TARGET_OBJECT_ORACLE_QUESTION_V1 = """
You are an intelligent embodied agent tasked with finding a specific target {target_object}.

You know the following facts about the target {target_object}:
<START_TARGET_PICTURE_FACTS> {facts_about_the_target_picture} <END_TARGET_PICTURE_FACTS> 

You recently detected another picture and asked to the human several question:
<START_OF_ORACLE_ANSWER>
{oracle_questions_answer}
<END_OF_ORACLE_ANSWER>

Task: Update the target facts with this new information. Be concise. Do not include information that are uncertain.

Ensure your output follows the following format exactly:
First, provide your step-by-step reasoning for updating the facts.
Then, output the final result enclosed in the YAML tags.

<reasoning step-by-step>

YAML_START
facts: <updated facts as a single text line>
YAML_END # must be present to get the information back"""


UNCERTAIN_ANSWER_CHOICE_PLACEHOLDER = "?=I don't know."
LLM_SELF_QUESTIONER_GIVEN_DISTRACTOR_DESCRIPTION = """
You are an intelligent embodied agent equipped with an RGB sensor, an object detector, and a Visual Question Answering (VQA) model. Your task is to explore an indoor environment to find a specific target {target_object}.
The detector has identified a {target_object}. The VQA model has provided the following description of the scene:

<START_OF_DESCRIPTION>
{distractor_object_description}
<END_OF_DESCRIPTION>

Based on your past interactions with the user, you know the following facts about the target picture: <START_TARGET_PICTURE_FACTS> {facts_about_the_target_picture} <END_TARGET_PICTURE_FACTS> 

Assume that the detected image description contains hallucinations. Your goal is to extract a set of specific atomic attributes from the description and rigorously verify them.

Vision-Language Models suffer from severe "Yes-Bias". To counteract this, you must be highly skeptical. 
Formally:
- Decompose the VQA model's initial description into a structured set of single attribute claims (e.g., color, material, location).
- Generate targeted follow-up questions to verify each extracted attribute.
- To counter "Yes-Bias", include mutually exclusive contrast questions or ask about opposite traits to verify the truth (e.g., if checking if a dresser is white, also ask "Is the dresser brown, dark, or wood-colored?").
- Every question MUST be in this exact format: "<question content>? You must answer only with Yes, No, or {uncertain_answer_choice_placeholder}" This allows us to access likelihood for the answers.


Ensure your output follows the following format exactly:
First, provide your step-by-step reasoning for generating the structured attributes and targeted questions.
Then, output the final result enclosed in the YAML tags.

<reasoning step-by-step>

YAML_START # must be present to get the information back
attributes_of_the_image:
    <attribute name>: "<attribute value>" # summarize all the known attributes from the description, enclosed in " "

questions_for_detected_object: # question for the detected object, if any
    <Question number>:  "<question>? You must answer only with Yes, No, or ?=I don't know."
reasoning_for_detected_object:
    <Question number>: <reasoning>
YAML_END # must be present to get the information back"""


LMM_RETRIEVE_FACTS_FROM_DESCRIPTION = """
You are an intelligent embodied agent equipped with an RGB sensor, an object detector, and a Visual Question Answering (VQA) model. 
Your task is to explore an indoor environment to find a specific target {target_object}.
The detector has identified a {target_object}. The VQA model has provided the following description of the scene:

<START_OF_DESCRIPTION>
{distractor_object_description}
<END_OF_DESCRIPTION>

Based on your past interactions with the user, you know the following facts about the target picture: 
<START_TARGET_PICTURE_FACTS> {facts_about_the_target_picture} <END_TARGET_PICTURE_FACTS> 

Your task is to:
- ask more question to the VQA model on the detected {target_object} to maximize information gain.

Ensure your output follows the following format exactly:
First, provide your step-by-step reasoning for asking questions.
Then, output the final result enclosed in the YAML tags.

<reasoning step-by-step>

YAML_START # must be present to get the information back
attributes_of_the_image:
    <attribute name>: "<attribute value>" # summarize all the known attributes from the description, enclosed in " "
questions:
        <question_number>: "<question content>"
YAML_END # must be present to get the information back"""


LLM_REFINE_DETECTED_OBJECT_DESCRIPTION = """
You are an intelligent embodied agent equipped with an RGB sensor, an object detector, and a Visual Question Answering (VQA) model. 
Your task is to refine an image description based on certainty estimates and user interactions.

Scenario:
The detector has identified a scene with a {target_object}. The VQA model provided this initial scene description:

<START_OF_DESCRIPTION>
{distractor_object_description}
<END_OF_DESCRIPTION>


Questions asked and responses:
<START_QUESTION_AND_RESPONSES>
{list_questions_answers_uncertainty_labels}
<END_QUESTION_AND_RESPONSES>

Task:
Using the questions/answer pairs with uncertainty labels, refine the image description. 
Since we have to find a {target_object}, put emphasis on it. 

CRITICAL RULE (UNCERTAINTY PRUNING): You must STRICTLY FILTER the semantic state. Any attribute or feature whose question was answered with uncertainty (e.g., "?=I don't know") MUST BE COMPLETELY REMOVED from the refined description. Do not mention that it is uncertain; simply drop it so that only verified, high-confidence attributes remain in the final text. If any answer explicitly contradicts what the {target_object} should look like, you must still include that contradicting verified trait.

Ensure your response follows the format below exactly:
First, provide your step-by-step reasoning for the extraction and pruning.
Then, output the final result enclosed in the YAML tags.

<reasoning step-by-step>

YAML_START # must be present to get the information back
attributes_of_the_image:
    <attribute name>: "<attribute value>" # summarize all the known attributes from the description, enclosed in " "
image_description_refined: <insert refined description>  # Ensure that the string does not contain a newline (\n) after the tag image_description_refined:
YAML_END # must be present to get the information back"""

LLM_SIMILARITY_SCORE_AND_QUESTION_TO_TARGET = """
You are an intelligent agent equipped with an RGB sensor, object detector, and Visual Question Answering (VQA) model.
Your goal is to identify a target {target_object} based on a scene description and prior knowledge of the target.

Scenario:
The object detector has identified a scene containing a {target_object}, and the VQA model has provided the following description:

<START_OF_DESCRIPTION>
{distractor_object_description}
<END_OF_DESCRIPTION>

Target object information: 
Based on previous interactions, you know the target picture has the following characteristics:
<START_TARGET_PICTURE_FACTS>
{facts_about_the_target_picture}
<END_TARGET_PICTURE_FACTS> 

Task:
1. Similarity analysis.
Analyze how closely the detected scene description aligns with the known facts about the target {target_object}. Provide a similarity score between 0 and 10, where:
- 0 to 4 = Contradiction or Weak Match: The detected object explicitly contradicts the target facts on primary features, or lacks sufficient basic similarities.
- 5 to 6 = Generic Match / Ambiguous: The object shares the base category or generic traits, but lacks the specific, unique identifying features of the target (meaning they are either missing or were pruned due to uncertainty).
- 7 to 10 = Verified Match: The scene description contains explicitly verified matches for the UNIQUE defining features of the target with no contradictions.
- If no information about the target is available, the score should be -1.

CRITICAL SCORING RULE: Do not assign a score of 7 or higher if you only have a generic match. A score of 7+ requires positive, verified identification of unique target traits.

CRITICAL CONTRADICTION RULE: Any contradiction regarding a primary feature (e.g., color, material, handles, specific shape, distinctive patterns) MUST result in a score of 4 or lower. Never "average" a contradiction with a generic category match. (e.g., If both are dressers but one has metal handles and the target has white handles, the contradiction overrides the category match -> Score MUST be 0-4).

2. Question Generation:
- The question is for the target object, not the detected one.
- Ask exactly one specific, relevant, and human-answerable question related to the target object that maximizes information gain for identifying the target {target_object}.
- CRITICAL UNCERTAINTY RULE: If the scene description explicitly notes any attribute as 'uncertain', 'unverified', or 'ambiguous', your question MUST target that exact attribute.
- Do not ask speculative or irrelevant questions.
- The question should be grounded in observable or known details from the scene, focusing on key characteristics that can help confirm or refute the identity of the target object.

Ensure your response follows the format below exactly:
First, provide your step-by-step reasoning for the similarity score and questions.
Then, output the final result enclosed in the YAML tags.

<reasoning step-by-step>

YAML_START # must be present to get the information back
similarity_score: <similarity score>
questions:
    <question_number>: <question_content>
YAML_END # must be present to get the information back"""
