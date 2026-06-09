#
dataset_list_orig = [
    # ==============================================================================
    # Translated
    ["nbertagnolli/counsel-chat", ["questionTitle", "questionText", "answerText"]],
    ["ShenLab/MentalChat16K", ["instruction", "input", "output"]],
    ["amaye15/suicide-descriptions", ["text"]],
    ["marmikpandya/mental-health", ["input", "instruction", "output"]],
    ["usham/mental-health-companion-new", ["input", "instruction", "output"]],
    # ==============================================================================
    # Not yet translated
    # ["Amod/mental_health_counseling_conversations", ["Context", "Response"]],
    # ["jsfactory/mental_health_reddit_posts", ["body"]],
    # ["heliosbrahma/mental_health_chatbot_dataset", ["text"]],
    # ["solomonk/reddit_mental_health_posts", ["body", "title"]],
    # ["kshitij230/emotional-support", ["user_input", "ai_response", "reflection"]],
    # ==============================================================================
    # specific format:
    #   ["facebook/empathetic_dialogues", ["prompt", "utterance"]],
    # field `conversations` is stored as json:
    #   ["hllzmz/synthetic-mental-health-convos", ["conversations"]]
]

dataset_list = [
    # ==============================================================================
    # Translated
    ["nbertagnolli/counsel-chat", ["questionText"]],
    ["ShenLab/MentalChat16K", ["input"]],
    ["amaye15/suicide-descriptions", ["text"]],
    ["marmikpandya/mental-health", ["input"]],
    ["usham/mental-health-companion-new", ["input"]],
    # ==============================================================================
    # Not yet translated
    # ["Amod/mental_health_counseling_conversations", ["Context", "Response"]],
    # ["jsfactory/mental_health_reddit_posts", ["body"]],
    # ["heliosbrahma/mental_health_chatbot_dataset", ["text"]],
    # ["solomonk/reddit_mental_health_posts", ["body", "title"]],
    # ["kshitij230/emotional-support", ["user_input", "ai_response", "reflection"]],
    # ==============================================================================
    # specific format:
    #   ["facebook/empathetic_dialogues", ["prompt", "utterance"]],
    # field `conversations` is stored as json:
    #   ["hllzmz/synthetic-mental-health-convos", ["conversations"]]
]
